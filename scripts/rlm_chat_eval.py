"""Evaluate an RLM chat (SFT) checkpoint on ChatCORE benchmarks."""

import argparse
from functools import partial
import json
import logging
import math
import os
from pathlib import Path
import time

from nanochat.paths import configure_runtime_paths, get_run_dir

ALL_TASKS = ("ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval")
BASELINE_ACCURACIES = {
    "ARC-Easy": 0.25,
    "ARC-Challenge": 0.25,
    "MMLU": 0.25,
    "GSM8K": 0.0,
    "HumanEval": 0.0,
}

logger = logging.getLogger(__name__)


def _parse_tasks(spec: str | None) -> tuple[str, ...]:
    if spec is None:
        return ALL_TASKS
    names = tuple(part.strip() for part in spec.split("|"))
    if not names or any(not name for name in names):
        raise ValueError("Task specification cannot contain empty task names")
    unknown = [name for name in names if name not in BASELINE_ACCURACIES]
    if unknown:
        raise ValueError(f"Unknown ChatCORE task(s): {', '.join(unknown)}")
    return names


def _build_task(task_name: str):
    from tasks.arc import ARC
    from tasks.gsm8k import GSM8K
    from tasks.humaneval import HumanEval
    from tasks.mmlu import MMLU

    factories = {
        "HumanEval": HumanEval,
        "MMLU": partial(MMLU, subset="all", split="test"),
        "ARC-Easy": partial(ARC, subset="ARC-Easy", split="test"),
        "ARC-Challenge": partial(ARC, subset="ARC-Challenge", split="test"),
        "GSM8K": partial(GSM8K, subset="main", split="test"),
    }
    return factories[task_name]()


def run_categorical_eval(
    task_name: str,
    task_object,
    tokenizer,
    model,
    batch_size: int,
    max_problems: int | None = None,
    step_hook=None,
) -> tuple[int, int]:
    from nanochat.rlm_inference import predict_next_logits

    bos = tokenizer.get_bos_token_id()
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    num_batches = -(-num_problems // batch_size)
    letter_to_id_cache: dict[str, int] = {}
    num_passed = 0
    total = 0

    for batch_idx in range(num_batches):
        if step_hook is not None:
            step_hook(batch_idx + 1)
        i0 = batch_idx * batch_size
        i1 = min((batch_idx + 1) * batch_size, num_problems)
        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conv) for conv in conversations]
        next_logits = predict_next_logits(model, prompt_ids, pad_token_id=bos)

        for idx, conversation in enumerate(conversations):
            letters = conversation["letters"]
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    if len(encoded_letter) != 1:
                        raise ValueError(f"Choice label {letter!r} must encode to a single token")
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])
            focus_logits = next_logits[idx, letter_ids]
            argmax_letter_id = focus_logits.argmax(dim=-1).item()
            predicted_letter = letters[argmax_letter_id]
            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

        if (batch_idx + 1) % 50 == 0 or batch_idx + 1 == num_batches:
            logger.info(
                "%s progress: %d/%d (%.2f%%)",
                task_name, num_passed, total, 100.0 * num_passed / total if total else 0.0,
            )

    return num_passed, total


def run_generative_eval(
    task_name: str,
    task_object,
    tokenizer,
    engine,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int = 42,
    max_problems: int | None = None,
    step_hook=None,
) -> tuple[int, int]:
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    num_passed = 0
    total = 0

    for i in range(num_problems):
        if step_hook is not None:
            step_hook(i + 1)
        conversation = task_object[i]
        encoded_prompt = tokenizer.render_for_completion(conversation)
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        )
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        num_passed += int(any(outcomes))
        total += 1

        if total % 25 == 0 or total == num_problems:
            logger.info(
                "%s progress: %d/%d (%.2f%%)",
                task_name, num_passed, total, 100.0 * num_passed / total,
            )

    return num_passed, total


def compute_chatcore(results: dict[str, float]) -> float | None:
    if not all(task_name in results for task_name in ALL_TASKS):
        return None
    centered_total = 0.0
    for task_name in ALL_TASKS:
        baseline = BASELINE_ACCURACIES[task_name]
        centered_total += (results[task_name] - baseline) / (1.0 - baseline)
    return centered_total / len(ALL_TASKS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=get_run_dir() / "sft" / "checkpoints",
        help="RLM checkpoint directory (default: selected run's sft/checkpoints)",
    )
    parser.add_argument("--step", type=int, help="checkpoint step (default: latest complete)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "-a", "--task-name",
        type=str,
        default=None,
        help="Task name(s). Default = all ChatCORE tasks. Use | to split multiple tasks.",
    )
    parser.add_argument("-t", "--temperature", type=float, default=0.0)
    parser.add_argument("-m", "--max-new-tokens", type=int, default=512)
    parser.add_argument("-n", "--num-samples", type=int, default=1)
    parser.add_argument("-k", "--top-k", type=int, default=50)
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size for categorical evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-x", "--max-problems", type=int, default=None, help="Max problems to evaluate per task")
    parser.add_argument("--output", type=Path, help="Write exact result JSON to a new file in an existing directory")
    args = parser.parse_args()

    try:
        task_names = _parse_tasks(args.task_name)
    except ValueError as exc:
        parser.error(str(exc))
    if args.batch_size <= 0 or args.num_samples <= 0 or args.max_new_tokens <= 0:
        parser.error("Require positive batch-size, num-samples, and max-new-tokens")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("Require a finite nonnegative temperature")
    if args.top_k is not None and args.top_k <= 0:
        parser.error("top-k must be positive when supplied")
    if args.max_problems is not None and args.max_problems <= 0:
        parser.error("max-problems must be positive when supplied")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("RLM chat evaluation supports one process; do not launch with torchrun")
    if args.output is not None:
        args.output = args.output.expanduser()
        if args.output.exists() or args.output.is_symlink():
            parser.error("--output must name a new file; existing files are never overwritten")
        if not args.output.parent.is_dir():
            parser.error("--output parent directory must already exist")

    os.environ["NANOCHAT_ATTENTION_IMPL"] = "sdpa"
    os.environ.setdefault("NANOCHAT_DTYPE", "float32" if args.device == "cpu" else "bfloat16")
    os.environ.setdefault("NANOCHAT_CUDA_MEMORY_LIMIT_GB", "80")
    os.environ.setdefault("NANOCHAT_MIN_SYSTEM_AVAILABLE_GB", "16")
    configure_runtime_paths()

    import torch
    from nanochat.common import compute_cleanup, compute_init
    from nanochat.rlm_inference import RLMEngine, load_rlm_checkpoint
    from nanochat.rlm_training import MemoryMonitor, cuda_metrics, synchronize
    from nanochat.training_runtime import check_training_memory

    _, _, _, _, device = compute_init(args.device)
    try:
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        with MemoryMonitor(os.environ["NANOCHAT_MIN_SYSTEM_AVAILABLE_GB"]) as monitor:
            def guard(index: int) -> None:
                check_training_memory(device, index)
                monitor.require_available(index)

            guard(0)
            checkpoint_dir = args.checkpoint_dir.expanduser()
            model, tokenizer, metadata = load_rlm_checkpoint(checkpoint_dir, args.step, device)
            engine = RLMEngine(model, tokenizer)
            synchronize(device)
            load_seconds = time.monotonic() - started
            load_memory = cuda_metrics(device)
            guard(0)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            evaluation_started = time.monotonic()
            accuracies: dict[str, float] = {}
            task_details: dict[str, dict] = {}
            for task_name in task_names:
                task_started = time.monotonic()
                logger.info("Starting %s evaluation", task_name)
                task_object = _build_task(task_name)
                if task_object.eval_type == "categorical":
                    passed, total = run_categorical_eval(
                        task_name,
                        task_object,
                        tokenizer,
                        model,
                        batch_size=args.batch_size,
                        max_problems=args.max_problems,
                        step_hook=guard,
                    )
                elif task_object.eval_type == "generative":
                    passed, total = run_generative_eval(
                        task_name,
                        task_object,
                        tokenizer,
                        engine,
                        num_samples=args.num_samples,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        seed=args.seed,
                        max_problems=args.max_problems,
                        step_hook=guard,
                    )
                else:
                    raise ValueError(f"Unsupported task evaluation type: {task_object.eval_type}")
                synchronize(device)
                task_seconds = time.monotonic() - task_started
                acc = passed / total
                baseline = BASELINE_ACCURACIES[task_name]
                centered = (acc - baseline) / (1.0 - baseline)
                accuracies[task_name] = acc
                task_details[task_name] = {
                    "eval_type": task_object.eval_type,
                    "passed": passed,
                    "total": total,
                    "accuracy": acc,
                    "baseline": baseline,
                    "centered_accuracy": centered,
                    "seconds": task_seconds,
                }
                logger.info(
                    "%s accuracy: %.2f%% (%d/%d, %.2fs)",
                    task_name, 100.0 * acc, passed, total, task_seconds,
                )

            synchronize(device)
            evaluation_seconds = time.monotonic() - evaluation_started
            evaluation_memory = cuda_metrics(device)
            monitor.require_available(len(task_names))

        chatcore = compute_chatcore(accuracies)
        if chatcore is not None:
            logger.info("ChatCORE metric: %.4f", chatcore)

        resources = {
            "load_seconds": load_seconds,
            "evaluation_seconds": evaluation_seconds,
            "total_seconds": time.monotonic() - started,
            "minimum_system_available_gb": monitor.minimum_observed / 1e9,
            "peak_cuda_allocated_gb": max(
                load_memory["cuda_peak_allocated_gb"], evaluation_memory["cuda_peak_allocated_gb"],
            ),
            "peak_cuda_reserved_gb": max(
                load_memory["cuda_peak_reserved_gb"], evaluation_memory["cuda_peak_reserved_gb"],
            ),
            "load_memory": load_memory,
            "evaluation_memory": evaluation_memory,
        }
        result = {
            "checkpoint_dir": str(checkpoint_dir),
            "step": metadata["step"],
            "stage": metadata.get("stage"),
            "settings": {
                "tasks": list(task_names),
                "batch_size": args.batch_size,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "num_samples": args.num_samples,
                "top_k": args.top_k,
                "seed": args.seed,
                "max_problems": args.max_problems,
            },
            "results": accuracies,
            "chatcore": chatcore,
            "tasks": task_details,
            "resources": resources,
        }
        serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(serialized)
        print(serialized, end="")
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()

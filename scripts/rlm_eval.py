"""Compare per-cycle RLM text loss with intact, zero, and shuffled memory."""

import argparse
from enum import Enum
import json
import os
from pathlib import Path
import time


class EvaluationSampler(str, Enum):
    BOS_BESTFIT = "bos_bestfit"
    CONTIGUOUS_DOCUMENTS = "contiguous_documents"

    def __str__(self):
        return self.value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--step", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampler", type=EvaluationSampler, choices=list(EvaluationSampler), default=EvaluationSampler.BOS_BESTFIT)
    parser.add_argument("--output", type=Path, help="Write exact result JSON to a new file in an existing directory")
    args = parser.parse_args()
    if args.batch_size < 2 or args.batches <= 0 or args.cycles < 2:
        parser.error("Require batch-size >= 2, batches > 0, and cycles >= 2")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("RLM evaluation supports one process; do not launch with torchrun")
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

    import torch
    from nanochat.common import compute_cleanup, compute_init
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
    from nanochat.rlm_data import contiguous_validation_batches
    from nanochat.rlm_inference import evaluate_memory_controls, load_rlm_checkpoint
    from nanochat.rlm_training import MemoryMonitor, cuda_metrics, synchronize
    from nanochat.training_runtime import check_training_memory

    _, _, _, _, device = compute_init(args.device)
    try:
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        with MemoryMonitor(os.environ["NANOCHAT_MIN_SYSTEM_AVAILABLE_GB"]) as monitor:
            check_training_memory(device, 0)
            monitor.require_available(0)
            model, tokenizer, metadata = load_rlm_checkpoint(args.checkpoint_dir.expanduser(), args.step, device)
            synchronize(device)
            load_seconds = time.monotonic() - started
            load_memory = cuda_metrics(device)
            check_training_memory(device, 0)
            monitor.require_available(0)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            evaluation_started = time.monotonic()
            episode_length = args.cycles * model.config.text_length
            if args.sampler is EvaluationSampler.CONTIGUOUS_DOCUMENTS:
                batches = contiguous_validation_batches(tokenizer, args.batch_size, episode_length, device)
            else:
                batches = tokenizing_distributed_data_loader_bos_bestfit(
                    tokenizer, args.batch_size, episode_length, "val", device=args.device,
                )

            def checked_batches():
                for index, batch in enumerate(batches, start=1):
                    check_training_memory(device, index)
                    monitor.require_available(index)
                    yield batch

            result = evaluate_memory_controls(model, checked_batches(), args.batches, args.cycles, args.seed).to_dict()
            synchronize(device)
            evaluation_seconds = time.monotonic() - evaluation_started
            evaluation_memory = cuda_metrics(device)
            monitor.require_available(args.batches)
        resources = {
            "load_seconds": load_seconds,
            "evaluation_seconds": evaluation_seconds,
            "total_seconds": time.monotonic() - started,
            "minimum_system_available_gb": monitor.minimum_observed / 1e9,
            "peak_cuda_allocated_gb": max(load_memory["cuda_peak_allocated_gb"], evaluation_memory["cuda_peak_allocated_gb"]),
            "peak_cuda_reserved_gb": max(load_memory["cuda_peak_reserved_gb"], evaluation_memory["cuda_peak_reserved_gb"]),
            "load_memory": load_memory,
            "evaluation_memory": evaluation_memory,
        }
        result.update({"step": metadata["step"], "split": "val", "sampler": args.sampler.value})
        if args.sampler is EvaluationSampler.CONTIGUOUS_DOCUMENTS:
            result["sampler_details"] = {
                "episode_selection": "one_prefix_per_document",
                "minimum_tokens_including_bos": episode_length + 1,
                "short_documents": "skipped",
                "final_partial_batch": "discarded",
                "validation_passes": 1,
            }
        result["resources"] = resources
        serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            # Exclusive creation is checked again atomically at publication so
            # a file created during evaluation cannot be overwritten either.
            with args.output.open("x", encoding="utf-8") as output:
                output.write(serialized)
        print(serialized, end="")
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()

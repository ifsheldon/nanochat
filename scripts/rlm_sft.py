"""Supervised fine-tuning of a pretrained RLM checkpoint on chat conversations.

NANOCHAT_RUN_NAME selects training_run/<name>; its tokenizer must match the
pretraining checkpoint. Output defaults to <run>/sft. Every cycle keeps the
pretraining layout [memory] [EOM] [text] [summary]; only assistant tokens and
the EOM readouts that predict an assistant token are supervised.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

from nanochat.paths import configure_runtime_paths, get_run_dir

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=get_run_dir() / "train" / "checkpoints",
                        help="pretraining checkpoint directory (default: selected run's train/checkpoints)")
    parser.add_argument("--step", type=int, help="pretraining step to load (default: latest complete)")
    parser.add_argument("--output-dir", type=Path, default=get_run_dir() / "sft",
                        help="new SFT output directory (default: selected run's sft directory)")
    parser.add_argument("--device-type", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cycles", type=int, default=4, help="text chunks per episode row")
    parser.add_argument("--device-batch-size", type=int, default=8, help="episode rows per forward/backward")
    parser.add_argument("--total-batch-size", type=int, default=None, help="text positions per optimizer update (default: inherit from pretraining)")
    parser.add_argument("--num-iterations", type=int, default=-1, help="optimizer updates (-1 = one epoch of the mixture)")
    parser.add_argument("--load-optimizer", type=int, default=1, help="warm-start optimizer moments from the pretraining checkpoint (0=no, 1=yes)")
    parser.add_argument("--embedding-lr", type=float, default=None, help="default: inherit from pretraining")
    parser.add_argument("--unembedding-lr", type=float, default=None, help="default: inherit from pretraining")
    parser.add_argument("--matrix-lr", type=float, default=None, help="default: inherit from pretraining")
    parser.add_argument("--scalar-lr", type=float, default=None, help="default: inherit from pretraining")
    parser.add_argument("--init-lr-frac", type=float, default=0.8, help="initial LR as fraction of the pretraining base LR")
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--warmdown-ratio", type=float, default=0.5)
    parser.add_argument("--final-lr-frac", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=50, help="evaluate held-out loss every N updates (-1 = disable)")
    parser.add_argument("--eval-batches", type=int, default=16, help="held-out microbatches per evaluation")
    parser.add_argument("--save-every", type=int, default=50, help="model-only checkpoint interval; the final checkpoint also saves the optimizer (-1 = final only)")
    parser.add_argument("--mmlu-epochs", type=int, default=3)
    parser.add_argument("--gsm8k-epochs", type=int, default=4)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.cycles < 1 or args.device_batch_size < 1:
        parser.error("cycles and device batch size must be positive")
    if args.num_iterations != -1 and args.num_iterations <= 0:
        parser.error("--num-iterations must be positive or -1 for one epoch")
    if args.eval_every == 0 or args.save_every == 0 or args.eval_batches <= 0:
        parser.error("eval/save intervals must be positive or -1, and eval batches must be positive")
    if not 0 < args.init_lr_frac <= 1:
        parser.error("--init-lr-frac must be in (0, 1]")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("RLM SFT supports one process; do not launch with torchrun")
    return args


def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def run(args):
    import torch
    from nanochat.checkpoint_manager import find_last_step, save_checkpoint
    from nanochat.common import COMPUTE_DTYPE, compute_init
    from nanochat.rlm_inference import load_rlm_checkpoint, tokenizer_fingerprint
    from nanochat.rlm_sft import evaluate_sft_loss, sft_update
    from nanochat.rlm_training import cuda_metrics, synchronize, write_json
    from nanochat.sft_dataloader import sft_data_loader_bos_bestfit
    from nanochat.sft_training import SFTProgress, sft_lr_multiplier
    from nanochat.training_runtime import check_training_memory
    from tasks.common import TaskMixture
    from tasks.gsm8k import GSM8K
    from tasks.mmlu import MMLU
    from tasks.smoltalk import SmolTalk

    if COMPUTE_DTYPE not in (torch.bfloat16, torch.float32):
        raise ValueError("RLM SFT supports BF16 or FP32 compute")
    _, _, _, _, device = compute_init(args.device_type)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    check_training_memory(device, 0)

    checkpoint_dir = args.checkpoint_dir.expanduser().absolute()
    step_loaded = find_last_step(checkpoint_dir) if args.step is None else args.step
    model, tokenizer, pretrain_metadata = load_rlm_checkpoint(checkpoint_dir, step_loaded, device)
    model.train()
    pretrain_config = pretrain_metadata["training_config"]
    for name in ("total_batch_size", "embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr"):
        if getattr(args, name) is None:
            setattr(args, name, pretrain_config[name])
            logger.info("Inherited %s=%s from pretraining", name, pretrain_config[name])
    text_length = model.config.text_length
    episode_length = args.cycles * text_length
    microbatch = args.device_batch_size * episode_length
    if args.total_batch_size % microbatch:
        raise ValueError(f"total batch {args.total_batch_size} must be divisible by {microbatch} text positions per microbatch")
    accumulation = args.total_batch_size // microbatch

    # Match pretraining's batch-scaled learning rates; pretraining decays weight decay to zero.
    batch_scale = math.sqrt(args.total_batch_size / 524_288)
    optimizer = model.setup_optimizer(
        embedding_lr=args.embedding_lr * batch_scale, unembedding_lr=args.unembedding_lr * batch_scale,
        matrix_lr=args.matrix_lr * batch_scale, scalar_lr=args.scalar_lr * batch_scale, weight_decay=0.0,
    )
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    optimizer_source = None
    if args.load_optimizer:
        optimizer_path = checkpoint_dir / f"optim_{step_loaded:06d}_rank0.pt"
        if optimizer_path.is_file():
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
            optimizer_source = str(optimizer_path)
            logger.info("Loaded optimizer moments from %s", optimizer_path)
        else:
            logger.warning("Optimizer checkpoint not found at %s; starting with fresh moments", optimizer_path)
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * args.init_lr_frac
        group["initial_lr"] = group["lr"]
        if group["kind"] == "muon":
            group["weight_decay"] = 0.0
    forward_cycle = torch.compile(model.forward_cycle, dynamic=False) if args.compile else model.forward_cycle

    train_dataset = TaskMixture([
        SmolTalk(split="train"),
        *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)],
        *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)],
    ])
    val_dataset = TaskMixture([
        SmolTalk(split="test"),
        MMLU(subset="all", split="test", stop=5200),
        GSM8K(subset="main", split="test", stop=420),
    ])
    logger.info("Training mixture: %d conversations (MMLU x%d, GSM8K x%d)", len(train_dataset), args.mmlu_epochs, args.gsm8k_epochs)

    def data_loader(dataset):
        return sft_data_loader_bos_bestfit(dataset, tokenizer, args.device_batch_size, episode_length, device)

    train_loader = data_loader(train_dataset)
    progress = SFTProgress(len(train_dataset), args.num_iterations)
    sft_lr_multiplier(0.0, args.warmup_ratio, args.warmdown_ratio, args.final_lr_frac)

    output_checkpoints = args.output_dir / "checkpoints"
    metrics_path = args.output_dir / "updates.jsonl"
    contract = {
        "source_checkpoint": {"dir": str(checkpoint_dir), "step": step_loaded, "optimizer": optimizer_source},
        "model_config": asdict(model.config), "sft_config": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "accumulation_steps": accumulation, "training_conversations": len(train_dataset),
        "supervision": "assistant_text_targets_and_eom_readouts_of_assistant_tokens",
        "tokenizer_sha256": tokenizer_fingerprint(), "compute_dtype": str(COMPUTE_DTYPE),
        "torch": torch.__version__, "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(args.output_dir / "run.json", contract)
    logger.info("%s", json.dumps(contract))

    def metadata(step, val_loss):
        return {
            "model_type": "rlm", "model_config": asdict(model.config), "step": step,
            "stage": "sft", "val_loss": val_loss,
            "consumed_conversations": progress.consumed, "epoch": progress.epoch,
            "tokenizer_sha256": contract["tokenizer_sha256"], "run_contract": contract,
            "memory_contract": {**pretrain_metadata.get("memory_contract", {}),
                                "loss": "assistant_text_and_assistant_eom_targets", "gradient_horizon": args.cycles},
        }

    val_loss = None
    step = 0
    training_seconds = 0.0
    while True:
        check_training_memory(device, step)
        last_step = progress.done
        if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
            val_loss = evaluate_sft_loss(forward_cycle, data_loader(val_dataset), args.eval_batches, text_length, args.cycles)
            logger.info("%s", json.dumps({"step": step, "val_loss": val_loss}))
        if last_step or (args.save_every > 0 and step > 0 and step % args.save_every == 0):
            synchronize(device)
            save_checkpoint(output_checkpoints, step, model.state_dict(),
                            optimizer.state_dict() if last_step else None, metadata(step, val_loss))
        if last_step:
            break
        lrm = sft_lr_multiplier(progress.fraction, args.warmup_ratio, args.warmdown_ratio, args.final_lr_frac)
        momentum = get_muon_momentum(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = momentum
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        metrics, data_state = sft_update(
            model, forward_cycle, optimizer, train_loader, accumulation, step,
            device_batch_size=args.device_batch_size, cycles=args.cycles,
        )
        progress.complete_update(data_state)
        step = progress.step
        training_seconds += metrics["duration_seconds"]
        metrics.update({
            "lr_multiplier": lrm, "progress": progress.fraction, "epoch": progress.epoch,
            "consumed_conversations": progress.consumed,
            "text_positions_per_second": args.total_batch_size / metrics["duration_seconds"],
            "training_seconds": training_seconds, **cuda_metrics(device),
        })
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        logger.info("%s", json.dumps(metrics))
        logger.info("Update %d (%.1f%%): loss %.4f, %.2f s", step, 100 * progress.fraction, metrics["loss"], metrics["duration_seconds"])
    report = {"status": "completed", "completed_updates": step, "val_loss": val_loss,
              "consumed_conversations": progress.consumed, "epoch": progress.epoch,
              "training_seconds": training_seconds, "checkpoint_dir": str(output_checkpoints)}
    write_json(args.output_dir / "result.json", report)
    logger.info("%s", json.dumps(report))


def main():
    args = parse_args()
    configure_runtime_paths()
    os.environ.setdefault("NANOCHAT_ATTENTION_IMPL", "sdpa")
    os.environ.setdefault("NANOCHAT_DTYPE", "bfloat16" if args.device_type == "cuda" else "float32")
    os.environ.setdefault("NANOCHAT_MIN_SYSTEM_AVAILABLE_GB", "16")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    args.output_dir = args.output_dir.expanduser().absolute()
    if args.output_dir.is_symlink():
        raise ValueError("SFT output must be an owned directory, not a symlink")
    args.output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    try:
        run(args)
    except BaseException as exc:
        from nanochat.rlm_training import write_json
        write_json(args.output_dir / "result.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()

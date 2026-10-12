"""Train or measure recurrent latent memory with explicit text-token budgets.

NANOCHAT_RUN_NAME selects training_run/<name>; NANOCHAT_BASE_DIR overrides that
artifact root, including its tokenizer. Shared data uses NANOCHAT_DATASETS_DIR.
Training output defaults to <run>/train. See dev/spark-training/SPARK_RLM.md for contracts.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

from nanochat.paths import configure_runtime_paths, get_run_dir
from nanochat.rlm_policy import InputEmbeddingFreeze

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=get_run_dir() / "train",
                        help="new training/probe directory (default: selected run's train directory)")
    parser.add_argument("--mode", choices=("probe", "capacity-probe", "train"), default="probe")
    parser.add_argument("--device-type", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--width", type=int, default=None, help="default: depth * 64, rounded to head dimension")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--text-length", type=int, default=1023)
    parser.add_argument("--memory-tokens", type=int, default=512)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--device-batch-size", type=int, default=16)
    parser.add_argument("--total-batch-size", type=int, default=1_047_552, help="real text targets per optimizer update; EOM adds one prediction per cycle")
    parser.add_argument("--num-iterations", type=int, default=None, help="required for training; probe defaults to two updates")
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--warmdown-ratio", type=float, default=0.65)
    parser.add_argument("--final-lr-frac", type=float, default=0.05)
    parser.add_argument("--embedding-lr", type=float, default=0.3)
    parser.add_argument("--unembedding-lr", type=float, default=0.008)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--scalar-lr", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--init-embedding-checkpoint-dir", type=Path, help="fresh runs only: checkpoint directory providing transformer.wte.weight")
    parser.add_argument("--init-embedding-step", type=int, help="published source checkpoint step for input embedding initialization")
    parser.add_argument("--init-embedding-tokenizer", type=Path, help="source tokenizer.pkl; must exactly match the training tokenizer")
    parser.add_argument("--init-embedding-bundle", type=Path, help="fresh runs only: verified embedding-only export directory")
    parser.add_argument("--input-embedding-freeze", type=InputEmbeddingFreeze,
                        choices=list(InputEmbeddingFreeze), default=InputEmbeddingFreeze.NONE,
                        help="none, scheduled (first 90%%), or permanent; EOM and summary inputs stay trainable")
    parser.add_argument("--save-every", type=int, help="training only; numbered checkpoint interval (default: 10), plus update 2 and the final/paused update; 0 disables periodic numbered saves")
    parser.add_argument("--keep-checkpoints", type=int, help="training only: numbered checkpoints to retain (default: 10), in addition to rolling latest")
    parser.add_argument("--checkpoint-mode", choices=("sync", "async"), default="async", help="async overlaps checkpoint writing with the next forward/backward; ordinary save errors warn, but a required 90%% checkpoint failure stops training")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume-from-step", type=int, default=None)
    resume.add_argument("--resume-latest", action="store_true", help="resume the complete model/optimizer state in checkpoints/latest.pt")
    parser.add_argument("--stop-after-step", type=int, default=None, help="save and stop at a completed update without changing schedule")
    args = parser.parse_args()
    if args.save_every is None:
        args.save_every = 10 if args.mode == "train" else 0
    if args.keep_checkpoints is None and args.mode == "train":
        args.keep_checkpoints = 10
    if args.num_iterations is None:
        if args.mode == "train":
            parser.error("training requires an explicit --num-iterations budget")
        args.num_iterations = 2
    resuming = args.resume_from_step is not None or args.resume_latest
    if args.mode == "probe" and (args.num_iterations < 2 or resuming
                                  or args.stop_after_step is not None or args.save_every
                                  or args.keep_checkpoints is not None):
        parser.error("probe requires at least two updates, with no resume, early stop, checkpoint saving, or retention")
    if args.mode == "capacity-probe":
        if resuming or args.stop_after_step is not None or args.save_every or args.keep_checkpoints is not None:
            parser.error("capacity-probe executes one fresh update without resume, early stop, or checkpoint saving")
        if args.input_embedding_freeze is InputEmbeddingFreeze.SCHEDULED:
            parser.error("capacity-probe requires input-embedding-freeze none or permanent")
    if args.save_every < 0:
        parser.error("--save-every cannot be negative")
    if args.keep_checkpoints is not None and args.keep_checkpoints <= 0:
        parser.error("--keep-checkpoints must be positive")
    initialization_args = (
        args.init_embedding_checkpoint_dir, args.init_embedding_step, args.init_embedding_tokenizer,
    )
    if any(value is not None for value in initialization_args):
        if args.init_embedding_bundle is not None:
            parser.error("choose an embedding bundle or source checkpoint, not both")
        if not all(value is not None for value in initialization_args):
            parser.error("embedding initialization requires checkpoint directory, step, and source tokenizer together")
        if args.init_embedding_step < 0:
            parser.error("--init-embedding-step must be nonnegative")
        if resuming:
            parser.error("embedding initialization is fresh-run only; resume restores its saved initialization provenance and trained weights")
    if args.init_embedding_bundle is not None and resuming:
        parser.error("embedding initialization is fresh-run only; resume restores its saved weights")
    if (args.input_embedding_freeze is not InputEmbeddingFreeze.NONE and not resuming
            and args.init_embedding_checkpoint_dir is None and args.init_embedding_bundle is None):
        parser.error("--input-embedding-freeze requires pretrained embedding initialization for a fresh run")
    if args.resume_from_step is not None and args.resume_from_step <= 0:
        parser.error("--resume-from-step must name a completed positive optimizer update")
    if args.stop_after_step is not None and not 0 < args.stop_after_step <= args.num_iterations:
        parser.error("--stop-after-step must be within the configured training horizon")
    if args.depth <= 0 or args.head_dim <= 0 or args.head_dim % 2:
        parser.error("depth must be positive and head dimension must be positive and even")
    if args.text_length <= 0 or args.memory_tokens <= 0:
        parser.error("text length and memory token count must be positive")
    if args.width is None:
        args.width = ((args.depth * 64 + args.head_dim - 1) // args.head_dim) * args.head_dim
    if args.width < 24 or args.width % args.head_dim:
        parser.error("width must be at least 24 and compatible with the attention head dimension")
    return args


def dataset_inventory():
    from nanochat.dataset import list_parquet_files
    entries = []
    paths = [Path(path) for path in list_parquet_files()]
    if len(paths) < 2:
        raise ValueError("RLM training requires at least one training shard and one validation shard")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Dataset shard is not a regular file: {path}")
        stat = path.stat()
        entries.append({"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": entries, "identity": "ordered_paths_sizes_and_mtimes"}


def record_checkpoint_retention(checkpoint_dir, step, keep_last, attempt):
    if keep_last is None:
        return
    from nanochat.rlm_checkpoints import prune_checkpoints

    try:
        retention = prune_checkpoints(checkpoint_dir, keep_last, step).to_dict()
        with (attempt / "checkpoint-retention.jsonl").open("a") as stream:
            stream.write(json.dumps({"published_step": step, **retention}) + "\n")
            stream.flush()
    except Exception as exc:
        logger.warning("%s", json.dumps({"checkpoint_warning": {
            "step": step, "phase": "resume_retention", "type": type(exc).__name__,
            "message": str(exc), "training_continues": True,
        }}))
        return
    logger.info("%s", json.dumps({"checkpoint_retention": retention}))


def run(args, attempt):
    import torch
    from nanochat.checkpoint_manager import checkpoint_is_complete, load_checkpoint
    from nanochat.rlm_checkpoints import recover_numbered_checkpoint, save_numbered_checkpoint
    from nanochat.rlm_checkpointing import CheckpointWriter
    from nanochat.common import COMPUTE_DTYPE, compute_init, get_base_dir
    from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
    from nanochat.rlm import RecurrentGPT, RLMConfig
    from nanochat.rlm_training import (
        TrainingConfig, apply_schedule, build_optimizer, checkpoint_metadata,
        cuda_metrics, file_digest, measured_update, MemoryMonitor,
        require_latest_resume_checkpoint, restore_rng,
        source_manifest, synchronize, write_json,
    )
    from nanochat.tokenizer import get_tokenizer
    from nanochat.training_runtime import check_training_memory
    from nanochat.rlm_latest import LatestCheckpoint

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("The initial RLM workflow supports one device per run")
    if COMPUTE_DTYPE not in (torch.bfloat16, torch.float32):
        raise ValueError("RLM training supports BF16 or FP32 compute; FP16 needs a separate scaling policy")
    _, _, _, _, device = compute_init(args.device_type)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    check_training_memory(device, 0)
    config = TrainingConfig(**{name: getattr(args, name) for name in TrainingConfig.__dataclass_fields__})
    accumulation = config.accumulation_steps(args.text_length)
    tokenizer_path = Path(get_base_dir()) / "tokenizer" / "tokenizer.pkl"
    tokenizer_sha256 = file_digest(tokenizer_path)
    tokenizer = get_tokenizer()
    checkpoint_dir = args.output_dir / "checkpoints"
    latest_checkpoint = None
    if args.mode == "train":
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        latest_checkpoint = LatestCheckpoint(checkpoint_dir)
    resuming = args.resume_from_step is not None or args.resume_latest
    metadata = None
    step = 0
    if resuming:
        if args.resume_latest:
            metadata = latest_checkpoint.metadata()
            step = metadata["step"]
        else:
            step = args.resume_from_step
        try:
            recover_numbered_checkpoint(checkpoint_dir)
        except Exception as exc:
            logger.warning("%s", json.dumps({"checkpoint_warning": {
                "step": step, "phase": "resume_publication", "type": type(exc).__name__,
                "message": str(exc), "training_continues": True,
            }}))
        require_latest_resume_checkpoint(checkpoint_dir, step)
        if metadata is None:
            metadata = json.loads((checkpoint_dir / f"meta_{step:06d}.json").read_text())
        if metadata.get("model_type") != "rlm" or metadata.get("step") != step:
            raise ValueError("Resume requires an RLM checkpoint at the requested completed update")
    initializer = None
    if resuming:
        initialization = metadata["run_contract"]["initialization"]
        if not isinstance(initialization, dict) or initialization.get("kind") not in (
            "random", "pretrained_input_token_embedding",
        ):
            raise ValueError("The recorded run has invalid initialization provenance")
    elif args.init_embedding_bundle is not None:
        from nanochat.rlm_initialization import TokenEmbeddingInitialization

        initializer = TokenEmbeddingInitialization.from_bundle(args.init_embedding_bundle, tokenizer_path)
        initialization = initializer.contract()
    elif args.init_embedding_checkpoint_dir is not None:
        from nanochat.rlm_initialization import TokenEmbeddingInitialization

        initializer = TokenEmbeddingInitialization.from_checkpoint(
            args.init_embedding_checkpoint_dir, args.init_embedding_step,
            args.init_embedding_tokenizer, tokenizer_path,
        )
        initialization = initializer.contract()
    else:
        initialization = {"kind": "random"}
    if config.input_embedding_freeze is not InputEmbeddingFreeze.NONE and initialization["kind"] != "pretrained_input_token_embedding":
        raise ValueError("Freezing the input embedding requires pretrained initialization provenance")
    model_config = RLMConfig(
        sequence_len=args.text_length + 1 + 2 * args.memory_tokens,
        vocab_size=tokenizer.get_vocab_size(), n_layer=args.depth,
        n_head=args.width // args.head_dim, n_kv_head=args.width // args.head_dim,
        n_embd=args.width, window_pattern="L", text_length=args.text_length,
        memory_tokens=args.memory_tokens, activation_checkpointing=args.activation_checkpointing,
    )
    source = source_manifest(Path(__file__).resolve().parents[1])
    inventory = dataset_inventory()
    try:
        triton_version = version("triton")
    except PackageNotFoundError:
        triton_version = None
    milestone_step = config.milestone_step
    if resuming:
        if latest_checkpoint.milestone_path.exists():
            milestone_step = latest_checkpoint.verify_milestone()["step"]
        elif metadata.get("milestone_checkpoint", False):
            # Finish the first requested preservation even if the horizon changed.
            milestone_step = step
        elif milestone_step < step:
            # A changed horizon cannot recreate a boundary already passed.
            milestone_step = 0
    contract = {
        "model_config": asdict(model_config), "training_config": asdict(config),
        "initialization": initialization,
        "checkpoint_policy": {
            "latest_every": 1, "early_step": 2,
            "save_every": args.save_every, "keep_last": args.keep_checkpoints,
            "mode": args.checkpoint_mode, "on_error": "warn",
            "milestone_step": milestone_step or None,
        } if args.mode == "train" else None,
        "source_sha256": source["sha256"], "tokenizer_sha256": tokenizer_sha256,
        "data_inventory_sha256": inventory["sha256"], "compute_dtype": str(COMPUTE_DTYPE),
        "device_type": device.type, "attention_impl": "sdpa",
        "runtime": {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.version.cuda, "triton": triton_version,
        },
    }
    write_json(attempt / "source.json", source)
    write_json(attempt / "data-inventory.json", inventory)
    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True).stdout.strip()
    diff = subprocess.run(["git", "diff", "HEAD", "--", "nanochat", "scripts/rlm_train.py"], cwd=repo, text=True, capture_output=True, check=True).stdout
    (attempt / "tracked-source.diff").write_text(diff)
    # Capture actual source bytes, including new untracked modules.
    for name in source["files"]:
        target = attempt / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo / name).read_bytes())
    write_json(attempt / "invocation.json", {
        "argv": sys.argv, "revision": revision, "contract": contract,
        "mode": args.mode, "torch_version": torch.__version__,
        "started_at": datetime.now(timezone.utc).isoformat(),
    })
    logger.info("%s", json.dumps({
        "configuration": contract, "accumulation_steps": accumulation,
        "cycle_lengths": [args.text_length + 1 + args.memory_tokens]
        + [args.text_length + 1 + 2 * args.memory_tokens] * (args.cycles - 2)
        + [args.text_length + 1 + args.memory_tokens],
        "prediction_targets_per_update": config.prediction_targets_per_update(args.text_length),
        "input_embedding_freeze": {
            "frozen_updates": config.input_embedding_frozen_updates,
            "first_trainable_update": (None if config.input_embedding_freeze is InputEmbeddingFreeze.PERMANENT
                                       else config.input_embedding_frozen_updates + 1),
        } if config.input_embedding_freeze is not InputEmbeddingFreeze.NONE else None,
    }))

    optimizer_data = None
    model_data = None
    if resuming:
        endpoint = args.stop_after_step or args.num_iterations
        if step >= endpoint:
            raise ValueError("Resume must continue beyond the checkpoint's completed update")
        if args.resume_latest:
            model_data, optimizer_data, loaded_metadata = latest_checkpoint.load(device)
        else:
            model_data, optimizer_data, loaded_metadata = load_checkpoint(
                checkpoint_dir, step, device, load_optimizer=True,
            )
        if loaded_metadata != metadata:
            raise ValueError("Resume checkpoint metadata changed after admission")
        # A stop during the less frequent numbered save must not strand a
        # complete update already published in latest. Repair that exact set.
        if args.resume_latest and metadata.get("numbered_checkpoint", False) and not checkpoint_is_complete(
            checkpoint_dir, step, load_optimizer=True,
        ):
            with MemoryMonitor(os.environ.get("NANOCHAT_MIN_SYSTEM_AVAILABLE_GB")) as monitor:
                monitor.require_available(step)
                try:
                    save_numbered_checkpoint(checkpoint_dir, step, model_data, optimizer_data, metadata)
                except Exception as exc:
                    logger.warning("%s", json.dumps({"checkpoint_warning": {
                        "step": step, "phase": "resume_numbered", "type": type(exc).__name__,
                        "message": str(exc), "training_continues": True,
                    }}))
                else:
                    write_json(attempt / "checkpoint-repair.json", {"step": step, "source": "latest.pt"})
                synchronize(device)
                monitor.require_available(step)
        if args.resume_latest and metadata.get("numbered_checkpoint", False):
            # A stop may also occur after publication but before retirement.
            record_checkpoint_retention(checkpoint_dir, step, args.keep_checkpoints, attempt)
    else:
        write_json(args.output_dir / "run.json", {"contract": contract, "revision": revision, "mode": args.mode})
    if resuming and milestone_step and step == milestone_step:
        # A stop after rolling publication can leave the required durable
        # copy unfinished. Repair it before any trainable-table forward.
        latest_checkpoint.preserve_milestone(step)
    with torch.device("meta"):
        model = RecurrentGPT(model_config)
    model.to_empty(device=device)
    model.init_weights()
    if model_data is not None:
        model.load_state_dict(model_data, strict=True, assign=True)
        del model_data
    elif initializer is not None:
        initialization_proof = initializer.initialize(model)
        write_json(attempt / "embedding-initialization.json", initialization_proof)
        logger.info("%s", json.dumps({"embedding_initialization": initialization_proof}))
        del initializer
    model.train()
    optimizer, decay = build_optimizer(model, config)
    parameter_counts = model.num_scaling_params()
    logger.info("%s", json.dumps({"parameter_counts": parameter_counts}))
    if optimizer_data is not None:
        optimizer.load_state_dict(optimizer_data)
        del optimizer_data
        restore_rng(metadata["rng_state"], device)
    forward_cycle = torch.compile(model.forward_cycle, dynamic=False) if args.compile else model.forward_cycle
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, args.device_batch_size, args.cycles * args.text_length,
        split="train", device=device.type,
        resume_state_dict=None if metadata is None else metadata["dataloader_state_dict"],
    )
    if args.mode == "capacity-probe":
        from nanochat.rlm_capacity import capacity_probe

        report = capacity_probe(
            model, forward_cycle, optimizer, loader, config, decay,
            tokenizer.get_bos_token_id(), attempt,
        )
        write_json(attempt / "result.json", report)
        write_json(args.output_dir / "result.json", report)
        (attempt / "updates.jsonl").write_text(json.dumps(report["updates"][0], allow_nan=False) + "\n")
        logger.info("%s", json.dumps(report))
        return
    setup_metrics = cuda_metrics(device)
    updates = []
    tokens_per_update = {
        "text_tokens": config.total_batch_size,
        "supervised_tokens": config.prediction_targets_per_update(args.text_length),
        "internal_positions": accumulation * config.device_batch_size
        * config.positions_per_episode(args.text_length, args.memory_tokens),
    }
    token_counts = {name: 0 if metadata is None else metadata[name] for name in tokens_per_update}
    writer = None if args.mode == "probe" else CheckpointWriter(
        checkpoint_dir, mode=args.checkpoint_mode, keep_last=args.keep_checkpoints,
        latest_step=latest_checkpoint.metadata()["step"] if latest_checkpoint.exists() else None,
    )
    checkpoint_measurements = [] if writer is None else writer.measurements
    endpoint = args.stop_after_step or args.num_iterations
    metrics_path = attempt / "updates.jsonl"
    execution_started = time.monotonic()
    with MemoryMonitor(os.environ.get("NANOCHAT_MIN_SYSTEM_AVAILABLE_GB")) as monitor:
        def record_checkpoint_measurements():
            for measurement in checkpoint_measurements:
                if "minimum_system_available_gb" not in measurement:
                    measurement.update({
                        "minimum_system_available_gb": monitor.minimum_observed / 1e9,
                        **cuda_metrics(device),
                    })
            if writer is not None:
                try:
                    write_json(attempt / "checkpoint-measurements.json", checkpoint_measurements)
                except OSError as exc:
                    logger.warning("%s", json.dumps({"checkpoint_warning": {
                        "step": step, "phase": "measurements", "type": type(exc).__name__,
                        "message": str(exc), "training_continues": True,
                    }}))

        def before_optimizer_step():
            writer.wait(completed_step=step)
            record_checkpoint_measurements()

        try:
            while step < endpoint:
                check_training_memory(device, step)
                monitor.require_available(step)
                lrm = apply_schedule(optimizer, config, step, decay)
                metrics, loader_state = measured_update(
                    model, forward_cycle, optimizer, loader, config, step, tokenizer.get_bos_token_id(),
                    os.environ.get("NANOCHAT_MIN_SYSTEM_AVAILABLE_GB"),
                    failure_path=attempt / "failed-update.json",
                    before_optimizer_step=before_optimizer_step if writer is not None else None,
                )
                step += 1
                for name, count in tokens_per_update.items():
                    token_counts[name] += count
                metrics["lr_multiplier"] = lrm
                updates.append(metrics)
                with metrics_path.open("a") as stream:
                    stream.write(json.dumps(metrics, allow_nan=False) + "\n")
                    stream.flush()
                logger.info("%s", json.dumps(metrics))
                logger.info("Update %d/%d: %.2f s", step, args.num_iterations, metrics["duration_seconds"])
                if writer is not None:
                    numbered = step == 2 or step == endpoint or bool(args.save_every and step % args.save_every == 0)
                    synchronize(device)
                    saved_metadata = checkpoint_metadata(model, config, step, loader_state, contract, token_counts)
                    saved_metadata["numbered_checkpoint"] = numbered
                    milestone = step == milestone_step
                    saved_metadata["milestone_checkpoint"] = milestone
                    writer.submit(model.state_dict(), optimizer.state_dict(), saved_metadata,
                                  numbered=numbered, milestone=milestone)
                    if milestone:
                        # Unfreezing happens at the start of the next forward,
                        # earlier than the usual pre-optimizer writer barrier.
                        writer.wait(completed_step=step)
                    record_checkpoint_measurements()
                    logger.info("%s", json.dumps({"checkpoint_status": writer.status(step)}))
        finally:
            if writer is not None:
                writer.close(completed_step=step)
                record_checkpoint_measurements()
        synchronize(device)
        monitor.require_available(step)
        execution_wall_seconds = time.monotonic() - execution_started
    measurements = updates + checkpoint_measurements
    training_peak_reserved = max([setup_metrics["cuda_peak_reserved_gb"]] + [item["cuda_peak_reserved_gb"] for item in updates])
    peak_reserved = max([setup_metrics["cuda_peak_reserved_gb"]] + [item["cuda_peak_reserved_gb"] for item in measurements])
    peak_allocated = max([setup_metrics["cuda_peak_allocated_gb"]] + [item["cuda_peak_allocated_gb"] for item in measurements])
    report = {
        "status": "completed" if step == args.num_iterations else "paused",
        "mode": args.mode, "completed_updates": step, "attempt_updates": len(updates),
        "parameter_counts": parameter_counts,
        "text_tokens": token_counts["text_tokens"],
        "supervised_tokens": token_counts["supervised_tokens"],
        "peak_cuda_reserved_gb": peak_reserved, "peak_cuda_allocated_gb": peak_allocated,
        "minimum_system_available_gb": monitor.minimum_observed / 1e9,
        "within_72gb_training_reserve_target": training_peak_reserved <= 72 if device.type == "cuda" else None,
        "evaluation_overhead_measured": False,
        "checkpoint_mode": args.checkpoint_mode if writer is not None else None,
        "checkpoint_status": writer.status(step) if writer is not None else None,
        "execution_wall_seconds": execution_wall_seconds,
        "checkpoint_overhead_measured": bool(checkpoint_measurements),
        "checkpoint_measurements": checkpoint_measurements,
        "checkpoint_dir": str(checkpoint_dir) if args.mode == "train" else None,
        "setup_memory": setup_metrics, "updates": updates,
    }
    write_json(attempt / "result.json", report)
    write_json(args.output_dir / "result.json", report)
    logger.info("%s", json.dumps({"result": str(attempt / "result.json"), "status": report["status"]}))


def main():
    args = parse_args()
    configure_runtime_paths()
    os.environ["NANOCHAT_ATTENTION_IMPL"] = "sdpa"
    os.environ.setdefault("NANOCHAT_DTYPE", "bfloat16" if args.device_type == "cuda" else "float32")
    os.environ.setdefault("NANOCHAT_CUDA_MEMORY_LIMIT_GB", "80")
    os.environ.setdefault("NANOCHAT_MIN_SYSTEM_AVAILABLE_GB", "16")
    os.environ.setdefault("NANOCHAT_CHECKPOINT_RESERVE_GB", "100")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    args.output_dir = args.output_dir.expanduser().absolute()
    if args.output_dir.is_symlink():
        raise ValueError("Run output must be an owned directory, not a symlink")
    if args.resume_from_step is None and not args.resume_latest:
        args.output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    elif not (args.output_dir / "run.json").is_file():
        raise ValueError("Resume requires an existing RLM run directory")
    from filelock import FileLock
    from nanochat.rlm_training import validate_run_directory, write_json
    validate_run_directory(args.output_dir)
    with FileLock(args.output_dir / ".run.lock", timeout=0):
        attempt = args.output_dir / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        attempt.mkdir(parents=True)
        try:
            run(args, attempt)
        except BaseException as exc:
            import torch
            from nanochat.rlm_capacity import is_cuda_out_of_memory
            from nanochat.rlm_training import cuda_metrics
            device = torch.device(args.device_type)
            status = {
                "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                "failure_kind": "cuda_out_of_memory" if is_cuda_out_of_memory(exc) else "error",
                "memory": cuda_metrics(device) if device.type == "cpu" or torch.cuda.is_initialized() else {},
            }
            write_json(attempt / "failure.json", status)
            write_json(args.output_dir / "result.json", status)
            raise


if __name__ == "__main__":
    main()

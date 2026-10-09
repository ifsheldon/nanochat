"""Profile one RLM episode and optimizer step using a recorded run configuration.

Warmup and profiling use real training data and the training optimizer, but each
diagnostic optimizer step consumes only one accumulation-scaled episode.
These artifacts measure operator costs, not full-update throughput or learning.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True, help="run.json from the training/probe configuration to profile")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for Chrome trace, operator tables, and measurements")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--warmup-episodes", type=int, default=3, help="at least two; the first initializes optimizer state")
    parser.add_argument("--record-shapes", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.warmup_episodes < 2:
        parser.error("--warmup-episodes must be at least two so a later warmup has resident optimizer state")
    return args


def diagnostic_episode(model, forward_cycle, optimizer, loader, config, monitor, optimizer_step, *, step):
    """Keep the loader's reusable storage alive until the entire unroll backpropagates."""
    import torch
    from nanochat.rlm_training import (
        apply_input_embedding_schedule, episode_loss, require_finite_gradients, require_finite_state,
    )
    from nanochat.training_runtime import require_finite_loss

    monitor.require_available(step)
    input_embedding_frozen = apply_input_embedding_schedule(model, config, step)
    optimizer.zero_grad(set_to_none=True)
    with torch.profiler.record_function("rlm.data.next_episode"):
        inputs, targets, loader_state = next(loader)
    expected_shape = (config.device_batch_size, config.cycles * model.config.text_length)
    if inputs.shape != expected_shape or targets.shape != expected_shape:
        raise ValueError("Profiling loader batch does not match the recorded episode configuration")
    if targets.ne(-1).sum().item() != inputs.numel():
        raise ValueError("Profiling requires the same fully supervised packed text as pretraining")
    cycle_index = 0

    def scoped_cycle(*args, **kwargs):
        nonlocal cycle_index
        cycle_index += 1
        with torch.profiler.record_function(f"rlm.forward.cycle_{cycle_index}"):
            return forward_cycle(*args, **kwargs)

    with torch.profiler.record_function("rlm.episode.forward"):
        loss, sums, counts = episode_loss(
            scoped_cycle, inputs, targets, model.config.text_length, config.cycles,
        )
    with torch.profiler.record_function("rlm.health.loss"):
        require_finite_loss(loss, step)
    accumulation = config.accumulation_steps(model.config.text_length)
    with torch.profiler.record_function("rlm.episode.backward"):
        (loss / accumulation).backward()
    with torch.profiler.record_function("rlm.health.gradients"):
        require_finite_gradients(
            model, step,
            frozen_parameter=model.transformer.wte.weight if input_embedding_frozen else None,
        )
    monitor.require_available(step)
    if optimizer_step:
        with torch.profiler.record_function("rlm.optimizer.step"):
            optimizer.step()
        with torch.profiler.record_function("rlm.health.model_optimizer_state"):
            require_finite_state(model, optimizer, step)
    monitor.require_available(step)
    result = {
        "loss": loss.item(), "cycle_losses": (sums / counts).tolist(),
        "real_text_tokens": inputs.numel(), "supervised_predictions": counts.sum().item(),
        "internal_positions": config.device_batch_size * config.positions_per_episode(
            model.config.text_length, model.config.memory_tokens,
        ),
        "optimizer_step": optimizer_step, "dataloader_state": dict(loader_state),
        "input_embedding_frozen": input_embedding_frozen,
    }
    with torch.profiler.record_function("rlm.gradients.release"):
        optimizer.zero_grad(set_to_none=True)
    return result


def export_device_events(trace_path, directory):
    """Exclude nested GPU annotation ranges from actual kernel/transfer costs."""
    from nanochat.rlm_training import write_json

    trace = json.loads(trace_path.read_text())
    categories = {"kernel", "gpu_memcpy", "gpu_memset"}
    groups = {}
    annotations = []
    for event in trace["traceEvents"]:
        category = event.get("cat")
        if category == "gpu_user_annotation":
            annotations.append({"name": event["name"], "duration_us": event.get("dur", 0)})
        if category not in categories or event.get("ph") != "X":
            continue
        aggregate = groups.setdefault((category, event["name"]), {
            "category": category, "name": event["name"], "calls": 0, "total_us": 0.0,
        })
        aggregate["calls"] += 1
        aggregate["total_us"] += event["dur"]
    events = sorted(groups.values(), key=lambda event: event["total_us"], reverse=True)
    write_json(directory / "device-events.json", events)
    write_json(directory / "device-annotations.json", annotations)
    return {
        "cuda_device_event_groups": len(events),
        "cuda_device_event_total_us": sum(event["total_us"] for event in events),
        "cuda_kernel_total_us": sum(event["total_us"] for event in events if event["category"] == "kernel"),
        "cuda_event_accounting": "Chrome trace kernel, gpu_memcpy, and gpu_memset events only; excludes nested GPU annotations; concurrent events can overlap",
    }


def export_profile(profiler, directory, record_shapes):
    """Keep operator attribution and raw device-event costs separately inspectable."""
    from nanochat.rlm_training import write_json

    profiler.export_chrome_trace(str(directory / "trace.json"))
    averages = profiler.key_averages(group_by_input_shape=record_shapes)
    for filename, metric in (("operators-cpu.txt", "self_cpu_time_total"), ("operators-device.txt", "self_device_time_total")):
        (directory / filename).write_text(averages.table(sort_by=metric, row_limit=100) + "\n")
    operators = [{
        "name": event.key, "device_type": str(event.device_type), "calls": event.count,
        "input_shapes": event.input_shapes,
        "self_cpu_us": event.self_cpu_time_total, "total_cpu_us": event.cpu_time_total,
        "self_device_us": event.self_device_time_total, "total_device_us": event.device_time_total,
        "self_cpu_memory_bytes": event.self_cpu_memory_usage,
        "self_device_memory_bytes": event.self_device_memory_usage,
    } for event in averages]
    write_json(directory / "operators.json", operators)
    return {
        "operator_groups": len(operators),
        **export_device_events(directory / "trace.json", directory),
    }


def profile_workload(model, forward_cycle, optimizer, loader, config, initial_decay,
                     output_dir, warmup_episodes=3, record_shapes=True, minimum_available_gb=None):
    import torch
    from nanochat.rlm_training import (
        MemoryMonitor, apply_schedule, cuda_metrics, synchronize, write_json,
    )

    device = model.get_device()
    monitor = MemoryMonitor(minimum_available_gb)
    warmup = []
    try:
        with monitor:
            apply_schedule(optimizer, config, 0, initial_decay)
            for index in range(warmup_episodes):
                synchronize(device)
                started = time.monotonic()
                result = diagnostic_episode(
                    model, forward_cycle, optimizer, loader, config, monitor,
                    optimizer_step=index == 0, step=0 if index == 0 else 1,
                )
                synchronize(device)
                result["duration_seconds"] = time.monotonic() - started
                warmup.append(result)
                write_json(output_dir / "warmup.json", warmup)
                print(f"Warmup episode {index + 1}/{warmup_episodes}: {result['duration_seconds']:.3f}s", flush=True)
            if not optimizer.state:
                raise RuntimeError("The profiling warmup did not initialize optimizer state")
            apply_schedule(optimizer, config, 1, initial_decay)
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            graphs_before = dict(torch._dynamo.utils.counters["stats"])
            with torch.profiler.profile(
                activities=activities, record_shapes=record_shapes, profile_memory=True,
            ) as profiler:
                started = time.monotonic()
                result = diagnostic_episode(
                    model, forward_cycle, optimizer, loader, config, monitor, optimizer_step=True, step=1,
                )
                synchronize(device)
                result["profiled_work_seconds"] = time.monotonic() - started
            monitor.require_available(1)
            result.update({
                "compiler_stats_before": graphs_before,
                "compiler_stats_after": dict(torch._dynamo.utils.counters["stats"]),
                "minimum_system_available_gb_including_warmup": monitor.minimum_observed / 1e9,
                **cuda_metrics(device),
            })
        result.update(export_profile(profiler, output_dir, record_shapes))
        if device.type == "cuda" and result["cuda_device_event_groups"] == 0:
            raise RuntimeError("CUDA profiling captured no device events; inspect profiler/CUPTI support before interpreting timings")
        result.update({
            "status": "completed", "warmup_episodes": warmup_episodes,
            "optimizer_steps_before_profile": 1, "profiled_optimizer_steps": 1,
            "production_accumulation_steps": config.accumulation_steps(model.config.text_length),
            "gradient_scale": 1 / config.accumulation_steps(model.config.text_length),
            "production_global_text_tokens": config.total_batch_size,
            "scope": "One episode plus one optimizer step; production amortizes the optimizer over all accumulation episodes",
            "timing_limit": "Profiler instrumentation adds overhead; use the ordinary two-update probe for throughput comparisons",
        })
        write_json(output_dir / "result.json", result)
        return result
    except BaseException as exc:
        write_json(output_dir / "failure.json", {
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
            "completed_warmup_episodes": len(warmup),
            "minimum_system_available_gb": monitor.minimum_observed / 1e9,
            **cuda_metrics(device),
        })
        raise


def run(args):
    import torch
    from nanochat.common import COMPUTE_DTYPE, compute_cleanup, compute_init, get_base_dir
    from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
    from nanochat.rlm import RecurrentGPT, RLMConfig
    from nanochat.rlm_training import TrainingConfig, build_optimizer, file_digest, source_manifest, write_json
    from nanochat.tokenizer import get_tokenizer
    from nanochat.training_runtime import check_training_memory
    from scripts.rlm_train import dataset_inventory

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("RLM profiling supports one device per run")
    if COMPUTE_DTYPE not in (torch.bfloat16, torch.float32):
        raise ValueError("RLM profiling requires the training workflow's BF16 or FP32 compute")
    contract = json.loads(args.run_config.read_text())["contract"]
    config = TrainingConfig(**contract["training_config"])
    model_config = RLMConfig(**contract["model_config"])
    config.accumulation_steps(model_config.text_length)
    _, _, _, _, device = compute_init(args.device)
    try:
        check_training_memory(device, 0)
        torch.manual_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(config.seed)
        tokenizer_sha256 = file_digest(Path(get_base_dir()) / "tokenizer" / "tokenizer.pkl")
        if tokenizer_sha256 != contract["tokenizer_sha256"]:
            raise ValueError("Profiling requires the recorded run's tokenizer bytes")
        if dataset_inventory()["sha256"] != contract["data_inventory_sha256"]:
            raise ValueError("Profiling requires the recorded run's ordered data inventory")
        tokenizer = get_tokenizer()
        if tokenizer.get_vocab_size() != model_config.vocab_size:
            raise ValueError("Profiling tokenizer vocabulary does not match the recorded model")
        write_json(args.output_dir / "invocation.json", {
            "argv": sys.argv, "reference_run_config": str(args.run_config),
            "reference_contract": contract, "model_config": asdict(model_config),
            "training_config": asdict(config), "device": str(device),
            "compute_dtype": str(COMPUTE_DTYPE), "torch_version": torch.__version__,
            "source": source_manifest(Path(__file__).resolve().parents[1]),
            "profiler_source_sha256": file_digest(Path(__file__)),
            "record_shapes": args.record_shapes, "profile_memory": True,
            "cuda_memory_limit_gb": os.environ["NANOCHAT_CUDA_MEMORY_LIMIT_GB"],
            "minimum_system_available_gb": os.environ["NANOCHAT_MIN_SYSTEM_AVAILABLE_GB"],
            "initialization": "fresh model; checkpoint weights are not loaded",
        })
        with torch.device("meta"):
            model = RecurrentGPT(model_config)
        model.to_empty(device=device)
        model.init_weights()
        model.train()
        optimizer, decay = build_optimizer(model, config)
        forward = torch.compile(model.forward_cycle, dynamic=False) if config.compile else model.forward_cycle
        loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tokenizer, config.device_batch_size, config.cycles * model_config.text_length,
            split="train", device=device.type,
        )
        return profile_workload(
            model, forward, optimizer, loader, config, decay, args.output_dir,
            args.warmup_episodes, args.record_shapes,
            os.environ["NANOCHAT_MIN_SYSTEM_AVAILABLE_GB"],
        )
    finally:
        compute_cleanup()


def main():
    args = parse_args()
    args.run_config = args.run_config.expanduser().absolute()
    args.output_dir = args.output_dir.expanduser().absolute()
    args.output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.environ["NANOCHAT_ATTENTION_IMPL"] = "sdpa"
    os.environ.setdefault("NANOCHAT_DTYPE", "bfloat16" if args.device == "cuda" else "float32")
    os.environ.setdefault("NANOCHAT_CUDA_MEMORY_LIMIT_GB", "80")
    os.environ.setdefault("NANOCHAT_MIN_SYSTEM_AVAILABLE_GB", "16")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        result = run(args)
    except BaseException as exc:
        if not (args.output_dir / "failure.json").exists():
            from nanochat.rlm_training import write_json
            write_json(args.output_dir / "failure.json", {
                "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
            })
        raise
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    print(f"Profiler artifacts: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

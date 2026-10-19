"""Measure a complete RLM update after model and optimizer compilation settle."""

import os
from pathlib import Path
import time

import torch

from nanochat.flash_attention import attention_backend
from nanochat.rlm_capacity import resident_probe_state
from nanochat.rlm_policy import InputEmbeddingFreeze
from nanochat.rlm_training import MemoryMonitor, apply_schedule, measured_update


MAX_WARMUP_UPDATES = 5


def compiler_snapshot():
    """Include lazy backward compilation and runtime kernel autotuning events."""
    return {
        "stats": dict(torch._dynamo.utils.counters["stats"]),
        "timing_events": {
            name: len(values)
            for name, values in torch._dynamo.utils.compilation_time_metrics.items()
        },
    }


def attention_probe(model, forward_cycle, optimizer, loader, config, initial_decay,
                    bos_token_id: int, attempt: Path):
    """Discard warmup weights and report exactly one compilation-free update.

    Both candidates must use the same production recipe and an explicit embedding
    phase. Their optimizer and pinned checkpoint staging allocations remain
    resident, but the probe never writes a checkpoint or initializes training.
    """
    if config.input_embedding_freeze is InputEmbeddingFreeze.SCHEDULED:
        raise ValueError("Attention probes require an explicit none or permanent embedding freeze policy")
    minimum = os.environ.get("NANOCHAT_MIN_SYSTEM_AVAILABLE_GB")
    started = time.monotonic()
    warmups = []
    stable_warmup = False
    with MemoryMonitor(minimum) as monitor:
        with resident_probe_state(model, optimizer, config, monitor) as (state, setup_memory):
            for step in range(MAX_WARMUP_UPDATES + 1):
                # A short standalone horizon must not unfreeze a permanently
                # frozen table or run a schedule past its configured endpoint.
                schedule_step = min(step, config.num_iterations - 1)
                multiplier = apply_schedule(optimizer, config, schedule_step, initial_decay)
                before = compiler_snapshot()
                metrics, _ = measured_update(
                    model, forward_cycle, optimizer, loader, config, schedule_step,
                    bos_token_id, minimum, failure_path=attempt / "failed-update.json",
                )
                after = compiler_snapshot()
                metrics.update({
                    "lr_multiplier": multiplier,
                    "completed_updates": step + 1,
                    "compilation_before": before,
                    "compilation_after": after,
                })
                monitor.require_available(step + 1)
                compiled = before != after
                if stable_warmup and not compiled:
                    break
                warmups.append(metrics)
                stable_warmup = not compiled
            else:
                raise RuntimeError(
                    f"Attention probe did not produce a compilation-free timed update "
                    f"after at most {MAX_WARMUP_UPDATES} warmup updates"
                )
            measurements = [setup_memory, *warmups, metrics]
            return {
                "status": "completed", "mode": "attention-probe",
                "attention_impl": attention_backend().value,
                "completed_updates": len(warmups) + 1,
                "attempt_updates": len(warmups) + 1,
                "attention_probe": {
                    **state,
                    "duration_seconds": metrics["duration_seconds"],
                    "warmup_updates": len(warmups),
                    "compilation_excluded": True,
                },
                "peak_cuda_reserved_gb": max(item["cuda_peak_reserved_gb"] for item in measurements),
                "peak_cuda_allocated_gb": max(item["cuda_peak_allocated_gb"] for item in measurements),
                "minimum_system_available_gb": monitor.minimum_observed / 1e9,
                "execution_wall_seconds": time.monotonic() - started,
                "checkpoint_overhead_measured": False,
                "setup_memory": setup_memory,
                "warmup_updates": warmups,
                "updates": [metrics],
            }

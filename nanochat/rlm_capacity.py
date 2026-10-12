"""One-update capacity measurement with optimizer and staging memory resident."""

import os
from pathlib import Path
import time

import torch
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions

from nanochat.rlm_policy import InputEmbeddingFreeze
from nanochat.rlm_training import (
    MemoryMonitor, apply_input_embedding_schedule, apply_schedule,
    cuda_metrics, measured_update, synchronize,
)


def is_cuda_out_of_memory(error: BaseException) -> bool:
    """Recognize allocation errors even when compilation wraps their cause."""
    visited = set()
    while error is not None and id(error) not in visited:
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
        visited.add(id(error))
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return False


def capacity_probe(model, forward_cycle, optimizer, loader, config, initial_decay,
                   bos_token_id: int, attempt: Path):
    """Preallocate state without changing weights, then execute one full update.

    State is allocated through the optimizer's ordinary initialization routines.
    CUDA probes retain the production pinned staging buffers throughout backward.
    The probe's updated weights are discarded, never used to initialize training.
    """
    if config.input_embedding_freeze is InputEmbeddingFreeze.SCHEDULED:
        raise ValueError("Capacity probes require an explicit none or permanent embedding freeze policy")
    device = model.get_device()
    minimum = os.environ.get("NANOCHAT_MIN_SYSTEM_AVAILABLE_GB")
    stager = None
    staged = None
    started = time.monotonic()
    with MemoryMonitor(minimum) as monitor:
        try:
            monitor.require_available(0)
            frozen = apply_input_embedding_schedule(model, config, 0)
            optimizer.initialize_state()
            if device.type == "cuda":
                stager = DefaultStager(StagingOptions(
                    use_pinned_memory=True, use_shared_memory=False,
                    use_async_staging=True, use_non_blocking_copy=True,
                ))
                synchronize(device)
                staged = stager.stage({
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                }).result()
            monitor.require_available(0)
            setup_memory = cuda_metrics(device)
            multiplier = apply_schedule(optimizer, config, 0, initial_decay)
            metrics, _ = measured_update(
                model, forward_cycle, optimizer, loader, config, 0, bos_token_id,
                minimum, failure_path=attempt / "failed-update.json",
            )
            metrics["lr_multiplier"] = multiplier
            synchronize(device)
            monitor.require_available(1)
            report = {
                "status": "completed", "mode": "capacity-probe",
                "completed_updates": 1, "attempt_updates": 1,
                "capacity_probe": {
                    "optimizer_state_preallocated": True,
                    "checkpoint_staging_preallocated": staged is not None,
                    "input_embedding_frozen": frozen,
                    "production_weights_reused": False,
                },
                "peak_cuda_reserved_gb": max(setup_memory["cuda_peak_reserved_gb"], metrics["cuda_peak_reserved_gb"]),
                "peak_cuda_allocated_gb": max(setup_memory["cuda_peak_allocated_gb"], metrics["cuda_peak_allocated_gb"]),
                "minimum_system_available_gb": monitor.minimum_observed / 1e9,
                "execution_wall_seconds": time.monotonic() - started,
                "checkpoint_overhead_measured": False,
                "setup_memory": setup_memory, "updates": [metrics],
            }
        finally:
            if stager is not None:
                # A failed stage may have queued copies; drain them before
                # releasing pinned destinations or live optimizer tensors.
                synchronize(device)
                stager.close()
    return report

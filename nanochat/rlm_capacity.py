"""One-update capacity measurement with optimizer and staging memory resident."""

from contextlib import contextmanager
import os
from pathlib import Path
import time

import torch
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions

from nanochat.rlm_policy import InputEmbeddingFreeze
from nanochat.rlm_distributed import DistributedRuntime
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


@contextmanager
def resident_probe_state(model, optimizer, config, monitor, runtime=None):
    """Keep ordinary optimizer state and production checkpoint buffers resident."""
    device = model.get_device()
    runtime = runtime or DistributedRuntime.single(device)
    stager = None
    staged = None
    full_optimizer = None
    completed = False
    try:
        def prepare_optimizer():
            monitor.require_available(0)
            frozen = apply_input_embedding_schedule(model, config, 0)
            optimizer.initialize_state()
            return frozen

        frozen = runtime.collective(prepare_optimizer, "allocate probe optimizer state")
        if runtime.world_size > 1:
            from nanochat.optim_checkpoint import OptimizerStateLayout
            from nanochat.rlm_distributed_checkpoint import CheckpointStateStager

            def stage_worker():
                nonlocal stager
                stager = CheckpointStateStager(device)
                synchronize(device)
                return stager.stage(model.state_dict() if runtime.is_primary else {}, optimizer.state_dict())

            staged = runtime.collective(stage_worker, "stage probe checkpoint")
            # Keep both local staging and the root's complete optimizer buffers
            # alive through the measured update, matching checkpoint residency.
            full_optimizer = OptimizerStateLayout.from_optimizer(optimizer).gather(
                staged["optimizer"], runtime.checkpoint_group,
            )
        elif device.type == "cuda":
            stager = DefaultStager(StagingOptions(
                use_pinned_memory=True, use_shared_memory=False,
                use_async_staging=True, use_non_blocking_copy=True,
            ))
            synchronize(device)
            staged = stager.stage({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            }).result()
        runtime.collective(lambda: monitor.require_available(0), "check probe staging headroom")
        yield {
            "optimizer_state_preallocated": True,
            "checkpoint_staging_preallocated": staged is not None,
            "complete_optimizer_staging_preallocated": full_optimizer is not None,
            "input_embedding_frozen": frozen,
            "production_weights_reused": False,
        }, runtime.aggregate_metrics(cuda_metrics(device))
        completed = True
    finally:
        def release_staging():
            if stager is not None:
                # A failed stage may have queued copies; drain them before
                # releasing pinned destinations or live optimizer tensors.
                synchronize(device)
                stager.close()

        if completed:
            runtime.collective(release_staging, "release probe staging")
        else:
            # An exception may mean the process group is broken. Drain local
            # copies without starting another rendezvous on the failure path.
            release_staging()


def capacity_probe(model, forward_cycle, optimizer, loader, config, initial_decay,
                   bos_token_id: int, attempt: Path, runtime=None):
    """Preallocate state without changing weights, then execute one full update.

    State is allocated through the optimizer's ordinary initialization routines.
    CUDA probes retain the production pinned staging buffers throughout backward.
    The probe's updated weights are discarded, never used to initialize training.
    """
    if config.input_embedding_freeze is InputEmbeddingFreeze.SCHEDULED:
        raise ValueError("Capacity probes require an explicit none or permanent embedding freeze policy")
    device = model.get_device()
    runtime = runtime or DistributedRuntime.single(device)
    minimum = os.environ.get("NANOCHAT_MIN_SYSTEM_AVAILABLE_GB")
    started = time.monotonic()
    with MemoryMonitor(minimum, runtime=runtime) as monitor:
        with resident_probe_state(model, optimizer, config, monitor, runtime) as (state, setup_memory):
            multiplier = apply_schedule(optimizer, config, 0, initial_decay)
            metrics, _ = measured_update(
                model, forward_cycle, optimizer, loader, config, 0, bos_token_id,
                minimum, failure_path=attempt / ("failed-update.json" if runtime.is_primary else f"failed-update-rank{runtime.rank}.json"),
                runtime=runtime,
            )
            metrics["lr_multiplier"] = multiplier
            def finish_probe():
                synchronize(device)
                monitor.require_available(1)

            runtime.collective(finish_probe, "finish capacity probe")
            report = {
                "status": "completed", "mode": "capacity-probe",
                "completed_updates": 1, "attempt_updates": 1,
                "capacity_probe": state,
                "peak_cuda_reserved_gb": max(setup_memory["cuda_peak_reserved_gb"], metrics["cuda_peak_reserved_gb"]),
                "peak_cuda_allocated_gb": max(setup_memory["cuda_peak_allocated_gb"], metrics["cuda_peak_allocated_gb"]),
                "minimum_system_available_gb": monitor.minimum_observed / 1e9,
                "execution_wall_seconds": time.monotonic() - started,
                "checkpoint_overhead_measured": False,
                "setup_memory": setup_memory, "updates": [metrics],
            }
    return runtime.aggregate_metrics(report)

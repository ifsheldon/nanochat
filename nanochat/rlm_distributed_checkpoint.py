"""Stage worker-local snapshots and publish one portable checkpoint on rank zero."""

from concurrent.futures import Future, ThreadPoolExecutor
import copy
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from threading import Event
import time

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions

from nanochat.optim_checkpoint import OptimizerCheckpointError, OptimizerStateLayout
from nanochat.rlm_checkpointing import (
    CheckpointMode, CheckpointStagingSynchronizationError, CheckpointWriter,
    MilestoneCheckpointError, _snapshot_state,
)
from nanochat.rlm_latest import LatestCheckpoint


logger = logging.getLogger(__name__)


class DistributedCheckpointError(RuntimeError):
    """A fatal checkpoint failure was reported by another worker."""


class CheckpointStateStager:
    """Reusable production/probe staging storage, created on the training thread.

    Callers synchronize the training stream before staging and keep source
    tensors unchanged until stage returns. CUDA copies use a dedicated stream;
    CPU copies also own independent storage. Pass an empty model on non-root
    workers, which only stage their local optimizer moments.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        self._stager = None
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                self._stager = DefaultStager(StagingOptions(
                    use_pinned_memory=True, use_shared_memory=False,
                    use_async_staging=True, use_non_blocking_copy=True,
                ))

    def stage(self, model_data, optimizer_data):
        state = {"model": model_data, "optimizer": optimizer_data}
        if self._stager is None:
            return copy.deepcopy(state)
        try:
            return self._stager.stage(state).result()
        except BaseException:
            try:
                torch.cuda.synchronize(self.device)
            except Exception as error:
                raise CheckpointStagingSynchronizationError(
                    "CUDA could not synchronize after distributed checkpoint staging failed; "
                    "live tensors cannot safely be updated"
                ) from error
            raise

    def close(self):
        if self._stager is not None:
            self._stager.close()
            self._stager = None


@dataclass(frozen=True)
class _Request:
    step: int
    model: dict
    optimizer: dict
    metadata: dict
    numbered: bool
    milestone: bool
    snapshot_seconds: float


def _error(error, phase, rank):
    return {
        "phase": phase, "type": type(error).__name__, "message": str(error),
        "rank": rank,
        "fatal": isinstance(error, CheckpointStagingSynchronizationError) or not isinstance(error, Exception),
    }


def _raise_fatal(error):
    message = f"Checkpoint {error['phase']} failed on rank {error['rank']}: {error['message']}"
    kind = error["type"]
    if kind == "CheckpointStagingSynchronizationError":
        raise CheckpointStagingSynchronizationError(message)
    if kind == "KeyboardInterrupt":
        raise KeyboardInterrupt(message)
    if kind == "SystemExit":
        raise SystemExit(message)
    raise DistributedCheckpointError(message)


class DistributedCheckpointWriter:
    """Publish complete optimizer state through a dedicated CPU process group.

    All ranks call submit in the same order. A control-group admission exchange
    precedes background work, so snapshot and thread-launch failures cannot
    strand peers in checkpoint collectives. Admitted workers exclusively own
    the checkpoint group through staging, optimizer gathering, and publication.
    The main thread may run the next forward/backward but must wait before
    optimizer mutation or embedding unfreezing. Groups need finite timeouts for
    process death or transport failure, which are fatal rather than warnings.
    """

    def __init__(self, checkpoint_dir: Path, *, mode, keep_last, optimizer,
                 checkpoint_group, control_group, latest_step=None):
        self.mode = CheckpointMode(mode)
        self.store = LatestCheckpoint(checkpoint_dir)
        self.checkpoint_group = checkpoint_group
        self.control_group = control_group
        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank(checkpoint_group) if self.distributed else 0
        self.world_size = dist.get_world_size(checkpoint_group) if self.distributed else 1
        self.layout = OptimizerStateLayout.from_optimizer(optimizer)
        self.device = optimizer.param_groups[0]["params"][0].device
        self.latest_step = latest_step
        self.measurements = []
        self._writer = (CheckpointWriter(checkpoint_dir, mode="sync", keep_last=keep_last,
                                         latest_step=latest_step) if self.rank == 0 else None)
        self._executor = (ThreadPoolExecutor(max_workers=1, thread_name_prefix="rlm-distributed-checkpoint")
                          if self.mode is CheckpointMode.ASYNC else None)
        self._stager = None
        self._pending = None
        self._pending_step = None
        self._closed = False

    def status(self, completed_step):
        return {
            "mode": self.mode.value, "completed_step": completed_step,
            "latest_path": str(self.store.path), "latest_step": self.latest_step,
            "latest_lag_completed_updates": (None if self.latest_step is None
                                             else completed_step - self.latest_step),
            "pending_step": self._pending_step,
        }

    def _exchange(self, value, group):
        if not self.distributed:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value, group=group)
        return values

    def _failed(self, request, errors):
        result = CheckpointWriter._measurement(
            self, request.step, request.numbered, request.snapshot_seconds, request.milestone,
        )
        result["status"] = "warning"
        result["errors"] = errors
        return result

    def submit(self, model_data, optimizer_data, metadata, *, numbered, milestone=False):
        if self._closed:
            raise RuntimeError("Cannot submit a checkpoint after the writer is closed")
        if self._pending is not None:
            raise RuntimeError("Wait for the preceding checkpoint before submitting another update")
        step = metadata.get("step")
        if type(step) is not int or step <= 0:
            raise ValueError("Checkpoint metadata must identify a positive completed update")
        started = time.monotonic()
        request = _Request(step, {}, {}, {}, numbered, milestone, 0.0)
        failure = None
        phase = "snapshot"
        gate = Event()
        admission = {}
        try:
            request = _Request(
                step, _snapshot_state(model_data) if self.rank == 0 else {},
                _snapshot_state(optimizer_data), json.loads(json.dumps(metadata, allow_nan=False)),
                numbered, milestone, time.monotonic() - started,
            )
            phase = "staging_setup"
            if self._stager is None:
                self._stager = CheckpointStateStager(self.device)
            if self._executor is not None:
                phase = "submit"
                self._pending = self._executor.submit(self._admitted_run, gate, admission, request)
        except BaseException as error:
            failure = _error(error, phase, self.rank)
        try:
            errors = [error for error in self._exchange(failure, self.control_group) if error is not None]
            admission["errors"] = errors
        except BaseException:
            # Release any parked worker without allowing checkpoint collectives.
            admission["errors"] = [{"phase": "admission", "rank": self.rank,
                                    "type": "DistributedCheckpointError", "fatal": True,
                                    "message": "Checkpoint admission collective failed"}]
            raise
        finally:
            gate.set()
        self._pending_step = step
        if self._pending is None:
            self._pending = Future()
            try:
                result = self._failed(request, errors) if errors else self._run(request)
                self._pending.set_result(result)
            except BaseException as error:
                self._pending.set_exception(error)
        if self.mode is CheckpointMode.SYNC:
            return self.wait(step)
        return None

    def _admitted_run(self, gate, admission, request):
        gate.wait()
        if admission["errors"]:
            return self._failed(request, admission["errors"])
        return self._run(request)

    def _run(self, request):
        started = time.monotonic()
        staged = None
        failure = None
        try:
            staged = self._stager.stage(request.model, request.optimizer)
            rank_state = {name: request.metadata[name]
                          for name in ("dataloader_state_dict", "rng_state")}
        except BaseException as error:
            failure = _error(error, "staging", self.rank)
            rank_state = None
        statuses = self._exchange((failure, rank_state), self.checkpoint_group)
        errors = [error for error, _ in statuses if error is not None]
        if errors:
            return self._failed(request, errors)
        staging_seconds = time.monotonic() - started
        gathering_started = time.monotonic()
        try:
            optimizer = self.layout.gather(staged["optimizer"], self.checkpoint_group)
        except OptimizerCheckpointError as error:
            # The layout exchanged validation/allocation errors before tensors.
            return self._failed(request, [_error(error, "optimizer_gather", 0)])
        # Transport failures remain fatal. A broken group cannot coordinate a
        # warning/retry; finite group timeout bounds the other workers' wait.
        gathering_seconds = time.monotonic() - gathering_started
        result = None
        if self.rank == 0:
            try:
                metadata = dict(request.metadata)
                metadata["distributed"] = {
                    "world_size": self.world_size,
                    "rank_states": [state for _, state in statuses],
                }
                try:
                    result = self._writer.submit(
                        staged["model"], optimizer, metadata,
                        numbered=request.numbered, milestone=request.milestone,
                    )
                except MilestoneCheckpointError:
                    # Broadcast durability failure before any rank raises it.
                    result = self._writer.measurements[-1]
                result = dict(result)
                result["snapshot_duration_seconds"] += request.snapshot_seconds
                result["staging_duration_seconds"] = staging_seconds
                result["gathering_duration_seconds"] = gathering_seconds
                result["duration_seconds"] = request.snapshot_seconds + time.monotonic() - started
            except BaseException as error:
                result = self._failed(request, [_error(error, "publication", self.rank)])
        if self.distributed:
            values = [result]
            root = dist.get_global_rank(self.checkpoint_group, 0) if self.checkpoint_group is not None else 0
            dist.broadcast_object_list(values, src=root, group=self.checkpoint_group)
            result = values[0]
        return result

    def wait(self, completed_step):
        if self._pending is None:
            return None
        pending, self._pending = self._pending, None
        self._pending_step = None
        started = time.monotonic()
        result = pending.result()
        if result["latest_published"]:
            self.latest_step = result["step"]
        result["join_wait_seconds"] = time.monotonic() - started
        result.update(self.status(completed_step))
        self.measurements.append(result)
        if self.rank == 0:
            logger.log(logging.WARNING if result["status"] == "warning" else logging.INFO,
                       json.dumps({"checkpoint_status": result}, allow_nan=False))
        for error in result["errors"]:
            if error.get("fatal"):
                _raise_fatal(error)
        if result["milestone"] and not result["milestone_published"]:
            detail = "; ".join(error["message"] for error in result["errors"])
            raise MilestoneCheckpointError(
                f"Required milestone at update {result['step']} was not durably preserved; "
                f"training stopped before the next update: {detail}"
            )
        return result

    def _shutdown(self):
        self._closed = True
        try:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
        finally:
            try:
                stager, self._stager = self._stager, None
                if stager is not None:
                    stager.close()
            finally:
                if self._writer is not None:
                    self._writer.close(self.latest_step or 0)

    def close(self, completed_step):
        if self._closed:
            return None
        try:
            return self.wait(completed_step)
        finally:
            self._shutdown()

    def abort(self):
        """Drain already-admitted work locally while preserving a training error."""
        if self._closed:
            return
        try:
            if self._pending is not None:
                self._pending.result()
        except BaseException:
            logger.exception("Checkpoint worker failed while aborting training")
        finally:
            self._pending = None
            self._pending_step = None
            try:
                self._shutdown()
            except BaseException:
                logger.exception("Checkpoint cleanup failed while aborting training")

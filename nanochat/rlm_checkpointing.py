"""Overlap one immutable checkpoint view with the next update's gradient work."""

from concurrent.futures import Future, ThreadPoolExecutor
import copy
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import time

import torch
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions

from nanochat.rlm_checkpoints import prune_checkpoints, save_numbered_checkpoint
from nanochat.rlm_latest import LatestCheckpoint


class CheckpointMode(str, Enum):
    SYNC = "sync"
    ASYNC = "async"


class CheckpointStagingSynchronizationError(RuntimeError):
    """CUDA could not drain a failed copy, so mutating live tensors is unsafe."""


def _snapshot_state(value):
    """Copy mutable containers and scalars while retaining tensor storage.

    Optimizer schedules can modify parameter groups while the worker runs. Their
    values must be frozen, as must the loader and RNG metadata. Tensor storage is
    protected separately by the caller's wait before the next optimizer step.
    """
    memo = {}
    visited = set()

    def collect(item):
        if id(item) in visited:
            return
        visited.add(id(item))
        if isinstance(item, torch.Tensor):
            memo[id(item)] = item.detach()
        elif isinstance(item, dict):
            for key, child in item.items():
                collect(key)
                collect(child)
            if hasattr(item, "_metadata"):
                collect(item._metadata)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)

    collect(value)
    return copy.deepcopy(value, memo)


@dataclass(frozen=True)
class _CheckpointRequest:
    step: int
    model: dict
    optimizer: dict
    metadata: dict
    numbered: bool
    snapshot_seconds: float


def _emit(name, value):
    print(json.dumps({name: value}, allow_nan=False), flush=True)


class CheckpointWriter:
    """Save latest, then a numbered checkpoint, then retire old numbered sets.

    The caller holds the run lock for this object's lifetime and synchronizes
    device work before submit. It may then compute the next forward/backward,
    but must call wait before any saved tensor can change, including optimizer
    mutation. close must run in finally so exception exits also drain the worker.
    Async CUDA saves stage into reusable pinned CPU storage on a separate copy
    stream. Successful staging waits only for that stream. A failed stage drains
    the device before allowing tensor mutation; a failed drain is a CUDA error.
    The worker does not run a memory monitor or peak reset.

    Ordinary snapshot/save/retention errors are warning results. The next update
    can retry publication, so durable latest may lag by more than one update.
    KeyboardInterrupt and other BaseException subclasses still propagate.
    """

    def __init__(self, checkpoint_dir: Path, *, mode: str | CheckpointMode,
                 keep_last: int | None, latest_step: int | None = None):
        self.mode = CheckpointMode(mode)
        if keep_last is not None and (type(keep_last) is not int or keep_last <= 0):
            raise ValueError("keep_last must be a positive integer or None")
        if latest_step is not None and (type(latest_step) is not int or latest_step <= 0):
            raise ValueError("latest_step must identify a positive completed update")
        self.store = LatestCheckpoint(checkpoint_dir)
        self.keep_last = keep_last
        self.latest_step = latest_step
        self.measurements = []
        self._executor = (ThreadPoolExecutor(max_workers=1, thread_name_prefix="rlm-checkpoint")
                          if self.mode is CheckpointMode.ASYNC else None)
        self._pending: Future | None = None
        self._pending_step: int | None = None
        self._stager: DefaultStager | None = None
        self._staging_device: torch.device | None = None
        self._closed = False

    def status(self, completed_step: int):
        return {
            "mode": self.mode.value,
            "completed_step": completed_step,
            "latest_path": str(self.store.path),
            "latest_step": self.latest_step,
            "latest_lag_completed_updates": (None if self.latest_step is None
                                              else completed_step - self.latest_step),
            "pending_step": self._pending_step,
        }

    def _measurement(self, step, numbered, snapshot_seconds):
        return {
            "step": step, "mode": self.mode.value, "numbered": numbered,
            "status": "saved", "errors": [],
            "snapshot_duration_seconds": snapshot_seconds,
            "duration_seconds": snapshot_seconds,
            "staging_duration_seconds": 0.0,
            "latest_duration_seconds": 0.0,
            "numbered_duration_seconds": 0.0,
            "retention_duration_seconds": 0.0,
            "latest_published": False, "numbered_published": False,
        }

    def _warning(self, result, phase, error):
        detail = {"phase": phase, "type": type(error).__name__, "message": str(error)}
        result["status"] = "warning"
        result["errors"].append(detail)
        _emit("checkpoint_warning", {
            "step": result["step"], "latest_path": str(self.store.path),
            "training_continues": True, **detail,
        })

    def submit(self, model_data, optimizer_data, metadata, *, numbered: bool):
        if self._closed:
            raise RuntimeError("Cannot submit a checkpoint after the writer is closed")
        if self._pending is not None:
            raise RuntimeError("Wait for the preceding checkpoint before submitting another update")
        step = metadata.get("step")
        if type(step) is not int or step <= 0:
            raise ValueError("Checkpoint metadata must identify a positive completed update")
        started = time.monotonic()
        phase = "snapshot"
        try:
            model_snapshot = _snapshot_state(model_data)
            optimizer_snapshot = _snapshot_state(optimizer_data)
            metadata_snapshot = json.loads(json.dumps(metadata, allow_nan=False))
            phase = "staging"
            self._prepare_stager(model_snapshot)
            request = _CheckpointRequest(
                step, model_snapshot, optimizer_snapshot, metadata_snapshot, numbered,
                time.monotonic() - started,
            )
        except Exception as exc:
            result = self._measurement(step, numbered, time.monotonic() - started)
            self._warning(result, phase, exc)
            self._pending = Future()
            self._pending.set_result(result)
        else:
            if self._executor is None:
                result = self._write(request)
                self._pending = Future()
                self._pending.set_result(result)
            else:
                try:
                    self._pending = self._executor.submit(self._write, request)
                except Exception as exc:
                    result = self._measurement(step, numbered, request.snapshot_seconds)
                    self._warning(result, "submit", exc)
                    self._pending = Future()
                    self._pending.set_result(result)
        self._pending_step = step
        if self.mode is CheckpointMode.SYNC:
            return self.wait(step)
        return None

    def _prepare_stager(self, model_data):
        """Create the dedicated staging stream on the training thread."""
        if self.mode is CheckpointMode.SYNC:
            return
        devices = {tensor.device for tensor in model_data.values()
                   if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda"}
        if not devices:
            return
        if len(devices) != 1:
            raise ValueError("Async RLM checkpoints require one CUDA device")
        device, = devices
        if self._stager is None:
            with torch.cuda.device(device):
                self._stager = DefaultStager(StagingOptions(
                    use_pinned_memory=True, use_shared_memory=False,
                    use_async_staging=True, use_non_blocking_copy=True,
                ))
            self._staging_device = device
        elif self._staging_device != device:
            raise ValueError("An RLM checkpoint writer cannot change CUDA devices")

    def _stage(self, request):
        if self._stager is None:
            return request.model, request.optimizer
        try:
            staged = self._stager.stage({
                "model": request.model, "optimizer": request.optimizer,
            }).result()
        except BaseException:
            # DefaultStager synchronizes its copy stream only on success. A
            # later allocation/copy can fail after earlier copies were queued.
            # Keep the request's source tensors alive until those copies drain.
            try:
                torch.cuda.synchronize(self._staging_device)
            except Exception as exc:
                raise CheckpointStagingSynchronizationError(
                    "CUDA could not synchronize after checkpoint staging failed; "
                    "live model and optimizer tensors cannot safely be updated"
                ) from exc
            raise
        return staged["model"], staged["optimizer"]

    def _write(self, request):
        result = self._measurement(request.step, request.numbered, request.snapshot_seconds)
        started = time.monotonic()
        phase = "staging"
        phase_started = started
        try:
            model_data, optimizer_data = self._stage(request)
            result["staging_duration_seconds"] = time.monotonic() - phase_started
            phase = "latest"
            phase_started = time.monotonic()
            self.store.save(model_data, optimizer_data, request.metadata)
            result["latest_published"] = True
            result["latest_duration_seconds"] = time.monotonic() - phase_started
            _emit("checkpoint_latest", {
                "step": request.step, "path": str(self.store.path), "durable": True,
            })
            if request.numbered:
                phase = "numbered"
                phase_started = time.monotonic()
                save_numbered_checkpoint(
                    self.store.checkpoint_dir, request.step,
                    model_data, optimizer_data, request.metadata,
                )
                result["numbered_published"] = True
                result["numbered_duration_seconds"] = time.monotonic() - phase_started
                if self.keep_last is not None:
                    phase = "retention"
                    phase_started = time.monotonic()
                    retention = prune_checkpoints(
                        self.store.checkpoint_dir, self.keep_last, request.step,
                    ).to_dict()
                    result["retention"] = retention
                    result["retention_duration_seconds"] = time.monotonic() - phase_started
                    _emit("checkpoint_retention", {"published_step": request.step, **retention})
        except CheckpointStagingSynchronizationError:
            raise
        except Exception as exc:
            result[f"{phase}_duration_seconds"] = time.monotonic() - phase_started
            self._warning(result, phase, exc)
        result["duration_seconds"] = request.snapshot_seconds + time.monotonic() - started
        return result

    def wait(self, completed_step: int):
        if self._pending is None:
            return None
        pending = self._pending
        self._pending = None
        self._pending_step = None
        started = time.monotonic()
        result = pending.result()
        if result["latest_published"]:
            self.latest_step = result["step"]
        result["join_wait_seconds"] = time.monotonic() - started
        result.update(self.status(completed_step))
        self.measurements.append(result)
        _emit("checkpoint_status", result)
        return result

    def close(self, completed_step: int):
        if self._closed:
            return None
        try:
            return self.wait(completed_step)
        finally:
            self._closed = True
            try:
                if self._executor is not None:
                    self._executor.shutdown(wait=True)
            finally:
                stager, self._stager = self._stager, None
                if stager is not None:
                    try:
                        stager.close()
                    except Exception as exc:
                        self._warning(self._measurement(completed_step, False, 0.0),
                                      "staging_close", exc)

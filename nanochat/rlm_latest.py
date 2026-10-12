"""Durable replacement of the latest complete RLM update, under the run lock."""

from dataclasses import dataclass
import ctypes
import json
import os
from pathlib import Path
import re
import shutil
import stat
from uuid import uuid4

import torch

from nanochat.checkpoint_manager import _tensor_bytes
from nanochat.rlm_checkpoints import CheckpointRetentionError, _open_directory
from nanochat.training_runtime import positive_gb


class LatestCheckpointError(RuntimeError):
    """A rolling checkpoint could not be safely read or durably published."""


def _publish_no_replace(directory_fd, source, destination):
    """Publish an immutable milestone atomically without overwriting a rival file."""
    # RLM recovery already requires Linux descriptor-backed mmap. RENAME_NOREPLACE
    # also avoids a check/rename race without introducing temporary hardlinks.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(directory_fd, os.fsencode(source), directory_fd, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination)


def _file_identity(status):
    return status.st_dev, status.st_ino, status.st_mtime_ns, status.st_size


def _read_bundle(stream, device, *, metadata_only=False):
    # The descriptor pins the verified inode across an atomic path replacement.
    source = f"/proc/self/fd/{stream.fileno()}" if metadata_only else stream
    return _bundle(torch.load(source, map_location=device, weights_only=True, mmap=metadata_only))


def _metadata(value):
    """Snapshot JSON metadata and validate the state needed to resume an update."""
    try:
        # In particular, torch.__version__ is a str subclass that the restricted
        # torch unpickler does not accept. Match the existing JSON checkpoints.
        value = json.loads(json.dumps(value, allow_nan=False))
        if not isinstance(value, dict) or value.get("model_type") != "rlm":
            raise ValueError("metadata must describe an RLM checkpoint")
        step = value.get("step")
        if type(step) is not int or step <= 0:
            raise ValueError("step must be a positive completed-update count")
        for name in ("model_config", "training_config", "dataloader_state_dict", "rng_state", "run_contract", "memory_contract"):
            if not isinstance(value.get(name), dict):
                raise ValueError(f"{name} must be a dictionary")
        contract = value["run_contract"]
        for name in ("model_config", "training_config", "tokenizer_sha256"):
            if name not in value or value[name] != contract.get(name):
                raise ValueError(f"{name} disagrees with run_contract")
        if not isinstance(value["tokenizer_sha256"], str) or not value["tokenizer_sha256"]:
            raise ValueError("tokenizer_sha256 must identify the tokenizer")
        model, training = value["model_config"], value["training_config"]
        for config, names in ((model, ("text_length", "memory_tokens")), (training, ("total_batch_size", "device_batch_size", "cycles"))):
            for name in names:
                if type(config.get(name)) is not int or config[name] <= 0:
                    raise ValueError(f"{name} must be a positive integer")
        text, memory = model["text_length"], model["memory_tokens"]
        batch, cycles = training["total_batch_size"], training["cycles"]
        microbatch = training["device_batch_size"] * cycles * text
        if cycles < 2 or batch % microbatch:
            raise ValueError("training batch is inconsistent with complete RLM episodes")
        counts = {
            "text_tokens": step * batch,
            "supervised_tokens": step * (batch + batch // text),
            "internal_positions": step * batch // (cycles * text)
            * (cycles * (text + 1) + 2 * (cycles - 1) * memory),
        }
        for name, expected in counts.items():
            if type(value.get(name)) is not int or value[name] != expected:
                raise ValueError(f"{name} is inconsistent with step and training configuration")
        rng = value["rng_state"]
        for name in ("cpu", "cuda"):
            state = rng.get(name)
            if name == "cuda" and state is None and name in rng:
                continue
            if not isinstance(state, list) or not state or any(type(byte) is not int or not 0 <= byte <= 255 for byte in state):
                raise ValueError(f"{name} RNG state must contain bytes")
        if contract.get("device_type") == "cuda" and rng["cuda"] is None:
            raise ValueError("CUDA checkpoint requires CUDA RNG state")
        if value["memory_contract"].get("gradient_horizon") != cycles:
            raise ValueError("memory gradient horizon disagrees with training cycles")
        if value.get("data_resume") != "approximate_row_group":
            raise ValueError("data_resume must describe the RLM loader state")
        return value
    except (ValueError, TypeError, KeyError) as exc:
        raise LatestCheckpointError(f"Invalid latest checkpoint metadata: {exc}") from exc


def _bundle(value):
    if not isinstance(value, dict) or set(value) != {"model", "optimizer", "metadata"}:
        raise LatestCheckpointError("Latest checkpoint must contain model, optimizer, and metadata together")
    model, optimizer = value["model"], value["optimizer"]
    if not isinstance(model, dict) or not model or any(not isinstance(name, str) or not isinstance(tensor, torch.Tensor) for name, tensor in model.items()):
        raise LatestCheckpointError("Latest checkpoint model must contain named tensors")
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict) or not isinstance(optimizer.get("param_groups"), list) or not optimizer["param_groups"]:
        raise LatestCheckpointError("Latest checkpoint requires complete optimizer state and parameter groups")
    return model, optimizer, _metadata(value["metadata"])


@dataclass(frozen=True)
class LatestCheckpoint:
    """One atomic bundle; callers must hold the run lock throughout save/resume.

    Serialization is synchronous and uses the supplied tensor storage directly.
    The previous bundle remains published until its replacement has been flushed
    and fsynced. Readers therefore observe either complete update, never a mix.
    """

    checkpoint_dir: Path

    @property
    def path(self):
        return self.checkpoint_dir / "latest.pt"

    @property
    def milestone_path(self):
        return self.checkpoint_dir / "pretrain-90pct.pt"

    def _directory_fd(self):
        try:
            descriptor = _open_directory(self.checkpoint_dir)
        except CheckpointRetentionError as exc:
            raise LatestCheckpointError(str(exc)) from exc
        if os.fstat(descriptor).st_uid != os.geteuid():
            os.close(descriptor)
            raise LatestCheckpointError(f"Checkpoint directory must be owned by the current user: {self.checkpoint_dir}")
        return descriptor

    @staticmethod
    def _require_owned_file(status, name, *, nonempty=True):
        if not stat.S_ISREG(status.st_mode) or status.st_uid != os.geteuid() or status.st_nlink != 1:
            raise LatestCheckpointError(f"Checkpoint file must be an owned regular file without symlinks or hardlinks: {name}")
        if nonempty and status.st_size <= 0:
            raise LatestCheckpointError(f"Checkpoint file must be nonempty: {name}")

    def _inspect_file(self, directory_fd, name):
        try:
            status = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        self._require_owned_file(status, self.checkpoint_dir / name)
        return status

    def _inspect_latest(self, directory_fd):
        return self._inspect_file(directory_fd, self.path.name)

    def exists(self):
        directory_fd = self._directory_fd()
        try:
            return self._inspect_latest(directory_fd) is not None
        except OSError as exc:
            raise LatestCheckpointError(f"Cannot inspect latest checkpoint {self.path}: {exc}") from exc
        finally:
            os.close(directory_fd)

    def _clear_staging(self, directory_fd, stem="latest"):
        candidates = []
        for name in os.listdir(directory_fd):
            if not name.startswith(f".{stem}."):
                continue
            if re.fullmatch(rf"\.{re.escape(stem)}\.[0-9a-f]{{32}}\.tmp", name) is None:
                raise LatestCheckpointError(f"Unexpected rolling checkpoint staging name: {name}")
            status = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            self._require_owned_file(status, name, nonempty=False)
            candidates.append((name, status))
        # Validate every candidate before removing any interrupted staging file.
        for name, original in candidates:
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if (current.st_dev, current.st_ino, current.st_mtime_ns) != (original.st_dev, original.st_ino, original.st_mtime_ns):
                raise LatestCheckpointError(f"Rolling checkpoint staging file changed during cleanup: {name}")
            os.unlink(name, dir_fd=directory_fd)

    def save(self, model_data, optimizer_data, metadata):
        model_data, optimizer_data, metadata = _bundle({
            "model": model_data, "optimizer": optimizer_data, "metadata": metadata,
        })
        directory_fd = self._directory_fd()
        temporary = None
        published = False
        try:
            self._inspect_latest(directory_fd)
            self._clear_staging(directory_fd)
            reserve = os.environ.get("NANOCHAT_CHECKPOINT_RESERVE_GB")
            if reserve is not None:
                required = positive_gb(reserve, "NANOCHAT_CHECKPOINT_RESERVE_GB")
                required += 1.05 * (_tensor_bytes(model_data) + _tensor_bytes(optimizer_data))
                required += len(json.dumps(metadata).encode("utf-8"))
                filesystem = os.fstatvfs(directory_fd)
                if filesystem.f_bavail * filesystem.f_frsize < required:
                    raise LatestCheckpointError("Insufficient disk space for latest checkpoint staging and configured free-space reserve")
            name = f".latest.{uuid4().hex}.tmp"
            descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
            temporary = name
            with os.fdopen(descriptor, "wb") as stream:
                torch.save({"model": model_data, "optimizer": optimizer_data, "metadata": metadata}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            self._inspect_latest(directory_fd)
            os.replace(temporary, "latest.pt", src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            temporary = None
            published = True
            os.fsync(directory_fd)
            return self.path
        except Exception as exc:
            detail = "was replaced but directory durability could not be confirmed" if published else "was not replaced"
            raise LatestCheckpointError(f"Latest checkpoint {detail}: {exc}") from exc
        finally:
            try:
                if temporary is not None:
                    os.unlink(temporary, dir_fd=directory_fd)
            finally:
                os.close(directory_fd)

    def _load(self, device, *, metadata_only=False, name="latest.pt"):
        directory_fd = self._directory_fd()
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
            with os.fdopen(descriptor, "rb") as stream:
                self._require_owned_file(os.fstat(stream.fileno()), self.checkpoint_dir / name)
                return _read_bundle(stream, device, metadata_only=metadata_only)
        except Exception as exc:
            raise LatestCheckpointError(f"Cannot load checkpoint {self.checkpoint_dir / name}: {exc}") from exc
        finally:
            os.close(directory_fd)

    def load(self, device):
        return self._load(device)

    def metadata(self):
        """Read resume metadata without allocating the model/optimizer storage."""
        return self._load(torch.device("cpu"), metadata_only=True)[2]

    def verify_milestone(self, step, *, run_contract=None):
        """Validate the retained boundary, including the caller's resume contract."""
        if type(step) is not int or step <= 0:
            raise ValueError("Milestone step must identify a positive completed update")
        metadata = self._load(torch.device("cpu"), metadata_only=True, name=self.milestone_path.name)[2]
        if metadata["step"] != step:
            raise LatestCheckpointError(f"Milestone checkpoint must preserve update {step}, found {metadata['step']}")
        if run_contract is not None and metadata["run_contract"] != run_contract:
            raise LatestCheckpointError("Milestone checkpoint belongs to a different run contract")
        return metadata

    def _matching_milestone(self, directory_fd, source, metadata):
        """A retry may reuse only the exact already-published source bundle."""
        if self._inspect_file(directory_fd, self.milestone_path.name) is None:
            return False
        descriptor = os.open(self.milestone_path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
        with os.fdopen(descriptor, "rb") as existing:
            status = os.fstat(existing.fileno())
            self._require_owned_file(status, self.milestone_path)
            if _read_bundle(existing, torch.device("cpu"), metadata_only=True)[2] != metadata:
                raise LatestCheckpointError("Existing milestone checkpoint conflicts with latest metadata")
            source.seek(0)
            existing.seek(0)
            while True:
                block = source.read(8 * 1024 * 1024)
                if block != existing.read(8 * 1024 * 1024):
                    raise LatestCheckpointError("Existing milestone checkpoint conflicts with latest bundle")
                if not block:
                    break
            current = self._inspect_file(directory_fd, self.milestone_path.name)
            if current is None or _file_identity(current) != _file_identity(status):
                raise LatestCheckpointError("Milestone checkpoint changed during verification")
            os.fsync(existing.fileno())
        return True

    def preserve_milestone(self, step):
        """Keep an exact durable copy of latest without GPU work or reserialization.

        This file survives both rolling replacement and numbered retention. The
        caller holds the run lock and must stop before unfreezing if this fails.
        """
        if type(step) is not int or step <= 0:
            raise ValueError("Milestone step must identify a positive completed update")
        directory_fd = self._directory_fd()
        temporary = None
        published = False
        try:
            self._clear_staging(directory_fd, "pretrain-90pct")
            descriptor = os.open(self.path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
            with os.fdopen(descriptor, "rb") as source:
                original = os.fstat(source.fileno())
                self._require_owned_file(original, self.path)
                metadata = _read_bundle(source, torch.device("cpu"), metadata_only=True)[2]
                if metadata["step"] != step:
                    raise LatestCheckpointError(f"Latest checkpoint must be update {step} to preserve the milestone, found {metadata['step']}")
                if self._matching_milestone(directory_fd, source, metadata):
                    current = self._inspect_latest(directory_fd)
                    if current is None or _file_identity(current) != _file_identity(original):
                        raise LatestCheckpointError("Latest checkpoint changed while verifying the milestone")
                    os.fsync(directory_fd)
                    return self.milestone_path
                reserve = os.environ.get("NANOCHAT_CHECKPOINT_RESERVE_GB")
                required = original.st_size + (0 if reserve is None else positive_gb(reserve, "NANOCHAT_CHECKPOINT_RESERVE_GB"))
                filesystem = os.fstatvfs(directory_fd)
                if filesystem.f_bavail * filesystem.f_frsize < required:
                    raise LatestCheckpointError("Insufficient disk space for retained milestone and configured free-space reserve")
                name = f".pretrain-90pct.{uuid4().hex}.tmp"
                descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
                temporary = name
                with os.fdopen(descriptor, "wb") as target:
                    source.seek(0)
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
                    if os.fstat(target.fileno()).st_size != original.st_size:
                        raise LatestCheckpointError("Milestone copy does not match latest checkpoint size")
                current = self._inspect_latest(directory_fd)
                if current is None or _file_identity(current) != _file_identity(original):
                    raise LatestCheckpointError("Latest checkpoint changed while preserving the milestone")
                if self._matching_milestone(directory_fd, source, metadata):
                    os.unlink(temporary, dir_fd=directory_fd)
                else:
                    _publish_no_replace(directory_fd, temporary, self.milestone_path.name)
                temporary = None
                published = True
                os.fsync(directory_fd)
                return self.milestone_path
        except Exception as exc:
            detail = "was published but directory durability could not be confirmed" if published else "was not published"
            raise LatestCheckpointError(f"Milestone checkpoint {detail}: {exc}") from exc
        finally:
            try:
                if temporary is not None:
                    os.unlink(temporary, dir_fd=directory_fd)
            finally:
                os.close(directory_fd)

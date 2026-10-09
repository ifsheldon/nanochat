"""Retain complete checkpoints within one locked RLM run directory."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import stat
from uuid import uuid4


class CheckpointRetentionError(RuntimeError):
    """Retention stopped because checkpoint safety or completeness was not proven."""


class NumberedCheckpointError(RuntimeError):
    """A numbered checkpoint could not be safely published or recovered."""


@dataclass(frozen=True)
class CheckpointRetention:
    """Deletion counts and paths cover only work performed by this invocation."""

    retained_steps: tuple[int, ...]
    deleted_steps: tuple[int, ...]
    deleted_bytes: int
    deleted_paths: tuple[Path, ...]

    def to_dict(self):
        return {
            "retained_steps": list(self.retained_steps),
            "deleted_steps": list(self.deleted_steps),
            "deleted_bytes": self.deleted_bytes,
            "deleted_paths": [str(path) for path in self.deleted_paths],
        }


@dataclass(frozen=True)
class _CheckpointFile:
    path: Path
    size: int
    identity: tuple[int, int, int]

    @classmethod
    def inspect(cls, directory_fd, path, *, missing_ok=False, nonempty=True):
        try:
            status = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError as exc:
            if missing_ok:
                return None
            raise CheckpointRetentionError(f"Cannot inspect checkpoint file {path}: {exc}") from exc
        except OSError as exc:
            raise CheckpointRetentionError(f"Cannot inspect checkpoint file {path}: {exc}") from exc
        if not stat.S_ISREG(status.st_mode) or status.st_uid != os.geteuid() or status.st_nlink != 1:
            raise CheckpointRetentionError(f"Checkpoint file must be an owned regular file without symlinks or hardlinks: {path}")
        if nonempty and status.st_size <= 0:
            raise CheckpointRetentionError(f"Checkpoint file must be a nonempty regular file: {path}")
        return cls(path, status.st_size, (status.st_dev, status.st_ino, status.st_mtime_ns))

    def to_dict(self):
        return {"name": self.path.name, "size": self.size, "identity": list(self.identity)}

    @classmethod
    def from_dict(cls, value, expected_path):
        if not isinstance(value, dict) or set(value) != {"name", "size", "identity"}:
            raise CheckpointRetentionError("Retirement journal contains an invalid checkpoint file record")
        if value["name"] != expected_path.name:
            raise CheckpointRetentionError("Retirement journal must name exactly the canonical checkpoint files")
        if type(value["size"]) is not int or value["size"] <= 0:
            raise CheckpointRetentionError("Retirement journal file sizes must be positive integers")
        identity = value["identity"]
        if not isinstance(identity, list) or len(identity) != 3 or any(type(part) is not int or part < 0 for part in identity):
            raise CheckpointRetentionError("Retirement journal file identities must contain device, inode, and modification time")
        return cls(expected_path, value["size"], tuple(identity))


@dataclass(frozen=True)
class _Checkpoint:
    step: int
    files: tuple[_CheckpointFile, ...]

    @staticmethod
    def step_from_path(path):
        match = re.fullmatch(r"(model|meta)_([0-9]+)\.(pt|json)|optim_([0-9]+)_rank0\.pt", path.name)
        if match is None:
            return None
        kind = "optim" if match[4] is not None else match[1]
        step = int(match[4] if kind == "optim" else match[2])
        suffix = "_rank0.pt" if kind == "optim" else ".json" if kind == "meta" else ".pt"
        canonical = f"{kind}_{step:06d}{suffix}"
        if path.name != canonical:
            raise CheckpointRetentionError(f"Noncanonical checkpoint filename: {path}")
        return step

    @classmethod
    def paths(cls, directory, step):
        return (
            directory / f"meta_{step:06d}.json",
            directory / f"model_{step:06d}.pt",
            directory / f"optim_{step:06d}_rank0.pt",
        )

    @classmethod
    def inspect(cls, directory_fd, directory, step):
        paths = cls.paths(directory, step)
        files = tuple(_CheckpointFile.inspect(directory_fd, path) for path in paths)
        try:
            metadata_fd = os.open(paths[0].name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
            with os.fdopen(metadata_fd, encoding="utf-8") as stream:
                status = os.fstat(stream.fileno())
                if (status.st_dev, status.st_ino, status.st_mtime_ns) != files[0].identity or status.st_size != files[0].size:
                    raise CheckpointRetentionError(f"Checkpoint metadata changed during inspection: {paths[0]}")
                metadata = json.load(stream)
        except (OSError, ValueError) as exc:
            raise CheckpointRetentionError(f"Cannot read published checkpoint metadata {paths[0]}: {exc}") from exc
        if not isinstance(metadata, dict) or type(metadata.get("step")) is not int or metadata["step"] != step:
            raise CheckpointRetentionError(f"Checkpoint metadata step does not match filename: {paths[0]}")
        return cls(step, files)

    def to_dict(self):
        return {"step": self.step, "files": [file.to_dict() for file in self.files]}

    @classmethod
    def from_dict(cls, directory, value):
        if not isinstance(value, dict) or set(value) != {"step", "files"}:
            raise CheckpointRetentionError("Retirement journal contains an invalid checkpoint record")
        step, files = value["step"], value["files"]
        if type(step) is not int or step < 0 or not isinstance(files, list) or len(files) != 3:
            raise CheckpointRetentionError("Retirement journal checkpoints require a step and three file identities")
        return cls(step, tuple(_CheckpointFile.from_dict(file, path) for file, path in zip(files, cls.paths(directory, step))))


def _open_directory(directory: Path):
    if not isinstance(directory, Path) or not directory.is_absolute() or ".." in directory.parts:
        raise CheckpointRetentionError("Checkpoint directory must be an absolute Path without parent traversal")
    if directory == Path(directory.anchor):
        raise CheckpointRetentionError("A filesystem root cannot be a checkpoint directory")
    descriptor = None
    try:
        descriptor = os.open(directory.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for component in directory.parts[1:]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise CheckpointRetentionError(f"Cannot open checkpoint directory without symlinks {directory}: {exc}") from exc


def _unique_json_keys(pairs):
    value = {}
    for name, item in pairs:
        if name in value:
            raise CheckpointRetentionError(f"Retirement journal contains a duplicate JSON key: {name}")
        value[name] = item
    return value


@dataclass(frozen=True)
class _Retirement:
    published_step: int
    keep_last: int
    retained_steps: tuple[int, ...]
    retired: tuple[_Checkpoint, ...]

    def to_dict(self):
        return {
            "version": 1, "published_step": self.published_step, "keep_last": self.keep_last,
            "retained_steps": list(self.retained_steps),
            "retired": [checkpoint.to_dict() for checkpoint in self.retired],
        }

    @classmethod
    def read(cls, directory_fd, directory, published_step, keep_last):
        path = directory / ".retention.json"
        journal = _CheckpointFile.inspect(directory_fd, path, missing_ok=True)
        if journal is None:
            return None, None
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
        try:
            with os.fdopen(descriptor, encoding="utf-8") as stream:
                status = os.fstat(stream.fileno())
                if (status.st_dev, status.st_ino, status.st_mtime_ns) != journal.identity or status.st_size != journal.size:
                    raise CheckpointRetentionError("Retirement journal changed during inspection")
                value = json.load(stream, object_pairs_hook=_unique_json_keys)
        except ValueError as exc:
            raise CheckpointRetentionError(f"Invalid retirement journal JSON: {exc}") from exc
        required = {"version", "published_step", "keep_last", "retained_steps", "retired"}
        if not isinstance(value, dict) or set(value) != required:
            raise CheckpointRetentionError("Retirement journal has missing or unexpected fields")
        if type(value["version"]) is not int or value["version"] != 1:
            raise CheckpointRetentionError("Unsupported retirement journal version")
        if type(value["published_step"]) is not int or not 0 <= value["published_step"] <= published_step:
            raise CheckpointRetentionError("Retirement journal published_step is newer than the requested retirement")
        if type(value["keep_last"]) is not int or value["keep_last"] != keep_last:
            raise CheckpointRetentionError("Retirement journal keep_last differs from the requested retirement")
        retained = value["retained_steps"]
        if not isinstance(retained, list) or not retained or any(type(step) is not int or step < 0 for step in retained):
            raise CheckpointRetentionError("Retirement journal must identify retained checkpoint steps")
        records = value["retired"]
        if not isinstance(records, list) or not records:
            raise CheckpointRetentionError("Retirement journal must identify retired checkpoints")
        retired = tuple(_Checkpoint.from_dict(directory, record) for record in records)
        steps = [checkpoint.step for checkpoint in retired] + retained
        if steps != sorted(set(steps)) or len(retained) != keep_last or retained[-1] != value["published_step"]:
            raise CheckpointRetentionError("Retirement journal must preserve exactly the newest requested checkpoint steps")
        return cls(value["published_step"], keep_last, tuple(retained), retired), journal

    def publish(self, directory_fd, directory):
        path = directory / ".retention.json"
        staging = directory / ".retention.json.tmp"
        # A stop before journal publication cannot have deleted checkpoint files.
        # Only this one owned staging file may be discarded after full validation.
        stale = _CheckpointFile.inspect(directory_fd, staging, missing_ok=True, nonempty=False)
        if stale is not None:
            os.unlink(staging.name, dir_fd=directory_fd)
        descriptor = os.open(staging.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging.name, path.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
        return _CheckpointFile.inspect(directory_fd, path)

    def validate_remaining(self, directory_fd, directory, existing_steps):
        known = set(self.retained_steps) | {checkpoint.step for checkpoint in self.retired}
        newer = existing_steps - known
        if any(step <= self.published_step for step in newer):
            raise CheckpointRetentionError("Unlisted checkpoint steps appeared during an interrupted retirement")
        for step in (*self.retained_steps, *sorted(newer)):
            _Checkpoint.inspect(directory_fd, directory, step)
        for checkpoint in self.retired:
            for file in checkpoint.files:
                actual = _CheckpointFile.inspect(directory_fd, file.path, missing_ok=True)
                if actual is not None and actual != file:
                    raise CheckpointRetentionError(f"Checkpoint file changed during interrupted retirement: {file.path}")


def prune_checkpoints(checkpoint_dir: Path, keep_last: int, published_step: int) -> CheckpointRetention:
    """Retain the newest complete sets after a successful save, under the run lock.

    Every managed set must contain regular nonempty model, rank-zero optimizer,
    and valid metadata files before anything is removed. Partial or corrupt sets
    stop retention rather than being silently skipped. Unrelated files remain
    untouched. A durable journal records exact file identities before retirement.
    A retry may finish its listed, partly removed sets after validating all kept
    sets and every remaining retired file. Newer published sets are validated
    before finishing the old retirement and applying the current retention.
    An unjournaled partial set still fails.
    Metadata is removed first. Returned steps, bytes, and paths count only files
    actually removed by this invocation, excluding deletions before a prior stop.
    """
    if type(keep_last) is not int or keep_last <= 0:
        raise CheckpointRetentionError("keep_last must be a positive integer")
    if type(published_step) is not int or published_step < 0:
        raise CheckpointRetentionError("published_step must be a nonnegative integer")
    directory_fd = _open_directory(checkpoint_dir)
    deleted_paths = []
    deleted_bytes = 0
    deleted_steps = []
    try:
        if os.fstat(directory_fd).st_uid != os.geteuid():
            raise CheckpointRetentionError("Checkpoint directory must be owned by the current user")
        steps = set()
        for name in os.listdir(directory_fd):
            step = _Checkpoint.step_from_path(checkpoint_dir / name)
            if step is not None:
                steps.add(step)
        if not steps or published_step != max(steps):
            raise CheckpointRetentionError(f"Just-published checkpoint {published_step} is not the newest managed checkpoint in {checkpoint_dir}")
        plan, journal = _Retirement.read(directory_fd, checkpoint_dir, published_step, keep_last)
        if plan is None:
            checkpoints = [_Checkpoint.inspect(directory_fd, checkpoint_dir, step) for step in sorted(steps)]
            retired, retained = checkpoints[:-keep_last], checkpoints[-keep_last:]
            if not retired:
                return CheckpointRetention(tuple(checkpoint.step for checkpoint in retained), (), 0, ())
            plan = _Retirement(published_step, keep_last, tuple(checkpoint.step for checkpoint in retained), tuple(retired))
            journal = plan.publish(directory_fd, checkpoint_dir)
        else:
            plan.validate_remaining(directory_fd, checkpoint_dir, steps)
            # Also cover a previous failure to fsync the journal's publication.
            os.fsync(directory_fd)
        for checkpoint in plan.retired:
            removed_step = False
            for file in checkpoint.files:
                # Detect a replacement before unlinking. The caller's run lock
                # must remain held throughout validation and retirement.
                actual = _CheckpointFile.inspect(directory_fd, file.path, missing_ok=True)
                if actual is None:
                    continue
                if actual != file:
                    raise CheckpointRetentionError(f"Checkpoint file changed during retention: {file.path}")
                os.unlink(file.path.name, dir_fd=directory_fd)
                deleted_paths.append(file.path)
                deleted_bytes += file.size
                removed_step = True
            if removed_step:
                deleted_steps.append(checkpoint.step)
        # Persist all removals before forgetting the only authority to finish a
        # partial set. A crash before clearing the journal safely repeats this.
        os.fsync(directory_fd)
        if _CheckpointFile.inspect(directory_fd, journal.path) != journal:
            raise CheckpointRetentionError("Retirement journal changed before completion")
        os.unlink(journal.path.name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        if plan.published_step != published_step:
            # The previous save warned and continued. Finish its exact journal
            # before planning a new retirement for the newer published update.
            remaining = prune_checkpoints(checkpoint_dir, keep_last, published_step)
            return CheckpointRetention(
                remaining.retained_steps, tuple(deleted_steps) + remaining.deleted_steps,
                deleted_bytes + remaining.deleted_bytes, tuple(deleted_paths) + remaining.deleted_paths,
            )
        return CheckpointRetention(plan.retained_steps, tuple(deleted_steps), deleted_bytes, tuple(deleted_paths))
    except (OSError, CheckpointRetentionError) as exc:
        removed = ", ".join(str(path) for path in deleted_paths) or "none"
        raise CheckpointRetentionError(f"Checkpoint retention stopped: {exc}; files already removed: {removed}") from exc
    finally:
        os.close(directory_fd)


@dataclass(frozen=True)
class _Publication:
    """Exact staged inodes authorized for one numbered checkpoint publication."""

    token: str
    checkpoint: _Checkpoint

    def staging_path(self, file):
        kind = file.path.name.split("_", 1)[0]
        return file.path.parent / f".numbered.{self.token}.{kind}.tmp"

    def to_dict(self):
        return {"version": 1, "token": self.token, "checkpoint": self.checkpoint.to_dict()}

    @classmethod
    def read(cls, directory_fd, directory):
        path = directory / ".numbered-publication.json"
        journal = _CheckpointFile.inspect(directory_fd, path, missing_ok=True)
        if journal is None:
            return None, None
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            status = os.fstat(stream.fileno())
            if (status.st_dev, status.st_ino, status.st_mtime_ns) != journal.identity or status.st_size != journal.size:
                raise NumberedCheckpointError("Numbered publication journal changed during inspection")
            value = json.load(stream, object_pairs_hook=_unique_json_keys)
        if not isinstance(value, dict) or set(value) != {"version", "token", "checkpoint"}:
            raise NumberedCheckpointError("Numbered publication journal has missing or unexpected fields")
        if type(value["version"]) is not int or value["version"] != 1:
            raise NumberedCheckpointError("Unsupported numbered publication journal version")
        if not isinstance(value["token"], str) or re.fullmatch(r"[0-9a-f]{32}", value["token"]) is None:
            raise NumberedCheckpointError("Numbered publication journal has an invalid staging token")
        return cls(value["token"], _Checkpoint.from_dict(directory, value["checkpoint"])), journal

    def publish_journal(self, directory_fd, directory):
        staging = directory / ".numbered-publication.json.tmp"
        stale = _CheckpointFile.inspect(directory_fd, staging, missing_ok=True, nonempty=False)
        if stale is not None:
            os.unlink(staging.name, dir_fd=directory_fd)
        descriptor = os.open(staging.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        path = directory / ".numbered-publication.json"
        os.replace(staging.name, path.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
        return _CheckpointFile.inspect(directory_fd, path)

    def finish(self, directory_fd, directory, journal):
        pending = []
        # Validate the entire publication before replacing any file. A target
        # may exist only if it is precisely the staged inode already published.
        for expected in self.checkpoint.files:
            source = self.staging_path(expected)
            staged = _CheckpointFile.inspect(directory_fd, source, missing_ok=True)
            target = _CheckpointFile.inspect(directory_fd, expected.path, missing_ok=True)
            if target is not None:
                if target != expected or staged is not None:
                    raise NumberedCheckpointError(f"Numbered publication target changed: {expected.path}")
            elif staged is None or (staged.size, staged.identity) != (expected.size, expected.identity):
                raise NumberedCheckpointError(f"Numbered publication staging file changed or disappeared: {source}")
            else:
                pending.append((source, expected.path))
        # Also handles a prior failure while syncing journal publication.
        os.fsync(directory_fd)
        # Metadata is the final publication marker. Its presence always implies
        # that both tensor files have already been moved into place.
        pending.sort(key=lambda pair: pair[1].suffix == ".json")
        for source, target in pending:
            os.replace(source.name, target.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
        _Checkpoint.inspect(directory_fd, directory, self.checkpoint.step)
        if _CheckpointFile.inspect(directory_fd, journal.path) != journal:
            raise NumberedCheckpointError("Numbered publication journal changed before completion")
        os.unlink(journal.path.name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        return self.checkpoint.step


def _clear_numbered_staging(directory_fd, directory):
    candidates = []
    for name in os.listdir(directory_fd):
        if not name.startswith(".numbered."):
            continue
        if re.fullmatch(r"\.numbered\.[0-9a-f]{32}\.(meta|model|optim)\.tmp", name) is None:
            raise NumberedCheckpointError(f"Unexpected numbered checkpoint staging name: {name}")
        candidates.append(_CheckpointFile.inspect(directory_fd, directory / name, nonempty=False))
    for original in candidates:
        if _CheckpointFile.inspect(directory_fd, original.path, nonempty=False) != original:
            raise NumberedCheckpointError(f"Numbered checkpoint staging file changed during cleanup: {original.path}")
        os.unlink(original.path.name, dir_fd=directory_fd)


def recover_numbered_checkpoint(checkpoint_dir: Path) -> int | None:
    """Finish an exact journaled publication under the run lock, if one exists.

    Only the recorded staged inodes may be published. No existing checkpoint is
    deleted or replaced, and unjournaled partial numbered sets are not repaired.
    """
    directory_fd = None
    try:
        directory_fd = _open_directory(checkpoint_dir)
        if os.fstat(directory_fd).st_uid != os.geteuid():
            raise NumberedCheckpointError("Checkpoint directory must be owned by the current user")
        plan, journal = _Publication.read(directory_fd, checkpoint_dir)
        return None if plan is None else plan.finish(directory_fd, checkpoint_dir, journal)
    except Exception as exc:
        raise NumberedCheckpointError(f"Numbered checkpoint recovery failed: {exc}") from exc
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def save_numbered_checkpoint(checkpoint_dir: Path, step: int, model_data, optimizer_data, metadata):
    """Durably publish one complete numbered set without retiring older sets.

    Tensor and metadata files are fully staged and synced before recording a
    durable publication journal. If publication fails, a subsequent call or
    explicit recovery finishes the exact staged update before writing a new one.
    Failure before journal publication leaves only hidden staging files, which
    the next save safely removes. The caller must hold the run directory lock.
    """
    if type(step) is not int or step < 0:
        raise NumberedCheckpointError("Checkpoint step must be a nonnegative integer")
    if not isinstance(metadata, dict) or type(metadata.get("step")) is not int or metadata["step"] != step:
        raise NumberedCheckpointError("Checkpoint metadata step must match the numbered checkpoint")
    if not isinstance(model_data, dict) or not model_data or not isinstance(optimizer_data, dict) or not optimizer_data:
        raise NumberedCheckpointError("Numbered checkpoint requires model and optimizer state dictionaries")
    recovered = recover_numbered_checkpoint(checkpoint_dir)
    if recovered == step:
        return step
    directory_fd = None
    try:
        import torch
        from nanochat.checkpoint_manager import _tensor_bytes
        from nanochat.training_runtime import positive_gb

        directory_fd = _open_directory(checkpoint_dir)
        paths = _Checkpoint.paths(checkpoint_dir, step)
        for path in paths:
            if _CheckpointFile.inspect(directory_fd, path, missing_ok=True, nonempty=False) is not None:
                raise NumberedCheckpointError(f"Refusing to replace an existing numbered checkpoint file: {path}")
        _clear_numbered_staging(directory_fd, checkpoint_dir)
        reserve = os.environ.get("NANOCHAT_CHECKPOINT_RESERVE_GB")
        if reserve is not None:
            required = positive_gb(reserve, "NANOCHAT_CHECKPOINT_RESERVE_GB")
            required += 1.05 * (_tensor_bytes(model_data) + _tensor_bytes(optimizer_data))
            required += len(json.dumps(metadata, allow_nan=False).encode("utf-8"))
            filesystem = os.fstatvfs(directory_fd)
            if filesystem.f_bavail * filesystem.f_frsize < required:
                raise NumberedCheckpointError("Insufficient disk space for numbered checkpoint staging and configured free-space reserve")
        token = uuid4().hex
        files = []
        for path, value in zip(paths, (metadata, model_data, optimizer_data)):
            kind = path.name.split("_", 1)[0]
            staging = checkpoint_dir / f".numbered.{token}.{kind}.tmp"
            descriptor = os.open(staging.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
            with os.fdopen(descriptor, "wb") as stream:
                if kind == "meta":
                    stream.write(json.dumps(value, allow_nan=False).encode("utf-8"))
                else:
                    torch.save(value, stream)
                stream.flush()
                os.fsync(stream.fileno())
            staged = _CheckpointFile.inspect(directory_fd, staging)
            files.append(_CheckpointFile(path, staged.size, staged.identity))
        plan = _Publication(token, _Checkpoint(step, tuple(files)))
        journal = plan.publish_journal(directory_fd, checkpoint_dir)
        return plan.finish(directory_fd, checkpoint_dir, journal)
    except Exception as exc:
        raise NumberedCheckpointError(f"Numbered checkpoint {step} could not be published: {exc}") from exc
    finally:
        if directory_fd is not None:
            os.close(directory_fd)

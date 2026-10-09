"""Explicit, hash-bound approval of a reviewed checkpoint workflow change."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat


class ResumeTransitionError(ValueError):
    """A receipt does not authorize this checkpoint and target run contract."""


class _Digest(str):
    def __new__(cls, value):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ResumeTransitionError("SHA-256 identities must contain 64 lowercase hexadecimal characters")
        return super().__new__(cls, value)

    @classmethod
    def of_bytes(cls, value):
        return cls(hashlib.sha256(value).hexdigest())


@dataclass(frozen=True)
class _JSON:
    encoded: bytes

    @classmethod
    def from_value(cls, value):
        try:
            return cls(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        except (TypeError, ValueError) as exc:
            raise ResumeTransitionError(f"Transition records must contain finite JSON values: {exc}") from exc

    def value(self):
        return json.loads(self.encoded)

    @property
    def sha256(self):
        return _Digest.of_bytes(self.encoded)


@dataclass(frozen=True)
class _SourceManifest:
    sha256: _Digest
    files: tuple[tuple[PurePosixPath, _Digest], ...]

    @classmethod
    def from_value(cls, value):
        if not isinstance(value, dict) or set(value) != {"sha256", "files"}:
            raise ResumeTransitionError("A source manifest must contain sha256 and files")
        if not isinstance(value["files"], dict) or not value["files"]:
            raise ResumeTransitionError("A source manifest must contain a nonempty file map")
        files = []
        for name, digest in value["files"].items():
            if not isinstance(name, str) or not name or "\\" in name:
                raise ResumeTransitionError("Source filenames must be canonical relative paths")
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != name or name == ".":
                raise ResumeTransitionError(f"Unsafe source manifest path: {name}")
            files.append((path, _Digest(digest)))
        # Match source_manifest(), whose file-map encoding uses JSON's default
        # separators. Receipt/contract hashes use the canonical encoding above.
        actual = _Digest.of_bytes(json.dumps(value["files"], sort_keys=True).encode())
        if actual != _Digest(value["sha256"]):
            raise ResumeTransitionError("Source manifest fingerprint does not match its file map")
        return cls(actual, tuple(sorted(files)))

    def to_dict(self):
        return {"sha256": str(self.sha256), "files": {str(path): str(digest) for path, digest in self.files}}


def _absolute_path(path):
    if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
        raise ResumeTransitionError("Transition paths must be absolute pathlib.Path values without parent traversal")
    return path


def _open_directory(path):
    path = _absolute_path(path)
    descriptor = None
    try:
        descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise ResumeTransitionError(f"Cannot open transition directory without symlinks {path}: {exc}") from exc


def _run_directory(path):
    path = _absolute_path(path)
    if path == Path(path.anchor):
        raise ResumeTransitionError("A filesystem root cannot identify an RLM run")
    descriptor = _open_directory(path)
    try:
        if os.fstat(descriptor).st_uid != os.geteuid():
            raise ResumeTransitionError(f"Run directory must be owned by the current user: {path}")
    finally:
        os.close(descriptor)
    return path


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ResumeTransitionError(f"Duplicate JSON key in transition receipt: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class ResumeTransition:
    """An explicit approval for one old checkpoint and one reviewed target.

    This is an audit receipt, not a signature or an automatic proof of semantic
    equivalence. Its creator must review the source change before preparation.
    All fields outside source identity and checkpoint policy remain identical.
    Existing checkpoints and the run's founding contract are never rewritten.
    """

    run_dir: Path
    checkpoint_step: int
    checkpoint_metadata_sha256: _Digest
    previous_contract: _JSON
    current_contract: _JSON
    previous_source: _SourceManifest
    current_source: _SourceManifest
    reason: str

    @staticmethod
    def _require_workflow_change(previous, current):
        required = {
            "model_config", "training_config", "initialization", "checkpoint_policy",
            "source_sha256", "tokenizer_sha256", "data_inventory_sha256",
            "compute_dtype", "device_type", "attention_impl", "runtime",
        }
        if not isinstance(previous, dict) or not isinstance(current, dict):
            raise ResumeTransitionError("Resume contracts must be JSON objects")
        if not required <= previous.keys() or not required <= current.keys():
            raise ResumeTransitionError("Resume contracts lack required training provenance")
        if _JSON.from_value(previous) == _JSON.from_value(current):
            raise ResumeTransitionError("A transition receipt is redundant for identical resume contracts")
        allowed = {"source_sha256", "checkpoint_policy"}
        old = {key: value for key, value in previous.items() if key not in allowed}
        new = {key: value for key, value in current.items() if key not in allowed}
        if _JSON.from_value(old) != _JSON.from_value(new):
            raise ResumeTransitionError("A workflow transition cannot change model, training, data, tokenizer, precision, runtime, initialization, or other semantic fields")
        if not isinstance(previous["checkpoint_policy"], dict) or not isinstance(current["checkpoint_policy"], dict):
            raise ResumeTransitionError("Training checkpoint policies must be JSON objects")

    @classmethod
    def prepare(cls, run_dir: Path, metadata: dict, current_contract: dict,
                old_source: dict, new_source: dict, reason: str):
        """Prepare after reviewing source changes and verifying archived bytes."""
        run_dir = _run_directory(run_dir)
        if not isinstance(metadata, dict) or metadata.get("model_type") != "rlm":
            raise ResumeTransitionError("Transition metadata must describe an RLM checkpoint")
        step = metadata.get("step")
        if type(step) is not int or step <= 0:
            raise ResumeTransitionError("Transition checkpoint step must be a positive integer")
        previous = metadata.get("run_contract")
        cls._require_workflow_change(previous, current_contract)
        previous_source = _SourceManifest.from_value(old_source)
        current_source = _SourceManifest.from_value(new_source)
        if previous_source.sha256 != previous["source_sha256"] or current_source.sha256 != current_contract["source_sha256"]:
            raise ResumeTransitionError("Source manifests do not match the old and new contract identities")
        if not isinstance(reason, str) or not reason.strip():
            raise ResumeTransitionError("A transition receipt requires a nonempty review reason")
        return cls(
            run_dir, step, _JSON.from_value(metadata).sha256,
            _JSON.from_value(previous), _JSON.from_value(current_contract),
            previous_source, current_source, reason,
        )

    def to_dict(self):
        return {
            "version": 1, "run_dir": str(self.run_dir), "checkpoint_step": self.checkpoint_step,
            "checkpoint_metadata_sha256": str(self.checkpoint_metadata_sha256),
            "previous_contract": self.previous_contract.value(),
            "previous_contract_sha256": str(self.previous_contract.sha256),
            "current_contract": self.current_contract.value(),
            "current_contract_sha256": str(self.current_contract.sha256),
            "previous_source": self.previous_source.to_dict(),
            "current_source": self.current_source.to_dict(), "reason": self.reason,
        }

    @classmethod
    def from_file(cls, path: Path):
        """Read an owned receipt without following file or ancestor symlinks."""
        path = _absolute_path(path)
        directory_fd = _open_directory(path.parent)
        try:
            descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
            with os.fdopen(descriptor, encoding="utf-8") as stream:
                status = os.fstat(stream.fileno())
                if not stat.S_ISREG(status.st_mode) or status.st_uid != os.geteuid() or status.st_nlink != 1:
                    raise ResumeTransitionError("Transition receipt must be an owned regular file without symlinks or hardlinks")
                record = json.load(stream, object_pairs_hook=_unique_keys)
        except (OSError, ValueError) as exc:
            raise ResumeTransitionError(f"Cannot read transition receipt {path}: {exc}") from exc
        finally:
            os.close(directory_fd)
        required = {
            "version", "run_dir", "checkpoint_step", "checkpoint_metadata_sha256",
            "previous_contract", "previous_contract_sha256", "current_contract",
            "current_contract_sha256", "previous_source", "current_source", "reason",
        }
        if not isinstance(record, dict) or set(record) != required:
            raise ResumeTransitionError("Transition receipt has unexpected or missing fields")
        if type(record["version"]) is not int or record["version"] != 1:
            raise ResumeTransitionError("Unsupported transition receipt version")
        if not isinstance(record["run_dir"], str):
            raise ResumeTransitionError("Transition receipt run_dir must contain a filesystem path")
        step = record["checkpoint_step"]
        if type(step) is not int or step <= 0:
            raise ResumeTransitionError("Transition checkpoint step must be a positive integer")
        previous, current = record["previous_contract"], record["current_contract"]
        cls._require_workflow_change(previous, current)
        previous_json, current_json = _JSON.from_value(previous), _JSON.from_value(current)
        if previous_json.sha256 != _Digest(record["previous_contract_sha256"]) or current_json.sha256 != _Digest(record["current_contract_sha256"]):
            raise ResumeTransitionError("Transition contract digests do not match their records")
        previous_source = _SourceManifest.from_value(record["previous_source"])
        current_source = _SourceManifest.from_value(record["current_source"])
        if previous_source.sha256 != previous["source_sha256"] or current_source.sha256 != current["source_sha256"]:
            raise ResumeTransitionError("Source manifests do not match the old and new contract identities")
        if not isinstance(record["reason"], str) or not record["reason"].strip():
            raise ResumeTransitionError("A transition receipt requires a nonempty review reason")
        return cls(
            _run_directory(Path(record["run_dir"])), step, _Digest(record["checkpoint_metadata_sha256"]),
            previous_json, current_json, previous_source, current_source, record["reason"],
        )

    def validate(self, run_dir: Path, metadata: dict, current_contract: dict):
        """Admit only this exact checkpoint and reviewed target; return evidence."""
        if _run_directory(run_dir) != self.run_dir:
            raise ResumeTransitionError("Transition receipt belongs to a different run directory")
        if not isinstance(metadata, dict) or metadata.get("model_type") != "rlm":
            raise ResumeTransitionError("Transition metadata must describe an RLM checkpoint")
        if type(metadata.get("step")) is not int or metadata["step"] != self.checkpoint_step:
            raise ResumeTransitionError("Transition receipt names a different checkpoint step")
        if _JSON.from_value(metadata).sha256 != self.checkpoint_metadata_sha256:
            raise ResumeTransitionError("Checkpoint metadata differs from the reviewed transition")
        previous = metadata.get("run_contract")
        self._require_workflow_change(previous, current_contract)
        if _JSON.from_value(previous) != self.previous_contract:
            raise ResumeTransitionError("Checkpoint contract differs from the reviewed transition")
        if _JSON.from_value(current_contract) != self.current_contract:
            raise ResumeTransitionError("Current contract or source differs from the reviewed transition")
        return {"receipt_sha256": str(_JSON.from_value(self.to_dict()).sha256), **self.to_dict()}

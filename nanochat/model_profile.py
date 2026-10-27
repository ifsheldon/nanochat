"""RLM architecture profiles and their original commented checkpoint documents."""

from dataclasses import dataclass, fields
import os
from pathlib import Path
import stat
from typing import TYPE_CHECKING
from uuid import uuid4

from nanochat.rlm_checkpoints import CheckpointRetentionError, _open_directory

if TYPE_CHECKING:
    from nanochat.rlm import RLMConfig


CHECKPOINT_PROFILE_NAME = "model-profile.jsonc"


@dataclass(frozen=True)
class RLMModelProfile:
    """Validated architecture; training and execution settings live elsewhere."""

    n_layer: int
    n_embd: int
    n_head: int
    n_kv_head: int
    text_length: int
    memory_tokens: int
    eom_tokens: int

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Model profile {field.name} must be a positive integer")
        if self.eom_tokens != 1:
            raise ValueError("Model profile eom_tokens must be exactly 1")
        if self.n_embd < 24:
            raise ValueError("Model profile n_embd must be at least 24 for the smear gate")
        if self.n_embd % self.n_head:
            raise ValueError("Model profile n_embd must be divisible by n_head")
        if self.n_kv_head > self.n_head or self.n_head % self.n_kv_head:
            raise ValueError("Model profile n_kv_head must divide n_head and cannot exceed it")
        if (self.n_embd // self.n_head) % 2:
            raise ValueError("Model profile attention head dimension must be even for rotary embeddings")

    @classmethod
    def from_jsonc(cls, text: str) -> "RLMModelProfile":
        # H100 launcher argument parsing runs before dependencies are installed.
        import json5

        try:
            value = json5.loads(text, allow_duplicate_keys=False)
        except ValueError as exc:
            raise ValueError(f"Invalid model profile JSONC: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("Model profile must be a JSONC object")
        required = {field.name for field in fields(cls)}
        missing, unknown = required - value.keys(), value.keys() - required
        if missing or unknown:
            raise ValueError(
                f"Model profile fields differ from the schema: missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        return cls(**value)

    @property
    def sequence_len(self) -> int:
        return self.text_length + self.eom_tokens + 2 * self.memory_tokens

    def model_config(self, *, vocab_size: int, activation_checkpointing: bool = False) -> "RLMConfig":
        from nanochat.rlm import RLMConfig

        return RLMConfig(
            sequence_len=self.sequence_len, vocab_size=vocab_size,
            n_layer=self.n_layer, n_embd=self.n_embd,
            n_head=self.n_head, n_kv_head=self.n_kv_head,
            text_length=self.text_length, memory_tokens=self.memory_tokens,
            activation_checkpointing=activation_checkpointing,
        )


@dataclass(frozen=True)
class ModelProfileDocument:
    """Retain comments and line endings from the document selected for a run."""

    profile: RLMModelProfile
    text: str

    @classmethod
    def read(cls, path: Path) -> "ModelProfileDocument":
        try:
            text = path.read_bytes().decode("utf-8")
            return cls(RLMModelProfile.from_jsonc(text), text)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read model profile {path}: {exc}") from exc

    def _matches_existing(self, directory_fd: int, name: str) -> bool:
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
        except FileNotFoundError:
            return False
        with os.fdopen(descriptor, "rb") as stream:
            status = os.fstat(stream.fileno())
            if not stat.S_ISREG(status.st_mode) or status.st_uid != os.geteuid() or status.st_nlink != 1:
                raise ValueError(f"Saved model profile must be an owned regular file without links: {name}")
            profile = RLMModelProfile.from_jsonc(stream.read().decode("utf-8"))
        if profile != self.profile:
            raise ValueError(f"Saved model profile has different architecture fields: {name}")
        return True

    def copy_once(self, destination: Path) -> None:
        """Publish a complete original document, or leave an equivalent copy intact."""
        directory_fd = None
        staging = None
        try:
            directory_fd = _open_directory(destination.absolute().parent)
            if self._matches_existing(directory_fd, destination.name):
                return
            staging = f".{destination.name}.{uuid4().hex}.tmp"
            descriptor = os.open(
                staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600, dir_fd=directory_fd,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(self.text.encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(staging, destination.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            except FileExistsError:
                if not self._matches_existing(directory_fd, destination.name):
                    raise ValueError(f"Saved model profile disappeared during publication: {destination}")
            os.unlink(staging, dir_fd=directory_fd)
            staging = None
            os.fsync(directory_fd)
        except (OSError, CheckpointRetentionError) as exc:
            raise ValueError(f"Cannot save model profile {destination}: {exc}") from exc
        finally:
            if directory_fd is not None:
                try:
                    if staging is not None:
                        os.unlink(staging, dir_fd=directory_fd)
                finally:
                    os.close(directory_fd)

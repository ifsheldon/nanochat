"""Copy a verified pretrained input embedding into a freshly initialized RLM."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import stat

import torch

from nanochat.rlm import RecurrentGPT


@dataclass(frozen=True)
class _SourceFile:
    path: Path
    sha256: str
    identity: tuple[int, int, int, int, int] = field(repr=False)

    @staticmethod
    def _identity(path: Path):
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError(f"Initialization source cannot contain symlinks: {path}")
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Initialization source must be a regular file: {path}")
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    @classmethod
    def from_path(cls, path: Path):
        if not isinstance(path, Path):
            raise TypeError("Initialization file paths must be pathlib.Path values")
        if ".." in path.parts:
            raise ValueError(f"Initialization source cannot contain parent traversal: {path}")
        path = path.expanduser().absolute()
        identity = cls._identity(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        source = cls(path, digest.hexdigest(), identity)
        source.require_unchanged()
        return source

    def require_unchanged(self):
        if self._identity(self.path) != self.identity:
            raise ValueError(f"Initialization source changed after validation: {self.path}")

    def contract(self):
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class TokenEmbeddingInitialization:
    """Validated source identity and a CPU mapping of only the table to transfer.

    Construct before recording a fresh run contract, then call ``initialize``
    after the RLM's ordinary random initialization and before its optimizer.
    Resuming an RLM restores its own checkpoint instead of applying this again.
    """

    checkpoint_dir: Path
    step: int
    vocab_size: int
    n_embd: int
    checkpoint: _SourceFile
    metadata: _SourceFile
    source_tokenizer: _SourceFile
    target_tokenizer: _SourceFile
    embedding_sha256: str
    _embedding: torch.Tensor = field(repr=False, compare=False)

    @staticmethod
    def _tensor_sha256(tensor: torch.Tensor):
        values = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
        return hashlib.sha256(memoryview(values).cast("B")).hexdigest()

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: Path, step: int,
                        source_tokenizer: Path, target_tokenizer: Path):
        if not isinstance(checkpoint_dir, Path):
            raise TypeError("Checkpoint directory must be a pathlib.Path value")
        if type(step) is not int or step < 0:
            raise ValueError("Initialization checkpoint step must be a nonnegative integer")
        source_tokenizer_file = _SourceFile.from_path(source_tokenizer)
        target_tokenizer_file = _SourceFile.from_path(target_tokenizer)
        if source_tokenizer_file.sha256 != target_tokenizer_file.sha256:
            raise ValueError("Source and target tokenizer SHA-256 identities do not match")
        metadata = _SourceFile.from_path(checkpoint_dir / f"meta_{step:06d}.json")
        with metadata.path.open(encoding="utf-8") as stream:
            contents = json.load(stream)
        metadata.require_unchanged()
        if not isinstance(contents, dict) or type(contents.get("step")) is not int or contents["step"] != step:
            raise ValueError("Initialization metadata must publish the requested checkpoint step")
        config = contents.get("model_config")
        if not isinstance(config, dict):
            raise ValueError("Initialization metadata must contain a model_config object")
        for name in ("vocab_size", "n_embd"):
            if type(config.get(name)) is not int or config[name] <= 0:
                raise ValueError(f"Initialization model_config {name} must be a positive integer")
        checkpoint = _SourceFile.from_path(checkpoint_dir / f"model_{step:06d}.pt")
        state = torch.load(checkpoint.path, map_location="cpu", weights_only=True, mmap=True)
        checkpoint.require_unchanged()
        if not isinstance(state, dict):
            raise ValueError("Initialization checkpoint must contain a model state dictionary")
        embedding = state.get("transformer.wte.weight")
        if not isinstance(embedding, torch.Tensor):
            raise ValueError("Initialization checkpoint lacks transformer.wte.weight tensor")
        if embedding.layout != torch.strided or embedding.ndim != 2:
            raise ValueError("Initialization token embedding must be a dense two-dimensional tensor")
        if embedding.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Initialization token embedding must use float32 or bfloat16")
        if embedding.shape[0] < config["vocab_size"] or embedding.shape[1] != config["n_embd"]:
            raise ValueError("Initialization token embedding shape does not match source model metadata")
        if not torch.isfinite(embedding).all().item():
            raise ValueError("Initialization token embedding contains non-finite values")
        embedding_sha256 = cls._tensor_sha256(embedding)
        checkpoint.require_unchanged()
        return cls(
            checkpoint.path.parent, step, config["vocab_size"], config["n_embd"],
            checkpoint, metadata, source_tokenizer_file, target_tokenizer_file,
            embedding_sha256, embedding,
        )

    def contract(self):
        """Return JSON provenance without loading a model or exposing tensors."""
        return {
            "kind": "pretrained_input_token_embedding",
            "parameter": "transformer.wte.weight",
            "checkpoint_dir": str(self.checkpoint_dir), "step": self.step,
            "checkpoint": self.checkpoint.contract(), "metadata": self.metadata.contract(),
            "source_tokenizer": self.source_tokenizer.contract(),
            "target_tokenizer": self.target_tokenizer.contract(),
            "source_model_config": {"vocab_size": self.vocab_size, "n_embd": self.n_embd},
            "embedding_shape": list(self._embedding.shape),
            "embedding_dtype": str(self._embedding.dtype),
            "embedding_sha256": self.embedding_sha256,
        }

    @torch.no_grad()
    def initialize(self, model: RecurrentGPT):
        """Copy only the main table, keeping parameter identity and trainability."""
        for source in (self.checkpoint, self.metadata, self.source_tokenizer, self.target_tokenizer):
            source.require_unchanged()
        if model.config.vocab_size != self.vocab_size or model.config.n_embd != self.n_embd:
            raise ValueError("Source and target model vocabulary size and embedding width must match")
        target = model.transformer.wte.weight
        if target.shape != self._embedding.shape:
            raise ValueError("Source and target padded token embedding shapes must match exactly")
        if target.dtype != self._embedding.dtype:
            raise ValueError("Source and target token embedding dtypes must match exactly")
        if target.device.type == "meta":
            raise ValueError("Target token embedding must be materialized before initialization")
        if not all(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Token embedding initialization requires every model parameter to remain trainable")
        if self._tensor_sha256(self._embedding) != self.embedding_sha256:
            raise ValueError("Initialization token embedding changed after validation")
        target.copy_(self._embedding)
        copied_sha256 = self._tensor_sha256(target)
        if copied_sha256 != self.embedding_sha256:
            raise RuntimeError("Copied input token embedding failed exact SHA-256 verification")
        return {
            "parameter": "transformer.wte.weight", "parameters": target.numel(),
            "source_sha256": self.embedding_sha256, "copied_sha256": copied_sha256,
            "trainable": target.requires_grad,
        }

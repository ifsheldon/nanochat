"""Copy a verified pretrained input embedding into a freshly initialized RLM."""

import copy
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
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
class _EmbeddingBundleManifest:
    step: int
    vocab_size: int
    shape: tuple[int, int]
    dtype: torch.dtype
    embedding_sha256: str
    checkpoint_sha256: str
    file_hashes: dict[str, str]

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 1:
            raise ValueError("Unsupported input embedding bundle version")
        if value.get("kind") != "pretrained_input_token_embedding" or value.get("parameter") != "transformer.wte.weight":
            raise ValueError("Bundle must contain a pretrained input token embedding")
        step = value.get("step")
        vocab_size = value.get("vocab_size")
        shape = value.get("embedding_shape")
        if type(step) is not int or step < 0:
            raise ValueError("Embedding bundle step must be a nonnegative integer")
        if type(vocab_size) is not int or vocab_size <= 0:
            raise ValueError("Embedding bundle vocabulary size must be a positive integer")
        if not isinstance(shape, list) or len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape):
            raise ValueError("Embedding bundle shape must contain two positive integers")
        if shape[0] < vocab_size:
            raise ValueError("Embedding bundle shape cannot be smaller than its vocabulary")
        dtype_name = value.get("embedding_dtype")
        dtype = {"torch.float32": torch.float32, "torch.bfloat16": torch.bfloat16}.get(dtype_name) if isinstance(dtype_name, str) else None
        if dtype is None:
            raise ValueError("Embedding bundle dtype must be torch.float32 or torch.bfloat16")
        checkpoint = value.get("source_checkpoint")
        if not isinstance(checkpoint, dict) or checkpoint.get("name") != f"model_{step:06d}.pt":
            raise ValueError("Embedding bundle checkpoint provenance does not match its step")
        file_hashes = value.get("files")
        expected = {f"model_{step:06d}.pt", "tokenizer.pkl", "token_bytes.pt", f"meta_{step:06d}.json"}
        if not isinstance(file_hashes, dict) or set(file_hashes) != expected:
            raise ValueError("Embedding bundle files must use the exact canonical filenames")
        digests = [value.get("embedding_sha256"), checkpoint.get("sha256"), *file_hashes.values()]
        if any(not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) for digest in digests):
            raise ValueError("Embedding bundle SHA-256 identities must be lowercase hexadecimal digests")
        return cls(step, vocab_size, tuple(shape), dtype, digests[0], digests[1], dict(file_hashes))

    def to_dict(self):
        return {
            "version": 1, "kind": "pretrained_input_token_embedding",
            "parameter": "transformer.wte.weight", "step": self.step,
            "vocab_size": self.vocab_size, "embedding_shape": list(self.shape),
            "embedding_dtype": str(self.dtype), "embedding_sha256": self.embedding_sha256,
            "source_checkpoint": {"name": f"model_{self.step:06d}.pt", "sha256": self.checkpoint_sha256},
            "files": dict(self.file_hashes),
        }


@dataclass(frozen=True)
class TokenEmbeddingInitialization:
    """Validated source identity and a CPU mapping of only the table to transfer.

    Construct before recording a fresh run contract, then call ``initialize``
    after the RLM's ordinary random initialization and before its optimizer.
    Resuming an RLM restores its own checkpoint instead of applying this again.
    """

    step: int
    vocab_size: int
    n_embd: int
    embedding_sha256: str
    _embedding: torch.Tensor = field(repr=False, compare=False)
    _sources: tuple[_SourceFile, ...] = field(repr=False)
    _provenance: dict = field(repr=False)

    @staticmethod
    def _tensor_sha256(tensor: torch.Tensor):
        values = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
        return hashlib.sha256(memoryview(values).cast("B")).hexdigest()

    @staticmethod
    def _validate_embedding(embedding, vocab_size, n_embd):
        if not isinstance(embedding, torch.Tensor):
            raise ValueError("Initialization checkpoint lacks transformer.wte.weight tensor")
        if embedding.layout != torch.strided or embedding.ndim != 2:
            raise ValueError("Initialization token embedding must be a dense two-dimensional tensor")
        if embedding.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Initialization token embedding must use float32 or bfloat16")
        if embedding.shape[0] < vocab_size or embedding.shape[1] != n_embd:
            raise ValueError("Initialization token embedding shape does not match source model metadata")
        if not torch.isfinite(embedding).all().item():
            raise ValueError("Initialization token embedding contains non-finite values")

    @staticmethod
    def _read_metadata(metadata, step):
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
        return config

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: Path, step: int,
                        source_tokenizer: Path, target_tokenizer: Path):
        """Read either a complete model or an embedding-only state dictionary."""
        if not isinstance(checkpoint_dir, Path):
            raise TypeError("Checkpoint directory must be a pathlib.Path value")
        if type(step) is not int or step < 0:
            raise ValueError("Initialization checkpoint step must be a nonnegative integer")
        source_tokenizer_file = _SourceFile.from_path(source_tokenizer)
        target_tokenizer_file = _SourceFile.from_path(target_tokenizer)
        if source_tokenizer_file.sha256 != target_tokenizer_file.sha256:
            raise ValueError("Source and target tokenizer SHA-256 identities do not match")
        metadata = _SourceFile.from_path(checkpoint_dir / f"meta_{step:06d}.json")
        config = cls._read_metadata(metadata, step)
        checkpoint = _SourceFile.from_path(checkpoint_dir / f"model_{step:06d}.pt")
        state = torch.load(checkpoint.path, map_location="cpu", weights_only=True, mmap=True)
        checkpoint.require_unchanged()
        if not isinstance(state, dict):
            raise ValueError("Initialization checkpoint must contain a model state dictionary")
        embedding = state.get("transformer.wte.weight")
        cls._validate_embedding(embedding, config["vocab_size"], config["n_embd"])
        embedding_sha256 = cls._tensor_sha256(embedding)
        checkpoint.require_unchanged()
        provenance = {
            "checkpoint_dir": str(checkpoint.path.parent), "checkpoint": checkpoint.contract(),
            "metadata": metadata.contract(), "source_tokenizer": source_tokenizer_file.contract(),
            "target_tokenizer": target_tokenizer_file.contract(),
        }
        return cls(
            step, config["vocab_size"], config["n_embd"], embedding_sha256, embedding,
            (checkpoint, metadata, source_tokenizer_file, target_tokenizer_file), provenance,
        )

    @classmethod
    def from_bundle(cls, bundle_dir: Path, target_tokenizer: Path):
        """Verify an exported table and tokenizer without the original checkpoint."""
        if not isinstance(bundle_dir, Path):
            raise TypeError("Embedding bundle directory must be a pathlib.Path value")
        manifest_file = _SourceFile.from_path(bundle_dir / "manifest.json")
        manifest = _EmbeddingBundleManifest.from_dict(json.loads(manifest_file.path.read_text(encoding="utf-8")))
        manifest_file.require_unchanged()
        files = {}
        for name, expected_hash in manifest.file_hashes.items():
            source = _SourceFile.from_path(bundle_dir / name)
            if source.sha256 != expected_hash:
                raise ValueError(f"Embedding bundle file SHA-256 mismatch: {name}")
            files[name] = source
        target = _SourceFile.from_path(target_tokenizer)
        if target.sha256 != files["tokenizer.pkl"].sha256:
            raise ValueError("Source and target tokenizer SHA-256 identities do not match")
        initialization = cls.from_checkpoint(bundle_dir, manifest.step, files["tokenizer.pkl"].path, target.path)
        if initialization.vocab_size != manifest.vocab_size or initialization.n_embd != manifest.shape[1]:
            raise ValueError("Embedding bundle dimensions do not match source model metadata")
        embedding = initialization._embedding
        if tuple(embedding.shape) != manifest.shape or embedding.dtype != manifest.dtype:
            raise ValueError("Embedding bundle tensor shape or dtype does not match its manifest")
        if initialization.embedding_sha256 != manifest.embedding_sha256:
            raise ValueError("Embedding bundle table SHA-256 mismatch")
        sources = (manifest_file, *files.values(), target)
        for source in sources:
            source.require_unchanged()
        provenance = {
            **initialization._provenance,
            "bundle": manifest_file.contract(),
            "source_checkpoint": manifest.to_dict()["source_checkpoint"],
            "token_bytes": files["token_bytes.pt"].contract(),
        }
        return cls(manifest.step, manifest.vocab_size, manifest.shape[1], manifest.embedding_sha256,
                   embedding, sources, provenance)

    @classmethod
    def export_from_checkpoint(cls, checkpoint_dir: Path, step: int, tokenizer_dir: Path, output_dir: Path):
        """Publish an embedding-only bundle; manifest.json commits a complete export.

        The destination is reserved exclusively. Failed exports are removed and
        an interrupted export without its manifest cannot be loaded as a bundle.
        """
        if not isinstance(tokenizer_dir, Path) or not isinstance(output_dir, Path):
            raise TypeError("Tokenizer and output directories must be pathlib.Path values")
        if ".." in output_dir.parts:
            raise ValueError("Embedding bundle destination cannot contain parent traversal")
        output_dir = output_dir.expanduser().absolute()
        if any(part.is_symlink() for part in (output_dir, *output_dir.parents)):
            raise ValueError("Embedding bundle destination cannot contain symlinks")
        if output_dir.exists():
            raise FileExistsError(f"Refusing to replace existing embedding bundle destination: {output_dir}")
        if not output_dir.parent.is_dir():
            raise FileNotFoundError("Embedding bundle destination parent directory must already exist")
        source = cls.from_checkpoint(checkpoint_dir, step, tokenizer_dir / "tokenizer.pkl", tokenizer_dir / "tokenizer.pkl")
        token_bytes = _SourceFile.from_path(tokenizer_dir / "token_bytes.pt")
        checkpoint, metadata, tokenizer, _ = source._sources
        output_dir.mkdir(mode=0o700)
        try:
            # clone is essential: saving a view can serialize its entire backing
            # storage, including unrelated source parameters.
            torch.save({"transformer.wte.weight": source._embedding.detach().clone().contiguous()},
                       output_dir / checkpoint.path.name)
            for asset, name in ((metadata, metadata.path.name), (tokenizer, "tokenizer.pkl"), (token_bytes, "token_bytes.pt")):
                asset.require_unchanged()
                shutil.copyfile(asset.path, output_dir / name)
                asset.require_unchanged()
                if _SourceFile.from_path(output_dir / name).sha256 != asset.sha256:
                    raise ValueError(f"Embedding bundle copy SHA-256 mismatch: {name}")
            file_hashes = {name: _SourceFile.from_path(output_dir / name).sha256 for name in
                           (checkpoint.path.name, metadata.path.name, "tokenizer.pkl", "token_bytes.pt")}
            manifest = _EmbeddingBundleManifest(
                step, source.vocab_size, tuple(source._embedding.shape), source._embedding.dtype,
                source.embedding_sha256, checkpoint.sha256, file_hashes,
            )
            for asset in (*source._sources, token_bytes):
                asset.require_unchanged()
            for name in file_hashes:
                with (output_dir / name).open("rb") as stream:
                    os.fsync(stream.fileno())
            staging_manifest = output_dir / ".manifest.tmp"
            with staging_manifest.open("x", encoding="utf-8") as stream:
                json.dump(manifest.to_dict(), stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            staging_manifest.replace(output_dir / "manifest.json")
            for directory in (output_dir, output_dir.parent):
                directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except BaseException:
            shutil.rmtree(output_dir)
            raise
        return output_dir

    def contract(self):
        """Return JSON provenance without loading a model or exposing tensors."""
        return {
            "kind": "pretrained_input_token_embedding",
            "parameter": "transformer.wte.weight",
            "step": self.step,
            **copy.deepcopy(self._provenance),
            "source_model_config": {"vocab_size": self.vocab_size, "n_embd": self.n_embd},
            "embedding_shape": list(self._embedding.shape),
            "embedding_dtype": str(self._embedding.dtype),
            "embedding_sha256": self.embedding_sha256,
        }

    @torch.no_grad()
    def initialize(self, model: RecurrentGPT):
        """Copy only the main table, keeping parameter identity and trainability."""
        for source in self._sources:
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

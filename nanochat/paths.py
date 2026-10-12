"""Project-local run artifacts and shared datasets, independent of the cwd."""

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunName:
    value: str

    def __post_init__(self):
        if (not isinstance(self.value, str) or not self.value or self.value in (".", "..")
                or any(character in self.value for character in ("/", "\\", "\0"))
                or self.value != self.value.strip()):
            raise ValueError("Run name must be one nonempty directory name without separators or surrounding whitespace")

    def __str__(self):
        return self.value


def _directory(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if ".." in path.parts:
        raise ValueError(f"Storage directory cannot contain parent traversal: {path}")
    path = path.absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"Storage directory cannot contain symlinks: {path}")
    for part in (path, *path.parents):
        if part.exists() and not part.is_dir():
            raise ValueError(f"Storage path must be a directory: {part}")
    return path


def get_run_dir(run_name: RunName | None = None) -> Path:
    """Resolve a named run, or the explicit environment-selected artifact root."""
    if run_name is not None and not isinstance(run_name, RunName):
        raise TypeError("Run names must be validated RunName values")
    if run_name is None:
        override = os.environ.get("NANOCHAT_BASE_DIR")
        if override:
            return _directory(override)
        run_name = RunName(os.environ.get("NANOCHAT_RUN_NAME", "pretrain-1"))
    return _directory(PROJECT_ROOT / "training_run" / run_name.value)


def get_datasets_dir() -> Path:
    """Resolve downloaded data independently of the selected training run."""
    return _directory(os.environ.get("NANOCHAT_DATASETS_DIR") or PROJECT_ROOT / "datasets")


def configure_runtime_paths(run_dir: Path | None = None) -> None:
    """Set process-local library-cache defaults without changing user settings.

    Call before importing compiler, kernel-download, or tracking libraries.
    Directory creation remains with their owners, so this does not admit a run
    or overwrite an existing output directory merely by resolving its paths.
    """
    run = get_run_dir() if run_dir is None else _directory(run_dir)
    defaults = {
        "UV_CACHE_DIR": run / "cache" / "uv",
        "TORCH_HOME": run / "cache" / "torch",
        "TORCHINDUCTOR_CACHE_DIR": run / "cache" / "torchinductor",
        "TRITON_CACHE_DIR": run / "cache" / "triton",
        "CUDA_CACHE_PATH": run / "cache" / "cuda",
        "HF_HUB_CACHE": run / "cache" / "huggingface" / "hub",
        "HF_XET_CACHE": run / "cache" / "huggingface" / "xet",
        "TIKTOKEN_CACHE_DIR": run / "cache" / "tiktoken",
        "WANDB_DIR": run,
        "WANDB_CACHE_DIR": run / "cache" / "wandb",
        "WANDB_DATA_DIR": run / "wandb" / "artifacts",
    }
    # Validate every generated destination before publishing any defaults.
    # Existing library overrides are intentional and remain the caller's choice.
    pending = {
        name: str(_directory(path))
        for name, path in defaults.items()
        if name not in os.environ
    }
    for name, path in pending.items():
        os.environ.setdefault(name, path)

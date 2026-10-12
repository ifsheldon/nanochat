# Shared storage and logging for run scripts. Source with a default run name and
# optional fixed ClimbMix training-shard count before installing dependencies.
nanochat_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || return
cd -- "$nanochat_repo_root" || return
export NANOCHAT_RUN_NAME="${NANOCHAT_RUN_NAME:-${1:?A default run name is required}}"
# This imports only the standard library, before uv or the training environment
# is available. shlex.quote keeps explicit paths data when exported into Bash.
nanochat_path_exports="$(python3 - <<'PY'
import os
import shlex
from nanochat.paths import configure_runtime_paths, get_datasets_dir, get_run_dir

os.environ["NANOCHAT_BASE_DIR"] = str(get_run_dir())
os.environ["NANOCHAT_DATASETS_DIR"] = str(get_datasets_dir())
if (get_run_dir() / "logs").is_symlink():
    raise ValueError("The run log directory must not be a symlink")
configure_runtime_paths()
for name in (
    "NANOCHAT_BASE_DIR", "NANOCHAT_DATASETS_DIR", "TORCH_HOME",
    "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
    "HF_HUB_CACHE", "HF_XET_CACHE", "UV_CACHE_DIR", "TIKTOKEN_CACHE_DIR", "WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR",
):
    print(f"export {name}={shlex.quote(os.environ[name])}")
PY
)" || return
eval "$nanochat_path_exports"
if [[ -n "${2:-}" ]]; then
    export NANOCHAT_TRAIN_SHARDS="$2"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
mkdir -p -- "$NANOCHAT_BASE_DIR/logs" "$NANOCHAT_DATASETS_DIR" || return
NANOCHAT_LOG_FILE="$(mktemp "$NANOCHAT_BASE_DIR/logs/${1}-$(date -u +%Y%m%dT%H%M%S)-XXXXXX.log")" || return
export NANOCHAT_LOG_FILE
exec > >(tee "$NANOCHAT_LOG_FILE") 2>&1
echo "Run artifacts: $NANOCHAT_BASE_DIR"
echo "Shared datasets: $NANOCHAT_DATASETS_DIR"
echo "Console log: $NANOCHAT_LOG_FILE"

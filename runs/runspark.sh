#!/usr/bin/env bash
# Run from the repository root: bash runs/runspark.sh [all|prepare|smoke|demo|sft]
set -euo pipefail

stage="${1:-all}"
case "$stage" in
    all|prepare|smoke|demo|sft) ;;
    *) echo "Usage: bash runs/runspark.sh [all|prepare|smoke|demo|sft]" >&2; exit 2 ;;
esac

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat/spark}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "$NANOCHAT_BASE_DIR/logs"
log_file="$NANOCHAT_BASE_DIR/logs/${stage}-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee "$log_file") 2>&1
trap 'echo "Failed at line $LINENO; see $log_file" >&2' ERR

if [[ "$stage" == all || "$stage" == prepare ]]; then
    uv sync --extra gpu --group dev --locked
fi
source .venv/bin/activate

python - <<'PY'
import torch
from nanochat.common import COMPUTE_DTYPE
from nanochat.flash_attention import USE_FA3

if not torch.cuda.is_available():
    raise RuntimeError("The Spark experiment requires a working CUDA PyTorch installation")
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name()}, capability: {torch.cuda.get_device_capability()}")
print(f"Compute dtype: {COMPUTE_DTYPE}, Flash Attention 3: {USE_FA3}")
x = torch.randn(256, 256, device="cuda", dtype=COMPUTE_DTYPE)
if not (x @ x).isfinite().all().item():
    raise RuntimeError("CUDA matrix multiplication produced non-finite values")
torch.cuda.synchronize()
PY

if [[ "$stage" == all || "$stage" == prepare ]]; then
    python -m nanochat.dataset -n 8
    # The upstream downloader reports failures without a nonzero exit status.
    python - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq
from nanochat.dataset import DATA_DIR, MAX_SHARD, index_to_filename

for index in [*range(8), MAX_SHARD]:
    path = Path(DATA_DIR) / index_to_filename(index)
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows == 0 or "text" not in parquet.schema.names:
        raise RuntimeError(f"Empty or invalid training shard: {path}")
print("Verified all eight training shards and the validation shard")
PY
    python -m scripts.tok_train --max-chars=2000000000
    python -m scripts.tok_eval
fi

# Same architecture and batch sizes as runs/runcpu.sh, executed on CUDA.
base_args=(
    --device-type=cuda --depth=6 --head-dim=64 --window-pattern=L
    --max-seq-len=512 --device-batch-size=32 --total-batch-size=16384
    --core-metric-every=-1 --run=dummy
)

if [[ "$stage" == all || "$stage" == smoke ]]; then
    python -m scripts.base_train "${base_args[@]}" \
        --model-tag=spark-smoke-d6-torch214 --num-iterations=20 --warmup-steps=2 \
        --eval-every=20 --eval-tokens=16384 --sample-every=-1
    # Load the saved checkpoint in a fresh process and exercise inference.
    python -m scripts.base_eval --device-type=cuda --model-tag=spark-smoke-d6-torch214 \
        --device-batch-size=1 --split-tokens=16384 --max-per-task=16
    python -m scripts.chat_sft --device-type=cuda --model-tag=spark-smoke-d6-torch214 \
        --num-iterations=20 --eval-every=20 --eval-tokens=16384 \
        --chatcore-every=-1 --run=dummy
    python -m scripts.chat_cli --device-type=cuda --model-tag=spark-smoke-d6-torch214 \
        -p "What is the capital of France?"
fi

if [[ "$stage" == all || "$stage" == demo ]]; then
    python -m scripts.base_train "${base_args[@]}" \
        --model-tag=spark-d6-torch214 --num-iterations=5000 --save-every=1000 \
        --eval-every=100 --eval-tokens=524288 --sample-every=100
    python -m scripts.base_eval --device-type=cuda --model-tag=spark-d6-torch214 \
        --device-batch-size=1 --split-tokens=16384 --max-per-task=16
fi

if [[ "$stage" == all || "$stage" == demo || "$stage" == sft ]]; then
    python -m scripts.chat_sft --device-type=cuda --model-tag=spark-d6-torch214 \
        --eval-every=200 --eval-tokens=524288 --num-iterations=1500 --run=dummy \
        --chatcore-every=1500 --chatcore-max-cat=16 --chatcore-max-sample=4
    python -m scripts.chat_cli --device-type=cuda --model-tag=spark-d6-torch214 \
        -p "What is the capital of France?"
fi

echo "Completed $stage; log: $log_file"

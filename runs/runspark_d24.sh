#!/usr/bin/env bash
# Run from the repository root. See dev/SPARK_D24.md.
set -euo pipefail
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat/spark-d24}"
export NANOCHAT_CUDA_MEMORY_LIMIT_GB=80
export NANOCHAT_MIN_SYSTEM_AVAILABLE_GB=16
export NANOCHAT_CHECKPOINT_RESERVE_GB=100
export NANOCHAT_DTYPE=bfloat16
export NANOCHAT_ATTENTION_IMPL=sdpa
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
uv run --locked --extra gpu --group dev python -m scripts.spark_d24 "$@"

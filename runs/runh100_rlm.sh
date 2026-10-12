#!/usr/bin/env bash
# Single-H100 preparation, ordered capacity probes, and training. See dev/H100_RLM.md.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Resolve cache locations with the standard library before uv can download.
# argparse handles --help anywhere without importing Torch or invoking uv.
exec python3 - "$@" <<'PY'
import os
import shutil
import sys

from nanochat.paths import configure_runtime_paths, get_run_dir
from scripts.h100_rlm import checked_path, parse_args

args = parse_args(sys.argv[1:])
run_dir = checked_path(args.output_dir or get_run_dir(args.run_name), directory=True, must_exist=False)
configure_runtime_paths(run_dir)
uv = shutil.which("uv")
if uv is None:
    raise SystemExit("uv is required. Install it using https://docs.astral.sh/uv/getting-started/installation/ and rerun this command.")
os.execvpe(uv, [
    uv, "run", "--locked", "--python", "3.10.21", "--extra", "gpu", "--no-default-groups",
    "python", "-m", "scripts.h100_rlm", *sys.argv[1:],
], os.environ)
PY

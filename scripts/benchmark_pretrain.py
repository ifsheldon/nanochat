"""Compare D24 attention/linear precision with one warmup and one measured update."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Literal

from nanochat.paths import configure_runtime_paths, get_run_dir


@dataclass(frozen=True)
class BenchmarkCase:
    window: Literal["L", "SSSL"]
    attention: Literal["sdpa", "fa4"]
    precision: Literal["bf16", "fp8"]

    @property
    def name(self):
        return f"{self.window.lower()}-{self.attention}-{self.precision}"

    def arguments(self, output: Path, batch_size: int):
        args = [
            "--device-type=cuda", "--depth=24", "--head-dim=128",
            "--max-seq-len=2048", f"--window-pattern={self.window}",
            f"--device-batch-size={batch_size}", "--total-batch-size=1048576",
            "--target-param-data-ratio=8", "--num-iterations=5568",
            "--run=dummy", f"--benchmark-output={output}",
        ]
        if self.precision == "fp8":
            args.extend(["--fp8", "--fp8-recipe=tensorwise"])
        return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=get_run_dir() / "benchmarks" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
                        help="new results/log directory (default: selected run benchmarks/<UTC timestamp>)")
    parser.add_argument("--device-batch-size", type=int, default=8, help="same microbatch for every case; must divide 512")
    args = parser.parse_args()
    if args.device_batch_size <= 0 or 512 % args.device_batch_size:
        parser.error("--device-batch-size must be a positive divisor of 512")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Run the matrix directly in one process, not with torchrun")

    configure_runtime_paths()
    import torch
    from nanochat.common import get_base_dir, get_datasets_dir

    if not torch.cuda.is_available():
        parser.error("The D24 benchmark requires CUDA")
    run_root = Path(get_base_dir())
    data_root = Path(get_datasets_dir())
    for source in (run_root / "tokenizer", data_root / "base_data_climbmix"):
        if not source.is_dir():
            parser.error(f"Missing {source}; prepare the selected run tokenizer and shared datasets first")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_dir = args.output_dir.resolve()
    repo = Path(__file__).resolve().parents[1]
    packages = {}
    for name in ("torch", "triton", "flash-attn-4", "nvidia-cutlass-dsl", "quack-kernels"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "packages": packages,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "run_root": str(run_root.resolve()),
        "data_root": str(data_root.resolve()),
        "training_shards": os.environ.get("NANOCHAT_TRAIN_SHARDS", "170"),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "cases": [],
    }
    # Preserve the actual source under test, including uncommitted changes.
    (output_dir / "source.diff").write_text(subprocess.check_output(["git", "diff", "HEAD"], cwd=repo, text=True))
    sources = output_dir / "source"
    sources.mkdir()
    for name in ("nanochat/flash_attention.py", "nanochat/fp8.py", "scripts/base_train.py", "scripts/benchmark_pretrain.py"):
        destination = sources / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((repo / name).read_bytes())

    started = time.perf_counter()
    for window, attention, precision in itertools.product(("L", "SSSL"), ("sdpa", "fa4"), ("bf16", "fp8")):
        case = BenchmarkCase(window, attention, precision)
        result_path = output_dir / f"{case.name}.json"
        log_path = output_dir / f"{case.name}.log"
        command = [sys.executable, "-m", "scripts.base_train", *case.arguments(result_path, args.device_batch_size)]
        env = os.environ.copy()
        env.update(NANOCHAT_BASE_DIR=str(run_root), NANOCHAT_DATASETS_DIR=str(data_root),
                   NANOCHAT_TRAIN_SHARDS=os.environ.get("NANOCHAT_TRAIN_SHARDS", "170"), NANOCHAT_DTYPE="bfloat16",
                   NANOCHAT_ATTENTION_IMPL=case.attention, OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        env.setdefault("NANOCHAT_CUDA_MEMORY_LIMIT_GB", "80")
        env.setdefault("NANOCHAT_MIN_SYSTEM_AVAILABLE_GB", "16")
        print(f"Starting {case.name}; log: {log_path}", flush=True)
        case_started = time.perf_counter()
        with log_path.open("x") as log:
            completed = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        record = {
            "name": case.name, "command": command, "exit_code": completed.returncode,
            "wall_seconds": time.perf_counter() - case_started, "log": str(log_path),
        }
        if completed.returncode == 0:
            record["result"] = json.loads(result_path.read_text())
            seconds = record["result"]["measured_update_seconds"]
            print(f"Finished {case.name}: measured update {seconds:.3f}s", flush=True)
        else:
            print(f"Failed {case.name} (exit {completed.returncode}); see {log_path}", flush=True)
        manifest["cases"].append(record)
        manifest["wall_seconds"] = time.perf_counter() - started
        (output_dir / "results.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return int(any(case["exit_code"] != 0 for case in manifest["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())

"""Prepare, capacity-test, and launch the fixed single-H100 RLM recipe.

Use runs/runh100_rlm.sh to install the locked environment first.
See dev/h100-training/H100_RLM.md for the asset bundle, resource requirements, and recovery.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys

from nanochat.paths import RunName, configure_runtime_paths, get_datasets_dir, get_run_dir
from nanochat.rlm_policy import InputEmbeddingFreeze


REPOSITORY = Path(__file__).resolve().parents[1]
GB = 1_000_000_000
EXPECTED_SHARDS = tuple(f"shard_{index:05d}.parquet" for index in (*range(170), 6542))
TOKENIZER_FILES = ("tokenizer.pkl", "token_bytes.pt")


@dataclass(frozen=True)
class Recipe:
    name: str
    device_batch_size: int
    accumulation_steps: int
    probe_freeze: InputEmbeddingFreeze
    training_freeze: InputEmbeddingFreeze


RECIPES = (
    Recipe("b32-unfrozen", 32, 8, InputEmbeddingFreeze.NONE, InputEmbeddingFreeze.SCHEDULED),
    Recipe("b32-frozen", 32, 8, InputEmbeddingFreeze.PERMANENT, InputEmbeddingFreeze.PERMANENT),
    Recipe("b16-unfrozen", 16, 16, InputEmbeddingFreeze.NONE, InputEmbeddingFreeze.SCHEDULED),
)


def checked_path(path, *, directory=False, must_exist=True):
    """Accept explicit local paths, but never follow symlinks in owned artifacts."""
    if ".." in path.parts:
        raise ValueError(f"Path must not contain parent-directory traversal: {path}")
    path = Path(os.path.abspath(path.expanduser()))
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError(f"Path must not contain symlinks: {path}")
    if must_exist and (not path.is_dir() if directory else not path.is_file()):
        raise ValueError(f"Expected {'directory' if directory else 'regular file'}: {path}")
    if path.exists() and not directory and not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Expected regular file: {path}")
    return path


def digest(path):
    checked_path(path)
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    checked_path(path, must_exist=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    checked_path(temporary, must_exist=False)
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, help="verified embedding-only seed bundle; required for a fresh run")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--run-name", type=RunName, help="run name under repository training_run/; default: NANOCHAT_RUN_NAME or pretrain-1")
    destination.add_argument("--output-dir", type=Path, help="explicit run directory instead of training_run/<run-name>")
    parser.add_argument("--datasets-dir", type=Path, help="shared dataset root; defaults to NANOCHAT_DATASETS_DIR or repository datasets/")
    parser.add_argument("--resume", action="store_true", help="reuse the saved recipe and resume train/checkpoints/latest.pt; no probes or reinitialization")
    args = parser.parse_args(argv)
    if not args.resume and args.assets is None:
        parser.error("fresh training requires --assets pointing to the embedding-only seed bundle")
    if args.resume and args.assets is not None:
        parser.error("--resume restores checkpoint weights; omit --assets")
    return args


def training_arguments(recipe, output_dir, *, assets=None, probe=False, resume=False):
    arguments = [
        "--mode", "capacity-probe" if probe else "train", "--device-type", "cuda",
        "--depth", "24", "--width", "1536", "--head-dim", "128",
        "--text-length", "1023", "--memory-tokens", "512", "--cycles", "4",
        "--device-batch-size", str(recipe.device_batch_size), "--total-batch-size", "1047552",
        "--num-iterations", "5574", "--warmup-steps", "40", "--warmdown-ratio", "0.65",
        "--final-lr-frac", "0.05", "--embedding-lr", "0.3", "--unembedding-lr", "0.008",
        "--matrix-lr", "0.02", "--scalar-lr", "0.5", "--weight-decay", "0.28",
        "--seed", "42", "--compile", "--activation-checkpointing",
        "--input-embedding-freeze", (recipe.probe_freeze if probe else recipe.training_freeze).value,
        "--checkpoint-mode", "async", "--output-dir", str(output_dir),
    ]
    if not probe:
        arguments.extend(("--save-every", "10", "--keep-checkpoints", "10"))
    if resume:
        arguments.append("--resume-latest")
    elif assets is not None:
        arguments.extend(("--init-embedding-bundle", str(assets)))
    else:
        raise ValueError("Fresh training and probes require an embedding bundle")
    return arguments


def check_gpu():
    import psutil
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check nvidia-smi and the NVIDIA driver's support for this repository's locked CUDA 13.2 PyTorch build; do not substitute a different training environment silently.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES before launching.")
    properties = torch.cuda.get_device_properties(0)
    if "H100" not in properties.name or "MIG" in properties.name or not 75 * GB <= properties.total_memory <= 90 * GB:
        raise RuntimeError(f"This launcher requires one full H100 80 GB GPU; found {properties.name} with {properties.total_memory / GB:.1f} GB.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU/runtime does not support the required BF16 compute.")
    free, total = torch.cuda.mem_get_info(0)
    ceiling = min(total, free) - 2 * GB
    if ceiling <= 0:
        raise RuntimeError("The selected GPU has no usable memory after the 2 GB external-allocation reserve.")
    memory = psutil.virtual_memory()
    if memory.available < 32 * GB:
        raise RuntimeError(f"At least 32 GB of host RAM must be currently available; found {memory.total / GB:.1f} GB total and {memory.available / GB:.1f} GB available.")
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True,
        ).strip().splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Cannot query the NVIDIA driver with nvidia-smi.") from exc
    return {
        "name": properties.name, "total_bytes": total, "free_bytes": free,
        "allocator_limit_gb": ceiling / GB, "driver_versions": driver,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "host_total_bytes": memory.total, "host_available_bytes": memory.available,
    }


def runtime_environment(output_dir, datasets_dir, gpu):
    environment = dict(os.environ)
    environment.update({
        "NANOCHAT_BASE_DIR": str(output_dir), "NANOCHAT_DATASETS_DIR": str(datasets_dir),
        "NANOCHAT_TRAIN_SHARDS": "170", "NANOCHAT_DTYPE": "bfloat16",
        "NANOCHAT_ATTENTION_IMPL": "sdpa", "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        "NANOCHAT_CUDA_MEMORY_LIMIT_GB": str(gpu["allocator_limit_gb"]),
        "NANOCHAT_MIN_SYSTEM_AVAILABLE_GB": "12", "NANOCHAT_CHECKPOINT_RESERVE_GB": "100",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True", "WORLD_SIZE": "1",
    })
    return environment


def require_disk_space(output_dir, datasets_dir):
    checkpoint_dir = output_dir / "train" / "checkpoints"
    existing = 0
    if checkpoint_dir.exists():
        checked_path(checkpoint_dir, directory=True)
        for path in checkpoint_dir.rglob("*"):
            checked_path(path, directory=path.is_dir())
            if path.is_file():
                existing += path.stat().st_size
    dataset_dir = datasets_dir / "base_data_climbmix"
    missing = sum(not (dataset_dir / name).is_file() for name in EXPECTED_SHARDS)
    requirements = (
        (output_dir, 100 * GB + max(0, 156 * GB - existing)),
        (datasets_dir, missing * 110_000_000 + GB),
    )
    volumes = {}
    for path, required in requirements:
        device = path.stat().st_dev
        previous_path, previous_required = volumes.get(device, (path, 0))
        volumes[device] = previous_path, previous_required + required
    for path, required in volumes.values():
        available = shutil.disk_usage(path).free
        if available < required:
            raise RuntimeError(f"Insufficient free disk space at {path}: need {required / GB:.1f} GB, have {available / GB:.1f} GB. This includes retained checkpoints, staging, and the 100 GB checkpoint reserve.")


def run_logged(command, log_path, environment):
    checked_path(log_path, must_exist=False)
    print(f"Running: {shlex.join(command)}\nLog: {log_path}", flush=True)
    with log_path.open("x") as log:
        with subprocess.Popen(command, cwd=REPOSITORY, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return process.wait()


def verify_dataset(datasets_dir):
    import pyarrow.parquet as parquet

    directory = checked_path(datasets_dir / "base_data_climbmix", directory=True)
    missing = [name for name in EXPECTED_SHARDS if not (directory / name).exists()]
    if missing:
        raise ValueError(f"Dataset requires training shards 00000..00169 and validation shard 06542; missing={missing}.")
    inventory = {}
    for name in EXPECTED_SHARDS:
        path = checked_path(directory / name)
        source = parquet.ParquetFile(path)
        if not source.metadata.num_rows or "text" not in source.schema.names:
            raise ValueError(f"Empty or invalid dataset shard: {path}")
        inventory[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    return inventory


def tokenizer_inventory(output_dir):
    return {name: digest(output_dir / "tokenizer" / name) for name in TOKENIZER_FILES}


def prepare_assets(assets, output_dir):
    # The bundle loader validates exact file hashes, tensor shape/dtype, and source provenance.
    from nanochat.rlm_initialization import TokenEmbeddingInitialization

    initialization = TokenEmbeddingInitialization.from_bundle(assets, assets / "tokenizer.pkl")
    if initialization.contract()["embedding_shape"] != [32768, 1536]:
        raise ValueError("The H100 recipe requires the original 32768 x 1536 input embedding table.")
    tokenizer_dir = output_dir / "tokenizer"
    checked_path(tokenizer_dir, directory=True, must_exist=False)
    tokenizer_dir.mkdir(exist_ok=True)
    for name in TOKENIZER_FILES:
        source = checked_path(assets / name)
        destination = checked_path(tokenizer_dir / name, must_exist=False)
        if destination.exists():
            if digest(destination) != digest(source):
                raise ValueError(f"Existing tokenizer differs from the seed bundle: {destination}; use a new run directory.")
        else:
            with source.open("rb") as incoming, destination.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
    return {"manifest_sha256": digest(assets / "manifest.json"), "tokenizer": tokenizer_inventory(output_dir)}


def choose_recipe(output_dir, assets, environment, log_dir, runner=None):
    runner = run_logged if runner is None else runner
    for index, recipe in enumerate(RECIPES, start=1):
        probe_dir = checked_path(
            output_dir / "probes" / f"{index:02d}-{recipe.name}", directory=True, must_exist=False,
        )
        if probe_dir.exists():
            raise FileExistsError(f"A previous probe already exists at {probe_dir}; preserve this run and choose a new --output-dir.")
        arguments = training_arguments(recipe, probe_dir, assets=assets, probe=True)
        code = runner([sys.executable, "-m", "scripts.rlm_train", *arguments],
                      log_dir / f"probe-{index:02d}-{recipe.name}.log", environment)
        result_path = probe_dir / "result.json"
        if not result_path.exists():
            raise RuntimeError(f"Probe exited with status {code} without a structured result; inspect {log_dir}. No recipe fallback was selected.")
        result = json.loads(checked_path(result_path).read_text())
        updates = result.get("updates", [])
        capacity = result.get("capacity_probe", {})
        expected_frozen = recipe.probe_freeze is InputEmbeddingFreeze.PERMANENT
        if (code == 0 and result.get("status") == "completed"
                and result.get("mode") == "capacity-probe"
                and result.get("completed_updates") == 1 and result.get("attempt_updates") == 1
                and capacity.get("optimizer_state_preallocated") is True
                and capacity.get("checkpoint_staging_preallocated") is True
                and capacity.get("input_embedding_frozen") is expected_frozen
                and len(updates) == 1
                and updates[0].get("finite_loss_and_gradients") is True
                and updates[0].get("finite_model_and_optimizer_state") is True):
            return recipe
        if code > 0 and result.get("status") == "failed" and result.get("failure_kind") == "cuda_out_of_memory":
            print(f"Capacity probe {recipe.name} exceeded memory; trying the next authorized recipe.", flush=True)
            continue
        raise RuntimeError(f"Probe {recipe.name} failed for a reason other than GPU memory capacity: {result}. Inspect {log_dir} before retrying.")
    raise RuntimeError("WARNING: all three authorized H100 capacity probes failed. Training was not started; no smaller batch or alternate precision was selected.")


def load_recipe(output_dir):
    path = checked_path(output_dir / "selected-recipe.json")
    saved = json.loads(path.read_text())
    if not isinstance(saved, dict) or saved.get("schema_version") != 2:
        raise ValueError("Unsupported selected-recipe.json schema; refusing to resume.")
    for recipe in RECIPES:
        if saved.get("recipe") == asdict(recipe):
            expected = training_arguments(recipe, output_dir / "train", assets=Path(saved["assets_directory"]))
            if saved.get("production_arguments") != expected:
                raise ValueError("Saved production arguments do not match the supported H100 recipe; refusing to resume.")
            return recipe, saved
    raise ValueError("Saved recipe is not one of the three authorized H100 recipes.")


def launch(args):
    from filelock import FileLock

    if any(name in os.environ for name in ("RANK", "LOCAL_RANK")) or os.environ.get("WORLD_SIZE", "1") != "1":
        raise ValueError("Run the launcher directly on one GPU, outside torchrun or a distributed worker environment.")
    output_dir = checked_path(args.output_dir or get_run_dir(args.run_name), directory=True, must_exist=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = checked_path(output_dir / ".launcher.lock", must_exist=False)
    with FileLock(str(lock_path), timeout=0):
        recipe_path = output_dir / "selected-recipe.json"
        recipe = None
        saved = None
        if args.resume:
            recipe, saved = load_recipe(output_dir)
            datasets_dir = checked_path(Path(saved["datasets_directory"]), directory=True)
            if args.datasets_dir is not None and checked_path(args.datasets_dir, directory=True) != datasets_dir:
                raise ValueError("Resume must reuse the datasets directory recorded in selected-recipe.json.")
            checked_path(output_dir / "train" / "checkpoints" / "latest.pt")
            assets = None
        else:
            if recipe_path.exists() or (output_dir / "train").exists():
                raise FileExistsError("This output directory already has a selected recipe or training artifacts; use --resume or a new --output-dir.")
            datasets_dir = checked_path(args.datasets_dir or get_datasets_dir(), directory=True, must_exist=False)
            assets = checked_path(args.assets, directory=True)
        datasets_dir.mkdir(parents=True, exist_ok=True)
        log_dir = checked_path(
            output_dir / "logs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
            directory=True, must_exist=False,
        )
        log_dir.mkdir(parents=True)
        configure_runtime_paths(output_dir)
        gpu = check_gpu()
        environment = runtime_environment(output_dir, datasets_dir, gpu)
        require_disk_space(output_dir, datasets_dir)
        write_json(log_dir / "preflight.json", gpu)
        if args.resume:
            if tokenizer_inventory(output_dir) != saved["assets"]["tokenizer"]:
                raise ValueError("Tokenizer changed since recipe selection; refusing to resume.")
        else:
            assets_record = prepare_assets(assets, output_dir)
        dataset_lock = checked_path(datasets_dir / ".h100-datasets.lock", must_exist=False)
        with FileLock(str(dataset_lock), timeout=0):
            if args.resume:
                if verify_dataset(datasets_dir) != saved["dataset"]:
                    raise ValueError("Dataset content changed since recipe selection; refusing to resume.")
            else:
                dataset_dir = checked_path(datasets_dir / "base_data_climbmix", directory=True, must_exist=False)
                dataset_dir.mkdir(exist_ok=True)
                for name in EXPECTED_SHARDS:
                    checked_path(dataset_dir / name, must_exist=False)
                    checked_path(dataset_dir / (name + ".tmp"), must_exist=False)
                code = run_logged([sys.executable, "-m", "nanochat.dataset", "-n", "170"],
                                  log_dir / "download.log", environment)
                if code:
                    raise RuntimeError(f"Dataset downloader failed with status {code}; inspect {log_dir}.")
                dataset = verify_dataset(datasets_dir)
        if not args.resume:
            recipe = choose_recipe(output_dir, assets, environment, log_dir)
            production_arguments = training_arguments(recipe, output_dir / "train", assets=assets)
            saved = {
                "schema_version": 2, "recipe": asdict(recipe),
                "assets_directory": str(assets), "datasets_directory": str(datasets_dir),
                "assets": assets_record, "dataset": dataset, "gpu": gpu,
                "production_arguments": production_arguments,
                "environment": {key: environment[key] for key in (
                    "NANOCHAT_BASE_DIR", "NANOCHAT_DATASETS_DIR", "NANOCHAT_TRAIN_SHARDS",
                    "NANOCHAT_DTYPE", "NANOCHAT_ATTENTION_IMPL",
                    "NANOCHAT_CUDA_MEMORY_LIMIT_GB", "NANOCHAT_MIN_SYSTEM_AVAILABLE_GB",
                    "NANOCHAT_CHECKPOINT_RESERVE_GB", "OMP_NUM_THREADS", "PYTORCH_ALLOC_CONF",
                )},
                "python": sys.version,
            }
            write_json(recipe_path, saved)
            print(f"Selected {recipe.name}: batch {recipe.device_batch_size} x {recipe.accumulation_steps}, embedding policy {recipe.training_freeze}.", flush=True)
        arguments = training_arguments(recipe, output_dir / "train", assets=assets, resume=args.resume)
        write_json(log_dir / "invocation.json", {"arguments": arguments, "resume": args.resume})
        code = run_logged([sys.executable, "-m", "scripts.rlm_train", *arguments],
                          log_dir / "training.log", environment)
        if code:
            raise RuntimeError(f"Training exited with status {code}; inspect {log_dir / 'training.log'}. To resume a completed checkpoint, rerun with --resume --output-dir {shlex.quote(str(output_dir))}.")


def main():
    args = parse_args()
    try:
        launch(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"H100 launcher: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

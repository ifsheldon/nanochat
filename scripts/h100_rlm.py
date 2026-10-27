"""Prepare, capacity-test, and launch RLM profiles on 1, 2, or 4 H100s.

Use runs/runh100_rlm.sh to install the locked environment first.
See dev/h100-training/H100_RLM.md for the asset bundle, resource requirements, and recovery.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys

from nanochat.model_profile import CHECKPOINT_PROFILE_NAME, ModelProfileDocument
from nanochat.paths import RunName, configure_runtime_paths, get_datasets_dir, get_run_dir
from nanochat.rlm_policy import AttentionBackend, InputEmbeddingFreeze


REPOSITORY = Path(__file__).resolve().parents[1]
GB = 1_000_000_000
EXPECTED_SHARDS = tuple(f"shard_{index:05d}.parquet" for index in (*range(170), 6542))
TOKENIZER_FILES = ("tokenizer.pkl", "token_bytes.pt")


@dataclass(frozen=True)
class Recipe:
    name: str
    device_batch_size: int
    probe_freeze: InputEmbeddingFreeze
    training_freeze: InputEmbeddingFreeze


RECIPES = (
    Recipe("b32-unfrozen", 32, InputEmbeddingFreeze.NONE, InputEmbeddingFreeze.SCHEDULED),
    Recipe("b32-frozen", 32, InputEmbeddingFreeze.PERMANENT, InputEmbeddingFreeze.PERMANENT),
    Recipe("b16-unfrozen", 16, InputEmbeddingFreeze.NONE, InputEmbeddingFreeze.SCHEDULED),
)


def accumulation_steps(recipe, text_length, num_gpus):
    tokens_per_microbatch = recipe.device_batch_size * 4 * text_length * num_gpus
    steps, remainder = divmod(1047552, tokens_per_microbatch)
    if remainder or steps < 1:
        raise ValueError("The H100 global batch of 1047552 text tokens must be divisible by device batch size x 4 cycles x profile text_length x GPU count.")
    return steps


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
    parser.add_argument("--num-gpus", type=int, choices=(1, 2, 4), required=True,
                        help="use the first N visible H100s; required for fresh training and resume")
    parser.add_argument("--assets", type=Path, help="verified embedding-only seed bundle; required for a fresh run")
    parser.add_argument("--model-profile", type=Path, required=True,
                        help="JSONC architecture profile; required for fresh training and resume")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--run-name", type=RunName, help="run name under repository training_run/; default: NANOCHAT_RUN_NAME or pretrain-1")
    destination.add_argument("--output-dir", type=Path, help="explicit run directory instead of training_run/<run-name>")
    parser.add_argument("--datasets-dir", type=Path, help="shared dataset root; defaults to the saved directory on resume, otherwise NANOCHAT_DATASETS_DIR or repository datasets/")
    parser.add_argument("--resume", action="store_true", help="reuse the saved recipe and resume train/checkpoints/latest.pt; no probes or reinitialization")
    args = parser.parse_args(argv)
    if not args.resume and args.assets is None:
        parser.error("fresh training requires --assets pointing to the embedding-only seed bundle")
    if args.resume and args.assets is not None:
        parser.error("--resume restores checkpoint weights; omit --assets")
    return args


def training_arguments(recipe, output_dir, *, model_profile, assets=None, mode="train", resume=False):
    probe = mode != "train"
    arguments = [
        "--mode", mode, "--device-type", "cuda",
        "--model-profile", str(model_profile), "--cycles", "4",
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


def check_gpu(num_gpus):
    import psutil
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check nvidia-smi and the NVIDIA driver's support for this repository's locked PyTorch 2.14.1 CUDA 12.6 build; do not substitute a different training environment silently.")
    available = torch.cuda.device_count()
    if available < num_gpus:
        raise RuntimeError(f"Requested {num_gpus} H100 GPUs, but only {available} CUDA devices are visible.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    device_ids = visible.split(",") if visible is not None else [str(index) for index in range(available)]
    devices = []
    for index in range(num_gpus):
        properties = torch.cuda.get_device_properties(index)
        if "H100" not in properties.name or "MIG" in properties.name or properties.total_memory < 75 * GB:
            raise RuntimeError(f"Selected device {index} must be a full H100 GPU with at least 75 GB of VRAM; found {properties.name} with {properties.total_memory / GB:.1f} GB.")
        with torch.cuda.device(index):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(f"Selected GPU {index}/runtime does not support the required BF16 compute.")
        free, total = torch.cuda.mem_get_info(index)
        devices.append({"device": device_ids[index], "name": properties.name,
                        "total_bytes": total, "free_bytes": free})
    ceiling = min(min(device["total_bytes"], device["free_bytes"]) for device in devices) - 2 * GB
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
        "num_gpus": num_gpus, "devices": devices,
        "visible_devices": ",".join(device_ids[:num_gpus]),
        "allocator_limit_gb": ceiling / GB, "driver_versions": driver,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "host_total_bytes": memory.total, "host_available_bytes": memory.available,
    }


def runtime_environment(output_dir, datasets_dir, gpu):
    environment = dict(os.environ)
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK",
                 "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    for name in tuple(environment):
        if name.startswith("TORCHELASTIC_"):
            environment.pop(name)
    environment.update({
        "NANOCHAT_BASE_DIR": str(output_dir), "NANOCHAT_DATASETS_DIR": str(datasets_dir),
        "NANOCHAT_TRAIN_SHARDS": "170", "NANOCHAT_DTYPE": "bfloat16",
        "NANOCHAT_ATTENTION_IMPL": "sdpa", "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        "NANOCHAT_CUDA_MEMORY_LIMIT_GB": str(gpu["allocator_limit_gb"]),
        "NANOCHAT_MIN_SYSTEM_AVAILABLE_GB": "12", "NANOCHAT_CHECKPOINT_RESERVE_GB": "100",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "CUDA_VISIBLE_DEVICES": gpu["visible_devices"],
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


def training_command(arguments, num_gpus):
    if num_gpus == 1:
        return [sys.executable, "-m", "scripts.rlm_train", *arguments]
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc-per-node={num_gpus}", "-m", "scripts.rlm_train", *arguments]


def choose_recipe(output_dir, assets, environment, log_dir, runner=None, *, num_gpus=1, model_profile):
    runner = run_logged if runner is None else runner
    for index, recipe in enumerate(RECIPES, start=1):
        probe_dir = checked_path(
            output_dir / "probes" / f"{index:02d}-{recipe.name}", directory=True, must_exist=False,
        )
        if probe_dir.exists():
            raise FileExistsError(f"A previous probe already exists at {probe_dir}; preserve this run and choose a new --output-dir.")
        arguments = training_arguments(recipe, probe_dir, assets=assets, mode="capacity-probe", model_profile=model_profile)
        code = runner(training_command(arguments, num_gpus),
                      log_dir / f"probe-{index:02d}-{recipe.name}.log", environment)
        result_path = probe_dir / "result.json"
        if not result_path.exists():
            raise RuntimeError(f"Probe exited with status {code} without a structured result; inspect {log_dir}. No recipe fallback was selected.")
        result = json.loads(checked_path(result_path).read_text())
        updates = result.get("updates", [])
        capacity = result.get("capacity_probe", {})
        expected_frozen = recipe.probe_freeze is InputEmbeddingFreeze.PERMANENT
        if (code == 0 and result.get("status") == "completed"
                and result.get("world_size") == num_gpus
                and result.get("mode") == "capacity-probe"
                and result.get("completed_updates") == 1 and result.get("attempt_updates") == 1
                and capacity.get("optimizer_state_preallocated") is True
                and capacity.get("checkpoint_staging_preallocated") is True
                and capacity.get("input_embedding_frozen") is expected_frozen
                and len(updates) == 1
                and updates[0].get("finite_loss_and_gradients") is True
                and updates[0].get("finite_model_and_optimizer_state") is True):
            return recipe
        if (code > 0 and result.get("world_size") == num_gpus
                and result.get("status") == "failed" and result.get("failure_kind") == "cuda_out_of_memory"):
            print(f"Capacity probe {recipe.name} exceeded memory; trying the next authorized recipe.", flush=True)
            continue
        raise RuntimeError(f"Probe {recipe.name} failed for a reason other than GPU memory capacity: {result}. Inspect {log_dir} before retrying.")
    raise RuntimeError("WARNING: all three authorized H100 capacity probes failed. Training was not started; no smaller batch or alternate precision was selected.")


def choose_attention(recipe, output_dir, assets, environment, log_dir, runner=None, *, num_gpus=1, model_profile):
    """Compare isolated, compilation-free updates; an unusable FA3 is optional."""
    runner = run_logged if runner is None else runner
    measurements = {}
    for backend in AttentionBackend:
        probe_dir = checked_path(
            output_dir / "probes" / f"attention-{backend.value}", directory=True, must_exist=False,
        )
        if probe_dir.exists():
            raise FileExistsError(f"An attention probe already exists at {probe_dir}; choose a new --output-dir.")
        log_path = log_dir / f"attention-{backend.value}.log"
        arguments = training_arguments(recipe, probe_dir, assets=assets, mode="attention-probe", model_profile=model_profile)
        probe_environment = dict(environment, NANOCHAT_ATTENTION_IMPL=backend.value)
        code = None
        try:
            code = runner(training_command(arguments, num_gpus), log_path, probe_environment)
            if code != 0:
                raise RuntimeError(f"Attention probe exited with status {code}; inspect {log_path}.")
            result = json.loads(checked_path(probe_dir / "result.json").read_text())
            measurement = result["attention_probe"]
            updates = result["updates"]
            duration = measurement["duration_seconds"]
            if (result.get("status") != "completed" or result.get("mode") != "attention-probe"
                    or result.get("world_size") != num_gpus
                    or AttentionBackend(result["attention_impl"]) is not backend
                    or measurement.get("compilation_excluded") is not True
                    or measurement.get("optimizer_state_preallocated") is not True
                    or measurement.get("checkpoint_staging_preallocated") is not True
                    or measurement.get("input_embedding_frozen") is not (recipe.probe_freeze is InputEmbeddingFreeze.PERMANENT)
                    or type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0
                    or len(updates) != 1
                    or updates[0].get("finite_loss_and_gradients") is not True
                    or updates[0].get("finite_model_and_optimizer_state") is not True):
                raise ValueError("Attention probe did not report a healthy compilation-free update for the requested backend and recipe")
            measurements[backend.value] = {"status": "completed", **measurement}
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError) as exc:
            if backend is AttentionBackend.SDPA:
                raise RuntimeError(f"SDPA attention probe failed: {exc}") from exc
            measurements[backend.value] = {
                "status": "failed", "exit_code": code, "error": str(exc), "log": str(log_path),
            }
            print(f"FlashAttention probe failed; continuing with SDPA: {exc}", flush=True)
    selected = AttentionBackend.SDPA
    flash = measurements[AttentionBackend.FA3.value]
    if (flash["status"] == "completed"
            and flash["duration_seconds"] < measurements[selected.value]["duration_seconds"]):
        selected = AttentionBackend.FA3
    print(f"Selected attention {selected.value}: {measurements[selected.value]['duration_seconds']:.3f} s per compiled update.", flush=True)
    return {"selected": selected.value, "measurements": measurements}


def load_recipe(output_dir, text_length):
    path = checked_path(output_dir / "selected-recipe.json")
    saved = json.loads(path.read_text())
    if not isinstance(saved, dict) or saved.get("schema_version") != 3:
        raise ValueError("Unsupported selected-recipe.json schema; refusing to resume.")
    if type(saved.get("num_gpus")) is not int or saved["num_gpus"] not in (1, 2, 4):
        raise ValueError("Saved recipe must record a supported GPU count.")
    for recipe in RECIPES:
        if (saved.get("recipe") == asdict(recipe)
                and saved.get("accumulation_steps") == accumulation_steps(recipe, text_length, saved["num_gpus"])):
            return recipe, saved
    raise ValueError("Saved recipe is not one of the three authorized H100 recipes.")


def launch(args):
    from filelock import FileLock

    if any(name in os.environ for name in ("RANK", "LOCAL_RANK")) or os.environ.get("WORLD_SIZE", "1") != "1":
        raise ValueError("Run the launcher directly, outside torchrun or a distributed worker environment.")
    output_dir = checked_path(args.output_dir or get_run_dir(args.run_name), directory=True, must_exist=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = checked_path(output_dir / ".launcher.lock", must_exist=False)
    with FileLock(str(lock_path), timeout=0):
        recipe_path = output_dir / "selected-recipe.json"
        recipe = None
        saved = None
        requested = ModelProfileDocument.read(checked_path(args.model_profile))
        if args.resume:
            checkpoint_profile = checked_path(
                output_dir / "train" / "checkpoints" / CHECKPOINT_PROFILE_NAME, must_exist=False,
            )
            if checkpoint_profile.exists():
                document = ModelProfileDocument.read(checkpoint_profile)
                if requested.profile != document.profile:
                    raise ValueError("Explicit model profile differs from the saved checkpoint profile.")
                model_profile = checkpoint_profile
            else:
                document = requested
                model_profile = checked_path(output_dir / CHECKPOINT_PROFILE_NAME, must_exist=False)
                document.copy_once(model_profile)
                document = ModelProfileDocument.read(model_profile)
            recipe, saved = load_recipe(output_dir, document.profile.text_length)
            datasets_dir = checked_path(args.datasets_dir or Path(saved["datasets_directory"]), directory=True)
            checked_path(output_dir / "train" / "checkpoints" / "latest.pt")
            assets = None
        else:
            if recipe_path.exists() or (output_dir / "train").exists():
                raise FileExistsError("This output directory already has a selected recipe or training artifacts; use --resume or a new --output-dir.")
            datasets_dir = checked_path(args.datasets_dir or get_datasets_dir(), directory=True, must_exist=False)
            assets = checked_path(args.assets, directory=True)
            document = requested
            model_profile = checked_path(output_dir / CHECKPOINT_PROFILE_NAME, must_exist=False)
            document.copy_once(model_profile)
            document = ModelProfileDocument.read(model_profile)
        datasets_dir.mkdir(parents=True, exist_ok=True)
        log_dir = checked_path(
            output_dir / "logs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
            directory=True, must_exist=False,
        )
        log_dir.mkdir(parents=True)
        configure_runtime_paths(output_dir)
        gpu = check_gpu(args.num_gpus)
        environment = runtime_environment(output_dir, datasets_dir, gpu)
        if args.resume:
            environment["NANOCHAT_ATTENTION_IMPL"] = AttentionBackend(saved["environment"]["NANOCHAT_ATTENTION_IMPL"]).value
        require_disk_space(output_dir, datasets_dir)
        write_json(log_dir / "preflight.json", gpu)
        if not args.resume:
            assets_record = prepare_assets(assets, output_dir)
            dataset_lock = checked_path(datasets_dir / ".h100-datasets.lock", must_exist=False)
            with FileLock(str(dataset_lock), timeout=0):
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
            recipe = choose_recipe(output_dir, assets, environment, log_dir, num_gpus=args.num_gpus, model_profile=model_profile)
            attention = choose_attention(recipe, output_dir, assets, environment, log_dir, num_gpus=args.num_gpus, model_profile=model_profile)
            environment["NANOCHAT_ATTENTION_IMPL"] = attention["selected"]
            production_arguments = training_arguments(recipe, output_dir / "train", assets=assets, model_profile=model_profile)
            saved = {
                "schema_version": 3, "recipe": asdict(recipe),
                "num_gpus": args.num_gpus, "accumulation_steps": accumulation_steps(recipe, document.profile.text_length, args.num_gpus),
                "assets_directory": str(assets), "datasets_directory": str(datasets_dir),
                "assets": assets_record, "dataset": dataset, "gpu": gpu,
                "production_arguments": production_arguments,
                "attention_probe": attention,
                "environment": {key: environment[key] for key in (
                    "NANOCHAT_BASE_DIR", "NANOCHAT_DATASETS_DIR", "NANOCHAT_TRAIN_SHARDS",
                    "NANOCHAT_DTYPE", "NANOCHAT_ATTENTION_IMPL",
                    "NANOCHAT_CUDA_MEMORY_LIMIT_GB", "NANOCHAT_MIN_SYSTEM_AVAILABLE_GB",
                    "NANOCHAT_CHECKPOINT_RESERVE_GB", "OMP_NUM_THREADS", "PYTORCH_ALLOC_CONF",
                    "CUDA_VISIBLE_DEVICES",
                )},
                "python": sys.version,
            }
            write_json(recipe_path, saved)
        accumulation = accumulation_steps(recipe, document.profile.text_length, args.num_gpus)
        print(f"Selected {recipe.name}: {args.num_gpus} GPUs x batch {recipe.device_batch_size} x accumulation {accumulation}, embedding policy {recipe.training_freeze}.", flush=True)
        arguments = training_arguments(recipe, output_dir / "train", assets=assets, resume=args.resume, model_profile=model_profile)
        command = training_command(arguments, args.num_gpus)
        write_json(log_dir / "invocation.json", {"arguments": arguments, "command": command,
                                               "resume": args.resume, "num_gpus": args.num_gpus,
                                               "accumulation_steps": accumulation})
        code = run_logged(command,
                          log_dir / "training.log", environment)
        if code:
            raise RuntimeError(f"Training exited with status {code}; inspect {log_dir / 'training.log'}. To resume a completed checkpoint, rerun with --num-gpus {args.num_gpus} --resume --output-dir {shlex.quote(str(output_dir))} --model-profile {shlex.quote(str(output_dir / 'train' / 'checkpoints' / CHECKPOINT_PROFILE_NAME))}.")


def main():
    args = parse_args()
    try:
        launch(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"H100 launcher: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

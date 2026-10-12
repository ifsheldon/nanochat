"""Exercise the H100 handoff's ordered decisions and checkpoint resume on CPU."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import h100_rlm as launcher


def argument(command, name):
    return command[command.index(name) + 1]


def passed_result(frozen=False):
    return {
        "status": "completed", "mode": "capacity-probe", "completed_updates": 1,
        "attempt_updates": 1, "capacity_probe": {
            "optimizer_state_preallocated": True, "checkpoint_staging_preallocated": True,
            "input_embedding_frozen": frozen,
        }, "updates": [{"finite_loss_and_gradients": True, "finite_model_and_optimizer_state": True}],
    }


def fake_probe_runner(outcomes, commands):
    def run(command, log_path, environment):
        commands.append(command)
        destination = Path(argument(command, "--output-dir"))
        destination.mkdir(parents=True)
        outcome = outcomes[len(commands) - 1]
        if outcome == "success":
            result, code = passed_result(argument(command, "--input-embedding-freeze") == "permanent"), 0
        elif outcome == "oom":
            result, code = {"status": "failed", "failure_kind": "cuda_out_of_memory"}, 1
        else:
            result, code = outcome, 1
        (destination / "result.json").write_text(json.dumps(result))
        return code
    return run


@pytest.mark.parametrize("outcomes,selected", [
    (["success"], 0), (["oom", "success"], 1), (["oom", "oom", "success"], 2),
])
def test_capacity_selection_respects_order_and_records_correct_production_phase(tmp_path, outcomes, selected):
    commands = []
    recipe = launcher.choose_recipe(tmp_path / "output", tmp_path / "assets", {}, tmp_path,
                                    fake_probe_runner(outcomes, commands))
    assert recipe == launcher.RECIPES[selected]
    assert len(commands) == len(outcomes)
    for command, expected in zip(commands, launcher.RECIPES):
        assert argument(command, "--mode") == "capacity-probe"
        assert argument(command, "--input-embedding-freeze") == expected.probe_freeze
        assert argument(command, "--device-batch-size") == str(expected.device_batch_size)
        assert argument(command, "--num-iterations") == "5574"
        assert int(argument(command, "--total-batch-size")) == expected.device_batch_size * expected.accumulation_steps * 4 * 1023
        assert "--save-every" not in command
    production = launcher.training_arguments(recipe, tmp_path / "train", assets=tmp_path / "assets")
    assert argument(production, "--input-embedding-freeze") == ("permanent" if selected == 1 else "scheduled")
    assert argument(production, "--save-every") == "10"


def test_all_capacity_failures_abort_without_fourth_recipe(tmp_path):
    commands = []
    with pytest.raises(RuntimeError, match="WARNING: all three"):
        launcher.choose_recipe(tmp_path / "output", tmp_path / "assets", {}, tmp_path,
                               fake_probe_runner(["oom"] * 3, commands))
    assert len(commands) == 3
    assert not (tmp_path / "output" / "selected-recipe.json").exists()


@pytest.mark.parametrize("failure", [
    {"status": "failed", "failure_kind": "error", "error_type": "RuntimeError", "error": "CUDA driver mismatch"},
    {"status": "failed", "error_type": "OutOfMemoryError"},
    {"status": "failed", "failure_kind": "error", "error": "Insufficient host memory"},
    {"status": "completed", "mode": "probe", "completed_updates": 2},
])
def test_unclassified_failures_do_not_silently_change_recipe(tmp_path, failure):
    commands = []
    with pytest.raises(RuntimeError, match="other than GPU memory capacity"):
        launcher.choose_recipe(tmp_path / "output", tmp_path / "assets", {}, tmp_path,
                               fake_probe_runner([failure], commands))
    assert len(commands) == 1


@pytest.mark.parametrize("missing_field", ["optimizer_state_preallocated", "checkpoint_staging_preallocated"])
def test_probe_without_resident_memory_contract_is_not_accepted(tmp_path, missing_field):
    calls = []
    def run(command, log_path, environment):
        calls.append(command)
        path = Path(argument(command, "--output-dir"))
        path.mkdir(parents=True)
        result = passed_result()
        del result["capacity_probe"][missing_field]
        (path / "result.json").write_text(json.dumps(result))
        return 0
    with pytest.raises(RuntimeError, match="other than GPU memory capacity"):
        launcher.choose_recipe(tmp_path / "output", tmp_path / "assets", {}, tmp_path, run)
    assert len(calls) == 1


@pytest.mark.parametrize("destination_kind", ["explicit", "named", "default"])
def test_launch_and_resume_reuse_recipe_without_probes_download_or_initialization(tmp_path, monkeypatch, destination_kind):
    output = tmp_path / "output"
    assets = tmp_path / "assets"
    assets.mkdir()
    calls = []
    environments = []
    monkeypatch.setattr(launcher, "get_datasets_dir", lambda: tmp_path / "datasets")
    if destination_kind != "explicit":
        def resolve_run(name):
            assert name == (launcher.RunName("h100-test") if destination_kind == "named" else None)
            return output
        monkeypatch.setattr(launcher, "get_run_dir", resolve_run)
    destination = {
        "explicit": ["--output-dir", str(output)], "named": ["--run-name", "h100-test"], "default": [],
    }[destination_kind]
    inventories = {"dataset": {"shard": "hash"}, "tokenizer": {"tokenizer.pkl": "sha", "token_bytes.pt": "sha"}}
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    configured = []
    monkeypatch.setattr(launcher, "configure_runtime_paths", configured.append)
    def check_gpu():
        assert configured[-1] == output
        return {"allocator_limit_gb": 80}
    monkeypatch.setattr(launcher, "check_gpu", check_gpu)
    monkeypatch.setattr(launcher, "require_disk_space", lambda *_: None)
    def prepare_assets(source, run_dir):
        assert source == assets and run_dir == output
        return {"manifest_sha256": "seed", "tokenizer": inventories["tokenizer"].copy()}
    def verify_dataset(datasets_dir):
        assert datasets_dir == tmp_path / "datasets"
        return inventories["dataset"].copy()
    def tokenizer_inventory(run_dir):
        assert run_dir == output
        return inventories["tokenizer"].copy()
    monkeypatch.setattr(launcher, "prepare_assets", prepare_assets)
    monkeypatch.setattr(launcher, "verify_dataset", verify_dataset)
    monkeypatch.setattr(launcher, "tokenizer_inventory", tokenizer_inventory)

    def run(command, log_path, environment):
        calls.append(command)
        environments.append(environment)
        if command[2] == "nanochat.dataset":
            assert command[3:] == ["-n", "170"]
            return 0
        destination = Path(argument(command, "--output-dir"))
        destination.mkdir(parents=True, exist_ok=True)
        if argument(command, "--mode") == "capacity-probe":
            result = passed_result(argument(command, "--input-embedding-freeze") == "permanent")
            if argument(command, "--input-embedding-freeze") == "none":
                result = {"status": "failed", "failure_kind": "cuda_out_of_memory"}
            (destination / "result.json").write_text(json.dumps(result))
            return int(result["status"] == "failed")
        (destination / "checkpoints").mkdir(exist_ok=True)
        (destination / "checkpoints" / "latest.pt").write_bytes(b"checkpoint")
        return 0
    monkeypatch.setattr(launcher, "run_logged", run)
    launcher.launch(launcher.parse_args(["--assets", str(assets), *destination]))
    saved = json.loads((output / "selected-recipe.json").read_text())
    assert saved["recipe"] == asdict(launcher.RECIPES[1])
    assert saved["datasets_directory"] == str(tmp_path / "datasets")
    assert saved["environment"]["NANOCHAT_BASE_DIR"] == str(output)
    assert saved["environment"]["NANOCHAT_DATASETS_DIR"] == str(tmp_path / "datasets")
    assert saved["environment"]["NANOCHAT_TRAIN_SHARDS"] == "170"
    assert not (output / "cache").exists()
    assert [command[2] for command in calls] == ["nanochat.dataset", "scripts.rlm_train", "scripts.rlm_train", "scripts.rlm_train"]
    assert "--init-embedding-bundle" in calls[-1]
    assert argument(calls[-1], "--input-embedding-freeze") == "permanent"
    calls.clear()
    environments.clear()
    inventories["dataset"]["shard"] = "changed"
    inventories["tokenizer"]["tokenizer.pkl"] = "changed"
    saved["production_arguments"][saved["production_arguments"].index("--num-iterations") + 1] = "10"
    (output / "selected-recipe.json").write_text(json.dumps(saved))
    monkeypatch.setattr(launcher, "prepare_assets", lambda *_: pytest.fail("resume must not initialize assets"))
    monkeypatch.setattr(launcher, "verify_dataset", lambda *_: pytest.fail("resume must not scan dataset identity"))
    monkeypatch.setattr(launcher, "tokenizer_inventory", lambda *_: pytest.fail("resume must not scan tokenizer identity"))
    launcher.launch(launcher.parse_args(["--resume", *destination]))
    assert len(calls) == 1
    assert calls[0][2] == "scripts.rlm_train"
    assert "--resume-latest" in calls[0]
    assert "--init-embedding-bundle" not in calls[0]
    assert argument(calls[0], "--input-embedding-freeze") == "permanent"
    assert argument(calls[0], "--device-batch-size") == "32"
    assert argument(calls[0], "--num-iterations") == "5574"
    assert environments[0]["NANOCHAT_DATASETS_DIR"] == str(tmp_path / "datasets")
    alternate = tmp_path / "other-datasets"
    (tmp_path / "datasets").rename(alternate)
    launcher.launch(launcher.parse_args(["--resume", *destination, "--datasets-dir", str(alternate)]))
    assert len(calls) == 2
    assert calls[1] == calls[0]
    assert environments[1]["NANOCHAT_DATASETS_DIR"] == str(alternate)


def test_dataset_verification_selects_required_shards_from_shared_root_and_hashes_contents(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = tmp_path / "base_data_climbmix"
    directory.mkdir()
    for name in launcher.EXPECTED_SHARDS:
        pq.write_table(pa.table({"text": ["training document"]}), directory / name)
    inventory = launcher.verify_dataset(tmp_path)
    pq.write_table(pa.table({"text": ["different document"]}), directory / launcher.EXPECTED_SHARDS[0])
    changed = launcher.verify_dataset(tmp_path)
    assert changed[launcher.EXPECTED_SHARDS[0]]["sha256"] != inventory[launcher.EXPECTED_SHARDS[0]]["sha256"]
    (directory / "shard_00170.parquet").write_bytes(b"extra")
    assert launcher.verify_dataset(tmp_path) == changed
    (directory / "shard_00170.parquet").unlink()
    (directory / "shard_06542.parquet").unlink()
    with pytest.raises(ValueError, match="missing=.*shard_06542"):
        launcher.verify_dataset(tmp_path)


def test_symlinked_artifact_paths_are_rejected_before_writes(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "link"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        launcher.checked_path(linked / "output", directory=True, must_exist=False)
    with pytest.raises(ValueError, match="symlinks"):
        launcher.write_json(linked / "output.json", {"value": 1})
    assert not (real / "output.json").exists()


def test_disk_preflight_adds_cache_and_output_budgets_on_same_volume(tmp_path, monkeypatch):
    output, cache = tmp_path / "output", tmp_path / "cache"
    output.mkdir()
    cache.mkdir()
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda _: SimpleNamespace(free=270 * launcher.GB))
    with pytest.raises(RuntimeError, match="Insufficient free disk"):
        launcher.require_disk_space(output, cache)
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda _: SimpleNamespace(free=280 * launcher.GB))
    launcher.require_disk_space(output, cache)


@pytest.mark.parametrize("field,value", [("input_embedding_frozen", True), ("finite_loss_and_gradients", False), ("finite_model_and_optimizer_state", False)])
def test_wrong_phase_or_failed_health_checks_cannot_select_recipe(tmp_path, field, value):
    def run(command, log_path, environment):
        path = Path(argument(command, "--output-dir"))
        path.mkdir(parents=True)
        result = passed_result()
        if field == "input_embedding_frozen":
            result["capacity_probe"][field] = value
        else:
            result["updates"][0][field] = value
        (path / "result.json").write_text(json.dumps(result))
        return 0
    with pytest.raises(RuntimeError, match="other than GPU memory capacity"):
        launcher.choose_recipe(tmp_path / "output", tmp_path / "assets", {}, tmp_path, run)


def test_parent_directory_components_are_rejected_before_normalization(tmp_path):
    with pytest.raises(ValueError, match="parent-directory traversal"):
        launcher.checked_path(tmp_path / "scratch" / ".." / "output", directory=True, must_exist=False)


def test_signal_termination_does_not_trigger_fallback_even_with_oom_result(tmp_path):
    calls = []
    writer = fake_probe_runner(["oom"], calls)
    def run(*args):
        writer(*args)
        return -9
    with pytest.raises(RuntimeError, match="other than GPU memory capacity"):
        launcher.choose_recipe(tmp_path / "output", tmp_path / "assets", {}, tmp_path, run)
    assert len(calls) == 1


def test_gpu_capacity_and_environment_override_spark_allocator_limit(monkeypatch):
    cuda = SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1,
        get_device_properties=lambda _: SimpleNamespace(name="NVIDIA H100 80GB HBM3", total_memory=85 * launcher.GB),
        is_bf16_supported=lambda: True,
        mem_get_info=lambda _: (83 * launcher.GB, 85 * launcher.GB),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, __version__="locked-torch", version=SimpleNamespace(cuda="13.2")))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(virtual_memory=lambda: SimpleNamespace(total=64 * launcher.GB, available=40 * launcher.GB)))
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *_args, **_kwargs: "driver-version\n")
    monkeypatch.setenv("NANOCHAT_CUDA_MEMORY_LIMIT_GB", "100")
    monkeypatch.setenv("NANOCHAT_DTYPE", "float32")
    gpu = launcher.check_gpu()
    environment = launcher.runtime_environment(Path("/run"), Path("/datasets"), gpu)
    assert float(environment["NANOCHAT_CUDA_MEMORY_LIMIT_GB"]) == 81
    assert environment["NANOCHAT_DTYPE"] == "bfloat16"
    cuda.get_device_properties = lambda _: SimpleNamespace(name="NVIDIA H100 80GB MIG", total_memory=85 * launcher.GB)
    with pytest.raises(RuntimeError, match="full H100"):
        launcher.check_gpu()


def test_probe_symlink_is_rejected_before_dispatch_or_external_writes(tmp_path):
    output = tmp_path / "output"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / "probes").symlink_to(outside, target_is_directory=True)
    def run(*_):
        pytest.fail("A probe must not launch through a symlinked output ancestor")
    with pytest.raises(ValueError, match="symlinks"):
        launcher.choose_recipe(output, tmp_path / "assets", {}, tmp_path, run)
    assert list(outside.iterdir()) == []


def test_log_symlink_is_rejected_before_external_directory_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "get_datasets_dir", lambda: tmp_path / "datasets")
    output = tmp_path / "output"
    outside = tmp_path / "outside"
    assets = tmp_path / "assets"
    output.mkdir()
    outside.mkdir()
    assets.mkdir()
    (output / "logs").symlink_to(outside, target_is_directory=True)
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(launcher, "check_gpu", lambda: pytest.fail("Symlink validation must precede GPU initialization"))
    with pytest.raises(ValueError, match="symlinks"):
        launcher.launch(launcher.parse_args(["--assets", str(assets), "--output-dir", str(output)]))
    assert list(outside.iterdir()) == []


def test_wrapper_sets_run_local_download_caches_before_uv_without_shell_interpolation(tmp_path):
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    uv = binary_dir / "uv"
    uv.write_text("#!/usr/bin/env python3\nimport json, os, sys\nprint(json.dumps({'args': sys.argv[1:], 'uv_cache': os.environ['UV_CACHE_DIR'], 'xet_cache': os.environ['HF_XET_CACHE']}))\n")
    uv.chmod(0o700)
    run = tmp_path / "run with $(touch UNEXPECTED_COMMAND_EXECUTION)"
    environment = dict(os.environ, PATH=str(binary_dir) + os.pathsep + os.environ["PATH"])
    for name in ("UV_CACHE_DIR", "HF_XET_CACHE"):
        environment.pop(name, None)
    result = subprocess.run(
        ["bash", "runs/runh100_rlm.sh", "--assets", str(tmp_path / "seed"), "--output-dir", str(run)],
        cwd=launcher.REPOSITORY, env=environment, capture_output=True, text=True, check=True,
    )
    recorded = json.loads(result.stdout)
    assert recorded["uv_cache"] == str(run / "cache" / "uv")
    assert recorded["xet_cache"] == str(run / "cache" / "huggingface" / "xet")
    assert recorded["args"][-4:] == ["--assets", str(tmp_path / "seed"), "--output-dir", str(run)]
    assert not run.exists()
    assert not (launcher.REPOSITORY / "UNEXPECTED_COMMAND_EXECUTION").exists()
    help_result = subprocess.run(
        ["bash", "runs/runh100_rlm.sh", "--assets", "missing", "--help"],
        cwd=launcher.REPOSITORY, env=environment, capture_output=True, text=True, check=True,
    )
    assert "--run-name" in help_result.stdout and '"uv_cache"' not in help_result.stdout


def test_wrapper_preserves_explicit_uv_and_xet_cache_locations(tmp_path):
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    uv = binary_dir / "uv"
    uv.write_text("#!/usr/bin/env python3\nimport json, os\nprint(json.dumps([os.environ['UV_CACHE_DIR'], os.environ['HF_XET_CACHE']]))\n")
    uv.chmod(0o700)
    environment = dict(os.environ, PATH=str(binary_dir) + os.pathsep + os.environ["PATH"],
                       UV_CACHE_DIR=str(tmp_path / "shared-uv"), HF_XET_CACHE=str(tmp_path / "shared-xet"))
    run = tmp_path / "run"
    result = subprocess.run(
        ["bash", "runs/runh100_rlm.sh", "--assets", str(tmp_path / "seed"), "--output-dir", str(run)],
        cwd=launcher.REPOSITORY, env=environment, capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout) == [str(tmp_path / "shared-uv"), str(tmp_path / "shared-xet")]
    assert not run.exists()

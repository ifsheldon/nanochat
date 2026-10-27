"""Exercise the H100 handoff's ordered decisions and checkpoint resume on CPU."""

from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import h100_rlm as launcher


TEST_MODEL_PROFILE = launcher.REPOSITORY / "model_profiles" / "rlm-d24.jsonc"


def argument(command, name):
    return command[command.index(name) + 1]


def passed_result(frozen=False):
    return {
        "world_size": 1, "status": "completed", "mode": "capacity-probe", "completed_updates": 1,
        "attempt_updates": 1, "capacity_probe": {
            "optimizer_state_preallocated": True, "checkpoint_staging_preallocated": True,
            "input_embedding_frozen": frozen,
        }, "updates": [{"finite_loss_and_gradients": True, "finite_model_and_optimizer_state": True}],
    }


def passed_attention_result(backend, duration, frozen=False):
    return {
        "world_size": 1, "status": "completed", "mode": "attention-probe", "attention_impl": backend,
        "attention_probe": {
            "duration_seconds": duration, "warmup_updates": 2, "compilation_excluded": True,
            "optimizer_state_preallocated": True, "checkpoint_staging_preallocated": True,
            "input_embedding_frozen": frozen,
        },
        "updates": [{"finite_loss_and_gradients": True, "finite_model_and_optimizer_state": True}],
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
            result, code = {"status": "failed", "failure_kind": "cuda_out_of_memory", "world_size": 1}, 1
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
                                    fake_probe_runner(outcomes, commands), model_profile=TEST_MODEL_PROFILE, total_batch_token_size=1047552)
    assert recipe == launcher.RECIPES[selected]
    assert len(commands) == len(outcomes)
    for command, expected in zip(commands, launcher.RECIPES):
        assert argument(command, "--mode") == "capacity-probe"
        assert argument(command, "--input-embedding-freeze") == expected.probe_freeze
        assert argument(command, "--device-batch-size") == str(expected.device_batch_size)
        assert argument(command, "--num-iterations") == "5574"
        assert int(argument(command, "--total-batch-token-size")) == 1047552
        assert argument(command, "--model-profile").endswith("model_profiles/rlm-d24.jsonc")
        assert "--depth" not in command and "--text-length" not in command
        assert "--save-every" not in command
    production = launcher.training_arguments(recipe, tmp_path / "train", assets=tmp_path / "assets", model_profile=TEST_MODEL_PROFILE, total_batch_token_size=1047552)
    assert argument(production, "--input-embedding-freeze") == ("permanent" if selected == 1 else "scheduled")
    assert argument(production, "--save-every") == "10"


def test_all_capacity_failures_abort_without_fourth_recipe(tmp_path):
    commands = []
    with pytest.raises(RuntimeError, match="WARNING: all three"):
        launcher.choose_recipe(tmp_path / "output", tmp_path / "assets", {}, tmp_path,
                               fake_probe_runner(["oom"] * 3, commands), model_profile=TEST_MODEL_PROFILE, total_batch_token_size=1047552)
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
                               fake_probe_runner([failure], commands), model_profile=TEST_MODEL_PROFILE, total_batch_token_size=1047552)
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
        launcher.choose_recipe(tmp_path / "output", tmp_path / "assets", {}, tmp_path, run, model_profile=TEST_MODEL_PROFILE, total_batch_token_size=1047552)
    assert len(calls) == 1


@pytest.mark.parametrize("profile_file,total_batch_token_size,one_gpu_accumulation", [
    ("rlm-d24.jsonc", 1047552, 8), ("rlm-d24-c511-m128.jsonc", 1046528, 16),
])
@pytest.mark.parametrize("initial_gpus,resume_gpus", [(1, 1), (1, 2), (2, 4), (4, 1)])
@pytest.mark.parametrize("destination_kind", ["explicit", "named", "default"])
@pytest.mark.parametrize("flash_crashes", [False, True])
def test_launch_and_resume_reuse_recipe_without_probes_download_or_initialization(tmp_path, monkeypatch, destination_kind, flash_crashes, initial_gpus, resume_gpus, profile_file, total_batch_token_size, one_gpu_accumulation):
    output = tmp_path / "output"
    assets = tmp_path / "assets"
    assets.mkdir()
    source_profile = tmp_path / "selected.jsonc"
    profile_text = "// Preserve this original profile comment.\n" + (TEST_MODEL_PROFILE.parent / profile_file).read_text()
    source_profile.write_text(profile_text)
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
    def check_gpu(num_gpus):
        assert configured[-1] == output
        return {"allocator_limit_gb": 80, "visible_devices": ",".join(str(index) for index in range(num_gpus))}
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
        assert argument(command, "--total-batch-token-size") == str(total_batch_token_size)
        assert "--total-batch-size" not in command
        selected_profile = Path(argument(command, "--model-profile"))
        assert selected_profile.read_text() == profile_text
        if "--resume-latest" in command:
            assert selected_profile == output / "train" / "checkpoints" / "model-profile.jsonc"
        else:
            assert selected_profile == output / "model-profile.jsonc"
        # A source edit after selection must never alter subsequent stages.
        source_profile.write_text("{}")
        workers = next((int(value.split("=")[1]) for value in command
                        if value.startswith("--nproc-per-node=")), 1)
        destination = Path(argument(command, "--output-dir"))
        destination.mkdir(parents=True, exist_ok=True)
        if argument(command, "--mode") == "capacity-probe":
            result = passed_result(argument(command, "--input-embedding-freeze") == "permanent")
            if argument(command, "--input-embedding-freeze") == "none":
                result = {"status": "failed", "failure_kind": "cuda_out_of_memory"}
            result["world_size"] = workers
            (destination / "result.json").write_text(json.dumps(result))
            return int(result["status"] == "failed")
        if argument(command, "--mode") == "attention-probe":
            backend = environment["NANOCHAT_ATTENTION_IMPL"]
            if flash_crashes and backend == "fa3":
                return -11
            result = passed_attention_result(backend, 12 if backend == "fa3" else 15, frozen=True)
            result["world_size"] = workers
            (destination / "result.json").write_text(json.dumps(result))
            return 0
        (destination / "checkpoints").mkdir(exist_ok=True)
        (destination / "checkpoints" / "latest.pt").write_bytes(b"checkpoint")
        (destination / "checkpoints" / "model-profile.jsonc").write_text(selected_profile.read_text())
        return 0
    monkeypatch.setattr(launcher, "run_logged", run)
    launcher.launch(launcher.parse_args(["--num-gpus", str(initial_gpus), "--assets", str(assets), "--model-profile", str(source_profile), "--total-batch-token-size", str(total_batch_token_size), *destination]))
    saved = json.loads((output / "selected-recipe.json").read_text())
    assert saved["recipe"] == asdict(launcher.RECIPES[1])
    assert saved["num_gpus"] == initial_gpus
    assert saved["total_batch_token_size"] == total_batch_token_size
    assert saved["accumulation_steps"] == one_gpu_accumulation // initial_gpus
    assert saved["datasets_directory"] == str(tmp_path / "datasets")
    assert saved["environment"]["NANOCHAT_BASE_DIR"] == str(output)
    assert saved["environment"]["NANOCHAT_DATASETS_DIR"] == str(tmp_path / "datasets")
    assert saved["environment"]["NANOCHAT_TRAIN_SHARDS"] == "170"
    assert not (output / "cache").exists()
    assert [command[7] if command[2] == "torch.distributed.run" else command[2] for command in calls] == ["nanochat.dataset", *["scripts.rlm_train"] * 5]
    selected = "sdpa" if flash_crashes else "fa3"
    assert saved["attention_probe"]["selected"] == selected
    assert saved["environment"]["NANOCHAT_ATTENTION_IMPL"] == selected
    assert environments[-1]["NANOCHAT_ATTENTION_IMPL"] == selected
    assert "--init-embedding-bundle" in calls[-1]
    assert argument(calls[-1], "--input-embedding-freeze") == "permanent"
    calls.clear()
    environments.clear()
    source_profile.unlink()
    (output / "model-profile.jsonc").unlink()
    inventories["dataset"]["shard"] = "changed"
    inventories["tokenizer"]["tokenizer.pkl"] = "changed"
    saved["production_arguments"][saved["production_arguments"].index("--num-iterations") + 1] = "10"
    (output / "selected-recipe.json").write_text(json.dumps(saved))
    monkeypatch.setattr(launcher, "prepare_assets", lambda *_: pytest.fail("resume must not initialize assets"))
    monkeypatch.setattr(launcher, "verify_dataset", lambda *_: pytest.fail("resume must not scan dataset identity"))
    monkeypatch.setattr(launcher, "tokenizer_inventory", lambda *_: pytest.fail("resume must not scan tokenizer identity"))
    launcher.launch(launcher.parse_args(["--num-gpus", str(resume_gpus), "--resume", *destination,
                                        "--model-profile", str(output / "train" / "checkpoints" / "model-profile.jsonc")]))
    assert len(calls) == 1
    expected_prefix = [sys.executable, "-m", "scripts.rlm_train"]
    if resume_gpus > 1:
        expected_prefix = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                           f"--nproc-per-node={resume_gpus}", "-m", "scripts.rlm_train"]
    assert calls[0][:len(expected_prefix)] == expected_prefix
    assert environments[0]["CUDA_VISIBLE_DEVICES"] == ",".join(str(index) for index in range(resume_gpus))
    invocations = [json.loads(path.read_text()) for path in sorted((output / "logs").glob("*/invocation.json"))]
    assert invocations[-1]["num_gpus"] == resume_gpus
    assert invocations[-1]["total_batch_token_size"] == total_batch_token_size
    assert invocations[-1]["accumulation_steps"] == one_gpu_accumulation // resume_gpus
    assert "--resume-latest" in calls[0]
    assert "--init-embedding-bundle" not in calls[0]
    assert argument(calls[0], "--input-embedding-freeze") == "permanent"
    assert argument(calls[0], "--device-batch-size") == "32"
    assert argument(calls[0], "--num-iterations") == "5574"
    assert environments[0]["NANOCHAT_DATASETS_DIR"] == str(tmp_path / "datasets")
    assert environments[0]["NANOCHAT_ATTENTION_IMPL"] == selected
    alternate = tmp_path / "other-datasets"
    (tmp_path / "datasets").rename(alternate)
    equivalent_profile = tmp_path / "equivalent.jsonc"
    equivalent_profile.write_text(profile_text.replace("Preserve this original profile comment.", "Equivalent profile, different comment."))
    launcher.launch(launcher.parse_args(["--num-gpus", str(resume_gpus), "--resume", *destination,
                                        "--datasets-dir", str(alternate), "--model-profile", str(equivalent_profile),
                                        "--total-batch-token-size", str(total_batch_token_size)]))
    assert len(calls) == 2
    assert calls[1] == calls[0]
    assert environments[1]["NANOCHAT_DATASETS_DIR"] == str(alternate)


@pytest.mark.parametrize("sdpa_seconds,fa3_seconds,selected", [
    (20, 15, "fa3"), (15, 20, "sdpa"), (15, 15, "sdpa"),
])
def test_attention_selection_compares_the_same_recipe_and_uses_fastest_valid_backend(
    tmp_path, sdpa_seconds, fa3_seconds, selected,
):
    calls = []
    environment = {"NANOCHAT_ATTENTION_IMPL": "sdpa"}

    def run(command, log_path, child_environment):
        backend = child_environment["NANOCHAT_ATTENTION_IMPL"]
        calls.append(backend)
        assert argument(command, "--mode") == "attention-probe"
        assert argument(command, "--num-iterations") == "5574"
        assert argument(command, "--input-embedding-freeze") == "none"
        assert argument(command, "--device-batch-size") == "32"
        assert "--save-every" not in command
        path = Path(argument(command, "--output-dir"))
        path.mkdir(parents=True)
        duration = sdpa_seconds if backend == "sdpa" else fa3_seconds
        (path / "result.json").write_text(json.dumps(passed_attention_result(backend, duration)))
        return 0

    result = launcher.choose_attention(launcher.RECIPES[0], tmp_path, tmp_path / "assets", environment, tmp_path, run, model_profile=TEST_MODEL_PROFILE, total_batch_token_size=1047552)
    assert result["selected"] == selected
    assert calls == ["sdpa", "fa3"]
    assert environment["NANOCHAT_ATTENTION_IMPL"] == "sdpa"


@pytest.mark.parametrize("failure", [
    "signal", "nonzero", "missing-result", "malformed-result", "wrong-backend",
    "compilation", "nonfinite-timing", "failed-health", "launch-error",
])
def test_flash_attention_probe_failures_preserve_successful_sdpa(tmp_path, failure):
    def run(command, log_path, environment):
        backend = environment["NANOCHAT_ATTENTION_IMPL"]
        path = Path(argument(command, "--output-dir"))
        path.mkdir(parents=True)
        result = passed_attention_result(backend, 10 if backend == "fa3" else 20)
        if backend == "fa3":
            if failure == "signal":
                return launcher.run_logged(
                    [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"],
                    log_path, environment,
                )
            if failure == "nonzero":
                return 1
            if failure == "missing-result":
                return 0
            if failure == "malformed-result":
                (path / "result.json").write_text("{incomplete")
                return 0
            if failure == "wrong-backend":
                result["attention_impl"] = "sdpa"
            if failure == "compilation":
                result["attention_probe"]["compilation_excluded"] = False
            if failure == "nonfinite-timing":
                result["attention_probe"]["duration_seconds"] = float("nan")
            if failure == "failed-health":
                result["updates"][0]["finite_loss_and_gradients"] = False
            if failure == "launch-error":
                raise OSError("injected subprocess failure")
        (path / "result.json").write_text(json.dumps(result))
        return 0

    result = launcher.choose_attention(launcher.RECIPES[0], tmp_path, tmp_path / "assets", {}, tmp_path, run, model_profile=TEST_MODEL_PROFILE, total_batch_token_size=1047552)
    assert result["selected"] == "sdpa"
    assert result["measurements"]["sdpa"]["duration_seconds"] == 20
    assert result["measurements"]["fa3"]["status"] == "failed"


def test_sdpa_attention_probe_failure_stops_before_optional_flash_probe(tmp_path):
    calls = []

    def run(command, log_path, environment):
        calls.append(environment["NANOCHAT_ATTENTION_IMPL"])
        return 1

    with pytest.raises(RuntimeError, match="SDPA attention probe failed"):
        launcher.choose_attention(launcher.RECIPES[0], tmp_path, tmp_path / "assets", {}, tmp_path, run, model_profile=TEST_MODEL_PROFILE, total_batch_token_size=1047552)
    assert calls == ["sdpa"]


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
        launcher.choose_recipe(tmp_path / "output", tmp_path / "assets", {}, tmp_path, run, model_profile=TEST_MODEL_PROFILE, total_batch_token_size=1047552)


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
        launcher.choose_recipe(tmp_path / "output", tmp_path / "assets", {}, tmp_path, run, model_profile=TEST_MODEL_PROFILE, total_batch_token_size=1047552)
    assert len(calls) == 1


@pytest.mark.parametrize("name,total_gb,free_gb,allocator_limit_gb", [
    ("NVIDIA H100 80GB HBM3", 85, 83, 81),
    ("NVIDIA H100 NVL", 94, 92, 90),
])
def test_gpu_capacity_and_environment_override_spark_allocator_limit(monkeypatch, name, total_gb, free_gb, allocator_limit_gb):
    cuda = SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1,
        get_device_properties=lambda _: SimpleNamespace(name=name, total_memory=total_gb * launcher.GB),
        is_bf16_supported=lambda: True, device=lambda _: nullcontext(),
        mem_get_info=lambda _: (free_gb * launcher.GB, total_gb * launcher.GB),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, __version__="locked-torch", version=SimpleNamespace(cuda="13.2")))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(virtual_memory=lambda: SimpleNamespace(total=64 * launcher.GB, available=40 * launcher.GB)))
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *_args, **_kwargs: "driver-version\n")
    monkeypatch.setenv("NANOCHAT_CUDA_MEMORY_LIMIT_GB", "100")
    monkeypatch.setenv("NANOCHAT_DTYPE", "float32")
    gpu = launcher.check_gpu(1)
    environment = launcher.runtime_environment(Path("/run"), Path("/datasets"), gpu)
    assert float(environment["NANOCHAT_CUDA_MEMORY_LIMIT_GB"]) == allocator_limit_gb
    assert environment["NANOCHAT_DTYPE"] == "bfloat16"
    cuda.get_device_properties = lambda _: SimpleNamespace(name="NVIDIA H100 80GB MIG", total_memory=85 * launcher.GB)
    with pytest.raises(RuntimeError, match="full H100"):
        launcher.check_gpu(1)


def test_probe_symlink_is_rejected_before_dispatch_or_external_writes(tmp_path):
    output = tmp_path / "output"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / "probes").symlink_to(outside, target_is_directory=True)
    def run(*_):
        pytest.fail("A probe must not launch through a symlinked output ancestor")
    with pytest.raises(ValueError, match="symlinks"):
        launcher.choose_recipe(output, tmp_path / "assets", {}, tmp_path, run, model_profile=TEST_MODEL_PROFILE, total_batch_token_size=1047552)
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
        launcher.launch(launcher.parse_args(["--num-gpus", "1", "--assets", str(assets), "--model-profile", str(TEST_MODEL_PROFILE), "--total-batch-token-size", "1047552", "--output-dir", str(output)]))
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
        ["bash", "runs/runh100_rlm.sh", "--num-gpus", "1", "--model-profile", str(TEST_MODEL_PROFILE), "--total-batch-token-size", "1047552", "--assets", str(tmp_path / "seed"), "--output-dir", str(run)],
        cwd=launcher.REPOSITORY, env=environment, capture_output=True, text=True, check=True,
    )
    recorded = json.loads(result.stdout)
    assert recorded["uv_cache"] == str(run / "cache" / "uv")
    assert recorded["xet_cache"] == str(run / "cache" / "huggingface" / "xet")
    assert recorded["args"][-4:] == ["--assets", str(tmp_path / "seed"), "--output-dir", str(run)]
    assert not run.exists()
    assert not (launcher.REPOSITORY / "UNEXPECTED_COMMAND_EXECUTION").exists()
    help_result = subprocess.run(
        ["bash", "runs/runh100_rlm.sh", "--num-gpus", "1", "--assets", "missing", "--help"],
        cwd=launcher.REPOSITORY, env=environment, capture_output=True, text=True, check=True,
    )
    assert "--model-profile" in help_result.stdout
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
        ["bash", "runs/runh100_rlm.sh", "--num-gpus", "1", "--model-profile", str(TEST_MODEL_PROFILE), "--total-batch-token-size", "1047552", "--assets", str(tmp_path / "seed"), "--output-dir", str(run)],
        cwd=launcher.REPOSITORY, env=environment, capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout) == [str(tmp_path / "shared-uv"), str(tmp_path / "shared-xet")]
    assert not run.exists()


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("count", [None, "0", "3", "8"])
def test_cli_requires_an_explicit_supported_gpu_count(resume, count):
    arguments = ["--resume"] if resume else ["--assets", "/seed"]
    arguments.extend(("--model-profile", str(TEST_MODEL_PROFILE), "--total-batch-token-size", "1047552"))
    if count is not None:
        arguments.extend(("--num-gpus", count))
    with pytest.raises(SystemExit) as exc:
        launcher.parse_args(arguments)
    assert exc.value.code == 2


@pytest.mark.parametrize("num_gpus", [1, 2, 4])
@pytest.mark.parametrize("visible", [None, "5,2,7,3", "GPU-a,GPU-b,GPU-c,GPU-d"])
def test_gpu_selection_respects_visibility_and_uses_smallest_selected_free_memory(monkeypatch, num_gpus, visible):
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    selected = []
    current = []
    class Device:
        def __init__(self, index):
            self.index = index
        def __enter__(self):
            current.append(self.index)
        def __exit__(self, *_):
            current.pop()
    def properties(index):
        selected.append(index)
        return SimpleNamespace(name="NVIDIA H100", total_memory=85 * launcher.GB)
    cuda = SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 4, get_device_properties=properties,
        device=Device, is_bf16_supported=lambda: bool(current),
        mem_get_info=lambda index: ((83 - index) * launcher.GB, 85 * launcher.GB),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, __version__="locked", version=SimpleNamespace(cuda="12.6")))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(virtual_memory=lambda: SimpleNamespace(total=128 * launcher.GB, available=100 * launcher.GB)))
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *_args, **_kwargs: "driver\n")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("MASTER_ADDR", "stale")
    gpu = launcher.check_gpu(num_gpus)
    environment = launcher.runtime_environment(Path("/run"), Path("/datasets"), gpu)
    tokens = visible.split(",") if visible is not None else ["0", "1", "2", "3"]
    assert environment["CUDA_VISIBLE_DEVICES"] == ",".join(tokens[:num_gpus])
    assert selected == list(range(num_gpus))
    assert float(environment["NANOCHAT_CUDA_MEMORY_LIMIT_GB"]) == 82 - num_gpus
    assert environment.get("WORLD_SIZE", "1") == "1"
    assert "LOCAL_WORLD_SIZE" not in environment
    assert "MASTER_ADDR" not in environment


@pytest.mark.parametrize("invalid", ["unavailable", "not-h100", "mig", "small", "bf16"])
def test_every_selected_gpu_is_validated_before_launch(monkeypatch, invalid):
    def properties(index):
        name = "NVIDIA H100"
        total = 85 * launcher.GB
        if index == 1:
            name = {"not-h100": "NVIDIA A100", "mig": "NVIDIA H100 MIG"}.get(invalid, name)
            if invalid == "small":
                total = 40 * launcher.GB
        return SimpleNamespace(name=name, total_memory=total)
    current = []
    class Device:
        def __init__(self, index):
            self.index = index
        def __enter__(self):
            current.append(self.index)
        def __exit__(self, *_):
            current.pop()
    cuda = SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1 if invalid == "unavailable" else 4,
        get_device_properties=properties, device=Device,
        is_bf16_supported=lambda: not (invalid == "bf16" and current[-1] == 1),
        mem_get_info=lambda _: (83 * launcher.GB, 85 * launcher.GB),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    with pytest.raises(RuntimeError, match="visible|full H100|BF16"):
        launcher.check_gpu(2)


@pytest.mark.parametrize("num_gpus", [2, 4])
def test_multi_gpu_capacity_probes_preserve_fallback_and_require_matching_worker_reports(tmp_path, num_gpus):
    calls = []
    def run(command, log_path, environment):
        calls.append(command)
        assert command[:8] == [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc-per-node={num_gpus}", "-m", "scripts.rlm_train"]
        destination = Path(argument(command, "--output-dir"))
        destination.mkdir(parents=True)
        result = {"status": "failed", "failure_kind": "cuda_out_of_memory", "world_size": num_gpus}
        if len(calls) == 2:
            result = passed_result(True)
            result["world_size"] = num_gpus
        (destination / "result.json").write_text(json.dumps(result))
        return 1 if len(calls) == 1 else 0
    recipe = launcher.choose_recipe(tmp_path, tmp_path / "assets", {}, tmp_path, run, num_gpus=num_gpus, model_profile=TEST_MODEL_PROFILE, total_batch_token_size=1047552)
    assert recipe == launcher.RECIPES[1]
    assert len(calls) == 2


@pytest.mark.parametrize("outcome", ["success", "oom"])
def test_capacity_probe_cannot_select_result_from_another_worker_count(tmp_path, outcome):
    with pytest.raises(RuntimeError, match="other than GPU memory capacity"):
        launcher.choose_recipe(tmp_path, tmp_path / "assets", {}, tmp_path, fake_probe_runner([outcome] * 3, []), num_gpus=2, model_profile=TEST_MODEL_PROFILE, total_batch_token_size=1047552)


@pytest.mark.parametrize("num_gpus", [2, 4])
def test_attention_probe_requires_matching_worker_count(tmp_path, num_gpus):
    calls = []
    def run(command, log_path, environment):
        calls.append(command)
        assert command[:8] == [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc-per-node={num_gpus}", "-m", "scripts.rlm_train"]
        destination = Path(argument(command, "--output-dir"))
        destination.mkdir(parents=True)
        result = passed_attention_result(environment["NANOCHAT_ATTENTION_IMPL"], 10)
        (destination / "result.json").write_text(json.dumps(result))
        return 0
    with pytest.raises(RuntimeError, match="SDPA attention probe failed"):
        launcher.choose_attention(launcher.RECIPES[0], tmp_path, tmp_path / "assets", {}, tmp_path, run, num_gpus=num_gpus, model_profile=TEST_MODEL_PROFILE, total_batch_token_size=1047552)
    assert len(calls) == 1


def test_wrapper_rejects_missing_gpu_count_before_running_uv(tmp_path):
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    uv = binary_dir / "uv"
    uv.write_text("#!/bin/sh\nexit 99\n")
    uv.chmod(0o700)
    result = subprocess.run(
        ["bash", "runs/runh100_rlm.sh", "--resume", "--model-profile", str(TEST_MODEL_PROFILE), "--output-dir", str(tmp_path / "run")],
        cwd=launcher.REPOSITORY, env=dict(os.environ, PATH=str(binary_dir) + os.pathsep + os.environ["PATH"]),
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "--num-gpus" in result.stderr
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("num_gpus,expected", [(1, 24), (2, 12), (4, 6)])
@pytest.mark.parametrize("recipe_index,factor", [(0, 1), (1, 1), (2, 2)])
def test_accumulation_uses_profile_text_length_and_worker_count(num_gpus, expected, recipe_index, factor):
    assert launcher.accumulation_steps(launcher.RECIPES[recipe_index], 341, num_gpus, 1047552) == expected * factor


def test_resume_rejects_conflicting_explicit_profile_before_gpu_work(tmp_path, monkeypatch):
    output = tmp_path / "output"
    checkpoints = output / "train" / "checkpoints"
    checkpoints.mkdir(parents=True)
    profile = {"n_layer": 24, "n_embd": 1536, "n_head": 12, "n_kv_head": 12,
               "text_length": 1023, "memory_tokens": 512, "eom_tokens": 1}
    (checkpoints / "model-profile.jsonc").write_text(json.dumps(profile))
    (checkpoints / "latest.pt").write_bytes(b"checkpoint")
    explicit = tmp_path / "other.jsonc"
    explicit.write_text(json.dumps(dict(profile, n_kv_head=4)))
    (tmp_path / "datasets").mkdir()
    (output / "selected-recipe.json").write_text(json.dumps({
        "schema_version": 4, "total_batch_token_size": 1047552, "recipe": asdict(launcher.RECIPES[0]), "num_gpus": 1,
        "accumulation_steps": 8, "datasets_directory": str(tmp_path / "datasets"),
    }))
    monkeypatch.setattr(launcher, "check_gpu", lambda *_: pytest.fail("profile mismatch must precede GPU work"))
    args = launcher.parse_args(["--num-gpus", "1", "--resume", "--output-dir", str(output),
                                "--model-profile", str(explicit)])
    with pytest.raises(ValueError, match="profile.*differ|differ.*profile"):
        launcher.launch(args)
    assert json.loads((checkpoints / "model-profile.jsonc").read_text()) == profile


@pytest.mark.parametrize("text_length", [342, 1047552])
def test_accumulation_rejects_profile_lengths_that_cannot_form_global_batch(text_length):
    with pytest.raises(ValueError, match="global batch.*divisible"):
        launcher.accumulation_steps(launcher.RECIPES[0], text_length, 1, 1047552)


def test_resume_can_supply_missing_checkpoint_profile_explicitly(tmp_path, monkeypatch):
    output = tmp_path / "output"
    checkpoints = output / "train" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "latest.pt").write_bytes(b"checkpoint")
    explicit = tmp_path / "selected.jsonc"
    text = "// Profile supplied when restoring a checkpoint.\n" + json.dumps({
        "n_layer": 24, "n_embd": 1536, "n_head": 12, "n_kv_head": 4,
        "text_length": 341, "memory_tokens": 512, "eom_tokens": 1,
    })
    explicit.write_text(text)
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (output / "selected-recipe.json").write_text(json.dumps({
        "schema_version": 4, "total_batch_token_size": 1047552, "recipe": asdict(launcher.RECIPES[0]), "num_gpus": 1,
        "accumulation_steps": 24, "datasets_directory": str(datasets),
        "environment": {"NANOCHAT_ATTENTION_IMPL": "sdpa"},
    }))
    monkeypatch.setattr(launcher, "check_gpu", lambda *_: {"allocator_limit_gb": 80, "visible_devices": "0,1"})
    monkeypatch.setattr(launcher, "require_disk_space", lambda *_: None)
    commands = []
    def run(command, log_path, environment):
        commands.append(command)
        selected = Path(argument(command, "--model-profile"))
        assert selected == output / "model-profile.jsonc"
        assert selected.read_text() == text
        assert "--resume-latest" in command and "--init-embedding-bundle" not in command
        return 0
    monkeypatch.setattr(launcher, "run_logged", run)
    launcher.launch(launcher.parse_args(["--num-gpus", "2", "--resume", "--output-dir", str(output),
                                        "--model-profile", str(explicit)]))
    assert len(commands) == 1
    invocation = next((output / "logs").glob("*/invocation.json"))
    assert json.loads(invocation.read_text())["accumulation_steps"] == 12


@pytest.mark.parametrize("resume", [False, True])
def test_model_profile_is_required_for_fresh_and_resumed_launches(resume):
    arguments = ["--num-gpus", "1"]
    arguments += ["--resume"] if resume else ["--assets", "/seed", "--total-batch-token-size", "1047552"]
    with pytest.raises(SystemExit) as error:
        launcher.parse_args(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize("budget", [None, "0", "-1", "1.5", "wrong"])
def test_fresh_launch_requires_a_positive_integer_text_token_budget(budget):
    arguments = ["--num-gpus", "1", "--assets", "/seed", "--model-profile", str(TEST_MODEL_PROFILE)]
    if budget is not None:
        arguments.extend(("--total-batch-token-size", budget))
    with pytest.raises(SystemExit) as error:
        launcher.parse_args(arguments)
    assert error.value.code == 2


@pytest.fixture
def saved_h100_run(tmp_path, monkeypatch):
    output = tmp_path / "output"
    checkpoints = output / "train" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "latest.pt").write_bytes(b"checkpoint")
    (checkpoints / "model-profile.jsonc").write_bytes(TEST_MODEL_PROFILE.read_bytes())
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    record = {
        "schema_version": 4, "recipe": asdict(launcher.RECIPES[0]), "num_gpus": 1,
        "total_batch_token_size": 1047552, "accumulation_steps": 8,
        "datasets_directory": str(datasets), "environment": {"NANOCHAT_ATTENTION_IMPL": "sdpa"},
    }
    (output / "selected-recipe.json").write_text(json.dumps(record))
    monkeypatch.setattr(launcher, "check_gpu", lambda *_: pytest.fail("invalid budget must fail before GPU initialization"))
    return output, record


@pytest.mark.parametrize("budget", [None, 0, -1, True, "1047552", 1047552.0])
def test_resume_rejects_missing_or_malformed_saved_budget(saved_h100_run, budget):
    output, record = saved_h100_run
    if budget is None:
        del record["total_batch_token_size"]
    else:
        record["total_batch_token_size"] = budget
    (output / "selected-recipe.json").write_text(json.dumps(record))
    args = launcher.parse_args(["--num-gpus", "1", "--resume", "--output-dir", str(output),
                                "--model-profile", str(output / "train" / "checkpoints" / "model-profile.jsonc")])
    with pytest.raises(ValueError, match="positive integer total_batch_token_size"):
        launcher.launch(args)


def test_resume_rejects_conflicting_token_budget_before_gpu_work(saved_h100_run):
    output, _ = saved_h100_run
    args = launcher.parse_args(["--num-gpus", "1", "--resume", "--output-dir", str(output),
                                "--model-profile", str(output / "train" / "checkpoints" / "model-profile.jsonc"),
                                "--total-batch-token-size", "1046528"])
    with pytest.raises(ValueError, match="total-batch-token-size.*differs"):
        launcher.launch(args)


@pytest.mark.parametrize("resume", [False, True])
def test_nonintegral_accumulation_fails_before_gpu_or_preparation(saved_h100_run, tmp_path, resume):
    output, record = saved_h100_run
    if resume:
        record.update(total_batch_token_size=130944, accumulation_steps=1)
        (output / "selected-recipe.json").write_text(json.dumps(record))
        arguments = ["--resume", "--output-dir", str(output)]
    else:
        assets = tmp_path / "assets"
        assets.mkdir()
        arguments = ["--assets", str(assets), "--output-dir", str(tmp_path / "fresh"),
                     "--total-batch-token-size", "130945"]
    args = launcher.parse_args(["--num-gpus", "4", "--model-profile", str(TEST_MODEL_PROFILE), *arguments])
    with pytest.raises(ValueError, match="global batch.*divisible"):
        launcher.launch(args)


@pytest.mark.parametrize("budget", ["0", "-1", "1.5", "wrong"])
def test_resume_rejects_invalid_explicit_token_budget(budget):
    with pytest.raises(SystemExit) as error:
        launcher.parse_args(["--num-gpus", "1", "--resume", "--model-profile", str(TEST_MODEL_PROFILE),
                             "--total-batch-token-size", budget])
    assert error.value.code == 2


def test_removed_batch_size_option_is_rejected():
    with pytest.raises(SystemExit) as error:
        launcher.parse_args(["--num-gpus", "1", "--resume", "--model-profile", str(TEST_MODEL_PROFILE),
                             "--total-batch-size", "1047552"])
    assert error.value.code == 2

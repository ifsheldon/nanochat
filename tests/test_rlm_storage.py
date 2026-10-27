"""RLM commands share a named run's tokenizer and keep explicit output overrides."""

from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import nanochat.paths as paths
from scripts import rlm_profile, rlm_train
from test_rlm_train_cli import arguments, training_environment


@pytest.mark.parametrize("explicit_output", [False, True])
def test_training_default_and_resume_keep_tokenizer_in_selected_outer_run(
    training_environment, tmp_path, monkeypatch, explicit_output,
):
    _, tokenizer_source, _ = training_environment
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setenv("NANOCHAT_RUN_NAME", "rlm-storage")
    outer = paths.PROJECT_ROOT / "training_run" / "rlm-storage"
    if explicit_output:
        monkeypatch.setenv("NANOCHAT_BASE_DIR", str(outer))
    else:
        monkeypatch.delenv("NANOCHAT_BASE_DIR", raising=False)
    tokenizer = outer / "tokenizer" / "tokenizer.pkl"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_bytes(tokenizer_source.read_bytes())
    monkeypatch.setenv("NANOCHAT_MIN_SYSTEM_AVAILABLE_GB", "0.000001")
    monkeypatch.setenv("NANOCHAT_CHECKPOINT_RESERVE_GB", "0.000001")
    output = tmp_path / "explicit-training" if explicit_output else outer / "train"

    def launch(*extra):
        arguments(
            monkeypatch, output, "--num-iterations", "2", "--save-every", "0",
            "--checkpoint-mode", "sync", *extra,
        )
        if not explicit_output:
            position = sys.argv.index("--output-dir")
            del sys.argv[position:position + 2]
        rlm_train.main()

    launch("--stop-after-step", "1")
    assert json.loads((output / "result.json").read_text())["completed_updates"] == 1
    assert (output / "checkpoints" / "latest.pt").is_file()
    launch("--resume-latest")
    result = json.loads((output / "result.json").read_text())
    assert result["completed_updates"] == 2
    assert Path(result["checkpoint_dir"]) == output / "checkpoints"
    assert len(list((output / "attempts").iterdir())) == 2
    assert paths.get_run_dir() == outer
    assert tokenizer.read_bytes() == tokenizer_source.read_bytes()
    assert not (output / "tokenizer").exists()
    assert os.environ.get("NANOCHAT_BASE_DIR") == (str(outer) if explicit_output else None)


@pytest.mark.parametrize("module_name", ["scripts.rlm_eval", "scripts.rlm_sample"])
@pytest.mark.parametrize("explicit_checkpoint", [False, True])
def test_inference_resolves_default_or_explicit_checkpoint_directory(
    tmp_path, monkeypatch, module_name, explicit_checkpoint,
):
    module = importlib.import_module(module_name)
    outer = tmp_path / "run"
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(outer))
    monkeypatch.setenv("NANOCHAT_MIN_SYSTEM_AVAILABLE_GB", "0.000001")
    monkeypatch.setenv("WORLD_SIZE", "1")
    expected = tmp_path / "external-checkpoints" if explicit_checkpoint else outer / "train" / "checkpoints"
    argv = [module_name, "--device", "cpu"]
    if explicit_checkpoint:
        argv.extend(["--checkpoint-dir", str(expected)])
    monkeypatch.setattr(sys, "argv", argv)

    def load_checkpoint(directory, step, device):
        assert directory == expected
        assert paths.get_run_dir() == outer
        raise RuntimeError("verified checkpoint selection")

    monkeypatch.setattr("nanochat.common.compute_init", lambda device: (False, 0, 0, 1, torch.device("cpu")))
    monkeypatch.setattr("nanochat.common.compute_cleanup", lambda: None)
    monkeypatch.setattr("nanochat.training_runtime.check_training_memory", lambda *args: None)
    monkeypatch.setattr("nanochat.rlm_inference.load_rlm_checkpoint", load_checkpoint)
    with pytest.raises(RuntimeError, match="verified checkpoint selection"):
        module.main()


@pytest.mark.parametrize("explicit_paths", [False, True])
def test_profile_defaults_follow_run_and_refuse_existing_output(tmp_path, monkeypatch, explicit_paths):
    outer = tmp_path / "run"
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(outer))
    timestamp = datetime(2026, 10, 12, 12, 30, 0, 123456, tzinfo=timezone.utc)

    class FixedClock:
        @staticmethod
        def now(tz):
            return timestamp

    monkeypatch.setattr(rlm_profile, "datetime", FixedClock)
    reference = tmp_path / "reference.json" if explicit_paths else outer / "train" / "run.json"
    output = tmp_path / "explicit-profile" if explicit_paths else outer / "profiles" / "20261012T123000.123456Z"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text('{"recorded":true}')
    argv = ["rlm_profile", "--device", "cpu"]
    if explicit_paths:
        argv.extend(["--run-config", str(reference), "--output-dir", str(output)])
    monkeypatch.setattr(sys, "argv", argv)

    def profile(args):
        assert args.run_config == reference
        assert json.loads(args.run_config.read_text()) == {"recorded": True}
        assert args.output_dir == output
        (args.output_dir / "trace.json").write_text('{"traceEvents":[]}')
        return {"status": "completed"}

    monkeypatch.setattr(rlm_profile, "run", profile)
    rlm_profile.main()
    with pytest.raises(FileExistsError):
        rlm_profile.main()
    assert (output / "trace.json").read_text() == '{"traceEvents":[]}'
    assert paths.get_run_dir() == outer


@pytest.mark.parametrize("module_name", [
    "scripts.rlm_train", "scripts.rlm_eval", "scripts.rlm_sample", "scripts.rlm_profile",
])
def test_cli_sets_runtime_cache_roots_before_importing_torch(tmp_path, module_name):
    outer = tmp_path / "run"
    script = r'''
import builtins
import importlib
import os
from pathlib import Path
import sys

module = importlib.import_module(sys.argv[1])
assert "torch" not in sys.modules
sys.argv = [sys.argv[1]]
if sys.argv[0] == "scripts.rlm_train":
    sys.argv.extend(["--model-profile", "model_profiles/rlm-d24.jsonc", "--total-batch-token-size", "1047552"])
ordinary_import = builtins.__import__

def checked_import(name, *args, **kwargs):
    if name in ("torch", "nanochat.common"):
        root = Path(os.environ["NANOCHAT_BASE_DIR"])
        assert Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]) == root / "cache" / "torchinductor"
        assert Path(os.environ["TRITON_CACHE_DIR"]) == root / "cache" / "triton"
        assert Path(os.environ["WANDB_DIR"]) == root
        raise SystemExit(0)
    return ordinary_import(name, *args, **kwargs)

builtins.__import__ = checked_import
module.main()
raise AssertionError("CLI did not reach its runtime import")
'''
    environment = {key: value for key, value in os.environ.items() if key not in (
        "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "WANDB_DIR",
    )}
    environment.update(NANOCHAT_BASE_DIR=str(outer), CUDA_VISIBLE_DEVICES="")
    subprocess.run([sys.executable, "-c", script, module_name], env=environment, check=True, capture_output=True, text=True)

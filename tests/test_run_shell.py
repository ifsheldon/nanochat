"""Shell entrypoints resolve safe run storage before dependencies or training."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


def test_shell_setup_quotes_paths_and_captures_both_streams_in_run_logs(tmp_path):
    helper = Path(__file__).resolve().parents[1] / "runs" / "_common.sh"
    run = tmp_path / "run with $(touch UNEXPECTED_COMMAND_EXECUTION)"
    shared = tmp_path / "shared data"
    environment = {key: value for key, value in os.environ.items() if key not in (
        "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
        "HF_HUB_CACHE", "TIKTOKEN_CACHE_DIR", "WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR",
    )}
    environment.update({"NANOCHAT_BASE_DIR": str(run), "NANOCHAT_DATASETS_DIR": str(shared)})
    script = '''source "$1" test-run 8 || exit $?
python3 -c 'import json, os; print(json.dumps({name: os.environ[name] for name in ("NANOCHAT_BASE_DIR", "NANOCHAT_DATASETS_DIR", "NANOCHAT_TRAIN_SHARDS", "TORCHINDUCTOR_CACHE_DIR", "WANDB_DIR")}))'
echo captured-error >&2
'''
    result = subprocess.run(["bash", "-c", script, "bash", str(helper)], env=environment,
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    state = json.loads(next(line for line in result.stdout.splitlines() if line.startswith("{")))
    assert state["NANOCHAT_BASE_DIR"] == str(run)
    assert state["NANOCHAT_DATASETS_DIR"] == str(shared)
    assert state["NANOCHAT_TRAIN_SHARDS"] == "8"
    assert state["TORCHINDUCTOR_CACHE_DIR"] == str(run / "cache" / "torchinductor")
    assert state["WANDB_DIR"] == str(run)
    logs = list((run / "logs").glob("test-run-*.log"))
    assert len(logs) == 1 and "captured-error" in logs[0].read_text()
    assert not (helper.parents[1] / "UNEXPECTED_COMMAND_EXECUTION").exists()


def test_shell_setup_rejects_symlinked_log_directory_before_writing(tmp_path):
    helper = Path(__file__).resolve().parents[1] / "runs" / "_common.sh"
    run, outside = tmp_path / "run", tmp_path / "outside"
    run.mkdir()
    outside.mkdir()
    (run / "logs").symlink_to(outside, target_is_directory=True)
    environment = dict(os.environ, NANOCHAT_BASE_DIR=str(run), NANOCHAT_DATASETS_DIR=str(tmp_path / "shared"))
    result = subprocess.run(["bash", "-c", 'source "$1" test-run || exit $?', "bash", str(helper)],
                            env=environment, capture_output=True, text=True)
    assert result.returncode != 0 and "must not be a symlink" in result.stderr
    assert not list(outside.iterdir())


@pytest.mark.parametrize(("tokenizer_state", "data_state", "expected_error"), [
    ("absent", "complete", None),
    ("complete", "complete", None),
    ("partial", "complete", "The run tokenizer is incomplete"),
    ("absent", "missing", "Shared ClimbMix data is unavailable"),
    ("complete", "validation_only", "requires training and validation shards"),
    ("absent", "selected_missing", "Selected ClimbMix shard is missing"),
    ("training_failure", "complete", "returned non-zero exit status"),
])
def test_scaling_sweep_prepares_run_tokenizer_before_training(tmp_path, tokenizer_state, data_state, expected_error):
    source = Path(__file__).resolve().parents[1]
    repo, run, shared = tmp_path / "repo", tmp_path / "run", tmp_path / "shared"
    for directory in (repo / "runs", repo / "nanochat", repo / "scripts", repo / ".venv" / "bin"):
        directory.mkdir(parents=True)
    for name in ("_common.sh", "scaling_laws.sh"):
        shutil.copyfile(source / "runs" / name, repo / "runs" / name)
    for name in ("paths.py", "dataset.py"):
        shutil.copyfile(source / "nanochat" / name, repo / "nanochat" / name)
    (repo / "nanochat" / "__init__.py").touch()
    (repo / "nanochat" / "common.py").write_text("from nanochat.paths import get_datasets_dir\n")

    # Stub only expensive commands; exercise the real shell and dataset selection.
    commands = repo / ".venv" / "bin"
    (commands / "python").write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} \"$@\"\n")
    (commands / "python").chmod(0o755)
    (commands / "activate").write_text(f"export PATH={shlex.quote(str(commands))}:\"$PATH\"\n")
    (commands / "torchrun").write_text("#!/bin/sh\necho Unexpected-training-command >&2\nexit 93\n")
    (commands / "column").write_text("#!/bin/sh\nexit 0\n")
    (commands / "torchrun").chmod(0o755)
    (commands / "column").chmod(0o755)
    (repo / "scripts" / "__init__.py").touch()
    (repo / "scripts" / "tok_train.py").write_text('''import json
import os
from pathlib import Path
import sys
run = Path(os.environ["NANOCHAT_BASE_DIR"])
(run / "tokenizer-command.json").write_text(json.dumps(sys.argv[1:]))
if os.environ.get("TOKENIZER_TRAINING_FAIL"):
    raise SystemExit(7)
(run / "tokenizer").mkdir()
for name in ("tokenizer.pkl", "token_bytes.pt"):
    (run / "tokenizer" / name).write_bytes(b"trained-tokenizer")
''')

    run.mkdir()
    if tokenizer_state in ("complete", "partial"):
        (run / "tokenizer").mkdir()
        (run / "tokenizer" / "tokenizer.pkl").write_bytes(b"existing-tokenizer")
        if tokenizer_state == "complete":
            (run / "tokenizer" / "token_bytes.pt").write_bytes(b"existing-token-bytes")
    if data_state != "missing":
        data = shared / "base_data_climbmix"
        data.mkdir(parents=True)
        (data / "shard_06542.parquet").touch()
        if data_state != "validation_only":
            (data / "shard_00000.parquet").touch()

    # Completed results skip the expensive sweep after preparation succeeds.
    results_dir = run / "scaling_laws_results_jan26"
    results_dir.mkdir()
    results_file = results_dir / "results.csv"
    completed = "".join(f"{flops},{depth},complete\n" for flops in ("1e18", "2.15e18", "4.64e18", "1e19")
                        for depth in (10, 12, 14, 16, 18, 20))
    results_file.write_text(completed)
    environment = dict(os.environ, NANOCHAT_BASE_DIR=str(run), NANOCHAT_DATASETS_DIR=str(shared))
    environment.pop("NANOCHAT_TRAIN_SHARDS", None)
    if data_state == "selected_missing":
        environment["NANOCHAT_TRAIN_SHARDS"] = "2"
    if tokenizer_state == "training_failure":
        environment["TOKENIZER_TRAINING_FAIL"] = "1"
    result = subprocess.run(["bash", str(repo / "runs" / "scaling_laws.sh")], env=environment,
                            capture_output=True, text=True, check=False)
    output = result.stdout + result.stderr
    assert "Unexpected-training-command" not in output
    assert results_file.read_text() == completed
    if expected_error is not None:
        assert result.returncode != 0 and expected_error in output
        assert "Scaling Laws Sweep Complete" not in output
    else:
        assert result.returncode == 0, output
        assert "Scaling Laws Sweep Complete" in output

    command_file = run / "tokenizer-command.json"
    if tokenizer_state in ("absent", "training_failure") and data_state == "complete":
        assert json.loads(command_file.read_text()) == ["--max-chars=2000000000", "--vocab-size=32768"]
        if tokenizer_state == "absent":
            assert (run / "tokenizer" / "tokenizer.pkl").read_bytes() == b"trained-tokenizer"
            assert (run / "tokenizer" / "token_bytes.pt").read_bytes() == b"trained-tokenizer"
    else:
        assert not command_file.exists()
    if tokenizer_state in ("complete", "partial"):
        assert (run / "tokenizer" / "tokenizer.pkl").read_bytes() == b"existing-tokenizer"

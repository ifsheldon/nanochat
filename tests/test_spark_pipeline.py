from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.spark_d24 import FINAL_STEP, Pipeline, write_json


def test_existing_production_marker_blocks_fresh_launch(tmp_path, monkeypatch):
    pipeline = Pipeline(tmp_path)
    monkeypatch.setattr(pipeline, "config", lambda: {"device_batch_size": 16})
    write_json(tmp_path / "production.json", {"device_batch_size": 16})
    def unexpected(*args):
        pytest.fail("existing run was launched again")
    monkeypatch.setattr(pipeline, "run", unexpected)
    with pytest.raises(FileExistsError, match="Production already started"):
        pipeline.pretrain()


def test_resume_accepts_changed_source_configuration_and_inventory(tmp_path, monkeypatch):
    from nanochat.checkpoint_manager import save_checkpoint

    write_json(tmp_path / "validated.json", {"device_batch_size": 16, "source_sha256": "previous-source"})
    write_json(tmp_path / "production.json", {"device_batch_size": 8, "source_sha256": "older-source"})
    write_json(tmp_path / "data-inventory.json", {"shard": {"sha256": "original"}})
    pipeline = Pipeline(tmp_path, resume_step=100)
    save_checkpoint(pipeline.checkpoint_dir("base"), 100, {}, {}, {
        "step": 100,
        "user_config": {"depth": 24, "head_dim": 128, "window_pattern": "L", "max_seq_len": 2048,
                        "device_batch_size": 8, "target_param_data_ratio": 8, "num_iterations": -1, "fp8": False},
        "total_batch_size": 524288,
    })
    monkeypatch.setattr(pipeline, "inventory", lambda: {"shard": {"sha256": "changed"}})
    monkeypatch.setattr(pipeline, "complete_step", lambda source: FINAL_STEP)
    calls = []
    monkeypatch.setattr(pipeline, "run", lambda module, arguments: calls.append((module, arguments)))
    pipeline.pretrain()
    assert len(calls) == 1
    module, arguments = calls[0]
    assert module == "scripts.base_train"
    assert "--resume-from-step=100" in arguments
    assert "--device-batch-size=16" in arguments


def test_incomplete_resume_never_launches_training(tmp_path, monkeypatch):
    pipeline = Pipeline(tmp_path, resume_step=100)
    monkeypatch.setattr(pipeline, "config", lambda: {"device_batch_size": 16})
    with pytest.raises(FileNotFoundError, match="incomplete"):
        pipeline.pretrain()


def test_inventory_keeps_run_tokenizer_and_shared_subset_separate(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    shared = tmp_path / "shared"
    data = shared / "base_data_climbmix"
    data.mkdir(parents=True)
    monkeypatch.setenv("NANOCHAT_DATASETS_DIR", str(shared))
    for index in (*range(171), 6542):
        pq.write_table(pa.table({"text": [f"document {index}"]}), data / f"shard_{index:05d}.parquet")
    run = tmp_path / "run"
    tokenizer = run / "tokenizer"
    tokenizer.mkdir(parents=True)
    for name in ("tokenizer.pkl", "token_bytes.pt"):
        (tokenizer / name).write_bytes(name.encode())
    pipeline = Pipeline(run)
    initial = pipeline.inventory()
    assert len(initial) == 173
    assert "datasets/base_data_climbmix/shard_00169.parquet" in initial
    assert "datasets/base_data_climbmix/shard_06542.parquet" in initial
    assert "datasets/base_data_climbmix/shard_00170.parquet" not in initial
    assert "tokenizer/tokenizer.pkl" in initial
    pq.write_table(pa.table({"text": ["another run replaced an unselected shard"]}), data / "shard_00170.parquet")
    assert pipeline.inventory() == initial
    (tokenizer / "tokenizer.pkl").write_bytes(b"changed tokenizer")
    assert pipeline.inventory() != initial
    assert not (run / "base_data_climbmix").exists()


@pytest.mark.parametrize("existing_tokenizer", [False, True])
def test_prepare_trains_only_missing_run_tokenizer_without_copying_old_cache(tmp_path, monkeypatch, existing_tokenizer):
    import nanochat.tokenizer as tokenizer_module
    import tasks.arc
    import tasks.gsm8k
    import tasks.humaneval
    import tasks.mmlu
    import tasks.smoltalk

    shared = tmp_path / "shared"
    (shared / "eval_bundle").mkdir(parents=True)
    monkeypatch.setenv("NANOCHAT_DATASETS_DIR", str(shared))
    run = tmp_path / "new-run"
    pipeline = Pipeline(run)
    calls = []
    tokenizer_dir = run / "tokenizer"
    if existing_tokenizer:
        tokenizer_dir.mkdir()
        for name in ("tokenizer.pkl", "token_bytes.pt"):
            (tokenizer_dir / name).write_bytes(b"explicit run tokenizer")

    def launch(module, arguments=()):
        calls.append((module, list(arguments)))
        if module == "scripts.tok_train":
            tokenizer_dir.mkdir()
            for name in ("tokenizer.pkl", "token_bytes.pt"):
                (tokenizer_dir / name).write_bytes(b"fresh run tokenizer")

    monkeypatch.setattr(pipeline, "run", launch)
    monkeypatch.setattr(pipeline, "inventory", lambda: {"tokenizer": "verified"})
    monkeypatch.setattr(tokenizer_module, "get_tokenizer", lambda: SimpleNamespace(get_vocab_size=lambda: 32768))
    for module, name in ((tasks.smoltalk, "SmolTalk"), (tasks.mmlu, "MMLU"), (tasks.gsm8k, "GSM8K"),
                         (tasks.arc, "ARC"), (tasks.humaneval, "HumanEval")):
        monkeypatch.setattr(module, name, lambda **kwargs: None)
    # If a future edit reintroduces automatic reuse of the old home cache, fail
    # before it can copy any file from that unrelated run.
    monkeypatch.setattr(Path, "home", lambda: pytest.fail("fresh preparation must not search the old cache"))
    pipeline.prepare()
    expected = [("nanochat.dataset", ["-n", "170"])]
    if not existing_tokenizer:
        expected.append(("scripts.tok_train", ["--max-chars=2000000000", "--vocab-size=32768"]))
    assert calls == expected
    assert (run / "data-inventory.json").is_file()
    assert not (run / "eval_bundle").exists()
    assert not (run / "task_data").exists()
    before = len(calls)
    pipeline.prepare()
    assert len(calls) == before

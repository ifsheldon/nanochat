"""Every published rolling checkpoint is a complete, independently resumable update."""

import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import select
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from nanochat.rlm import RecurrentGPT, RLMConfig
from nanochat.rlm_latest import LatestCheckpoint, LatestCheckpointError
from nanochat.rlm_training import TrainingConfig, checkpoint_metadata, restore_rng


@pytest.fixture
def payload(monkeypatch):
    monkeypatch.delenv("NANOCHAT_CHECKPOINT_RESERVE_GB", raising=False)
    torch.manual_seed(202)
    model = RecurrentGPT(RLMConfig(
        sequence_len=9, text_length=4, memory_tokens=2, vocab_size=32,
        n_layer=1, n_head=2, n_kv_head=2, n_embd=32,
    ))
    model.init_weights()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    tokens = torch.arange(8).reshape(1, 8)
    model(tokens, (tokens + 1) % 32).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    config = TrainingConfig(device_batch_size=1, total_batch_size=8, compile=False)
    contract = {
        "model_config": asdict(model.config), "training_config": asdict(config),
        "tokenizer_sha256": "a" * 64, "device_type": "cpu",
        "runtime": {"torch": torch.__version__},
    }
    metadata = checkpoint_metadata(model, config, 1, {"row_group": 7, "epoch": 0}, contract, {
        "text_tokens": config.total_batch_size,
        "supervised_tokens": config.prediction_targets_per_update(model.config.text_length),
        "internal_positions": config.accumulation_steps(model.config.text_length)
        * config.device_batch_size * config.positions_per_episode(model.config.text_length, model.config.memory_tokens),
    })
    return model, optimizer, metadata


def publish(store, payload, step=1):
    model, optimizer, metadata = payload
    metadata = copy.deepcopy(metadata)
    for name in ("step", "text_tokens", "supervised_tokens", "internal_positions"):
        metadata[name] *= step
    return store.save(model.state_dict(), optimizer.state_dict(), metadata)


def test_saved_model_optimizer_loader_and_rng_continue_identically(tmp_path, payload):
    model, optimizer, metadata = payload
    store = LatestCheckpoint(tmp_path)
    assert not store.exists()
    assert publish(store, payload) == tmp_path / "latest.pt"
    expected_random = torch.rand(8)
    loaded_model, loaded_optimizer, loaded_metadata = store.load(torch.device("cpu"))
    resumed = RecurrentGPT(model.config)
    resumed.init_weights()
    resumed.load_state_dict(loaded_model)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters())
    resumed_optimizer.load_state_dict(loaded_optimizer)
    restore_rng(loaded_metadata["rng_state"], torch.device("cpu"))
    assert torch.equal(torch.rand(8), expected_random)
    assert loaded_metadata == json.loads(json.dumps(metadata))
    assert loaded_metadata["dataloader_state_dict"] == {"row_group": 7, "epoch": 0}
    tokens = torch.arange(8).reshape(1, 8)
    for current_model, current_optimizer in ((model, optimizer), (resumed, resumed_optimizer)):
        current_model(tokens, (tokens + 1) % 32).backward()
        current_optimizer.step()
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], tensor, rtol=0, atol=0)
    for identifier, state in optimizer.state_dict()["state"].items():
        for name, tensor in state.items():
            torch.testing.assert_close(resumed_optimizer.state_dict()["state"][identifier][name], tensor, rtol=0, atol=0)


def test_metadata_inspection_uses_restricted_cpu_mmap(tmp_path, payload, monkeypatch):
    store = LatestCheckpoint(tmp_path)
    publish(store, payload)
    original = torch.load
    calls = []

    def record(source, **kwargs):
        calls.append((source, kwargs))
        return original(source, **kwargs)

    monkeypatch.setattr(torch, "load", record)
    assert store.metadata()["step"] == 1
    source, options = calls[0]
    assert source.startswith("/proc/self/fd/")
    assert options == {"map_location": torch.device("cpu"), "weights_only": True, "mmap": True}


@pytest.mark.parametrize("failure", ["serialize", "file_fsync", "replace"])
def test_unpublished_failure_preserves_previous_complete_update(tmp_path, payload, monkeypatch, failure):
    store = LatestCheckpoint(tmp_path)
    publish(store, payload)
    before = store.path.read_bytes()

    def fail_save(value, stream):
        stream.write(b"incomplete replacement")
        raise OSError("injected serialization failure")

    def fail(*args, **kwargs):
        raise OSError(f"injected {failure} failure")

    with monkeypatch.context() as patch:
        if failure == "serialize":
            patch.setattr(torch, "save", fail_save)
        else:
            patch.setattr(os, "fsync" if failure == "file_fsync" else "replace", fail)
        with pytest.raises(LatestCheckpointError, match="was not replaced.*injected"):
            publish(store, payload, 2)
    assert store.path.read_bytes() == before
    assert store.load(torch.device("cpu"))[2]["step"] == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == ["latest.pt"]


def test_publication_is_fsynced_before_and_after_atomic_replacement(tmp_path, payload, monkeypatch):
    store = LatestCheckpoint(tmp_path)
    events = []
    real_fsync, real_replace = os.fsync, os.replace

    def sync(descriptor):
        events.append("directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file")
        real_fsync(descriptor)

    def replace(*args, **kwargs):
        events.append("replace")
        real_replace(*args, **kwargs)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", replace)
    publish(store, payload)
    assert events == ["file", "replace", "directory"]
    assert store.load(torch.device("cpu"))[2]["step"] == 1


def test_directory_sync_failure_is_reported_after_complete_replacement(tmp_path, payload, monkeypatch):
    store = LatestCheckpoint(tmp_path)
    publish(store, payload)
    real_fsync = os.fsync

    def sync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected directory sync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", sync)
    with pytest.raises(LatestCheckpointError, match="was replaced but directory durability could not be confirmed"):
        publish(store, payload, 2)
    assert store.load(torch.device("cpu"))[2]["step"] == 2
    assert sorted(path.name for path in tmp_path.iterdir()) == ["latest.pt"]


def test_killed_writer_preserves_old_update_and_next_save_cleans_staging(tmp_path, payload):
    store = LatestCheckpoint(tmp_path)
    publish(store, payload)
    before = store.path.read_bytes()
    read_fd, write_fd = os.pipe()
    script = """
import os, signal, sys
from pathlib import Path
import torch
from nanochat.rlm_latest import LatestCheckpoint
store = LatestCheckpoint(Path(sys.argv[1]))
model, optimizer, metadata = store.load(torch.device('cpu'))
for name in ('step', 'text_tokens', 'supervised_tokens', 'internal_positions'):
    metadata[name] *= 2
def interrupted_write(value, stream):
    stream.write(b'partially serialized next update')
    stream.flush()
    os.fsync(stream.fileno())
    os.write(int(sys.argv[2]), b'ready')
    signal.pause()
torch.save = interrupted_write
store.save(model, optimizer, metadata)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path), str(write_fd)],
        pass_fds=(write_fd,), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    os.close(write_fd)
    try:
        readable, _, _ = select.select([read_fd], [], [], 20)
        assert readable and os.read(read_fd, 5) == b"ready", "Writer never reached its interrupted serialization"
        process.kill()
        process.wait(timeout=10)
    finally:
        os.close(read_fd)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        process.stderr.close()
    assert store.path.read_bytes() == before
    assert store.metadata()["step"] == 1
    assert len(list(tmp_path.glob(".latest.*.tmp"))) == 1
    publish(store, payload, 2)
    assert store.metadata()["step"] == 2
    assert sorted(path.name for path in tmp_path.iterdir()) == ["latest.pt"]


def test_repeated_saves_replace_only_latest_and_bound_storage(tmp_path, payload):
    store = LatestCheckpoint(tmp_path)
    unrelated = {
        "model_000001.pt": b"numeric model", "optim_000001_rank0.pt": b"numeric optimizer",
        "meta_000001.json": b"numeric metadata", "notes.txt": b"keep",
    }
    for name, value in unrelated.items():
        (tmp_path / name).write_bytes(value)
    for step in range(1, 12):
        publish(store, payload, step)
        assert store.metadata()["step"] == step
        assert set(path.name for path in tmp_path.iterdir()) == {*unrelated, "latest.pt"}
    assert {name: (tmp_path / name).read_bytes() for name in unrelated} == unrelated


def test_reserve_accounts_for_full_staging_while_old_bundle_stays_present(tmp_path, payload, monkeypatch):
    store = LatestCheckpoint(tmp_path)
    publish(store, payload)
    before = store.path.read_bytes()
    monkeypatch.setenv("NANOCHAT_CHECKPOINT_RESERVE_GB", "1")
    monkeypatch.setattr(os, "fstatvfs", lambda fd: SimpleNamespace(f_bavail=1_000_000_001, f_frsize=1))
    with pytest.raises(LatestCheckpointError, match="Insufficient disk space.*staging"):
        publish(store, payload, 2)
    assert store.path.read_bytes() == before
    assert store.metadata()["step"] == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == ["latest.pt"]


@pytest.mark.parametrize("location", ["directory", "ancestor", "latest", "staging"])
def test_symlinks_cannot_redirect_reads_writes_or_cleanup(tmp_path, payload, location):
    owned = tmp_path / "owned"
    owned.mkdir()
    directory = owned / "checkpoints"
    directory.mkdir()
    store = LatestCheckpoint(directory)
    publish(store, payload)
    before = store.path.read_bytes()
    outside = tmp_path / "outside"
    outside.write_bytes(b"external data")
    if location in ("directory", "ancestor"):
        link = tmp_path / "redirect"
        link.symlink_to(directory if location == "directory" else owned, target_is_directory=True)
        store = LatestCheckpoint(link if location == "directory" else link / "checkpoints")
    elif location == "latest":
        store.path.unlink()
        store.path.symlink_to(outside)
    else:
        (directory / f".latest.{'a' * 32}.tmp").symlink_to(outside)
    with pytest.raises(LatestCheckpointError, match="symlink"):
        publish(store, payload, 2)
    if location != "staging":
        with pytest.raises(LatestCheckpointError, match="symlink|symbolic"):
            store.metadata()
    if location != "latest":
        assert (directory / "latest.pt").read_bytes() == before
    assert outside.read_bytes() == b"external data"


@pytest.mark.parametrize("unsafe", ["relative", "parent", "noncanonical_staging", "directory_staging", "hardlink_staging"])
def test_unsafe_paths_fail_before_checkpoint_replacement(tmp_path, payload, unsafe):
    store = LatestCheckpoint(tmp_path)
    publish(store, payload)
    before = store.path.read_bytes()
    if unsafe == "relative":
        store = LatestCheckpoint(Path("checkpoints"))
    elif unsafe == "parent":
        (tmp_path / "unused").mkdir()
        store = LatestCheckpoint(tmp_path / "unused" / "..")
    elif unsafe == "noncanonical_staging":
        (tmp_path / ".latest.other.tmp").write_bytes(b"unrelated")
    elif unsafe == "directory_staging":
        (tmp_path / f".latest.{'a' * 32}.tmp").mkdir()
    else:
        os.link(store.path, tmp_path / f".latest.{'a' * 32}.tmp")
    with pytest.raises(LatestCheckpointError):
        publish(store, payload, 2)
    assert (tmp_path / "latest.pt").read_bytes() == before


@pytest.mark.parametrize("bad", ["non_rlm", "step", "token_counts", "contract", "rng", "missing_optimizer"])
def test_malformed_bundle_is_rejected_on_read_and_before_replacing_good_save(tmp_path, payload, bad):
    store = LatestCheckpoint(tmp_path)
    publish(store, payload)
    before = store.path.read_bytes()
    model, optimizer, metadata = store.load(torch.device("cpu"))
    if bad == "non_rlm":
        metadata["model_type"] = "base"
    elif bad == "step":
        metadata["step"] = True
    elif bad == "token_counts":
        metadata["text_tokens"] = -1
    elif bad == "contract":
        metadata["run_contract"]["training_config"]["total_batch_size"] *= 2
    elif bad == "rng":
        metadata["rng_state"]["cpu"] = [256]
    else:
        optimizer = None
    with pytest.raises(LatestCheckpointError):
        store.save(model, optimizer, metadata)
    assert store.path.read_bytes() == before
    torch.save({"model": model, "optimizer": optimizer, "metadata": metadata}, store.path)
    with pytest.raises(LatestCheckpointError):
        store.metadata()

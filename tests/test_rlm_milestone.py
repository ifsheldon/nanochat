"""A mandatory pre-unfreeze checkpoint survives rolling and numbered saves."""

import copy
import os
import stat
from types import SimpleNamespace

import pytest
import torch

from nanochat.rlm_checkpointing import CheckpointWriter, MilestoneCheckpointError
from nanochat.rlm_latest import LatestCheckpoint, LatestCheckpointError


@pytest.fixture
def checkpoint(monkeypatch):
    monkeypatch.delenv("NANOCHAT_CHECKPOINT_RESERVE_GB", raising=False)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    model(torch.ones(1, 4)).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model_config = {"text_length": 4, "memory_tokens": 2}
    training_config = {"total_batch_size": 8, "device_batch_size": 1, "cycles": 2}
    metadata = {
        "model_type": "rlm", "step": 1,
        "model_config": model_config, "training_config": training_config,
        "text_tokens": 8, "supervised_tokens": 10, "internal_positions": 14,
        "tokenizer_sha256": "test-tokenizer", "dataloader_state_dict": {"row_group": 7},
        "rng_state": {"cpu": torch.get_rng_state().tolist(), "cuda": None},
        "memory_contract": {"gradient_horizon": 2}, "data_resume": "approximate_row_group",
        "run_contract": {
            "model_config": model_config, "training_config": training_config,
            "tokenizer_sha256": "test-tokenizer", "device_type": "cpu",
        },
    }
    return model.state_dict(), optimizer.state_dict(), metadata


def at_step(checkpoint, step):
    model, optimizer, metadata = checkpoint
    metadata = copy.deepcopy(metadata)
    for name in ("step", "text_tokens", "supervised_tokens", "internal_positions"):
        metadata[name] *= step
    return model, optimizer, metadata


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_retained_bundle_survives_latest_replacement_and_numbered_retention(tmp_path, checkpoint, mode):
    store = LatestCheckpoint(tmp_path)
    writer = CheckpointWriter(tmp_path, mode=mode, keep_last=1)
    preserved = None
    try:
        for step in range(1, 6):
            writer.submit(*at_step(checkpoint, step), numbered=True, milestone=step == 2)
            writer.wait(step)
            if step == 2:
                preserved = store.milestone_path.read_bytes()
                assert preserved == store.path.read_bytes()
                assert writer.measurements[-1]["milestone_published"]
    finally:
        writer.close(5)
    assert store.metadata()["step"] == 5
    assert store.verify_milestone(2, run_contract=checkpoint[2]["run_contract"])["step"] == 2
    assert store.milestone_path.read_bytes() == preserved
    assert store.milestone_path.stat().st_nlink == store.path.stat().st_nlink == 1
    assert sorted(path.name for path in tmp_path.glob("meta_*.json")) == ["meta_000005.json"]
    bundle = torch.load(store.milestone_path, weights_only=True, map_location="cpu")
    torch.testing.assert_close(bundle["model"]["weight"], checkpoint[0]["weight"])
    torch.testing.assert_close(bundle["optimizer"]["state"][0]["exp_avg"], checkpoint[1]["state"][0]["exp_avg"])


def test_copy_and_retry_never_reserialize_or_replace_either_bundle(tmp_path, checkpoint, monkeypatch):
    store = LatestCheckpoint(tmp_path)
    store.save(*checkpoint)
    latest_inode = store.path.stat().st_ino

    def unexpected_save(*args, **kwargs):
        raise AssertionError("Milestone must reuse the already serialized bundle")

    monkeypatch.setattr(torch, "save", unexpected_save)
    store.preserve_milestone(1)
    milestone_inode = store.milestone_path.stat().st_ino
    store.preserve_milestone(1)
    assert store.milestone_path.stat().st_ino == milestone_inode
    assert store.path.stat().st_ino == latest_inode
    assert store.path.read_bytes() == store.milestone_path.read_bytes()


@pytest.mark.parametrize("conflict", ["step", "metadata", "weights"])
def test_existing_milestone_conflicts_are_not_overwritten(tmp_path, checkpoint, conflict):
    store = LatestCheckpoint(tmp_path)
    store.save(*checkpoint)
    store.preserve_milestone(1)
    before = store.milestone_path.read_bytes()
    model, optimizer, metadata = copy.deepcopy(checkpoint)
    if conflict == "step":
        model, optimizer, metadata = at_step(checkpoint, 2)
    elif conflict == "metadata":
        metadata["dataloader_state_dict"]["row_group"] = 8
    else:
        model["weight"].add_(1)
    store.save(model, optimizer, metadata)
    with pytest.raises(LatestCheckpointError, match="conflicts"):
        store.preserve_milestone(metadata["step"])
    assert store.milestone_path.read_bytes() == before
    assert store.metadata() == metadata


def test_wrong_boundary_or_resume_contract_is_rejected(tmp_path, checkpoint):
    store = LatestCheckpoint(tmp_path)
    store.save(*checkpoint)
    with pytest.raises(LatestCheckpointError, match="Latest checkpoint must be update 2"):
        store.preserve_milestone(2)
    store.preserve_milestone(1)
    with pytest.raises(LatestCheckpointError, match="must preserve update 2"):
        store.verify_milestone(2)
    contract = copy.deepcopy(checkpoint[2]["run_contract"])
    contract["tokenizer_sha256"] = "other"
    with pytest.raises(LatestCheckpointError, match="different run contract"):
        store.verify_milestone(1, run_contract=contract)


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
@pytest.mark.parametrize("name", ["latest.pt", "pretrain-90pct.pt", ".pretrain-90pct." + "a" * 32 + ".tmp"])
def test_unsafe_files_cannot_redirect_copy_or_cleanup(tmp_path, checkpoint, name, kind):
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    store = LatestCheckpoint(directory)
    store.save(*checkpoint)
    outside = tmp_path / "outside"
    outside.write_bytes(b"keep untouched")
    path = directory / name
    if path.exists():
        path.unlink()
    if kind == "symlink":
        path.symlink_to(outside)
    else:
        os.link(outside, path)
    with pytest.raises(LatestCheckpointError, match="symlink|symbolic|hardlink"):
        store.preserve_milestone(1)
    assert outside.read_bytes() == b"keep untouched"
    assert path.exists()


@pytest.mark.parametrize("failure", ["copy", "file_fsync", "publish"])
def test_failed_copy_keeps_latest_recoverable_and_allows_retry(tmp_path, checkpoint, monkeypatch, failure):
    store = LatestCheckpoint(tmp_path)
    store.save(*checkpoint)
    before = store.path.read_bytes()

    def fail(*args, **kwargs):
        if failure == "copy":
            args[1].write(b"partial copy")
        raise OSError("injected milestone failure")

    with monkeypatch.context() as patch:
        target = {"copy": "shutil.copyfileobj", "file_fsync": "os.fsync", "publish": "nanochat.rlm_latest._publish_no_replace"}[failure]
        patch.setattr(target, fail)
        with pytest.raises(LatestCheckpointError, match="was not published.*injected"):
            store.preserve_milestone(1)
    assert store.path.read_bytes() == before
    assert store.load("cpu")[2]["step"] == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == ["latest.pt"]
    store.preserve_milestone(1)
    assert store.milestone_path.read_bytes() == before


def test_directory_fsync_failure_reports_undurable_milestone_and_retry_fsyncs(tmp_path, checkpoint, monkeypatch):
    store = LatestCheckpoint(tmp_path)
    store.save(*checkpoint)
    original = os.fsync

    def fail_directory(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected directory durability failure")
        original(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail_directory)
        with pytest.raises(LatestCheckpointError, match="was published but directory durability"):
            store.preserve_milestone(1)
    assert store.path.read_bytes() == store.milestone_path.read_bytes()
    synced = []

    def record(descriptor):
        synced.append(stat.S_ISDIR(os.fstat(descriptor).st_mode))
        original(descriptor)

    monkeypatch.setattr(os, "fsync", record)
    store.preserve_milestone(1)
    assert synced == [False, True]


def test_latest_replacement_during_copy_prevents_wrong_boundary_publication(tmp_path, checkpoint, monkeypatch):
    import shutil

    store = LatestCheckpoint(tmp_path)
    store.save(*checkpoint)
    original = shutil.copyfileobj

    def replace_latest(source, target, **kwargs):
        original(source, target, **kwargs)
        store.save(*at_step(checkpoint, 2))

    monkeypatch.setattr(shutil, "copyfileobj", replace_latest)
    with pytest.raises(LatestCheckpointError, match="Latest checkpoint changed"):
        store.preserve_milestone(1)
    assert store.metadata()["step"] == 2
    assert not store.milestone_path.exists()
    assert not list(tmp_path.glob(".pretrain-90pct.*.tmp"))


def test_publication_never_overwrites_a_destination_created_during_copy(tmp_path, checkpoint, monkeypatch):
    store = LatestCheckpoint(tmp_path)
    store.save(*checkpoint)
    from nanochat.rlm_latest import _publish_no_replace

    def race(directory_fd, source, destination):
        store.milestone_path.write_bytes(b"racing foreign file")
        _publish_no_replace(directory_fd, source, destination)

    monkeypatch.setattr("nanochat.rlm_latest._publish_no_replace", race)
    with pytest.raises(LatestCheckpointError, match="File exists"):
        store.preserve_milestone(1)
    assert store.milestone_path.read_bytes() == b"racing foreign file"
    assert store.metadata()["step"] == 1
    assert not list(tmp_path.glob(".pretrain-90pct.*.tmp"))


def test_copy_reserves_additional_disk_space_and_cleans_interrupted_staging(tmp_path, checkpoint, monkeypatch):
    store = LatestCheckpoint(tmp_path)
    store.save(*checkpoint)
    staged = tmp_path / (".pretrain-90pct." + "a" * 32 + ".tmp")
    staged.write_bytes(b"interrupted copy")
    monkeypatch.setenv("NANOCHAT_CHECKPOINT_RESERVE_GB", "1")
    with monkeypatch.context() as patch:
        patch.setattr(os, "fstatvfs", lambda fd: SimpleNamespace(f_bavail=1_000_000_000, f_frsize=1))
        with pytest.raises(LatestCheckpointError, match="Insufficient disk space"):
            store.preserve_milestone(1)
    assert not staged.exists()
    assert store.metadata()["step"] == 1
    store.preserve_milestone(1)


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("phase", ["snapshot", "latest", "milestone"])
def test_required_milestone_failure_stops_writer_and_retry_recovers(tmp_path, checkpoint, monkeypatch, mode, phase):
    store = LatestCheckpoint(tmp_path)
    store.save(*checkpoint)
    writer = CheckpointWriter(tmp_path, mode=mode, keep_last=1, latest_step=1)
    model, optimizer, metadata = at_step(checkpoint, 2)
    try:
        with monkeypatch.context() as patch:
            if phase == "snapshot":
                metadata["bad"] = object()
            else:
                def fail(*args, **kwargs):
                    raise OSError("injected required checkpoint failure")
                patch.setattr(LatestCheckpoint, "save" if phase == "latest" else "preserve_milestone", fail)
            with pytest.raises(MilestoneCheckpointError, match="training stopped before the next update"):
                writer.submit(model, optimizer, metadata, numbered=False, milestone=True)
                writer.wait(2)
        assert writer.measurements[-1]["errors"][0]["phase"] == phase
        assert not writer.measurements[-1]["milestone_published"]
        assert store.metadata()["step"] == (2 if phase == "milestone" else 1)
        writer.submit(*at_step(checkpoint, 2), numbered=False, milestone=True)
        writer.wait(2)
        assert writer.measurements[-1]["milestone_published"]
        assert store.verify_milestone(2)["step"] == 2
    finally:
        writer.close(2)


def test_numbered_failure_after_durable_milestone_remains_warning_only(tmp_path, checkpoint, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("numbered checkpoint failure")

    monkeypatch.setattr("nanochat.rlm_checkpointing.save_numbered_checkpoint", fail)
    writer = CheckpointWriter(tmp_path, mode="sync", keep_last=1)
    try:
        result = writer.submit(*checkpoint, numbered=True, milestone=True)
        assert result["status"] == "warning"
        assert result["milestone_published"]
        assert LatestCheckpoint(tmp_path).verify_milestone(1)["step"] == 1
    finally:
        writer.close(1)

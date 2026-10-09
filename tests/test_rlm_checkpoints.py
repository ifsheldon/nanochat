"""Retention only retires validated, complete checkpoints from its current run."""

import json
import os
import stat
import pytest
import torch

from nanochat.checkpoint_manager import checkpoint_is_complete, save_checkpoint
from nanochat.rlm_checkpoints import (
    CheckpointRetentionError, NumberedCheckpointError, prune_checkpoints,
    recover_numbered_checkpoint, save_numbered_checkpoint,
)


def publish(directory, step):
    save_checkpoint(
        directory, step, {"weight": torch.tensor([step], dtype=torch.float32)},
        {"state": {0: {"step": torch.tensor(step), "momentum": torch.ones(2)}}},
        {"step": step, "model_type": "rlm"},
    )


def snapshot(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file() and not path.is_symlink()}


def test_eleven_real_checkpoints_keep_latest_ten_and_report_exact_deletions(tmp_path, monkeypatch):
    monkeypatch.delenv("NANOCHAT_CHECKPOINT_RESERVE_GB", raising=False)
    directory = tmp_path / "checkpoints"
    for step in range(1, 12):
        publish(directory, step)
    before = snapshot(directory)
    result = prune_checkpoints(directory, keep_last=10, published_step=11)
    assert result.retained_steps == tuple(range(2, 12))
    assert result.deleted_steps == (1,)
    assert [path.name for path in result.deleted_paths] == [
        "meta_000001.json", "model_000001.pt", "optim_000001_rank0.pt",
    ]
    assert result.deleted_bytes == sum(len(before[path.name]) for path in result.deleted_paths)
    for step in range(2, 12):
        assert checkpoint_is_complete(directory, step, load_optimizer=True)
    assert snapshot(directory) == {name: data for name, data in before.items() if "000001" not in name}
    assert result.to_dict()["deleted_paths"] == [str(path) for path in result.deleted_paths]
    assert prune_checkpoints(directory, 10, 11).deleted_steps == ()


@pytest.mark.parametrize("failure", ["missing_metadata", "missing_optimizer", "empty_model", "wrong_metadata_step", "invalid_json"])
def test_unpublished_or_broken_latest_never_deletes_older_checkpoint(tmp_path, failure):
    directory = tmp_path / "checkpoints"
    publish(directory, 1)
    publish(directory, 2)
    if failure == "missing_metadata":
        (directory / "meta_000002.json").unlink()
    elif failure == "missing_optimizer":
        (directory / "optim_000002_rank0.pt").unlink()
    elif failure == "empty_model":
        (directory / "model_000002.pt").write_bytes(b"")
    elif failure == "wrong_metadata_step":
        (directory / "meta_000002.json").write_text(json.dumps({"step": 1}))
    else:
        (directory / "meta_000002.json").write_text("{")
    before = snapshot(directory)
    with pytest.raises(CheckpointRetentionError):
        prune_checkpoints(directory, 1, 2)
    assert snapshot(directory) == before
    assert checkpoint_is_complete(directory, 1, load_optimizer=True)


def test_all_candidates_are_validated_before_any_checkpoint_is_retired(tmp_path):
    directory = tmp_path / "checkpoints"
    for step in (1, 2, 3):
        publish(directory, step)
    (directory / "optim_000002_rank0.pt").unlink()
    before = snapshot(directory)
    with pytest.raises(CheckpointRetentionError, match="optim_000002"):
        prune_checkpoints(directory, 1, 3)
    assert snapshot(directory) == before


@pytest.mark.parametrize("location", ["directory", "parent", "model", "metadata", "optimizer"])
def test_symlinks_cannot_redirect_retention_to_other_files(tmp_path, location):
    owned = tmp_path / "owned"
    directory = owned / "checkpoints"
    publish(directory, 1)
    publish(directory, 2)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"preserve this external file")
    if location in ("directory", "parent"):
        redirect = tmp_path / "redirect"
        redirect.symlink_to(directory if location == "directory" else owned, target_is_directory=True)
        supplied = redirect if location == "directory" else redirect / "checkpoints"
    else:
        name = {"model": "model_000001.pt", "metadata": "meta_000001.json", "optimizer": "optim_000001_rank0.pt"}[location]
        (directory / name).unlink()
        (directory / name).symlink_to(outside)
        supplied = directory
    before = snapshot(directory)
    with pytest.raises(CheckpointRetentionError, match="symlink"):
        prune_checkpoints(supplied, 1, 2)
    assert snapshot(directory) == before
    assert outside.read_bytes() == b"preserve this external file"


def test_retention_sorts_numeric_steps_and_leaves_unrelated_artifacts_untouched(tmp_path):
    directory = tmp_path / "checkpoints"
    for step in (9, 1000000, 999999):
        publish(directory, step)
    unrelated = ("model_000009.pt.backup", "meta_000009.json.notes", "optim_000009_rank1.pt", "training.log")
    for name in unrelated:
        (directory / name).write_text(name)
    nested = directory / "source"
    nested.mkdir()
    (nested / "model_000009.pt").write_text("nested artifact")
    sibling = tmp_path / "other_run" / "checkpoints"
    publish(sibling, 1)
    sibling_before = snapshot(sibling)
    result = prune_checkpoints(directory, 2, 1000000)
    assert result.retained_steps == (999999, 1000000)
    assert result.deleted_steps == (9,)
    for name in unrelated:
        assert (directory / name).read_text() == name
    assert (nested / "model_000009.pt").read_text() == "nested artifact"
    assert snapshot(sibling) == sibling_before


def test_wrong_published_step_or_noncanonical_names_do_not_prune(tmp_path):
    directory = tmp_path / "checkpoints"
    publish(directory, 1)
    publish(directory, 2)
    before = snapshot(directory)
    with pytest.raises(CheckpointRetentionError, match="not the newest"):
        prune_checkpoints(directory, 1, 1)
    assert snapshot(directory) == before
    (directory / "model_3.pt").write_bytes(b"not a canonical checkpoint name")
    before = snapshot(directory)
    with pytest.raises(CheckpointRetentionError, match="Noncanonical"):
        prune_checkpoints(directory, 1, 2)
    assert snapshot(directory) == before


def test_deletion_failure_invalidates_old_set_first_and_preserves_newest(tmp_path, monkeypatch):
    directory = tmp_path / "checkpoints"
    publish(directory, 1)
    publish(directory, 2)
    original_unlink = os.unlink
    removed = []

    def fail_on_old_model(path, *, dir_fd=None):
        if path == "model_000001.pt":
            raise PermissionError("simulated removal failure")
        removed.append(path)
        return original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", fail_on_old_model)
    with pytest.raises(CheckpointRetentionError, match="simulated removal failure.*files already removed:.*meta_000001"):
        prune_checkpoints(directory, 1, 2)
    assert removed == ["meta_000001.json"]
    assert not checkpoint_is_complete(directory, 1, load_optimizer=True)
    assert checkpoint_is_complete(directory, 2, load_optimizer=True)
    assert (directory / "model_000001.pt").is_file()
    assert (directory / "optim_000001_rank0.pt").is_file()


@pytest.mark.parametrize("keep_last", [0, -1, True, 1.5])
def test_invalid_retention_count_never_touches_files(tmp_path, keep_last):
    directory = tmp_path / "checkpoints"
    publish(directory, 1)
    before = snapshot(directory)
    with pytest.raises(CheckpointRetentionError, match="positive integer"):
        prune_checkpoints(directory, keep_last, 1)
    assert snapshot(directory) == before


def test_parent_traversal_is_rejected_even_when_destination_exists(tmp_path):
    directory = tmp_path / "checkpoints"
    publish(directory, 1)
    (tmp_path / "unused").mkdir()
    supplied = tmp_path / "unused" / ".." / "checkpoints"
    with pytest.raises(CheckpointRetentionError, match="parent traversal"):
        prune_checkpoints(supplied, 1, 1)
    assert checkpoint_is_complete(directory, 1, load_optimizer=True)


def interrupt_retirement(directory, monkeypatch, *, after_name="meta_000001.json"):
    original_unlink = os.unlink

    def stop_after_unlink(path, *, dir_fd=None):
        result = original_unlink(path, dir_fd=dir_fd)
        if path == after_name:
            raise KeyboardInterrupt("simulated process stop during retirement")
        return result

    with monkeypatch.context() as interrupted:
        interrupted.setattr(os, "unlink", stop_after_unlink)
        with pytest.raises(KeyboardInterrupt, match="process stop during retirement"):
            prune_checkpoints(directory, 1, 3)


@pytest.mark.parametrize("after_name", [
    "meta_000001.json", "model_000001.pt", "optim_000001_rank0.pt",
    "meta_000002.json", "model_000002.pt", "optim_000002_rank0.pt", ".retention.json",
])
def test_interrupted_retirement_recovers_after_each_unlink(tmp_path, monkeypatch, after_name):
    directory = tmp_path / "checkpoints"
    for step in (1, 2, 3):
        publish(directory, step)
    kept = {name: value for name, value in snapshot(directory).items() if "000003" in name}
    interrupt_retirement(directory, monkeypatch, after_name=after_name)
    remaining = snapshot(directory)
    removed = {name: value for name, value in remaining.items() if "000001" in name or "000002" in name}
    result = prune_checkpoints(directory, 1, 3)
    assert result.retained_steps == (3,)
    assert result.deleted_steps == tuple(sorted({int(name.split("_")[1].split(".")[0]) for name in removed}))
    assert {path.name for path in result.deleted_paths} == set(removed)
    assert result.deleted_bytes == sum(len(value) for value in removed.values())
    assert snapshot(directory) == kept
    assert checkpoint_is_complete(directory, 3, load_optimizer=True)
    assert prune_checkpoints(directory, 1, 3).deleted_paths == ()


def test_journal_durability_surrounds_checkpoint_deletion(tmp_path, monkeypatch):
    directory = tmp_path / "checkpoints"
    publish(directory, 1)
    publish(directory, 2)
    events = []
    original_sync, original_replace, original_unlink = os.fsync, os.replace, os.unlink

    def sync(descriptor):
        events.append("sync-directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "sync-journal")
        return original_sync(descriptor)

    def replace(source, target, **kwargs):
        events.append(f"publish:{target}")
        return original_replace(source, target, **kwargs)

    def unlink(path, **kwargs):
        events.append(f"unlink:{path}")
        return original_unlink(path, **kwargs)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(os, "unlink", unlink)
    prune_checkpoints(directory, 1, 2)
    assert events == [
        "sync-journal", "publish:.retention.json", "sync-directory",
        "unlink:meta_000001.json", "unlink:model_000001.pt", "unlink:optim_000001_rank0.pt",
        "sync-directory", "unlink:.retention.json", "sync-directory",
    ]


def test_interrupted_journal_staging_is_replaced_before_deleting_any_checkpoint(tmp_path, monkeypatch):
    directory = tmp_path / "checkpoints"
    publish(directory, 1)
    publish(directory, 2)
    before = snapshot(directory)

    def stop(descriptor):
        raise KeyboardInterrupt("stopped while syncing retirement journal")

    with monkeypatch.context() as interrupted:
        interrupted.setattr(os, "fsync", stop)
        with pytest.raises(KeyboardInterrupt, match="syncing retirement journal"):
            prune_checkpoints(directory, 1, 2)
    for name, value in before.items():
        assert (directory / name).read_bytes() == value
    assert (directory / ".retention.json.tmp").is_file()
    assert not (directory / ".retention.json").exists()
    result = prune_checkpoints(directory, 1, 2)
    assert result.deleted_steps == (1,)
    assert snapshot(directory) == {name: value for name, value in before.items() if "000002" in name}


@pytest.mark.parametrize("tamper", [
    "bad_json", "duplicate_key", "version", "published_step", "keep_last", "retained_step",
    "overlap", "traversal", "unlisted_file", "missing_file", "size", "identity",
])
def test_tampered_journal_never_authorizes_more_deletion(tmp_path, monkeypatch, tamper):
    directory = tmp_path / "checkpoints"
    for step in (1, 2, 3):
        publish(directory, step)
    interrupt_retirement(directory, monkeypatch)
    journal = directory / ".retention.json"
    record = json.loads(journal.read_text())
    if tamper == "bad_json":
        journal.write_text("{")
    elif tamper == "duplicate_key":
        journal.write_text('{"version":1,' + json.dumps(record)[1:])
    else:
        if tamper == "version":
            record["version"] = True
        elif tamper == "published_step":
            record["published_step"] = 2
        elif tamper == "keep_last":
            record["keep_last"] = 2
        elif tamper == "retained_step":
            record["retained_steps"] = [2]
        elif tamper == "overlap":
            record["retired"][-1]["step"] = 3
        elif tamper == "traversal":
            record["retired"][0]["files"][1]["name"] = "../model_000001.pt"
        elif tamper == "unlisted_file":
            record["retired"][0]["files"][1]["name"] = "model_000003.pt"
        elif tamper == "missing_file":
            record["retired"][0]["files"].pop()
        elif tamper == "size":
            record["retired"][0]["files"][1]["size"] += 1
        else:
            record["retired"][0]["files"][1]["identity"][2] += 1
        journal.write_text(json.dumps(record))
    before = snapshot(directory)
    with pytest.raises(CheckpointRetentionError):
        prune_checkpoints(directory, 1, 3)
    assert snapshot(directory) == before
    assert checkpoint_is_complete(directory, 3, load_optimizer=True)


@pytest.mark.parametrize("corruption", ["missing_retained", "corrupt_retained", "changed_retired", "unlisted_partial"])
def test_recovery_validates_every_retained_and_remaining_retired_file_before_deletion(tmp_path, monkeypatch, corruption):
    directory = tmp_path / "checkpoints"
    for step in (1, 2, 3):
        publish(directory, step)
    interrupt_retirement(directory, monkeypatch)
    if corruption == "missing_retained":
        (directory / "optim_000003_rank0.pt").unlink()
    elif corruption == "corrupt_retained":
        (directory / "meta_000003.json").write_text('{"step": 2}')
    elif corruption == "changed_retired":
        (directory / "optim_000002_rank0.pt").write_bytes(b"changed since the reviewed retirement")
    else:
        (directory / "model_000000.pt").write_bytes(b"unpublished and unlisted")
    before = snapshot(directory)
    with pytest.raises(CheckpointRetentionError):
        prune_checkpoints(directory, 1, 3)
    assert snapshot(directory) == before
    assert (directory / "model_000001.pt").is_file()


@pytest.mark.parametrize("location", ["journal", "staging", "retired", "retained"])
@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_retirement_rejects_redirected_journal_staging_and_checkpoint_files(tmp_path, monkeypatch, location, kind):
    directory = tmp_path / "checkpoints"
    for step in (1, 2, 3):
        publish(directory, step)
    if location != "staging":
        interrupt_retirement(directory, monkeypatch)
    name = {
        "journal": ".retention.json", "staging": ".retention.json.tmp",
        "retired": "model_000001.pt", "retained": "model_000003.pt",
    }[location]
    target = directory / name
    outside = tmp_path / "outside"
    outside.write_bytes(target.read_bytes() if target.exists() else b"unrelated staging content")
    target.unlink(missing_ok=True)
    if kind == "symlink":
        target.symlink_to(outside)
    else:
        os.link(outside, target)
    before = snapshot(directory)
    external_before = outside.read_bytes()
    with pytest.raises(CheckpointRetentionError, match="symlink|hardlink"):
        prune_checkpoints(directory, 1, 3)
    assert snapshot(directory) == before
    assert outside.read_bytes() == external_before


def test_failed_retirement_is_finished_after_newer_checkpoint_publication(tmp_path, monkeypatch):
    directory = tmp_path / "checkpoints"
    for step in (1, 2, 3):
        publish(directory, step)
    interrupt_retirement(directory, monkeypatch)
    publish(directory, 4)
    result = prune_checkpoints(directory, 1, 4)
    assert result.retained_steps == (4,)
    assert result.deleted_steps == (1, 2, 3)
    assert checkpoint_is_complete(directory, 4, load_optimizer=True)
    assert set(snapshot(directory)) == {"meta_000004.json", "model_000004.pt", "optim_000004_rank0.pt"}


def test_recovery_refuses_incomplete_newer_publication_before_finishing_old_retirement(tmp_path, monkeypatch):
    directory = tmp_path / "checkpoints"
    for step in (1, 2, 3):
        publish(directory, step)
    interrupt_retirement(directory, monkeypatch)
    (directory / "model_000004.pt").write_bytes(b"unpublished")
    before = snapshot(directory)
    with pytest.raises(CheckpointRetentionError, match="meta_000004"):
        prune_checkpoints(directory, 1, 4)
    assert snapshot(directory) == before


def durable_publish(directory, step):
    directory.mkdir(exist_ok=True)
    return save_numbered_checkpoint(
        directory, step, {"weight": torch.tensor([step], dtype=torch.float32)},
        {"state": {0: {"step": torch.tensor(step), "momentum": torch.ones(2)}}},
        {"step": step, "model_type": "rlm"},
    )


@pytest.mark.parametrize("after_target", [
    ".numbered-publication.json", "model_000002.pt", "optim_000002_rank0.pt", "meta_000002.json",
])
def test_next_numbered_save_recovers_exact_interrupted_publication(tmp_path, monkeypatch, after_target):
    directory = tmp_path / "checkpoints"
    durable_publish(directory, 1)
    older = snapshot(directory)
    original_replace = os.replace

    def fail_after_replace(source, target, **kwargs):
        result = original_replace(source, target, **kwargs)
        if target == after_target:
            raise OSError("simulated publication failure")
        return result

    with monkeypatch.context() as interrupted:
        interrupted.setattr(os, "replace", fail_after_replace)
        with pytest.raises(NumberedCheckpointError, match="simulated publication failure"):
            durable_publish(directory, 2)
    for name, content in older.items():
        assert (directory / name).read_bytes() == content
    durable_publish(directory, 3)
    for step in (1, 2, 3):
        assert checkpoint_is_complete(directory, step, load_optimizer=True)
        assert torch.load(directory / f"model_{step:06d}.pt", weights_only=True)["weight"].item() == step
        assert torch.load(directory / f"optim_{step:06d}_rank0.pt", weights_only=True)["state"][0]["step"].item() == step
    assert not any(path.name.startswith(".") for path in directory.iterdir())
    result = prune_checkpoints(directory, 1, 3)
    assert result.deleted_steps == (1, 2)


@pytest.mark.parametrize("fail_save", [1, 2])
def test_failed_tensor_staging_leaves_no_numbered_partial_and_next_save_proceeds(tmp_path, monkeypatch, fail_save):
    directory = tmp_path / "checkpoints"
    durable_publish(directory, 1)
    older = snapshot(directory)
    original_save = torch.save
    saves = 0

    def fail(value, stream):
        nonlocal saves
        saves += 1
        if saves == fail_save:
            stream.write(b"partial tensor archive")
            raise OSError("simulated tensor staging failure")
        return original_save(value, stream)

    with monkeypatch.context() as interrupted:
        interrupted.setattr(torch, "save", fail)
        with pytest.raises(NumberedCheckpointError, match="simulated tensor staging failure"):
            durable_publish(directory, 2)
    assert recover_numbered_checkpoint(directory) is None
    assert {name: data for name, data in snapshot(directory).items() if not name.startswith(".")} == older
    durable_publish(directory, 3)
    assert not any(path.name.startswith(".") for path in directory.iterdir())
    assert prune_checkpoints(directory, 1, 3).deleted_steps == (1,)


def test_numbered_publication_is_durable_before_old_checkpoint_deletion(tmp_path, monkeypatch):
    directory = tmp_path / "checkpoints"
    durable_publish(directory, 1)
    events = []
    original_sync, original_replace, original_unlink = os.fsync, os.replace, os.unlink

    def sync(descriptor):
        events.append("sync-directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "sync-file")
        return original_sync(descriptor)

    def replace(source, target, **kwargs):
        events.append(f"publish:{target}")
        return original_replace(source, target, **kwargs)

    def unlink(path, **kwargs):
        events.append(f"unlink:{path}")
        return original_unlink(path, **kwargs)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(os, "unlink", unlink)
    durable_publish(directory, 2)
    prune_checkpoints(directory, 1, 2)
    assert events[:4] == ["sync-file"] * 4
    metadata_publication = events.index("publish:meta_000002.json")
    assert events.index("publish:model_000002.pt") < metadata_publication
    assert events.index("publish:optim_000002_rank0.pt") < metadata_publication
    assert events[metadata_publication + 1:metadata_publication + 4] == [
        "sync-directory", "unlink:.numbered-publication.json", "sync-directory",
    ]
    assert events.index("unlink:meta_000001.json") > metadata_publication + 3


@pytest.mark.parametrize("location", ["journal", "staged", "target"])
@pytest.mark.parametrize("kind", ["symlink", "hardlink", "replacement"])
def test_numbered_recovery_rejects_changed_or_redirected_publication_files(tmp_path, monkeypatch, location, kind):
    directory = tmp_path / "checkpoints"
    durable_publish(directory, 1)
    original_replace = os.replace

    def fail_after_model(source, target, **kwargs):
        result = original_replace(source, target, **kwargs)
        if target == "model_000002.pt":
            raise OSError("stop after model publication")
        return result

    with monkeypatch.context() as interrupted:
        interrupted.setattr(os, "replace", fail_after_model)
        with pytest.raises(NumberedCheckpointError):
            durable_publish(directory, 2)
    journal = directory / ".numbered-publication.json"
    record = json.loads(journal.read_text())
    path = {
        "journal": journal,
        "staged": directory / f".numbered.{record['token']}.optim.tmp",
        "target": directory / "model_000002.pt",
    }[location]
    outside = tmp_path / "external"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    if kind == "symlink":
        path.symlink_to(outside)
    elif kind == "hardlink":
        os.link(outside, path)
    else:
        path.write_bytes(b"changed file")
    before = snapshot(directory)
    with pytest.raises(NumberedCheckpointError):
        recover_numbered_checkpoint(directory)
    assert snapshot(directory) == before
    assert checkpoint_is_complete(directory, 1, load_optimizer=True)


def test_numbered_save_refuses_overwriting_published_or_unjournaled_partial_set(tmp_path):
    directory = tmp_path / "checkpoints"
    durable_publish(directory, 1)
    before = snapshot(directory)
    with pytest.raises(NumberedCheckpointError, match="Refusing to replace"):
        durable_publish(directory, 1)
    assert snapshot(directory) == before
    (directory / "model_000002.pt").write_bytes(b"untracked partial")
    before = snapshot(directory)
    with pytest.raises(NumberedCheckpointError, match="Refusing to replace"):
        durable_publish(directory, 2)
    assert snapshot(directory) == before


@pytest.mark.parametrize("sync_number", range(1, 9))
def test_numbered_save_recovers_or_discards_staging_after_each_sync_failure(tmp_path, monkeypatch, sync_number):
    directory = tmp_path / "checkpoints"
    durable_publish(directory, 1)
    older = snapshot(directory)
    original_sync = os.fsync
    syncs = 0

    def fail_sync(descriptor):
        nonlocal syncs
        syncs += 1
        if syncs == sync_number:
            raise OSError("simulated sync failure")
        return original_sync(descriptor)

    with monkeypatch.context() as interrupted:
        interrupted.setattr(os, "fsync", fail_sync)
        with pytest.raises(NumberedCheckpointError, match="simulated sync failure"):
            durable_publish(directory, 2)
    for name, content in older.items():
        assert (directory / name).read_bytes() == content
    durable_publish(directory, 3)
    assert checkpoint_is_complete(directory, 1, load_optimizer=True)
    assert checkpoint_is_complete(directory, 3, load_optimizer=True)
    assert checkpoint_is_complete(directory, 2, load_optimizer=True) is (sync_number >= 5)
    assert not any(path.name.startswith(".") for path in directory.iterdir())
    result = prune_checkpoints(directory, 1, 3)
    assert result.deleted_steps == ((1, 2) if sync_number >= 5 else (1,))


@pytest.mark.parametrize("tamper", ["token", "path", "size", "identity", "duplicate_key", "step"])
def test_numbered_publication_journal_cannot_redirect_or_expand_recovery(tmp_path, monkeypatch, tamper):
    directory = tmp_path / "checkpoints"
    durable_publish(directory, 1)
    original_replace = os.replace

    def fail_after_journal(source, target, **kwargs):
        result = original_replace(source, target, **kwargs)
        if target == ".numbered-publication.json":
            raise OSError("stop after journal publication")
        return result

    with monkeypatch.context() as interrupted:
        interrupted.setattr(os, "replace", fail_after_journal)
        with pytest.raises(NumberedCheckpointError):
            durable_publish(directory, 2)
    journal = directory / ".numbered-publication.json"
    record = json.loads(journal.read_text())
    if tamper == "duplicate_key":
        journal.write_text('{"version":1,' + json.dumps(record)[1:])
    else:
        if tamper == "token":
            record["token"] = "../outside"
        elif tamper == "path":
            record["checkpoint"]["files"][0]["name"] = "../meta_000002.json"
        elif tamper == "size":
            record["checkpoint"]["files"][1]["size"] += 1
        elif tamper == "identity":
            record["checkpoint"]["files"][1]["identity"][2] += 1
        else:
            record["checkpoint"]["step"] = 1
        journal.write_text(json.dumps(record))
    before = snapshot(directory)
    with pytest.raises(NumberedCheckpointError):
        recover_numbered_checkpoint(directory)
    assert snapshot(directory) == before
    assert checkpoint_is_complete(directory, 1, load_optimizer=True)

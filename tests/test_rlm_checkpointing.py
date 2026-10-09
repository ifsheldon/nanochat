"""Checkpoint overlap preserves an update while warning-only failures can retry."""

import copy
from concurrent.futures import Future
import json
from threading import Event

import pytest
import torch

from nanochat.checkpoint_manager import load_checkpoint
from nanochat.rlm_checkpointing import CheckpointWriter
from nanochat.rlm_checkpoints import save_numbered_checkpoint
from nanochat.rlm_latest import LatestCheckpoint


def install_cpu_stager(monkeypatch, stager):
    """Exercise writer staging without allocating pinned memory or touching CUDA."""
    def prepare(writer, model_data):
        writer._stager = stager
        writer._staging_device = torch.device("cpu")

    monkeypatch.setattr(CheckpointWriter, "_prepare_stager", prepare)


@pytest.fixture(autouse=True)
def no_disk_reserve(monkeypatch):
    monkeypatch.delenv("NANOCHAT_CHECKPOINT_RESERVE_GB", raising=False)


def metadata(step, loader_state=None):
    model_config = {"text_length": 4, "memory_tokens": 2}
    training_config = {"total_batch_size": 8, "device_batch_size": 1, "cycles": 2}
    return {
        "model_type": "rlm", "step": step,
        "model_config": model_config, "training_config": training_config,
        "text_tokens": 8 * step, "supervised_tokens": 10 * step,
        "internal_positions": 14 * step,
        "tokenizer_sha256": "test-tokenizer",
        "dataloader_state_dict": {"row_group": step} if loader_state is None else loader_state,
        "rng_state": {"cpu": torch.get_rng_state().tolist(), "cuda": None},
        "memory_contract": {"gradient_horizon": 2},
        "data_resume": "approximate_row_group",
        "run_contract": {
            "model_config": model_config, "training_config": training_config,
            "tokenizer_sha256": "test-tokenizer", "device_type": "cpu",
        },
    }


def initialized_model():
    torch.manual_seed(941)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    model(torch.ones(1, 4)).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def test_async_freezes_schedule_loader_rng_and_containers_but_aliases_tensor_storage(tmp_path, monkeypatch):
    model, optimizer = initialized_model()
    model_state, optimizer_state = model.state_dict(), optimizer.state_dict()
    loader_state = {"row_group": 3, "nested": [9]}
    original_metadata = metadata(1, loader_state)
    expected_metadata = copy.deepcopy(original_metadata)
    expected_weights = model.weight.detach().clone()
    entered, release = Event(), Event()
    observed = {}
    original_save = LatestCheckpoint.save

    def delayed_save(store, saved_model, saved_optimizer, saved_metadata):
        entered.set()
        assert release.wait(5), "test did not release checkpoint writer"
        observed["model_pointer"] = saved_model["weight"].data_ptr()
        observed["moment_pointer"] = saved_optimizer["state"][0]["exp_avg"].data_ptr()
        return original_save(store, saved_model, saved_optimizer, saved_metadata)

    monkeypatch.setattr(LatestCheckpoint, "save", delayed_save)
    writer = CheckpointWriter(tmp_path, mode="async", keep_last=10)
    try:
        writer.submit(model_state, optimizer_state, original_metadata, numbered=False)
        assert entered.wait(5), "checkpoint worker did not start"
        optimizer.param_groups[0]["lr"] = 0.009
        optimizer_state["param_groups"][0]["betas"] = (0.4, 0.5)
        loader_state["row_group"] = 100
        loader_state["nested"].append(17)
        original_metadata["rng_state"]["cpu"][0] ^= 1
        model_state["weight"] = torch.zeros_like(model.weight)
        # This is the actual work permitted to overlap the saved tensor storage.
        model(torch.ones(2, 4)).square().mean().backward()
        torch.testing.assert_close(model.weight, expected_weights, rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="Wait for the preceding checkpoint"):
            writer.submit(model.state_dict(), optimizer.state_dict(), metadata(2), numbered=False)
        release.set()
        result = writer.wait(1)
        optimizer.step()
    finally:
        release.set()
        writer.close(1)
    saved_model, saved_optimizer, saved_metadata = LatestCheckpoint(tmp_path).load("cpu")
    torch.testing.assert_close(saved_model["weight"], expected_weights, rtol=0, atol=0)
    assert observed["model_pointer"] == model.weight.data_ptr()
    assert observed["moment_pointer"] == optimizer.state_dict()["state"][0]["exp_avg"].data_ptr()
    assert saved_optimizer["param_groups"][0]["lr"] == 0.02
    assert saved_optimizer["param_groups"][0]["betas"] == (0.9, 0.999)
    assert saved_metadata == expected_metadata
    assert result["latest_step"] == 1
    assert result["latest_lag_completed_updates"] == 0


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_failed_latest_warns_keeps_prior_update_and_next_save_recovers(tmp_path, monkeypatch, capsys, mode):
    model, optimizer = initialized_model()
    original = LatestCheckpoint.save

    def fail_second(store, model_data, optimizer_data, saved_metadata):
        if saved_metadata["step"] == 2:
            raise OSError("injected disk failure")
        return original(store, model_data, optimizer_data, saved_metadata)

    monkeypatch.setattr(LatestCheckpoint, "save", fail_second)
    writer = CheckpointWriter(tmp_path, mode=mode, keep_last=10)
    try:
        for step in (1, 2):
            writer.submit(model.state_dict(), optimizer.state_dict(), metadata(step), numbered=True)
            writer.wait(step)
        failed = writer.measurements[-1]
        assert failed["status"] == "warning"
        assert failed["errors"][0]["phase"] == "latest"
        assert failed["latest_step"] == 1
        assert failed["latest_lag_completed_updates"] == 1
        assert LatestCheckpoint(tmp_path).metadata()["step"] == 1
        assert not (tmp_path / "meta_000002.json").exists()
        writer.submit(model.state_dict(), optimizer.state_dict(), metadata(3), numbered=True)
    finally:
        writer.close(3)
    assert LatestCheckpoint(tmp_path).metadata()["step"] == 3
    assert writer.measurements[-1]["latest_lag_completed_updates"] == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    warnings = [event["checkpoint_warning"] for event in events if "checkpoint_warning" in event]
    assert len(warnings) == 1
    assert warnings[0]["training_continues"] is True
    assert warnings[0]["step"] == 2


@pytest.mark.parametrize("phase", ["numbered", "retention"])
def test_numbered_or_retention_failure_does_not_hide_durable_latest(tmp_path, monkeypatch, phase):
    model, optimizer = initialized_model()
    calls = []
    original_save = LatestCheckpoint.save

    def save_latest(store, *args):
        calls.append("latest")
        return original_save(store, *args)

    def save_numbered(*args):
        assert LatestCheckpoint(tmp_path).metadata()["step"] == 1
        calls.append("numbered")
        if phase == "numbered":
            raise OSError("numbered write failure")

    def retain(*args):
        calls.append("retention")
        raise OSError("retention failure")

    monkeypatch.setattr(LatestCheckpoint, "save", save_latest)
    monkeypatch.setattr("nanochat.rlm_checkpointing.save_numbered_checkpoint", save_numbered)
    monkeypatch.setattr("nanochat.rlm_checkpointing.prune_checkpoints", retain)
    writer = CheckpointWriter(tmp_path, mode="async", keep_last=10)
    writer.submit(model.state_dict(), optimizer.state_dict(), metadata(1), numbered=True)
    result = writer.close(1)
    assert result["status"] == "warning"
    assert result["latest_step"] == 1
    assert result["latest_published"] is True
    assert result["errors"][0]["phase"] == phase
    assert calls == (["latest", "numbered"] if phase == "numbered" else ["latest", "numbered", "retention"])


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_snapshot_failure_warns_and_next_update_can_publish(tmp_path, mode):
    model, optimizer = initialized_model()
    writer = CheckpointWriter(tmp_path, mode=mode, keep_last=10)
    invalid = metadata(1)
    invalid["unserializable"] = object()
    try:
        writer.submit(model.state_dict(), optimizer.state_dict(), invalid, numbered=False)
        writer.wait(1)
        assert writer.measurements[-1]["errors"][0]["phase"] == "snapshot"
        assert writer.status(1)["latest_step"] is None
        writer.submit(model.state_dict(), optimizer.state_dict(), metadata(2), numbered=False)
    finally:
        writer.close(2)
    assert LatestCheckpoint(tmp_path).metadata()["step"] == 2


def test_close_waits_for_final_save_on_exception_exit(tmp_path, monkeypatch):
    model, optimizer = initialized_model()
    entered, release = Event(), Event()
    original = LatestCheckpoint.save

    def delayed_save(store, *args):
        entered.set()
        assert release.wait(5)
        return original(store, *args)

    monkeypatch.setattr(LatestCheckpoint, "save", delayed_save)
    writer = CheckpointWriter(tmp_path, mode="async", keep_last=10)
    with pytest.raises(RuntimeError, match="training failed"):
        try:
            writer.submit(model.state_dict(), optimizer.state_dict(), metadata(1), numbered=False)
            assert entered.wait(5)
            release.set()
            raise RuntimeError("training failed")
        finally:
            writer.close(1)
    assert LatestCheckpoint(tmp_path).metadata()["step"] == 1
    assert writer.status(1)["pending_step"] is None
    assert writer.close(1) is None


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_interrupt_is_not_converted_to_checkpoint_warning(tmp_path, monkeypatch, mode):
    model, optimizer = initialized_model()

    def interrupted(*args):
        raise KeyboardInterrupt("interrupted save")

    monkeypatch.setattr(LatestCheckpoint, "save", interrupted)
    writer = CheckpointWriter(tmp_path, mode=mode, keep_last=10)
    with pytest.raises(KeyboardInterrupt, match="interrupted save"):
        try:
            writer.submit(model.state_dict(), optimizer.state_dict(), metadata(1), numbered=False)
        finally:
            writer.close(1)
    assert writer.measurements == []


def test_three_seeded_updates_match_between_sync_and_async(tmp_path):
    completed = {}
    for mode in ("sync", "async"):
        directory = tmp_path / mode
        directory.mkdir()
        model, optimizer = initialized_model()
        data = torch.randn(3, 2, 4, generator=torch.Generator().manual_seed(107))
        writer = CheckpointWriter(directory, mode=mode, keep_last=2)
        try:
            for step, inputs in enumerate(data, 1):
                optimizer.param_groups[0]["lr"] = 0.02 / step
                model(inputs).square().mean().backward()
                writer.wait(step - 1)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                saved_metadata = metadata(step)
                saved_metadata["numbered_checkpoint"] = True
                writer.submit(model.state_dict(), optimizer.state_dict(), saved_metadata, numbered=True)
        finally:
            writer.close(3)
        completed[mode] = LatestCheckpoint(directory).load("cpu")
        assert [path.name for path in sorted(directory.glob("meta_*.json"))] == ["meta_000002.json", "meta_000003.json"]
        assert len(writer.measurements) == 3
    sync_model, sync_optimizer, sync_metadata = completed["sync"]
    async_model, async_optimizer, async_metadata = completed["async"]
    for name, tensor in sync_model.items():
        torch.testing.assert_close(async_model[name], tensor, rtol=0, atol=0)
    for index, state in sync_optimizer["state"].items():
        for name, tensor in state.items():
            torch.testing.assert_close(async_optimizer["state"][index][name], tensor, rtol=0, atol=0)
    assert async_optimizer["param_groups"] == sync_optimizer["param_groups"]
    assert async_metadata == sync_metadata


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_cuda_checkpoint_stages_once_and_preserves_updates_aliases_and_layout(tmp_path, monkeypatch, mode):
    if not torch.cuda.is_available():
        pytest.skip("CUDA checkpoint staging requires a CUDA device")
    device = torch.device("cuda", torch.cuda.current_device())
    main_stream = torch.cuda.current_stream(device)
    producer_stream = torch.cuda.Stream(device=device)
    torch.manual_seed(941)
    model = torch.nn.Linear(4, 2, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    torch.cuda.synchronize(device)
    copies, publications = [], []
    original_copy = torch.UntypedStorage.copy_
    original_latest = LatestCheckpoint.save

    def record_copy(storage, source, non_blocking=False):
        if source.device.type == "cuda" and storage.device.type == "cpu":
            copies.append((torch.cuda.current_stream(device), non_blocking, storage.is_pinned()))
        return original_copy(storage, source, non_blocking=non_blocking)

    def inspect_publication(phase, saved_model, saved_optimizer, saved_metadata):
        weight = saved_model["weight"]
        publications.append((phase, saved_metadata["step"], weight.data_ptr()))
        assert weight.device.type == ("cpu" if mode == "async" else "cuda")
        if mode == "async":
            assert weight.is_pinned()
            assert all(tensor.device.type == "cpu"
                       for state in saved_optimizer["state"].values() for tensor in state.values())
        assert saved_model["weight_transposed"].stride() == (1, 4)
        assert saved_model["weight_slice"].stride() == (2,)
        assert saved_model["weight_slice"].storage_offset() == 1
        assert saved_model["weight_transposed"].untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
        assert saved_model["weight_slice"].untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
        assert saved_model._metadata == model.state_dict()._metadata

    def save_latest(store, saved_model, saved_optimizer, saved_metadata):
        inspect_publication("latest", saved_model, saved_optimizer, saved_metadata)
        return original_latest(store, saved_model, saved_optimizer, saved_metadata)

    def save_numbered(directory, step, saved_model, saved_optimizer, saved_metadata):
        inspect_publication("numbered", saved_model, saved_optimizer, saved_metadata)
        return save_numbered_checkpoint(directory, step, saved_model, saved_optimizer, saved_metadata)

    monkeypatch.setattr(torch.UntypedStorage, "copy_", record_copy)
    monkeypatch.setattr(LatestCheckpoint, "save", save_latest)
    monkeypatch.setattr("nanochat.rlm_checkpointing.save_numbered_checkpoint", save_numbered)
    writer = CheckpointWriter(tmp_path, mode=mode, keep_last=2)
    try:
        for step in (1, 2):
            with torch.cuda.stream(producer_stream):
                model(torch.full((2, 4), float(step), device=device)).square().mean().backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            # Match the trainer's publication boundary after real CUDA writes.
            torch.cuda.synchronize(device)
            model_state = model.state_dict()
            model_state["weight_transposed"] = model.weight.detach().t()
            model_state["weight_slice"] = model.weight.detach().reshape(-1)[1::2]
            expected_model = {name: value.cpu().clone() for name, value in model_state.items()}
            expected_optimizer = {
                index: {name: value.cpu().clone() for name, value in state.items()}
                for index, state in optimizer.state_dict()["state"].items()
            }
            saved_metadata = metadata(step)
            saved_metadata["run_contract"]["device_type"] = "cuda"
            saved_metadata["rng_state"]["cuda"] = torch.cuda.get_rng_state(device).tolist()
            writer.submit(model_state, optimizer.state_dict(), saved_metadata, numbered=True)
            writer.wait(step)
            assert writer.measurements[-1]["status"] == "saved"
            assert torch.cuda.current_stream(device) == main_stream
            for saved_model, saved_optimizer, actual_metadata in (
                LatestCheckpoint(tmp_path).load("cpu"),
                load_checkpoint(tmp_path, step, "cpu", load_optimizer=True),
            ):
                for name, expected in expected_model.items():
                    torch.testing.assert_close(saved_model[name], expected, rtol=0, atol=0)
                for index, state in expected_optimizer.items():
                    for name, expected in state.items():
                        torch.testing.assert_close(saved_optimizer["state"][index][name], expected, rtol=0, atol=0)
                assert actual_metadata == saved_metadata
                assert saved_model["weight_slice"].storage_offset() == 1
                assert saved_model["weight_slice"].stride() == (2,)
                assert saved_model["weight_transposed"].stride() == (1, 4)
                assert (saved_model["weight_transposed"].untyped_storage().data_ptr()
                        == saved_model["weight"].untyped_storage().data_ptr())
    finally:
        writer.close(2)
    assert [(phase, step) for phase, step, _ in publications] == [
        ("latest", 1), ("numbered", 1), ("latest", 2), ("numbered", 2),
    ]
    assert len({pointer for _, _, pointer in publications}) == 1
    assert copies
    assert all((stream != main_stream) is (mode == "async") for stream, _, _ in copies)
    assert all(non_blocking is (mode == "async") for _, non_blocking, _ in copies)
    if mode == "async":
        # Six distinct CUDA storages: two model parameters and two Adam moments each.
        # Both publication formats share the completed snapshot on each update.
        assert len(copies) == 12
        assert all(pinned for _, _, pinned in copies)


def test_async_staging_setup_failure_warns_without_publishing(tmp_path, monkeypatch):
    model, optimizer = initialized_model()

    def unavailable_stager(*args):
        raise RuntimeError("injected CUDA staging setup failure")

    monkeypatch.setattr(CheckpointWriter, "_prepare_stager", unavailable_stager)
    writer = CheckpointWriter(tmp_path, mode="async", keep_last=10)
    writer.submit(model.state_dict(), optimizer.state_dict(), metadata(1), numbered=True)
    result = writer.close(1)
    assert result["status"] == "warning"
    assert result["errors"][0]["phase"] == "staging"
    assert result["errors"][0]["message"] == "injected CUDA staging setup failure"
    assert result["latest_step"] is None
    assert not (tmp_path / "latest.pt").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned staging requires CUDA")
def test_reused_cuda_stager_includes_adam_state_created_after_unfreezing(tmp_path):
    model = torch.nn.Linear(4, 2, device="cuda")
    model.weight.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    writer = CheckpointWriter(tmp_path, mode="async", keep_last=2)
    stager = None
    try:
        for step in (1, 2):
            if step == 2:
                model.weight.requires_grad_(True)
            model(torch.ones(2, 4, device="cuda")).square().mean().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            saved_metadata = metadata(step)
            saved_metadata["run_contract"]["device_type"] = "cuda"
            saved_metadata["rng_state"]["cuda"] = torch.cuda.get_rng_state().tolist()
            writer.submit(model.state_dict(), optimizer.state_dict(), saved_metadata, numbered=True)
            result = writer.wait(step)
            assert result["status"] == "saved"
            if step == 1:
                assert model.weight not in optimizer.state
                stager = writer._stager
            else:
                assert writer._stager is stager
                assert optimizer.state[model.weight]["step"].item() == 1
            expected_optimizer = optimizer.state_dict()
            for saved_model, saved_optimizer, actual_metadata in (
                LatestCheckpoint(tmp_path).load("cpu"),
                load_checkpoint(tmp_path, step, "cpu", load_optimizer=True),
            ):
                assert actual_metadata == saved_metadata
                for name, expected in model.state_dict().items():
                    torch.testing.assert_close(saved_model[name], expected.cpu(), rtol=0, atol=0)
                assert saved_optimizer["param_groups"] == expected_optimizer["param_groups"]
                assert saved_optimizer["state"].keys() == expected_optimizer["state"].keys()
                for index, state in expected_optimizer["state"].items():
                    for name, expected in state.items():
                        torch.testing.assert_close(saved_optimizer["state"][index][name], expected.cpu(), rtol=0, atol=0)
    finally:
        writer.close(2)


def test_staging_future_completes_before_checkpoint_publication(tmp_path, monkeypatch):
    model, optimizer = initialized_model()
    staging_started, published = Event(), Event()
    staged = Future()
    saved = LatestCheckpoint.save
    close_calls = []

    class DelayedStager:
        def stage(self, bundle):
            staging_started.set()
            return staged

        def close(self):
            assert published.is_set(), "stager closed before publication finished"
            close_calls.append(True)

    def record_publication(store, *args):
        result = saved(store, *args)
        published.set()
        return result

    install_cpu_stager(monkeypatch, DelayedStager())
    monkeypatch.setattr(LatestCheckpoint, "save", record_publication)
    writer = CheckpointWriter(tmp_path, mode="async", keep_last=2)
    try:
        writer.submit(model.state_dict(), optimizer.state_dict(), metadata(1), numbered=False)
        assert staging_started.wait(5)
        assert not published.is_set()
        # Distinct values prove publication uses the completed stage result.
        model_state = copy.deepcopy(model.state_dict())
        model_state["weight"].fill_(42)
        staged.set_result({"model": model_state, "optimizer": optimizer.state_dict()})
        result = writer.close(1)
    finally:
        if not staged.done():
            staged.set_result({"model": model.state_dict(), "optimizer": optimizer.state_dict()})
        writer.close(1)
    actual, _, _ = LatestCheckpoint(tmp_path).load("cpu")
    torch.testing.assert_close(actual["weight"], model_state["weight"], rtol=0, atol=0)
    assert result["status"] == "saved"
    assert result["staging_duration_seconds"] > 0
    assert close_calls == [True]
    assert writer._stager is None


def test_reused_staging_buffers_cannot_be_overwritten_during_publication(tmp_path, monkeypatch):
    from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions

    model, optimizer = initialized_model()
    expected_first = model.weight.detach().clone()
    publication_started, release_publication = Event(), Event()
    calls, pointers = [], []
    saved = LatestCheckpoint.save
    # Use the same upstream cache and alias handling without requiring a GPU.
    cpu_stager = DefaultStager(StagingOptions(
        use_pinned_memory=False, use_shared_memory=False,
        use_async_staging=False, use_non_blocking_copy=False,
    ))

    class ReusingStager:
        def stage(self, bundle):
            calls.append("stage")
            result = cpu_stager.stage(bundle)
            pointers.append(result["model"]["weight"].data_ptr())
            future = Future()
            future.set_result(result)
            return future

        def close(self):
            calls.append("close")
            cpu_stager.close()

    def delayed_publication(store, saved_model, saved_optimizer, saved_metadata):
        if saved_metadata["step"] == 1:
            publication_started.set()
            assert release_publication.wait(5), "test did not release checkpoint publication"
            torch.testing.assert_close(saved_model["weight"], expected_first, rtol=0, atol=0)
        result = saved(store, saved_model, saved_optimizer, saved_metadata)
        calls.append("published")
        return result

    install_cpu_stager(monkeypatch, ReusingStager())
    monkeypatch.setattr(LatestCheckpoint, "save", delayed_publication)
    writer = CheckpointWriter(tmp_path, mode="async", keep_last=2)
    try:
        writer.submit(model.state_dict(), optimizer.state_dict(), metadata(1), numbered=False)
        assert publication_started.wait(5)
        with pytest.raises(RuntimeError, match="Wait for the preceding checkpoint"):
            writer.submit(model.state_dict(), optimizer.state_dict(), metadata(2), numbered=False)
        assert calls == ["stage"]
        release_publication.set()
        writer.wait(1)
        with torch.no_grad():
            model.weight.add_(7)
        writer.submit(model.state_dict(), optimizer.state_dict(), metadata(2), numbered=False)
    finally:
        release_publication.set()
        writer.close(2)
    actual, _, actual_metadata = LatestCheckpoint(tmp_path).load("cpu")
    torch.testing.assert_close(actual["weight"], expected_first + 7, rtol=0, atol=0)
    assert actual_metadata["step"] == 2
    assert pointers[0] == pointers[1]
    assert calls == ["stage", "published", "stage", "published", "close"]
    assert writer.close(2) is None
    assert calls[-1] == "close" and calls.count("close") == 1


def test_failed_staging_drains_queued_copies_and_next_update_recovers(tmp_path, monkeypatch):
    model, optimizer = initialized_model()
    calls = []

    class FailingOnceStager:
        def stage(self, bundle):
            calls.append("stage")
            future = Future()
            if calls.count("stage") == 1:
                future.set_exception(RuntimeError("allocation failed after queued copies"))
            else:
                future.set_result(copy.deepcopy(bundle))
            return future

        def close(self):
            calls.append("close")

    def drain(device):
        assert device == torch.device("cpu")
        calls.append("drain")

    install_cpu_stager(monkeypatch, FailingOnceStager())
    monkeypatch.setattr(torch.cuda, "synchronize", drain)
    writer = CheckpointWriter(tmp_path, mode="async", keep_last=2)
    try:
        writer.submit(model.state_dict(), optimizer.state_dict(), metadata(1), numbered=True)
        failed = writer.wait(1)
        assert calls == ["stage", "drain"]
        assert failed["status"] == "warning"
        assert failed["errors"][0]["phase"] == "staging"
        assert failed["latest_published"] is False
        assert not (tmp_path / "latest.pt").exists()
        assert not (tmp_path / "meta_000001.json").exists()
        writer.submit(model.state_dict(), optimizer.state_dict(), metadata(2), numbered=True)
    finally:
        writer.close(2)
    assert LatestCheckpoint(tmp_path).metadata()["step"] == 2
    assert calls == ["stage", "drain", "stage", "close"]


def test_failed_staging_drain_is_fatal_and_still_closes_resources(tmp_path, monkeypatch):
    from nanochat.rlm_checkpointing import CheckpointStagingSynchronizationError

    model, optimizer = initialized_model()
    calls = []

    class FailingStager:
        def stage(self, bundle):
            future = Future()
            future.set_exception(RuntimeError("copy failed"))
            return future

        def close(self):
            calls.append("close")

    def failed_drain(device):
        calls.append("drain")
        raise RuntimeError("device synchronization failed")

    install_cpu_stager(monkeypatch, FailingStager())
    monkeypatch.setattr(torch.cuda, "synchronize", failed_drain)
    writer = CheckpointWriter(tmp_path, mode="async", keep_last=2)
    writer.submit(model.state_dict(), optimizer.state_dict(), metadata(1), numbered=False)
    with pytest.raises(CheckpointStagingSynchronizationError):
        writer.close(1)
    assert calls == ["drain", "close"]
    assert writer.measurements == []
    assert writer._stager is None
    assert not (tmp_path / "latest.pt").exists()

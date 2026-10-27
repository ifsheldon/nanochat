"""Portable publication and coordinated failures across real CPU workers."""

import copy
from concurrent.futures import Future
from datetime import timedelta
import importlib
from pathlib import Path
from threading import Event
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nanochat.optim import MuonAdamW
from nanochat.rlm_checkpointing import CheckpointStagingSynchronizationError, MilestoneCheckpointError
from nanochat.rlm_latest import LatestCheckpoint


def _metadata(step, rank=0):
    model = {"text_length": 4, "memory_tokens": 2}
    training = {"total_batch_size": 32, "device_batch_size": 1, "cycles": 2}
    return {
        "model_type": "rlm", "step": step,
        "model_config": model, "training_config": training,
        "text_tokens": 32 * step, "supervised_tokens": 40 * step,
        "internal_positions": 56 * step, "tokenizer_sha256": "test",
        "dataloader_state_dict": {"epoch": 1, "pq_idx": 0, "rg_idx": rank + step},
        "rng_state": {"cpu": [rank + 1, step], "cuda": None},
        "memory_contract": {"gradient_horizon": 2}, "data_resume": "approximate_row_group",
        "run_contract": {"model_config": model, "training_config": training,
                         "tokenizer_sha256": "test", "device_type": "cpu"},
    }


def _optimizer():
    parameter = torch.nn.Parameter(torch.ones(16, 128))
    optimizer = MuonAdamW([{
        "params": [parameter], "kind": "adamw", "lr": 0.02,
        "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.1,
    }])
    optimizer.initialize_state()
    rank = dist.get_rank() if dist.is_initialized() else 0
    optimizer.state[parameter]["step"] = 7
    optimizer.state[parameter]["exp_avg"].fill_(rank + 1)
    optimizer.state[parameter]["exp_avg_sq"].fill_(rank + 11)
    return parameter, optimizer


def test_cpu_mapped_snapshot_keeps_verified_inode_after_atomic_replacement(tmp_path):
    _, optimizer = _optimizer()
    store = LatestCheckpoint(tmp_path)
    store.save({"weight": torch.ones(2)}, optimizer.state_dict(), _metadata(1))
    assert hasattr(store, "load_cpu_mapped"), "portable resume requires mapped CPU loading"
    first = store.load_cpu_mapped()
    store.save({"weight": torch.full((2,), 9.0)}, optimizer.state_dict(), _metadata(2))
    torch.testing.assert_close(first[0]["weight"], torch.ones(2))
    assert first[2]["step"] == 1
    assert store.load_cpu_mapped()[2]["step"] == 2


def test_async_snapshot_freezes_metadata_and_cpu_tensors(tmp_path):
    module = importlib.import_module("nanochat.rlm_distributed_checkpoint")
    parameter, optimizer = _optimizer()
    writer = module.DistributedCheckpointWriter(
        tmp_path, mode="async", keep_last=1, optimizer=optimizer,
        checkpoint_group=None, control_group=None,
    )
    entered, release = Event(), Event()
    original = LatestCheckpoint.save
    metadata = _metadata(1)
    expected = copy.deepcopy(metadata)

    def delayed_save(store, *args):
        entered.set()
        assert release.wait(10)
        return original(store, *args)

    with patch.object(LatestCheckpoint, "save", delayed_save):
        try:
            writer.submit({"weight": parameter}, optimizer.state_dict(), metadata, numbered=False)
            assert entered.wait(10)
            with torch.no_grad():
                parameter.fill_(90)
                optimizer.state[parameter]["exp_avg"].fill_(80)
            metadata["rng_state"]["cpu"][0] = 99
            optimizer.param_groups[0]["lr"] = 0.8
            release.set()
            result = writer.wait(1)
            assert result["latest_step"] == 1
        finally:
            release.set()
            writer.close(1)
    model, state, saved = LatestCheckpoint(tmp_path).load_cpu_mapped()
    torch.testing.assert_close(model["weight"], torch.ones_like(parameter))
    torch.testing.assert_close(state["state"][0]["exp_avg"], torch.ones_like(parameter))
    assert state["param_groups"][0]["lr"] == 0.02
    assert saved["rng_state"] == expected["rng_state"]


@pytest.mark.parametrize("drain_fails", [False, True])
def test_failed_cuda_stage_drains_copies_before_reporting_failure(drain_fails):
    module = importlib.import_module("nanochat.rlm_distributed_checkpoint")
    stager = module.CheckpointStateStager("cpu")
    failure = Future()
    failure.set_exception(torch.cuda.OutOfMemoryError("injected copy failure"))
    drained = []

    class FailedCopy:
        def stage(self, state):
            return failure

    def drain(device):
        drained.append(device)
        if drain_fails:
            raise RuntimeError("injected CUDA drain failure")

    stager._stager = FailedCopy()
    with patch.object(torch.cuda, "synchronize", drain):
        expected = CheckpointStagingSynchronizationError if drain_fails else torch.cuda.OutOfMemoryError
        with pytest.raises(expected):
            stager.stage({"weight": torch.ones(1)}, {})
    assert drained == [torch.device("cpu")]


def test_abort_drains_admitted_publication_and_preserves_training_failure(tmp_path):
    module = importlib.import_module("nanochat.rlm_distributed_checkpoint")
    parameter, optimizer = _optimizer()
    writer = module.DistributedCheckpointWriter(
        tmp_path, mode="async", keep_last=1, optimizer=optimizer,
        checkpoint_group=None, control_group=None,
    )
    writer.submit({"weight": parameter}, optimizer.state_dict(), _metadata(1), numbered=False)
    with patch.object(module.CheckpointStateStager, "close", side_effect=RuntimeError("cleanup failed")):
        try:
            raise ValueError("original training failure")
        except ValueError:
            writer.abort()
    assert LatestCheckpoint(tmp_path).metadata()["step"] == 1
    with pytest.raises(RuntimeError, match="closed"):
        writer.submit({"weight": parameter}, optimizer.state_dict(), _metadata(2), numbered=False)


def _distributed_worker(rank, world_size, directory, mode, failure):
    module = importlib.import_module("nanochat.rlm_distributed_checkpoint")
    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group("gloo", init_method=f"file://{directory / 'rendezvous'}",
                            rank=rank, world_size=world_size, timeout=timedelta(seconds=30))
    checkpoint_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=30))
    control_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=30))
    parameter, optimizer = _optimizer()
    writer = module.DistributedCheckpointWriter(
        directory, mode=mode, keep_last=1, optimizer=optimizer,
        checkpoint_group=checkpoint_group, control_group=control_group,
    )
    try:
        writer.submit({"weight": parameter}, optimizer.state_dict(), _metadata(1, rank), numbered=True)
        writer.wait(1)
        if failure:
            if failure == "launch":
                target = writer._executor
                name = "submit"
                failure_error = RuntimeError("injected thread launch failure")
            elif failure in ("stage", "fatal_stage"):
                target = module.CheckpointStateStager
                name = "stage"
                failure_error = (CheckpointStagingSynchronizationError("injected CUDA drain failure")
                                 if failure == "fatal_stage" else MemoryError("injected CPU staging failure"))
            elif failure == "snapshot":
                target = module
                name = "_snapshot_state"
                failure_error = ValueError("injected snapshot failure")
            else:
                target = LatestCheckpoint
                name = "preserve_milestone"
                failure_error = OSError("injected milestone failure")
            failing_rank = 0 if failure == "milestone" else world_size - 1
            with patch.object(target, name, side_effect=failure_error) if rank == failing_rank else _no_patch():
                if failure in ("milestone", "fatal_stage"):
                    expected = (CheckpointStagingSynchronizationError
                                if failure == "fatal_stage" else MilestoneCheckpointError)
                    with pytest.raises(expected):
                        writer.submit({"weight": parameter}, optimizer.state_dict(), _metadata(2, rank),
                                      numbered=False, milestone=failure == "milestone")
                        writer.wait(2)
                    if failure == "fatal_stage":
                        assert LatestCheckpoint(directory).metadata()["step"] == 1
                        writer.abort()
                        return
                else:
                    writer.submit({"weight": parameter}, optimizer.state_dict(), _metadata(2, rank), numbered=False)
                    writer.wait(2)
                    assert writer.measurements[-1]["status"] == "warning"
                    assert writer.status(2)["latest_step"] == 1
            writer.submit({"weight": parameter}, optimizer.state_dict(), _metadata(3, rank), numbered=True)
            writer.wait(3)
        assert writer.measurements[-1]["latest_published"]
        if rank == 0:
            _, state, metadata = LatestCheckpoint(directory).load_cpu_mapped()
            moments = state["state"][0]["exp_avg"]
            assert moments.shape == (16, 128)
            for worker in range(world_size):
                torch.testing.assert_close(moments.chunk(world_size)[worker],
                                           torch.full((16 // world_size, 128), worker + 1.0))
            assert state["state"][0]["step"] == 7
            assert metadata["distributed"]["world_size"] == world_size
            for worker, saved in enumerate(metadata["distributed"]["rank_states"]):
                assert saved["rng_state"]["cpu"][0] == worker + 1
            if failure:
                assert metadata["step"] == 3
                assert not (directory / "model_000001.pt").exists()
    finally:
        writer.close(3 if failure else 1)
        dist.destroy_process_group()


def _no_patch():
    from contextlib import nullcontext
    return nullcontext()


@pytest.mark.parametrize("world_size,mode,failure", [
    (2, "sync", None), (4, "async", None), (2, "async", "launch"),
    (2, "async", "stage"), (2, "sync", "milestone"), (2, "async", "milestone"),
    (2, "async", "fatal_stage"), (2, "async", "snapshot"),
])
def test_distributed_publication_and_failure_coordination(tmp_path, world_size, mode, failure):
    mp.spawn(_distributed_worker, args=(world_size, str(tmp_path), mode, failure),
             nprocs=world_size, join=True)

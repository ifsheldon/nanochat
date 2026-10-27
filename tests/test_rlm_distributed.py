"""Exercise real worker coordination, global batches, and portable resume state."""

import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nanochat.rlm_distributed import DistributedRuntime, DistributedRunError, resume_worker_state
from nanochat.rlm_training import MemoryMonitor, TrainingConfig


@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("batch", [16, 32])
def test_accumulation_preserves_global_text_batch(world_size, batch):
    config = TrainingConfig(cycles=4, device_batch_size=batch)
    accumulation = config.accumulation_steps(1023, world_size)
    assert accumulation * batch * 4 * 1023 * world_size == config.total_batch_size
    with pytest.raises(ValueError, match="divisible"):
        TrainingConfig(cycles=4, device_batch_size=batch, total_batch_size=1_047_551).accumulation_steps(
            1023, world_size,
        )


def _coordination_worker(rank, world_size, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size)
    runtime = DistributedRuntime.from_process_group(torch.device("cpu"))
    try:
        marker = Path(directory) / "coordinator.txt"
        value = runtime.primary(lambda: (marker.write_text(str(rank)), "published")[1], "publish test marker")
        assert value == "published"
        assert runtime.any(rank == world_size - 1)
        assert not runtime.any(False)
        result = runtime.aggregate_metrics({
            "duration_seconds": rank + 1.0,
            "loss": 2.0 + rank,
            "last_batch_loss": 3.0 + rank,
            "cycle_losses": [1.0 + rank, 3.0 + rank],
            "cycle_1_loss": 1.0 + rank,
            "cycle_2_loss": 3.0 + rank,
            "within_document_handoff_fraction": 0.5,
            "continuation_text_fraction": 0.25,
            "text_targets": 64,
            "internal_positions": 128,
            "minimum_system_available_gb": 20.0 - rank,
            "cuda_peak_allocated_gb": float(rank),
        })
        assert result["duration_seconds"] == world_size
        assert result["loss"] == 2.0 + (world_size - 1) / 2
        assert result["text_tokens_per_second"] == 64 / world_size
        assert result["minimum_system_available_gb"] == 21.0 - world_size
        assert result["cuda_peak_allocated_gb"] == world_size - 1

        def fail_one_worker():
            if rank == world_size - 1:
                raise ValueError("worker input failed")
            return rank

        with pytest.raises(DistributedRunError, match="worker input failed"):
            runtime.collective(fail_one_worker, "prepare batch")

        def oom_one_worker():
            if rank == 1:
                raise torch.cuda.OutOfMemoryError("test allocation")

        with pytest.raises(torch.cuda.OutOfMemoryError, match="test allocation"):
            runtime.collective(oom_one_worker, "capacity probe")

        def fail_publication():
            raise OSError("shared artifact write failed")

        with pytest.raises(DistributedRunError, match="shared artifact write failed"):
            runtime.primary(fail_publication, "publish result")
        monitor = MemoryMonitor(runtime=runtime)
        if rank == world_size - 1:
            def fail_thread_start():
                raise RuntimeError("cannot start monitoring thread")
            monitor._thread.start = fail_thread_start
        with pytest.raises(DistributedRunError, match="cannot start monitoring thread"):
            with monitor:
                pytest.fail("A failed monitor must stop every worker before forward")
        assert not monitor._thread.is_alive()
        Path(directory, f"rank-{rank}.json").write_text(json.dumps(result))
    finally:
        runtime.close()


@pytest.mark.parametrize("world_size", [2, 4])
def test_workers_coordinate_artifacts_metrics_and_failures(tmp_path, world_size):
    mp.spawn(_coordination_worker, args=(world_size, str(tmp_path / "rendezvous"), str(tmp_path)),
             nprocs=world_size, join=True)
    assert (tmp_path / "coordinator.txt").read_text() == "0"
    assert len(list(tmp_path.glob("rank-*.json"))) == world_size


def test_resume_preserves_rank_state_or_repartitions_from_earliest_cursor():
    states = [
        {"dataloader_state_dict": {"epoch": 2, "pq_idx": 0, "rg_idx": 0}, "rng_state": {"cpu": [10]}},
        {"dataloader_state_dict": {"epoch": 1, "pq_idx": 7, "rg_idx": 9}, "rng_state": {"cpu": [20]}},
    ]
    metadata = {**states[0], "distributed": {"world_size": 2, "rank_states": states}}
    for rank in range(2):
        assert resume_worker_state(metadata, rank, 2) == states[rank]
    for rank in range(4):
        restored = resume_worker_state(metadata, rank, 4)
        assert restored["dataloader_state_dict"] == states[1]["dataloader_state_dict"]
        assert restored["rng_state"] == states[rank % 2]["rng_state"]
    assert resume_worker_state(metadata, 0, 1)["dataloader_state_dict"] == states[1]["dataloader_state_dict"]


def test_single_worker_checkpoint_can_resume_on_multiple_workers():
    metadata = {"dataloader_state_dict": {"epoch": 1, "pq_idx": 3, "rg_idx": 7},
                "rng_state": {"cpu": [42], "cuda": None}}
    assert resume_worker_state(metadata, 3, 4) == metadata


def test_incomplete_rank_state_is_rejected_before_resume():
    metadata = {"distributed": {"world_size": 4, "rank_states": [{}]}}
    with pytest.raises(ValueError, match="worker state"):
        resume_worker_state(metadata, 0, 2)

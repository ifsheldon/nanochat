"""Worker allocation/compute failures must precede optimizer collectives."""

from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import nanochat.optim as optim_module
from nanochat.optim import MuonAdamW
from nanochat.optim_checkpoint import OptimizerStateLayout
from nanochat.rlm_distributed import DistributedRuntime
from test_rlm_distributed_training import _cpu_execution


def optimizer_with_gradients(gradient_rank):
    generator = torch.Generator().manual_seed(718)
    params = [torch.nn.Parameter(torch.randn(8, 128, generator=generator)),
              torch.nn.Parameter(torch.randn(7, generator=generator))]
    params += [torch.nn.Parameter(torch.randn(4, 8, generator=generator)) for _ in range(3)]
    params += [torch.nn.Parameter(torch.randn(8, 4, generator=generator))]
    optimizer = MuonAdamW([
        dict(kind="adamw", params=params[:2], lr=0.01, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1),
        dict(kind="muon", params=params[2:5], lr=0.02, momentum=0.9, beta2=0.95, ns_steps=5, weight_decay=0.1),
        dict(kind="muon", params=params[5:], lr=0.02, momentum=0.9, beta2=0.95, ns_steps=5, weight_decay=0.1),
    ])
    for index, parameter in enumerate(params):
        parameter.grad = torch.full_like(parameter, 0.2 + index * 0.03 + gradient_rank * 0.1)
    return optimizer, params


def distributed_worker(rank, world_size, rendezvous, directory, failure):
    torch.set_num_threads(1)
    timeout = timedelta(seconds=15)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size, timeout=timeout)
    control = dist.new_group(backend="gloo", timeout=timeout)
    checkpoint = dist.new_group(backend="gloo", timeout=timeout)
    runtime = DistributedRuntime(torch.device("cpu"), rank, world_size, control, checkpoint)
    try:
        with _cpu_execution():
            optimizer, params = optimizer_with_gradients(rank)
            original = [parameter.detach().clone() for parameter in params]
            if failure in ("adam-allocation", "muon-allocation"):
                name = "empty_like" if failure == "adam-allocation" else "stack"
                allocator = getattr(torch, name)

                def allocate(tensor, *args, **kwargs):
                    shape = tuple(tensor.shape) if name == "empty_like" else tuple(tensor[0].shape)
                    expected = (8 // world_size, 128) if name == "empty_like" else (4, 8)
                    if rank == 1 and shape == expected:
                        raise torch.cuda.OutOfMemoryError("rank-one reduction allocation OOM")
                    return allocator(tensor, *args, **kwargs)

                with patch.object(torch, name, allocate):
                    with pytest.raises(torch.cuda.OutOfMemoryError, match="rank 1.*allocation OOM"):
                        optimizer.step(coordinate=runtime.collective)
                for parameter, saved in zip(params, original):
                    torch.testing.assert_close(parameter, saved, rtol=0, atol=0)
            elif failure in ("adam-compute", "muon-compute"):
                name = "adamw_step_fused" if failure == "adam-compute" else "muon_step_fused"
                fused_step = getattr(optim_module, name)

                def compute(*args, **kwargs):
                    if rank == 1:
                        raise torch.cuda.OutOfMemoryError("rank-one fused compute OOM")
                    return fused_step(*args, **kwargs)

                with patch.object(optim_module, name, compute):
                    with pytest.raises(torch.cuda.OutOfMemoryError, match="rank 1.*compute OOM"):
                        optimizer.step(coordinate=runtime.collective)
            else:
                optimizer.step(coordinate=runtime.collective)
                reference = torch.load(Path(directory) / "reference.pt", weights_only=True)
                for parameter, expected in zip(params, reference["params"]):
                    torch.testing.assert_close(parameter, expected, rtol=1e-5, atol=1e-6)
                full = OptimizerStateLayout.from_optimizer(optimizer).gather(optimizer.state_dict(), checkpoint)
                if rank == 0:
                    for parameter_id, values in reference["optimizer"]["state"].items():
                        for field, expected in values.items():
                            if isinstance(expected, torch.Tensor):
                                torch.testing.assert_close(full["state"][parameter_id][field], expected, rtol=1e-5, atol=1e-6)
                            else:
                                assert full["state"][parameter_id][field] == expected
    finally:
        runtime.close()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("failure", ["adam-allocation", "muon-allocation", "adam-compute", "muon-compute"])
def test_rank_local_optimizer_oom_reaches_every_worker_without_hanging(tmp_path, world_size, failure):
    mp.spawn(distributed_worker, args=(world_size, str(tmp_path / "rendezvous"), str(tmp_path), failure), nprocs=world_size, join=True)


@pytest.mark.parametrize("world_size", [2, 4])
def test_coordinated_distributed_step_preserves_single_worker_update(tmp_path, world_size):
    with _cpu_execution():
        optimizer, params = optimizer_with_gradients((world_size - 1) / 2)
        optimizer.step()
        torch.save({"params": [parameter.detach() for parameter in params], "optimizer": optimizer.state_dict()}, tmp_path / "reference.pt")
    mp.spawn(distributed_worker, args=(world_size, str(tmp_path / "rendezvous"), str(tmp_path), None), nprocs=world_size, join=True)

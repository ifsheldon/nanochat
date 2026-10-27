"""Portable MuonAdamW moments across CPU worker counts."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nanochat.optim import MuonAdamW
from nanochat.optim_checkpoint import OptimizerCheckpointError, OptimizerStateLayout


def make_optimizer():
    params = [torch.nn.Parameter(torch.zeros(8, 128, dtype=torch.bfloat16)),
              torch.nn.Parameter(torch.zeros(7)),
              torch.nn.Parameter(torch.zeros(())),
              torch.nn.Parameter(torch.zeros(8, 128), requires_grad=False)]
    params += [torch.nn.Parameter(torch.zeros(2, 4)) for _ in range(3)]
    params += [torch.nn.Parameter(torch.zeros(4, 2))]
    return MuonAdamW([
        dict(kind="adamw", params=params[:4], lr=0.01, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1),
        dict(kind="muon", params=params[4:7], lr=0.02, momentum=0.9, beta2=0.95, ns_steps=5, weight_decay=0.1),
        dict(kind="muon", params=params[7:], lr=0.03, momentum=0.8, beta2=0.9, ns_steps=5, weight_decay=0.0),
    ])


def full_state():
    optimizer = make_optimizer()
    # Construct full moments even when a worker's default group is initialized.
    for group in optimizer.param_groups:
        if group["kind"] == "adamw":
            for parameter in group["params"]:
                if parameter.requires_grad:
                    optimizer._initialize_adamw_state(parameter, parameter)
        else:
            optimizer._initialize_muon_state(group, len(group["params"]))
    state = optimizer.state_dict()
    for parameter_id, values in state["state"].items():
        for field, tensor in values.items():
            if isinstance(tensor, torch.Tensor):
                tensor.copy_(torch.arange(tensor.numel()).reshape(tensor.shape) + parameter_id * 100 + (20 if field.endswith("sq") else 0))
        if "step" in values:
            values["step"] = 17
    return state


def assert_state_equal(actual, expected):
    assert actual["param_groups"] == expected["param_groups"]
    assert actual["state"].keys() == expected["state"].keys()
    for parameter_id, values in expected["state"].items():
        assert actual["state"][parameter_id].keys() == values.keys()
        for field, value in values.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual["state"][parameter_id][field], value, rtol=0, atol=0)
            else:
                assert actual["state"][parameter_id][field] == value


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_shard_preserves_real_rows_lazy_state_and_independent_storage(world_size):
    layout = OptimizerStateLayout.from_optimizer(make_optimizer())
    full = full_state()
    for rank in range(world_size):
        local = layout.shard(full, rank, world_size)
        assert 3 not in local["state"]  # Frozen embedding must remain lazy.
        rows = 8 // world_size
        for field in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(local["state"][0][field], full["state"][0][field][rank * rows:(rank + 1) * rows])
        for parameter_id in (1, 2):
            torch.testing.assert_close(local["state"][parameter_id]["exp_avg"], full["state"][parameter_id]["exp_avg"])
            assert local["state"][parameter_id]["step"] == 17
        for anchor, count in ((4, 3), (7, 1)):
            chunk = (count + world_size - 1) // world_size
            for field, value in full["state"][anchor].items():
                expected = torch.zeros(chunk, *value.shape[1:])
                owned = value[rank * chunk:(rank + 1) * chunk]
                expected[:len(owned)] = owned
                torch.testing.assert_close(local["state"][anchor][field], expected)
        restored = make_optimizer()
        restored.load_state_dict(local)
        assert_state_equal(restored.state_dict(), local)
        local["state"][0]["exp_avg"].fill_(-100)
        local["param_groups"][0]["lr"] = -1
        assert full["state"][0]["exp_avg"].min() >= 0
        assert full["param_groups"][0]["lr"] == 0.01


def test_single_worker_gather_and_reservation_keep_full_shapes_and_independence():
    layout = OptimizerStateLayout.from_optimizer(make_optimizer())
    full = full_state()
    gathered = layout.gather(full, None)
    assert_state_equal(gathered, full)
    reserved = layout.allocate_full(full)
    assert reserved["state"].keys() == full["state"].keys()
    for parameter_id, values in full["state"].items():
        for field, value in values.items():
            if isinstance(value, torch.Tensor):
                target = reserved["state"][parameter_id][field]
                assert target.shape == value.shape
                assert target.dtype == value.dtype
                assert target.device.type == "cpu"
    gathered["state"][4]["momentum_buffer"].zero_()
    assert full["state"][4]["momentum_buffer"].min() >= 400


def gather_worker(rank, world_size, rendezvous, output, failure=None):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size, timeout=timedelta(seconds=30))
    try:
        full = full_state()
        local = {"param_groups": full["param_groups"], "state": {}}
        # Derive shards independently from the production partitioning function.
        for parameter_id, values in full["state"].items():
            local["state"][parameter_id] = {}
            for field, value in values.items():
                if not isinstance(value, torch.Tensor):
                    local["state"][parameter_id][field] = value
                elif parameter_id == 0:
                    local["state"][parameter_id][field] = value.chunk(world_size)[rank].clone()
                elif parameter_id in (4, 7):
                    chunk = 2 if parameter_id == 4 and world_size == 2 else 1
                    buf = torch.full((chunk, *value.shape[1:]), -999.0)
                    owned = value[rank * chunk:(rank + 1) * chunk]
                    buf[:len(owned)].copy_(owned)
                    local["state"][parameter_id][field] = buf
                else:
                    local["state"][parameter_id][field] = value.clone()
        if failure == "step":
            local["state"][0]["step"] += rank
        elif failure == "shape" and rank == 1:
            local["state"][0]["exp_avg"] = torch.zeros(1)
        elif failure == "order" and rank == 1:
            local["state"][0] = dict(reversed(list(local["state"][0].items())))
        layout = OptimizerStateLayout.from_optimizer(make_optimizer())
        if failure == "allocation":
            def fail_allocation(*args, **kwargs):
                raise MemoryError("injected root allocation failure")
            if rank == 0:
                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(OptimizerStateLayout, "allocate_full", fail_allocation)
                    with pytest.raises(OptimizerCheckpointError, match="root allocation"):
                        layout.gather(local, dist.group.WORLD)
            else:
                with pytest.raises(OptimizerCheckpointError, match="root allocation"):
                    layout.gather(local, dist.group.WORLD)
            return
        if failure in ("step", "shape"):
            with pytest.raises(OptimizerCheckpointError, match=failure):
                layout.gather(local, dist.group.WORLD)
            return
        reserved = layout.allocate_full(local)
        assert reserved["state"][4]["momentum_buffer"].shape == (3, 2, 4)
        assert reserved["state"][7]["second_momentum_buffer"].shape == (1, 4, 1)
        def reject_tensor_pickling(tensor, protocol):
            raise AssertionError("Tensor payload must use direct collectives")
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch.Tensor, "__reduce_ex__", reject_tensor_pickling)
            gathered = layout.gather(local, dist.group.WORLD)
        if rank == 0:
            assert_state_equal(gathered, full)
            for target_world_size in (1, 2, 4):
                for target_rank in range(target_world_size):
                    resumed = make_optimizer()
                    resumed.load_state_dict(layout.shard(gathered, target_rank, target_world_size))
                    assert resumed.state[resumed.param_groups[0]["params"][0]]["step"] == 17
            torch.save(gathered, output)
        else:
            assert gathered is None
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_real_gloo_gather_captures_every_rank_and_trims_muon_padding(tmp_path, world_size):
    output = tmp_path / "complete.pt"
    mp.spawn(gather_worker, args=(world_size, str(tmp_path / "rendezvous"), str(output)), nprocs=world_size, join=True)
    assert_state_equal(torch.load(output, weights_only=True), full_state())


@pytest.mark.parametrize("failure", ["step", "shape", "allocation"])
def test_gather_preparation_failure_reaches_every_rank_before_tensor_collectives(tmp_path, failure):
    mp.spawn(gather_worker, args=(2, str(tmp_path / "rendezvous"), str(tmp_path / "unused"), failure), nprocs=2, join=True)


def test_tensor_parameter_group_metadata_is_rejected_before_object_serialization():
    layout = OptimizerStateLayout.from_optimizer(make_optimizer())
    state = full_state()
    state["param_groups"][0]["lr"] = torch.tensor(0.1)
    with pytest.raises(OptimizerCheckpointError, match="metadata"):
        layout.gather(state, None)



def test_gather_matches_fields_when_local_dictionary_insertion_order_differs(tmp_path):
    output = tmp_path / "complete.pt"
    mp.spawn(gather_worker, args=(2, str(tmp_path / "rendezvous"), str(output), "order"), nprocs=2, join=True)
    assert_state_equal(torch.load(output, weights_only=True), full_state())

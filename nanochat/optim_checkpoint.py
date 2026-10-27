"""Portable complete checkpoints for MuonAdamW's rank-local moments.

Adam moments shard along parameter rows above the optimizer's 1024-element
threshold. Muon moments shard along a group's parameter stack, whose last worker
may contain padding. A complete checkpoint stores only real rows in canonical
optimizer state-dict IDs and can be partitioned for a different worker count.
"""

from copy import deepcopy
from dataclasses import dataclass
from math import prod

import torch
import torch.distributed as dist


class OptimizerCheckpointError(ValueError):
    """A snapshot preparation failure reported coherently to every worker."""


def _validate_metadata(value):
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_metadata(key)
            _validate_metadata(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_metadata(item)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError("Optimizer parameter-group metadata must contain only scalar values")


@dataclass(frozen=True)
class _Parameter:
    kind: str
    shape: tuple[int, ...]
    count: int = 1

    def tensor_shape(self, field):
        if self.kind == "adamw":
            if field in ("exp_avg", "exp_avg_sq"):
                return self.shape
            if field == "step":
                return ()
        else:
            if field == "momentum_buffer":
                return (self.count, *self.shape)
            if field == "second_momentum_buffer":
                rows, columns = self.shape
                return (self.count, rows, 1) if rows >= columns else (self.count, 1, columns)
        raise ValueError(f"Unknown {self.kind} optimizer state field {field!r}")

    def sharded(self, field):
        return field != "step" and (self.kind == "muon" or prod(self.shape) >= 1024)

    def local_shape(self, field, world_size):
        shape = self.tensor_shape(field)
        if not self.sharded(field):
            return shape
        if self.kind == "muon":
            return ((self.count + world_size - 1) // world_size, *shape[1:])
        if shape[0] % world_size:
            raise ValueError("AdamW state sharding requires shape[0] divisible by world size")
        return (shape[0] // world_size, *shape[1:])


@dataclass(frozen=True)
class OptimizerStateLayout:
    """Capture canonical IDs and shapes without retaining model parameter storage."""

    parameters: dict[int, _Parameter]
    groups: tuple[tuple[str, tuple[int, ...]], ...]

    @classmethod
    def from_optimizer(cls, optimizer):
        parameters = {}
        groups = []
        saved_groups = optimizer.state_dict()["param_groups"]
        for live, saved in zip(optimizer.param_groups, saved_groups):
            kind = live["kind"]
            ids = tuple(saved["params"])
            if kind not in ("adamw", "muon"):
                raise ValueError(f"Unknown optimizer kind: {kind}")
            groups.append((kind, ids))
            if kind == "muon":
                shapes = {tuple(parameter.shape) for parameter in live["params"]}
                if len(shapes) != 1 or len(next(iter(shapes))) != 2:
                    raise ValueError("Muon checkpoints require a nonempty group of equally shaped matrices")
                parameters[ids[0]] = _Parameter(kind, next(iter(shapes)), len(ids))
            else:
                for parameter_id, parameter in zip(ids, live["params"]):
                    parameters[parameter_id] = _Parameter(kind, tuple(parameter.shape))
        return cls(parameters, tuple(groups))

    def _validate(self, state, world_size):
        _validate_metadata(state["param_groups"])
        if len(state["param_groups"]) != len(self.groups):
            raise ValueError("Optimizer checkpoint parameter groups do not match the model")
        for saved, (kind, ids) in zip(state["param_groups"], self.groups):
            if saved["kind"] != kind or tuple(saved["params"]) != ids:
                raise ValueError("Optimizer checkpoint parameter IDs or kinds do not match the model")
        for parameter_id, values in state["state"].items():
            if parameter_id not in self.parameters:
                raise ValueError(f"Unexpected optimizer state parameter ID {parameter_id}")
            parameter = self.parameters[parameter_id]
            expected = {"step", "exp_avg", "exp_avg_sq"} if parameter.kind == "adamw" else {"momentum_buffer", "second_momentum_buffer"}
            if values and values.keys() != expected:
                raise ValueError(f"Incomplete or unknown {parameter.kind} state for parameter {parameter_id}")
            for field, value in values.items():
                shape = parameter.local_shape(field, world_size)
                if field == "step":
                    if isinstance(value, torch.Tensor):
                        if value.shape != torch.Size([]):
                            raise ValueError("AdamW step must be a scalar")
                    elif not isinstance(value, int):
                        raise ValueError("AdamW step must be an integer or scalar tensor")
                elif not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                    raise ValueError(f"Invalid {field} shape for optimizer parameter {parameter_id}; expected {shape}")

    def shard(self, full_state, rank, world_size):
        """Return an independently owned CPU state suitable for load_state_dict."""
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Optimizer shard rank must belong to a positive world size")
        self._validate(full_state, 1)
        output = {"state": {}, "param_groups": deepcopy(full_state["param_groups"])}
        for parameter_id, values in full_state["state"].items():
            parameter = self.parameters[parameter_id]
            target = output["state"][parameter_id] = {}
            for field, value in values.items():
                if not isinstance(value, torch.Tensor):
                    target[field] = deepcopy(value)
                elif not parameter.sharded(field):
                    target[field] = value.detach().to("cpu", copy=True)
                else:
                    shape = parameter.local_shape(field, world_size)
                    tensor = torch.zeros(shape, dtype=value.dtype, device="cpu")
                    start = rank * shape[0]
                    owned = value[start:start + shape[0]]
                    tensor[:len(owned)].copy_(owned)
                    target[field] = tensor
        return output

    def allocate_full(self, local_state, group=None):
        """Reserve and touch root gather buffers, including Muon padding storage.

        Views in the returned ordinary state dict expose only real rows. Their
        underlying storage also holds any padded rows required by dist.gather.
        The optional group selects the same geometry as gather's process group.
        """
        world_size = dist.get_world_size(group) if dist.is_initialized() else 1
        self._validate(local_state, world_size)
        output = {"state": {}, "param_groups": deepcopy(local_state["param_groups"])}
        for parameter_id, values in local_state["state"].items():
            parameter = self.parameters[parameter_id]
            target = output["state"][parameter_id] = {}
            for field, value in values.items():
                if not isinstance(value, torch.Tensor):
                    target[field] = deepcopy(value)
                    continue
                shape = parameter.tensor_shape(field)
                padded_shape = shape
                if parameter.sharded(field):
                    padded_shape = (value.shape[0] * world_size, *shape[1:])
                storage = torch.zeros(padded_shape, dtype=value.dtype, device="cpu")
                target[field] = storage[:shape[0]] if shape else storage
        return output

    def gather(self, local_state, group=None):
        """Gather CPU tensor shards onto group rank zero, returning None elsewhere.

        Only shape/dtype descriptors and scalar metadata cross object collectives.
        Preparation and root allocation failures are exchanged before any tensor
        collective, so a failed snapshot cannot strand peers waiting for a shard.
        """
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size(group) if distributed else 1
        rank = dist.get_rank(group) if distributed else 0
        prepared = {}
        try:
            self._validate(local_state, world_size)
            metadata = {"param_groups": local_state["param_groups"], "state": {}}
            for parameter_id, values in local_state["state"].items():
                description = metadata["state"][parameter_id] = {}
                for field, value in values.items():
                    if isinstance(value, torch.Tensor):
                        prepared[parameter_id, field] = value.detach().to("cpu").contiguous()
                        description[field] = ("tensor", tuple(value.shape), value.dtype, value.item() if field == "step" else None)
                    else:
                        description[field] = ("scalar", value)
            status = (None, metadata)
        except Exception as error:
            status = (f"{type(error).__name__}: {error}", None)
        statuses = [None] * world_size
        if distributed:
            dist.all_gather_object(statuses, status, group=group)
        else:
            statuses[0] = status
        for worker, (error, _) in enumerate(statuses):
            if error is not None:
                raise OptimizerCheckpointError(f"Optimizer checkpoint preparation failed on rank {worker}: {error}")
        reference = statuses[0][1]
        if any(other[1] != reference for other in statuses[1:]):
            raise OptimizerCheckpointError("Optimizer checkpoint workers disagree on state, step counters, tensor descriptors, or parameter groups")

        output = None
        allocation_error = None
        if rank == 0:
            try:
                output = self.allocate_full(local_state, group)
            except Exception as error:
                allocation_error = f"{type(error).__name__}: {error}"
        statuses = [None] * world_size
        if distributed:
            dist.all_gather_object(statuses, allocation_error, group=group)
        else:
            statuses[0] = allocation_error
        if statuses[0] is not None:
            raise OptimizerCheckpointError(f"Optimizer checkpoint root allocation failed: {statuses[0]}")

        destination = dist.get_global_rank(group, 0) if group is not None else 0
        # Every worker must issue tensor collectives in the same canonical order,
        # independent of the insertion order used when its lazy state was created.
        for parameter_id in sorted(local_state["state"]):
            parameter = self.parameters[parameter_id]
            for field in sorted(local_state["state"][parameter_id]):
                value = local_state["state"][parameter_id][field]
                if not isinstance(value, torch.Tensor):
                    continue
                source = prepared[parameter_id, field]
                if parameter.sharded(field) and distributed and world_size > 1:
                    receives = None
                    if rank == 0:
                        tensor = output["state"][parameter_id][field]
                        padded = tensor.as_strided((source.shape[0] * world_size, *source.shape[1:]), tensor.stride())
                        receives = list(padded.chunk(world_size))
                    dist.gather(source, gather_list=receives, dst=destination, group=group)
                elif rank == 0:
                    output["state"][parameter_id][field].copy_(source)
        return output

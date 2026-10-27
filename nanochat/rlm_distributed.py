"""Worker coordination for RLM training and portable checkpoint resume."""

from dataclasses import dataclass
import os

import torch
import torch.distributed as dist


class DistributedRunError(RuntimeError):
    """A worker could not complete a phase needed by every worker."""


def _error_record(error):
    from nanochat.rlm_capacity import is_cuda_out_of_memory

    return {"type": type(error).__name__, "message": str(error),
            "cuda_oom": is_cuda_out_of_memory(error)}


def _raise_worker_errors(errors, phase):
    failures = [(rank, error) for rank, error in enumerate(errors) if error is not None]
    if not failures:
        return
    message = f"{phase} failed: " + "; ".join(
        f"rank {rank}: {error['type']}: {error['message']}" for rank, error in failures
    )
    if all(error["cuda_oom"] for _, error in failures):
        raise torch.cuda.OutOfMemoryError(message)
    raise DistributedRunError(message)


@dataclass
class DistributedRuntime:
    device: torch.device
    rank: int = 0
    world_size: int = 1
    control_group: object = None
    checkpoint_group: object = None

    @property
    def is_primary(self):
        return self.rank == 0

    @classmethod
    def single(cls, device):
        return cls(torch.device(device))

    @classmethod
    def initialize(cls, device_type):
        from nanochat.common import compute_init

        if device_type == "cpu" and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            dist.init_process_group("gloo")
        *_, device = compute_init(device_type)
        return cls.from_process_group(device)

    @classmethod
    def from_process_group(cls, device):
        if not dist.is_initialized():
            return cls.single(device)
        # Control messages must remain independent of both CUDA training and
        # background checkpoint traffic. Every rank creates groups in this order.
        return cls(torch.device(device), dist.get_rank(), dist.get_world_size(),
                   dist.new_group(backend="gloo"), dist.new_group(backend="gloo"))

    def collective(self, function, phase):
        """Complete local work and exchange failures before entering collectives.

        The callable itself must not start a collective before fallible local
        work completes; split such work into separate coordinated phases.
        """
        if self.world_size == 1:
            return function()
        result, error = None, None
        try:
            result = function()
        except Exception as exc:
            error = _error_record(exc)
        errors = [None] * self.world_size
        dist.all_gather_object(errors, error, group=self.control_group)
        _raise_worker_errors(errors, phase)
        return result

    def primary(self, function, phase):
        """Execute a shared artifact operation once and broadcast its outcome."""
        if self.world_size == 1:
            return function()
        payload = [None]
        if self.is_primary:
            try:
                payload[0] = {"value": function(), "error": None}
            except Exception as exc:
                payload[0] = {"value": None, "error": _error_record(exc)}
        dist.broadcast_object_list(payload, src=0, group=self.control_group)
        _raise_worker_errors([payload[0]["error"]], phase)
        return payload[0]["value"]

    def any(self, value):
        if self.world_size == 1:
            return bool(value)
        flag = torch.tensor(int(bool(value)), dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self.control_group)
        return bool(flag.item())

    def aggregate_metrics(self, metrics):
        """Report global means, slowest-worker timings, and worst memory usage."""
        values = [metrics]
        if self.world_size > 1:
            values = [None] * self.world_size
            dist.all_gather_object(values, metrics, group=self.control_group)
        result = dict(values[0])
        averages = {"loss", "last_batch_loss", "within_document_handoff_fraction",
                    "continuation_text_fraction"}
        for key in metrics:
            column = [value[key] for value in values]
            if key in averages or (key.startswith("cycle_") and key.endswith("_loss")):
                result[key] = sum(column) / len(column)
            elif key == "cycle_losses":
                result[key] = [sum(items) / len(items) for items in zip(*column)]
            elif key == "minimum_system_available_gb":
                result[key] = min(column)
            elif key.startswith(("cuda_", "peak_cuda_")) or key.endswith("_seconds"):
                result[key] = max(column)
            elif isinstance(column[0], bool):
                result[key] = all(column)
        if "first_episode_cycle_forwards" in metrics:
            result["first_episode_cycle_forwards"] = [
                {**items[0], "duration_seconds": max(item["duration_seconds"] for item in items)}
                for items in zip(*(value["first_episode_cycle_forwards"] for value in values))
            ]
        elapsed = result.get("duration_seconds")
        if elapsed is not None:
            for count, rate in (("text_targets", "text_tokens_per_second"),
                                ("internal_positions", "internal_positions_per_second")):
                if count in result:
                    result[rate] = result[count] / elapsed
        result["world_size"] = self.world_size
        return result

    def close(self):
        # Do not introduce a rendezvous during failure cleanup.
        for group in (self.checkpoint_group, self.control_group):
            if group is not None:
                dist.destroy_process_group(group)
        if dist.is_initialized():
            dist.destroy_process_group()


def resume_worker_state(metadata, rank, world_size):
    """Restore rank state, or repartition from the earliest saved row group."""
    distributed = metadata.get("distributed")
    if distributed is None:
        return {name: metadata[name] for name in ("dataloader_state_dict", "rng_state")}
    saved_world_size = distributed["world_size"]
    states = distributed["rank_states"]
    if (len(states) != saved_world_size or not states or any(
        "dataloader_state_dict" not in state or "rng_state" not in state for state in states
    )):
        raise ValueError("Checkpoint has incomplete saved worker state")
    if world_size == saved_world_size:
        return states[rank]
    cursor = min((state["dataloader_state_dict"] for state in states),
                 key=lambda value: (value["epoch"], value["pq_idx"], value["rg_idx"]))
    return {"dataloader_state_dict": dict(cursor), "rng_state": states[rank % saved_world_size]["rng_state"]}

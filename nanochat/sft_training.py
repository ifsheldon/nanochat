"""SFT loss configuration, update progress, and learning-rate schedules."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType


class LossWeighting(str, Enum):
    MICROBATCH = "microbatch"
    GROUP_TOKEN = "group-token"


@dataclass(frozen=True)
class SFTLossConfig:
    """An explicit reduction, with fixed cycle weights for group normalization."""

    mode: LossWeighting
    group_weights: Mapping[int, float] | None = None

    def __post_init__(self):
        if not isinstance(self.mode, LossWeighting):
            raise TypeError("SFT loss mode must be a LossWeighting value")
        if self.mode is LossWeighting.MICROBATCH:
            if self.group_weights is not None:
                raise ValueError("--group-loss-weights is only valid with --loss-weighting group-token")
            return
        if not self.group_weights:
            raise ValueError("--group-loss-weights is required with --loss-weighting group-token")
        weights = dict(self.group_weights)
        if any(type(cycles) is not int or cycles <= 0 for cycles in weights):
            raise ValueError("Group loss weights must use positive cycle counts")
        if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
            raise ValueError("Group loss weights must be finite and nonnegative")
        if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("Group loss weights must sum to one")
        object.__setattr__(self, "group_weights", MappingProxyType(weights))

    @classmethod
    def parse(cls, mode: str, weights: str | None, cycles: tuple[int, ...]):
        mode = LossWeighting(mode)
        parsed = None
        if weights is not None:
            if mode is LossWeighting.MICROBATCH:
                raise ValueError("--group-loss-weights is only valid with --loss-weighting group-token")
            parsed = {}
            for entry in weights.split(","):
                try:
                    cycle_text, weight_text = entry.split(":")
                    cycle, weight = int(cycle_text), float(weight_text)
                except ValueError as exc:
                    raise ValueError("Group loss weights must look like CYCLES:WEIGHT,...") from exc
                if cycle in parsed:
                    raise ValueError(f"Duplicate cycle count in group loss weights: {cycle}")
                parsed[cycle] = weight
        config = cls(mode, parsed)
        config.validate_cycles(cycles)
        return config

    def validate_cycles(self, cycles):
        if self.group_weights is not None and set(self.group_weights) != set(cycles):
            raise ValueError("Group loss weights must match every cycle count in --episode-mix exactly")


@dataclass
class SFTProgress:
    dataset_size: int
    num_iterations: int = -1
    step: int = 0
    consumed: int = 0
    epoch: int = 1

    def __post_init__(self):
        if self.dataset_size <= 0 or (self.num_iterations != -1 and self.num_iterations <= 0):
            raise ValueError("SFT requires a nonempty dataset and a positive update limit or -1 for one epoch")

    @property
    def fraction(self):
        value = self.step / self.num_iterations if self.num_iterations > 0 else self.consumed / self.dataset_size
        return min(max(value, 0.0), 1.0)

    @property
    def done(self):
        return self.step >= self.num_iterations if self.num_iterations > 0 else self.consumed >= self.dataset_size

    def complete_update(self, data_state):
        self.step += 1
        self.consumed = data_state.consumed
        self.epoch = data_state.epoch


def sft_lr_multiplier(progress, warmup_ratio, warmdown_ratio, final_lr_frac):
    if not 0 <= warmup_ratio <= 1 or not 0 <= warmdown_ratio <= 1 or warmup_ratio + warmdown_ratio > 1:
        raise ValueError("SFT warmup and warmdown must be in [0, 1] and sum to at most 1")
    if not 0 <= final_lr_frac <= 1:
        raise ValueError("SFT final learning-rate fraction must be in [0, 1]")
    progress = min(max(progress, 0.0), 1.0)
    if progress < warmup_ratio:
        return min((progress + 1e-8) / warmup_ratio, 1.0)
    if warmdown_ratio == 0 or progress <= 1 - warmdown_ratio:
        return 1.0
    decay = (progress - (1 - warmdown_ratio)) / warmdown_ratio
    return 1 - decay + decay * final_lr_frac

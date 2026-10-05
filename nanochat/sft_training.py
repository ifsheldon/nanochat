"""SFT progress counts completed updates and actually processed conversations."""

from dataclasses import dataclass


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

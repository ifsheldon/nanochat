import pytest
import torch

from nanochat.sft_dataloader import SFTDataState
from nanochat.sft_training import SFTProgress, sft_lr_multiplier


@pytest.mark.parametrize("updates,accumulation", [(1, 1), (1, 4), (5, 1), (5, 4)])
def test_fixed_horizon_performs_optimizer_updates_despite_prefetch_and_epoch_end(updates, accumulation):
    progress = SFTProgress(dataset_size=2, num_iterations=updates)
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    consumed = 0
    prefetched = SFTDataState(consumed=1, epoch=1)
    while not progress.done:
        for _ in range(accumulation):
            (parameter / accumulation).backward()
            processed = prefetched
            consumed += 1
            prefetched = SFTDataState(consumed=consumed + 1, epoch=2)
        optimizer.step()
        optimizer.zero_grad()
        progress.complete_update(processed)
    assert parameter.item() == -updates
    assert consumed == updates * accumulation
    assert progress.fraction == 1


def test_epoch_boundary_finishes_accumulation_and_ignores_prefetch():
    progress = SFTProgress(dataset_size=5)
    completed = []
    rates = []
    states = iter(SFTDataState(consumed=i, epoch=1) for i in range(1, 13))
    prefetched = next(states)
    while not progress.done:
        rates.append(sft_lr_multiplier(progress.fraction, 0, 0.5, 0))
        for _ in range(4):
            processed = prefetched
            completed.append(processed.consumed)
            prefetched = next(states)
        progress.complete_update(processed)
    assert completed == list(range(1, 9))
    assert progress.step == 2
    assert rates[0] == 1 and 0 < rates[1] < 1
    assert progress.consumed == 8 and prefetched.consumed == 9
    assert progress.fraction == 1


def test_excluded_examples_count_only_after_processed_batch():
    progress = SFTProgress(dataset_size=10)
    state = SFTDataState(consumed=12, epoch=2)  # includes filtered examples
    assert not progress.done
    progress.complete_update(state)
    assert progress.done and progress.fraction == 1


@pytest.mark.parametrize("fraction", [-1, 0, 0.5, 1, 2])
def test_learning_rate_stays_bounded(fraction):
    rate = sft_lr_multiplier(fraction, 0.1, 0.5, 0.05)
    assert 0 <= rate <= 1

"""Regression tests for SFT packing, supervision masks, and dataset progress."""

import pytest
import torch
import torch.nn.functional as F

from nanochat.sft_dataloader import sft_data_loader_bos_bestfit


class TokenizedConversations:
    """Use explicit tokens to make packing choices and loss boundaries observable."""

    def get_bos_token_id(self):
        return 0

    def render_conversation(self, conversation, max_tokens=2048):
        ids, mask = conversation
        return ids[:max_tokens], mask[:max_tokens]


def conversation(length, start=1):
    ids = [0, *range(start, start + length - 1)]
    return ids, [0, *([1] * (length - 1))]


def loader(dataset, **kwargs):
    options = dict(batch_size=1, sequence_len=7, device=torch.device("cpu"), buffer_size=len(dataset))
    options.update(kwargs)
    return sft_data_loader_bos_bestfit(dataset, TokenizedConversations(), **options)


def test_natural_fit_precedes_truncated_conversation():
    long = conversation(12, start=20)
    short = conversation(7)
    x, y, _ = next(loader([long, short]))
    assert x[0].tolist() == short[0]
    assert y[0].tolist() == [*short[0][1:], -1]


def test_partial_row_is_padded_instead_of_truncating_another_conversation():
    short = conversation(5)
    long = conversation(12, start=20)
    x, y, state = next(loader([short, long]))
    assert x[0].tolist() == [*short[0], 0, 0]
    assert y[0].tolist() == [*short[0][1:], -1, -1, -1]
    assert state.consumed == 1


def test_empty_row_truncates_the_shortest_oversized_conversation():
    longer = conversation(12, start=20)
    shorter = conversation(10)
    x, y, state = next(loader([longer, shorter]))
    assert x[0].tolist() == shorter[0][:7]
    assert y[0].tolist() == shorter[0][1:8]
    assert state.consumed == 1


def test_packed_conversations_preserve_assistant_masks_and_bos_boundaries():
    doc = ([0, 1, 2, 3], [0, 0, 1, 1])
    x, y, state = next(loader([doc]))
    assert x[0].tolist() == [0, 1, 2, 3, 0, 1, 2]
    assert y[0].tolist() == [-1, 2, 3, -1, -1, 2, 3]
    assert state.consumed == 2


def test_unusable_prefixes_are_excluded_and_counted_as_consumed():
    masked = ([0, 1, 2], [0, 0, 0])
    bos_only = ([0, 1, 2], [1, 0, 0])
    late_reply = (list(range(10)), [0] * 8 + [1, 1])
    usable = conversation(8, start=20)
    x, y, state = next(loader([masked, bos_only, late_reply, usable], buffer_size=1))
    assert x[0].tolist() == usable[0][:-1]
    assert y[0].tolist() == usable[0][1:]
    assert state.consumed == 4


def test_skipped_conversations_count_toward_distributed_progress():
    masked = ([0, 1, 2], [0, 0, 0])
    selected = conversation(8, start=20)
    dataset = [conversation(8), masked, conversation(8), selected, conversation(8), masked]
    x, _, state = next(loader(dataset, rank=1, world_size=2, buffer_size=1))
    assert x[0].tolist() == selected[0][:-1]
    assert state.consumed == 5


@pytest.mark.parametrize("world_size,rank", [(1, 0), (2, 1)])
def test_no_usable_targets_on_rank_raises_instead_of_looping(world_size, rank):
    masked = ([0, 1, 2], [0, 0, 0])
    dataset = [masked, masked] if world_size == 1 else [conversation(8), masked, conversation(8), masked]
    with pytest.raises(ValueError, match=f"SFT rank {rank} has no supervised targets"):
        next(loader(dataset, rank=rank, world_size=world_size))


def test_large_context_preserves_conversations_beyond_2048_tokens():
    doc = conversation(3000)
    x, y, _ = next(loader([doc], sequence_len=4096))
    assert x[0, :3000].tolist() == doc[0]
    assert y[0, :2999].tolist() == doc[0][1:]
    assert (y[0, 2999:] == -1).all()


def test_oversized_buffer_keeps_producing_finite_training_loss():
    batches = loader([conversation(10), conversation(12, start=20)], batch_size=3)
    for _ in range(8):
        _, targets, _ = next(batches)
        assert (targets != -1).any(dim=1).all()
        logits = torch.zeros(*targets.shape, 64)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-1)
        assert loss.isfinite()

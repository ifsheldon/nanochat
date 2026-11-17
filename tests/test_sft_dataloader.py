"""Regression tests for SFT packing, supervision masks, and dataset progress."""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from nanochat.sft_dataloader import (
    EpisodeShape, episode_mix_chunks, parse_episode_mix, sft_data_loader_bos_bestfit, sft_mixed_episode_loader,
)


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


def mixed_loader(dataset, episode_mix, **kwargs):
    options = dict(text_length=4, device=torch.device("cpu"), buffer_size=len(dataset))
    options.update(kwargs)
    return sft_mixed_episode_loader(dataset, TokenizedConversations(), episode_mix=parse_episode_mix(episode_mix), **options)


def test_episode_mix_parses_shapes_shortest_first():
    mix = parse_episode_mix("24x1:2, 4x8:29,12x2:2")
    assert mix == (EpisodeShape(4, 8, 29), EpisodeShape(12, 2, 2), EpisodeShape(24, 1, 2))
    assert episode_mix_chunks(mix) == 1024


@pytest.mark.parametrize("text", ["", "4x8", "4x8:0", "0x8:1", "4x8:1,4x2:1", "4x8:1;12x2:1"])
def test_invalid_episode_mix_is_rejected(text):
    with pytest.raises(ValueError):
        parse_episode_mix(text)


def test_single_length_mix_matches_original_packer_when_every_conversation_fits():
    dataset = [conversation(length, start=10 * index) for index, length in enumerate([3, 9, 5, 2, 7, 4, 6, 8], start=1)]
    original = sft_data_loader_bos_bestfit(dataset, TokenizedConversations(), 2, 8, torch.device("cpu"), buffer_size=4)
    mixed = mixed_loader(dataset, "2x2:1", buffer_size=4)
    for _ in range(6):
        x, y, state = next(original)
        mixed_x, mixed_y, mixed_state = next(mixed)
        assert torch.equal(x, mixed_x) and torch.equal(y, mixed_y)
        assert state == mixed_state


def test_long_conversation_opens_a_longer_episode_and_short_ones_fill_it():
    short, long, other = conversation(5), conversation(12, start=20), conversation(4, start=40)
    batches = mixed_loader([short, long, other], "2x1:1,4x1:1")
    x, y, _ = next(batches)
    assert x.shape == (1, 8)
    assert x[0].tolist() == [*short[0], *other[0]][:8]
    x, y, _ = next(batches)
    assert x.shape == (1, 16)
    assert x[0].tolist() == [*long[0], *short[0]][:16]
    assert y[0].tolist() == [*long[0][1:], -1, *short[0][1:]]


def test_long_conversations_never_displace_short_ones_from_the_buffer():
    long = [conversation(12, start=20 + index) for index in range(6)]
    dataset = [conversation(5), *long, conversation(4, start=40)]
    batches = mixed_loader(dataset, "2x1:3,4x1:1", buffer_size=2)
    for index in range(12):
        x, y, _ = next(batches)
        if index % 4 != 2:
            assert x.shape == (1, 8)
            assert (x[0] == 0).sum() == 2 and not ((x[0] >= 20) & (x[0] < 40)).any()
            assert (y[0] != -1).sum() == 7
        else:
            assert x.shape == (1, 16) and x[0, 1] >= 20


def test_longer_episode_without_a_waiting_long_conversation_uses_short_ones():
    dataset = [conversation(5), conversation(4, start=40)]
    batches = mixed_loader(dataset, "4x1:1,2x1:1")
    x, _, _ = next(batches)
    assert x.shape == (1, 8)
    x, y, _ = next(batches)
    assert x.shape == (1, 16)
    assert x[0].tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 40, 41, 42, 0, 0]
    assert (y[0] != -1).sum() == 11


def test_conversation_beyond_the_longest_episode_is_truncated_or_skipped():
    late_reply = (list(range(100, 130)), [0] * 20 + [1] * 10)
    early_reply = conversation(30, start=50)
    x, y, state = next(mixed_loader([late_reply, early_reply], "4x1:1", buffer_size=1))
    assert x[0].tolist() == early_reply[0][:16]
    assert y[0].tolist() == early_reply[0][1:17]
    assert state.consumed == 2


def test_mix_without_any_fitting_conversation_raises_instead_of_looping():
    with pytest.raises(ValueError, match="no conversation that fits a 2-cycle episode"):
        next(mixed_loader([conversation(12)], "2x1:1,4x1:1"))


def test_arrival_cropping_prevents_buffer_clog_and_matches_single_length_mixed_loader():
    late_reply = ([0, *range(100, 120)], [0] * 12 + [1] * 9)
    dataset = [
        conversation(12, start=10),
        conversation(4, start=30),
        late_reply,
        conversation(11, start=50),
        conversation(5, start=70),
        conversation(3, start=80),
    ]
    cropped = sft_data_loader_bos_bestfit(
        dataset, TokenizedConversations(), batch_size=2, sequence_len=8,
        device=torch.device("cpu"), buffer_size=2, crop_long_conversations=True,
    )
    mixed = mixed_loader(dataset, "2x2:1", buffer_size=2)
    for _ in range(6):
        x, y, state = next(cropped)
        mixed_x, mixed_y, mixed_state = next(mixed)
        assert torch.equal(x, mixed_x) and torch.equal(y, mixed_y)
        assert state == mixed_state


def test_chat_sft_cli_requires_explicit_crop_long_conversations_choice(tmp_path):
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "NANOCHAT_BASE_DIR": str(tmp_path / "run"),
        "NANOCHAT_DATASETS_DIR": str(tmp_path / "datasets"),
    }
    missing = subprocess.run(
        [sys.executable, "-m", "scripts.chat_sft"],
        env=env, capture_output=True, text=True, check=False,
    )
    assert missing.returncode == 2
    assert "--crop-long-conversations/--no-crop-long-conversations" in missing.stderr

    help_out = subprocess.run(
        [sys.executable, "-m", "scripts.chat_sft", "--help"],
        env=env, capture_output=True, text=True, check=True,
    )
    assert "--crop-long-conversations" in help_out.stdout
    assert "--no-crop-long-conversations" in help_out.stdout

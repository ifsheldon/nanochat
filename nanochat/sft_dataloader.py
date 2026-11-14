"""BOS-aligned SFT batches that prefer complete conversations over truncation."""

import itertools
import math
import re
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SFTDataState:
    consumed: int
    epoch: int


@dataclass(frozen=True)
class EpisodeShape:
    """Microbatches of one episode length inside every optimizer update."""

    cycles: int
    batch_size: int
    microbatches: int

    def __post_init__(self):
        for name in ("cycles", "batch_size", "microbatches"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Episode shape {name} must be a positive integer")


def parse_episode_mix(text: str) -> tuple[EpisodeShape, ...]:
    """Parse comma-separated CYCLESxBATCH:MICROBATCHES entries.

    '4x8:29,12x2:2,24x1:2' means that each optimizer update holds 29
    microbatches of 8 four-cycle episodes, 2 of 2 twelve-cycle episodes, and
    2 of 1 twenty-four-cycle episode. One entry selects a single episode
    length. Cycle counts must be distinct; shapes are returned shortest first.
    """
    shapes = []
    for entry in text.split(","):
        match = re.fullmatch(r"\s*(\d+)x(\d+):(\d+)\s*", entry)
        if match is None:
            raise ValueError(f"Episode mix entry must look like CYCLESxBATCH:MICROBATCHES, got {entry!r}")
        shapes.append(EpisodeShape(*map(int, match.groups())))
    return _ordered_shapes(shapes)


def _ordered_shapes(episode_mix) -> tuple[EpisodeShape, ...]:
    shapes = tuple(sorted(episode_mix, key=lambda shape: shape.cycles))
    if not shapes:
        raise ValueError("Episode mix must contain at least one shape")
    if len({shape.cycles for shape in shapes}) != len(shapes):
        raise ValueError("Episode mix cycle counts must be distinct")
    return shapes


def episode_mix_chunks(episode_mix) -> int:
    """Chunks one worker processes per optimizer update; multiply by text_length for text tokens."""
    return sum(shape.cycles * shape.batch_size * shape.microbatches for shape in episode_mix)


def sft_data_loader_bos_bestfit(
    dataset, tokenizer, batch_size: int, sequence_len: int, device: torch.device,
    rank: int = 0, world_size: int = 1, buffer_size: int = 100,
):
    """Yield inputs, assistant-only targets, and dataset progress.

    Pack complete conversations first. Only when none fits an empty row,
    truncate the shortest buffered conversation. Exclude conversations without
    a supervised target in their usable prefix, counting them as consumed.
    """
    if batch_size < 1 or sequence_len < 1 or buffer_size < 1:
        raise ValueError("SFT batch size, sequence length, and buffer size must be positive")
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("SFT rank must be in [0, world_size) with positive world_size")
    dataset_size = len(dataset)
    if dataset_size < world_size:
        raise ValueError("SFT dataset must contain at least one conversation per rank")

    row_capacity = sequence_len + 1
    bos_token = tokenizer.get_bos_token_id()
    conv_buffer = []
    cursor = rank
    consumed = rank
    epoch = 1
    # The cursor advances cyclically by world_size, matching the original loader.
    cycle_length = dataset_size // math.gcd(dataset_size, world_size)
    consecutive_skips = 0

    def refill_buffer():
        nonlocal cursor, consumed, epoch, consecutive_skips
        while len(conv_buffer) < buffer_size:
            # Preserve lengths so truncated conversations cannot outrank natural fits.
            ids, mask = tokenizer.render_conversation(dataset[cursor], max_tokens=None)
            cursor += world_size
            if cursor >= dataset_size:
                cursor %= dataset_size
                epoch += 1

            if not any(mask[1:row_capacity]):
                consumed += world_size
                consecutive_skips += 1
                if consecutive_skips >= cycle_length:
                    raise ValueError(
                        f"SFT rank {rank} has no supervised targets within the "
                        f"{sequence_len}-token context in its dataset cycle"
                    )
                continue
            consecutive_skips = 0
            conv_buffer.append((ids, mask))

    while True:
        rows, mask_rows = [], []
        for _ in range(batch_size):
            row, mask_row = [], []
            while len(row) < row_capacity:
                refill_buffer()
                remaining = row_capacity - len(row)
                best_idx = None
                best_len = 0
                for i, (conv, _) in enumerate(conv_buffer):
                    if best_len < len(conv) <= remaining:
                        best_idx, best_len = i, len(conv)

                if best_idx is not None:
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                    consumed += world_size
                elif not row:
                    # No complete conversation fits even an empty row.
                    shortest_idx = min(range(len(conv_buffer)), key=lambda i: len(conv_buffer[i][0]))
                    conv, conv_mask = conv_buffer.pop(shortest_idx)
                    row.extend(conv[:row_capacity])
                    mask_row.extend(conv_mask[:row_capacity])
                    consumed += world_size
                else:
                    # Keep the remaining conversations intact for subsequent rows.
                    row.extend([bos_token] * remaining)
                    mask_row.extend([0] * remaining)
            rows.append(row)
            mask_rows.append(mask_row)

        use_cuda = device.type == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda).contiguous()
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda).contiguous()
        mask_tensor = torch.tensor(mask_rows, dtype=torch.int8, pin_memory=use_cuda).to(device, non_blocking=use_cuda)
        targets[mask_tensor[:, 1:] == 0] = -1
        yield inputs, targets, SFTDataState(consumed=consumed, epoch=epoch)


def sft_mixed_episode_loader(
    dataset, tokenizer, text_length: int, episode_mix, device: torch.device,
    rank: int = 0, world_size: int = 1, buffer_size: int = 100,
):
    """Yield inputs, assistant-only targets, and progress for mixed episode lengths.

    Each conversation is routed, when it is read, to the shortest episode
    length that holds it whole. A conversation too long for a shorter episode
    therefore never waits in the shorter episodes' buffer. A longer episode
    opens with a waiting conversation that needs its length, when there is
    one, and the rest of it is filled best-fit from that length and all
    shorter ones. A conversation longer than the longest episode is truncated
    to it, or skipped and counted as consumed when no supervised target
    would remain. Conversations of a longer length wait, unbuffered, until an
    episode of that length takes them.

    One optimizer update is one pass over a fixed pattern in which every
    shape contributes its microbatches, evenly interleaved. Inputs have shape
    [batch_size, cycles * text_length]; the caller reads cycles from it.
    """
    if text_length < 1 or buffer_size < 1:
        raise ValueError("SFT text length and buffer size must be positive")
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("SFT rank must be in [0, world_size) with positive world_size")
    dataset_size = len(dataset)
    if dataset_size < world_size:
        raise ValueError("SFT dataset must contain at least one conversation per rank")

    shapes = _ordered_shapes(episode_mix)
    capacities = [shape.cycles * text_length + 1 for shape in shapes]
    order = sorted((
        ((index + 0.5) / shape.microbatches, level)
        for level, shape in enumerate(shapes) for index in range(shape.microbatches)
    ))
    pattern = [level for _, level in order]
    bos_token = tokenizer.get_bos_token_id()
    # waiting[0] is the best-fit buffer of the shortest length; longer levels are unbounded queues.
    waiting = [[] for _ in shapes]
    cursor = rank
    consumed = rank
    epoch = 1
    cycle_length = dataset_size // math.gcd(dataset_size, world_size)

    def refill_buffer():
        nonlocal cursor, consumed, epoch
        skips = 0
        reads_without_short = 0
        while len(waiting[0]) < buffer_size and reads_without_short < cycle_length:
            ids, mask = tokenizer.render_conversation(dataset[cursor], max_tokens=None)
            cursor += world_size
            if cursor >= dataset_size:
                cursor %= dataset_size
                epoch += 1

            ids, mask = ids[:capacities[-1]], mask[:capacities[-1]]
            if not any(mask[1:]):
                consumed += world_size
                skips += 1
                if skips >= cycle_length:
                    raise ValueError(
                        f"SFT rank {rank} has no supervised targets within the "
                        f"{shapes[-1].cycles}-cycle episode in its dataset cycle"
                    )
                continue
            skips = 0
            level = next(i for i, capacity in enumerate(capacities) if len(ids) <= capacity)
            waiting[level].append((ids, mask))
            reads_without_short = 0 if level == 0 else reads_without_short + 1

    def build_row(level):
        nonlocal consumed
        capacity = capacities[level]
        row, mask_row = [], []
        while len(row) < capacity:
            refill_buffer()
            remaining = capacity - len(row)
            opens_long_episode = not row and level > 0 and waiting[level]
            best = None
            for pool in ([level] if opens_long_episode else range(level + 1)):
                for i, (conv, _) in enumerate(waiting[pool]):
                    if len(conv) <= remaining and (best is None or len(conv) > best[0]):
                        best = (len(conv), pool, i)

            if best is not None:
                conv, conv_mask = waiting[best[1]].pop(best[2])
                row.extend(conv)
                mask_row.extend(conv_mask)
                consumed += world_size
            elif not row:
                raise ValueError(
                    f"SFT rank {rank} has no conversation that fits a {shapes[level].cycles}-cycle episode"
                )
            else:
                row.extend([bos_token] * remaining)
                mask_row.extend([0] * remaining)
        return row, mask_row

    use_cuda = device.type == "cuda"
    for level in itertools.cycle(pattern):
        rows, mask_rows = zip(*(build_row(level) for _ in range(shapes[level].batch_size)))
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda).contiguous()
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda).contiguous()
        mask_tensor = torch.tensor(mask_rows, dtype=torch.int8, pin_memory=use_cuda).to(device, non_blocking=use_cuda)
        targets[mask_tensor[:, 1:] == 0] = -1
        yield inputs, targets, SFTDataState(consumed=consumed, epoch=epoch)

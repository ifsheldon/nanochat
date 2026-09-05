"""BOS-aligned SFT batches that prefer complete conversations over truncation."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SFTDataState:
    consumed: int
    epoch: int


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

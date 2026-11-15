"""Assistant-only supervised fine-tuning of recurrent latent memory episodes."""

import time

import torch

from nanochat.rlm_training import require_finite_gradients, synchronize
from nanochat.training_runtime import require_finite_loss


def eom_supervision(targets, text_length, cycles):
    """Supervise a cycle's EOM readout only when its chunk's first token is supervised.

    Each cycle keeps the pretraining layout [memory] [EOM] [text] [summary], and
    the EOM slot predicts the first input token of its chunk. Targets are the
    inputs shifted left with -1 on unsupervised positions, so that first token
    is supervised exactly when the previous chunk's final target is not masked.
    The first chunk starts with BOS, which is never an assistant token.
    """
    if targets.ndim != 2 or targets.size(1) != cycles * text_length:
        raise ValueError("SFT targets must have shape [batch, cycles * text_length]")
    valid = torch.zeros((targets.size(0), cycles), dtype=torch.bool, device=targets.device)
    if cycles > 1:
        boundaries = torch.arange(1, cycles, device=targets.device) * text_length - 1
        valid[:, 1:] = targets[:, boundaries].ne(-1)
    return valid


def sft_episode_loss(forward_cycle, inputs, targets, text_length, cycles):
    """Average the loss over supervised text targets and supervised EOM readouts."""
    if inputs.ndim != 2 or inputs.shape != targets.shape:
        raise ValueError("Episode inputs and targets must have matching [batch, text] shapes")
    if inputs.shape[1] != cycles * text_length:
        raise ValueError("Episode length does not match cycles * text_length")
    eom_valid = eom_supervision(targets, text_length, cycles)
    memory = None
    sums = []
    counts = []
    for cycle in range(cycles):
        start = cycle * text_length
        stop = start + text_length
        chunk_targets = targets[:, start:stop]
        result = forward_cycle(
            inputs[:, start:stop], memory=memory,
            targets=chunk_targets, loss_reduction="none",
            write_memory=cycle < cycles - 1,
        )
        # Masked text positions already contribute zero through ignore_index.
        losses = result.loss.reshape(inputs.shape[0], text_length + 1)
        weight = eom_valid[:, cycle].to(losses.dtype)
        sums.append((losses[:, 0] * weight).sum() + losses[:, 1:].sum())
        counts.append(eom_valid[:, cycle].sum() + chunk_targets.ne(-1).sum())
        memory = result.memory
    cycle_sums = torch.stack(sums)
    cycle_counts = torch.stack(counts)
    return cycle_sums.sum() / cycle_counts.sum(), cycle_sums.detach(), cycle_counts


def sft_update(model, forward_cycle, optimizer, loader, accumulation, step, *, device_batch_size, cycles):
    """Accumulate complete episodes, then apply one optimizer update."""
    text_length = model.config.text_length
    device = model.get_device()
    optimizer.zero_grad(set_to_none=True)
    cycle_sums = torch.zeros(cycles, device=device)
    cycle_counts = torch.zeros(cycles, dtype=torch.int64, device=device)
    data_state = None
    synchronize(device)
    started = time.monotonic()
    for _ in range(accumulation):
        inputs, targets, data_state = next(loader)
        if inputs.shape != (device_batch_size, cycles * text_length):
            raise ValueError("Loader batch does not match the configured episode microbatch")
        loss, sums, counts = sft_episode_loss(forward_cycle, inputs.long(), targets, text_length, cycles)
        require_finite_loss(loss, step)
        (loss / accumulation).backward()
        cycle_sums += sums
        cycle_counts += counts
        del loss, inputs, targets
    require_finite_gradients(model, step)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    elapsed = time.monotonic() - started
    cycle_losses = (cycle_sums / cycle_counts.clamp(min=1)).tolist()
    return {
        "completed_updates": step + 1,
        "loss": (cycle_sums.sum() / cycle_counts.sum()).item(),
        "cycle_losses": cycle_losses,
        "supervised_targets": int(cycle_counts.sum().item()),
        "duration_seconds": elapsed,
    }, data_state


@torch.no_grad()
def evaluate_sft_loss(forward_cycle, loader, batches, text_length, cycles):
    """Mean loss per supervised target over a fixed number of held-out batches."""
    if batches <= 0:
        raise ValueError("Evaluation needs a positive number of batches")
    total = None
    count = None
    for _ in range(batches):
        inputs, targets, _ = next(loader)
        _, sums, counts = sft_episode_loss(forward_cycle, inputs.long(), targets, text_length, cycles)
        total = sums.sum() if total is None else total + sums.sum()
        count = counts.sum() if count is None else count + counts.sum()
    return (total / count).item()

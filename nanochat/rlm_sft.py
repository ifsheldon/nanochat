"""Assistant-only supervised fine-tuning of recurrent latent memory episodes."""

from collections import Counter
import time

import torch

from nanochat.rlm_training import require_finite_gradients, synchronize
from nanochat.sft_training import LossWeighting, SFTLossConfig
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


def _episode_cycles(inputs, targets, text_length):
    if inputs.ndim != 2 or inputs.shape != targets.shape:
        raise ValueError("Episode inputs and targets must have matching [batch, text] shapes")
    if inputs.size(1) == 0 or inputs.size(1) % text_length:
        raise ValueError("Episode width must be a positive multiple of text_length")
    return inputs.size(1) // text_length


def sft_update(model, forward_cycle, optimizer, loader, step, *, episode_mix, loss_config: SFTLossConfig):
    """Stage one mix on CPU, then backpropagate one episode microbatch at a time."""
    text_length = model.config.text_length
    device = model.get_device()
    if not episode_mix:
        raise ValueError("SFT update requires a nonempty episode mix")
    loss_config.validate_cycles(shape.cycles for shape in episode_mix)
    expected = Counter({(shape.cycles, shape.batch_size): shape.microbatches for shape in episode_mix})
    accumulation = sum(expected.values())
    remaining = expected.copy()
    group_targets = {shape.cycles: 0 for shape in episode_mix}
    batches = []
    data_state = None
    synchronize(device)
    started = time.monotonic()
    # Only token tensors are staged. Every forward graph is freed by its own backward.
    for _ in range(accumulation):
        inputs, targets, data_state = next(loader)
        cycles = _episode_cycles(inputs, targets, text_length)
        shape = (cycles, inputs.size(0))
        if remaining[shape] <= 0:
            raise ValueError("Loader batch does not match the configured episode mix")
        remaining[shape] -= 1
        inputs, targets = inputs.cpu(), targets.cpu()
        count = int(targets.ne(-1).sum() + eom_supervision(targets, text_length, cycles).sum())
        if count == 0:
            raise ValueError("SFT microbatch has no supervised targets")
        group_targets[cycles] += count
        batches.append((inputs, targets, cycles, count))

    optimizer.zero_grad(set_to_none=True)
    cycle_sums = {c: torch.zeros(c, device=device) for c in group_targets}
    cycle_counts = {c: torch.zeros(c, dtype=torch.int64, device=device) for c in group_targets}
    objective = torch.zeros((), device=device)
    for cpu_inputs, cpu_targets, cycles, count in batches:
        inputs = cpu_inputs.to(device=device, dtype=torch.long)
        targets = cpu_targets.to(device=device)
        loss, sums, counts = sft_episode_loss(forward_cycle, inputs, targets, text_length, cycles)
        require_finite_loss(loss, step)
        if loss_config.mode is LossWeighting.MICROBATCH:
            scale = 1.0 / accumulation
        else:
            scale = loss_config.group_weights[cycles] * count / group_targets[cycles]
        (loss * scale).backward()
        objective.add_(loss.detach(), alpha=scale)
        cycle_sums[cycles] += sums
        cycle_counts[cycles] += counts
        del loss, inputs, targets, sums, counts
    require_finite_gradients(model, step)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    elapsed = time.monotonic() - started
    total_sum = sum(sums.sum() for sums in cycle_sums.values())
    total_count = sum(counts.sum() for counts in cycle_counts.values())
    groups = {
        str(c): {
            "loss": (cycle_sums[c].sum() / cycle_counts[c].sum()).item(),
            "supervised_targets": int(cycle_counts[c].sum().item()),
            "cycle_losses": (cycle_sums[c] / cycle_counts[c].clamp(min=1)).tolist(),
        }
        for c in group_targets
    }
    return {
        "completed_updates": step + 1,
        "loss": objective.item(),
        "token_loss": (total_sum / total_count).item(),
        "episode_groups": groups,
        "supervised_targets": int(total_count.item()),
        "duration_seconds": elapsed,
    }, data_state


@torch.no_grad()
def evaluate_sft_loss(forward_cycle, loader, batches, text_length):
    """Global supervised-target mean across the requested mixed microbatches."""
    if batches <= 0:
        raise ValueError("Evaluation needs a positive number of batches")
    total = None
    count = None
    for _ in range(batches):
        inputs, targets, _ = next(loader)
        cycles = _episode_cycles(inputs, targets, text_length)
        _, sums, counts = sft_episode_loss(forward_cycle, inputs.long(), targets, text_length, cycles)
        total = sums.sum() if total is None else total + sums.sum()
        count = counts.sum() if count is None else count + counts.sum()
    return (total / count).item()

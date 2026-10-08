"""Checkpoint loading, bounded-context generation, and matched RLM evaluation."""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path

import torch

from nanochat.checkpoint_manager import find_last_step, load_checkpoint
from nanochat.common import get_base_dir
from nanochat.engine import sample_next_token
from nanochat.rlm import CycleOutput, RLMConfig, RecurrentGPT
from nanochat.tokenizer import RustBPETokenizer, get_tokenizer


def tokenizer_fingerprint() -> str:
    """Hash the tokenizer artifact used by the existing tokenizer loader."""
    path = Path(get_base_dir()) / "tokenizer" / "tokenizer.pkl"
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rlm_checkpoint(
    checkpoint_dir: Path, step: int | None, device: torch.device,
) -> tuple[RecurrentGPT, RustBPETokenizer, dict]:
    """Load the exact RLM contract; tokenizer and state mismatches are errors.

    Saved master-weight precision is preserved by assigning checkpoint tensors.
    CPU evaluation promotes reduced-precision weights to FP32 without rounding
    FP32 master weights through the initialization dtype.
    """
    if step is None:
        step = find_last_step(checkpoint_dir)
    if step < 0:
        raise ValueError("Checkpoint step must be nonnegative")
    state, _, metadata = load_checkpoint(checkpoint_dir, step, device)
    if metadata.get("model_type") != "rlm":
        raise ValueError("Checkpoint model_type must be 'rlm'")
    if not isinstance(metadata.get("model_config"), dict):
        raise ValueError("RLM checkpoint must contain model_config")
    if metadata.get("tokenizer_sha256") != tokenizer_fingerprint():
        raise ValueError("RLM checkpoint tokenizer fingerprint does not match tokenizer.pkl")
    config = RLMConfig(**metadata["model_config"])
    tokenizer = get_tokenizer()
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("Tokenizer vocabulary size does not match RLM checkpoint")
    if device.type == "cpu":
        state = {
            key: value.float() if value.dtype in (torch.bfloat16, torch.float16) else value
            for key, value in state.items()
        }
    with torch.device("meta"):
        model = RecurrentGPT(config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    return model, tokenizer, metadata


class RLMGenerator:
    """Generate one sequence, retaining only memory and the current text chunk.

    Each call recomputes the current chunk; no previous raw text or layer cache
    crosses a boundary. A completed chunk writes its summary, and a fresh EOM
    readout from that memory predicts the next chunk's first token.
    """

    def __init__(self, model: RecurrentGPT):
        self.model = model

    @torch.inference_mode()
    def generate(
        self,
        tokens: Sequence[int],
        max_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int = 42,
        stop_token_ids: frozenset[int] = frozenset(),
    ) -> Iterator[int]:
        if self.model.training:
            raise ValueError("RLM generation requires model.eval()")
        if not tokens:
            raise ValueError("RLM generation requires a nonempty prompt")
        if max_tokens < 0:
            raise ValueError("max_tokens must be nonnegative")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when supplied")
        vocab_size = self.model.config.vocab_size
        if any(type(token) is not int or not 0 <= token < vocab_size for token in tokens):
            raise ValueError("Prompt token IDs must be integers within the model vocabulary")
        if any(type(token) is not int or not 0 <= token < vocab_size for token in stop_token_ids):
            raise ValueError("Stop token IDs must be integers within the model vocabulary")
        if max_tokens == 0:
            return

        device = self.model.get_device()
        rng = torch.Generator(device=device).manual_seed(seed)
        text_length = self.model.config.text_length
        memory = None
        start = 0
        # Complete prompt chunks contribute only their compact memory. At an
        # exact boundary the next token comes from a fresh EOM readout.
        while len(tokens) - start >= text_length:
            ids = torch.tensor([tokens[start:start + text_length]], dtype=torch.long, device=device)
            memory = self.model.forward_cycle(ids, memory=memory).memory.clone()
            start += text_length
        current = list(tokens[start:])
        for _ in range(max_tokens):
            ids = torch.tensor([current], dtype=torch.long, device=device)
            output = self.model.forward_cycle(ids, memory=memory)
            if len(current) == text_length:
                # Own just the summary slots, releasing the completed chunk's
                # hidden storage before its memory conditions the EOM readout.
                memory = output.memory.clone()
                current = []
                del output
                ids = torch.empty((1, 0), dtype=torch.long, device=device)
                output = self.model.forward_cycle(ids, memory=memory)
            if output.logits is None:
                raise RuntimeError("RLM inference cycle did not return logits")
            token = sample_next_token(output.logits[:, -1], rng, temperature, top_k).item()
            if token in stop_token_ids:
                return
            current.append(token)
            del output
            yield token


@dataclass(frozen=True)
class PairedDelta:
    episodes: int
    mean: float
    standard_error: float | None


@dataclass(frozen=True)
class CycleLoss:
    cycle: int
    targets: int
    intact: float
    zero: float
    shuffled: float
    zero_minus_intact: PairedDelta
    shuffled_minus_intact: PairedDelta


@dataclass(frozen=True)
class MemoryEvaluation:
    batches: int
    episodes: int
    cycles: tuple[CycleLoss, ...]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["loss_unit"] = "nats_per_text_token"
        result["controls"] = "per_cycle_interventions_on_intact_history"
        result["delta_weighting"] = "equal_weight_per_episode_with_valid_targets"
        result["standard_error_assumption"] = "independent_episodes"
        return result


def _loss_per_episode(output: CycleOutput, targets: torch.Tensor) -> torch.Tensor:
    if output.loss is None:
        raise RuntimeError("RLM evaluation cycle did not return text loss")
    # Evaluation retains the shifted-text metric, excluding the additional EOM
    # training targets, which also predict each cycle's first input token.
    return output.loss.reshape(targets.size(0), targets.size(1) + 1)[:, 1:].double().sum(dim=1)


def _paired_delta(count: int, total: float, square_total: float) -> PairedDelta:
    mean = total / count
    # One observation cannot estimate sampling uncertainty. Roundoff may make
    # the centered sum of squares very slightly negative for constant deltas.
    standard_error = None
    if count > 1:
        centered_sum = max(0.0, square_total - total * mean)
        standard_error = math.sqrt(centered_sum / (count * (count - 1)))
    return PairedDelta(count, mean, standard_error)


@torch.inference_mode()
def evaluate_memory_controls(
    model: RecurrentGPT,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    num_batches: int,
    cycles: int,
    seed: int = 42,
) -> MemoryEvaluation:
    """Score the same rows with intact, zero, and wrong-row incoming memory.

    Each intervention uses the intact trajectory's incoming memory. Its output
    is discarded; only the intact output continues to the next cycle. A random
    cyclic permutation assigns every lane another lane's memory. Cycle one has
    no memory intervention and is reported identically for all three controls.
    Paired deltas average the per-token loss difference within each episode,
    then give each episode equal weight. Standard errors treat episodes as
    independent observations; they are descriptive when rows are correlated.
    """
    if num_batches <= 0 or cycles < 2:
        raise ValueError("Memory evaluation requires positive num_batches and at least two cycles")
    device = model.get_device()
    rng = torch.Generator(device=device).manual_seed(seed)
    loss_sums = torch.zeros((cycles, 3), device=device, dtype=torch.float64)
    counts = torch.zeros(cycles, device=device, dtype=torch.int64)
    delta_sums = torch.zeros((cycles, 2), device=device, dtype=torch.float64)
    delta_square_sums = torch.zeros_like(delta_sums)
    episode_counts = torch.zeros_like(counts)
    episodes = 0
    iterator = iter(batches)
    was_training = model.training
    model.eval()
    try:
        for _ in range(num_batches):
            try:
                inputs, targets = next(iterator)
            except StopIteration as exc:
                raise ValueError("Evaluation loader ended before num_batches") from exc
            if inputs.ndim != 2 or targets.shape != inputs.shape:
                raise ValueError("Evaluation inputs and targets must have matching [batch, text] shapes")
            if inputs.size(0) < 2:
                raise ValueError("Shuffled-memory evaluation requires at least two batch lanes")
            if inputs.size(1) != cycles * model.config.text_length:
                raise ValueError("Evaluation row length must equal cycles * text_length")
            if inputs.device != device or targets.device != device:
                raise ValueError("Evaluation inputs and targets must be on the model device")
            episodes += inputs.size(0)
            memory = None
            for cycle in range(cycles):
                start = cycle * model.config.text_length
                stop = start + model.config.text_length
                text = inputs[:, start:stop]
                target = targets[:, start:stop]
                row_counts = (target != -1).sum(dim=1)
                valid = row_counts > 0
                counts[cycle] += row_counts.sum()
                episode_counts[cycle] += valid.sum()
                output = model.forward_cycle(text, memory=memory, targets=target, loss_reduction="none")
                intact_losses = _loss_per_episode(output, target)
                loss_sums[cycle, 0] += intact_losses.sum()
                if memory is None:
                    loss_sums[cycle, 1:] += intact_losses.sum()
                else:
                    order = torch.randperm(memory.size(0), generator=rng, device=device)
                    permutation = torch.empty_like(order)
                    permutation[order] = order.roll(1)
                    alternatives = (torch.zeros_like(memory), memory[permutation])
                    for column, alternative in enumerate(alternatives, start=1):
                        control = model.forward_cycle(text, memory=alternative, targets=target, loss_reduction="none")
                        control_losses = _loss_per_episode(control, target)
                        loss_sums[cycle, column] += control_losses.sum()
                        deltas = (control_losses - intact_losses)[valid] / row_counts[valid]
                        delta_sums[cycle, column - 1] += deltas.sum()
                        delta_square_sums[cycle, column - 1] += deltas.square().sum()
                memory = output.memory
        if bool((counts == 0).any()):
            raise ValueError("Each evaluated cycle must contain at least one supervised target")
        averages = (loss_sums / counts[:, None]).tolist()
        if any(not math.isfinite(value) for row in averages for value in row):
            raise RuntimeError("RLM evaluation produced non-finite text loss")
        metrics = []
        statistics = zip(counts.tolist(), averages, episode_counts.tolist(), delta_sums.tolist(), delta_square_sums.tolist())
        for index, (count, losses, episode_count, totals, squares) in enumerate(statistics):
            deltas = [_paired_delta(episode_count, total, square) for total, square in zip(totals, squares)]
            metrics.append(CycleLoss(index + 1, count, *losses, *deltas))
        return MemoryEvaluation(num_batches, episodes, tuple(metrics))
    finally:
        model.train(was_training)

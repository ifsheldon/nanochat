"""Episode training, resource measurements, and run contracts for latent memory."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import time

import psutil
import torch

from nanochat.checkpoint_manager import _atomic_write, find_last_step
from nanochat.gpt import GPT, GPTConfig
from nanochat.rlm_policy import InputEmbeddingFreeze
from nanochat.training_runtime import positive_gb, require_finite_loss


@dataclass(frozen=True)
class TrainingConfig:
    """Training settings whose batch units are episodes and real-text targets.

    An episode contains ``cycles`` text chunks with differentiable memory
    between them; memory resets before the next episode. ``device_batch_size``
    counts episodes processed together per worker per accumulation step.
    ``total_batch_size``, exposed as ``--total-batch-token-size``, counts
    real-text targets per optimizer update across all workers and accumulation
    steps. It excludes additional EOM predictions and memory/summary positions.
    """

    cycles: int = 2
    device_batch_size: int = 16
    total_batch_size: int = 1_047_552
    num_iterations: int = 2
    warmup_steps: int = 40
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05
    embedding_lr: float = 0.3
    unembedding_lr: float = 0.008
    matrix_lr: float = 0.02
    scalar_lr: float = 0.5
    weight_decay: float = 0.28
    seed: int = 42
    compile: bool = True
    input_embedding_freeze: InputEmbeddingFreeze = InputEmbeddingFreeze.NONE

    def __post_init__(self):
        object.__setattr__(self, "input_embedding_freeze", InputEmbeddingFreeze(self.input_embedding_freeze))
        for name in ("device_batch_size", "total_batch_size", "num_iterations"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.cycles < 2:
            raise ValueError("Training needs at least two cycles to supervise the memory writer")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        for name in ("warmdown_ratio", "final_lr_frac"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between zero and one")
        for name in ("embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr", "weight_decay"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def accumulation_steps(self, text_length, world_size=1):
        """Derive complete microbatches per worker for one optimizer update.

        total_batch_size = world_size * device_batch_size * accumulation_steps
                           * cycles * text_length

        The budget must be an exact multiple of the global microbatch's text
        targets; episodes are neither split nor rounded to fit the budget.
        """
        microbatch_tokens = self.device_batch_size * self.cycles * text_length * world_size
        if self.total_batch_size % microbatch_tokens:
            raise ValueError(
                f"Global text batch {self.total_batch_size} must be divisible by "
                f"{microbatch_tokens} text targets per global episode microbatch"
            )
        return self.total_batch_size // microbatch_tokens

    def positions_per_episode(self, text_length, memory_tokens):
        """Count Transformer inputs across all cycles, including EOM and memory."""
        # The final cycle has no memory consumer, so it omits the summary suffix.
        return self.cycles * (text_length + 1) + 2 * (self.cycles - 1) * memory_tokens

    def prediction_targets_per_update(self, text_length):
        """Count real-text targets plus one extra EOM prediction per cycle."""
        self.accumulation_steps(text_length)
        return self.total_batch_size + self.total_batch_size // text_length

    @property
    def input_embedding_frozen_updates(self):
        """Number of complete updates that keep the input table fixed."""
        if self.input_embedding_freeze is InputEmbeddingFreeze.PERMANENT:
            return self.num_iterations
        if self.input_embedding_freeze is InputEmbeddingFreeze.SCHEDULED:
            return 9 * self.num_iterations // 10
        return 0

    @property
    def milestone_step(self):
        """Retain the 90% boundary for either pretrained-table freeze policy."""
        return 9 * self.num_iterations // 10 if self.input_embedding_freeze is not InputEmbeddingFreeze.NONE else 0

    def lr_multiplier(self, step):
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        warmdown = round(self.warmdown_ratio * self.num_iterations)
        if warmdown == 0 or step <= self.num_iterations - warmdown:
            return 1.0
        remaining = (self.num_iterations - step) / warmdown
        return remaining + (1 - remaining) * self.final_lr_frac

    def muon_momentum(self, step):
        if step < 400:
            return 0.85 + (0.97 - 0.85) * step / 400
        warmdown = round(self.warmdown_ratio * self.num_iterations)
        if warmdown and step >= self.num_iterations - warmdown:
            progress = (step - self.num_iterations + warmdown) / warmdown
            return 0.97 * (1 - progress) + 0.90 * progress
        return 0.97


def build_optimizer(model, config):
    """Preserve base training's batch/width scaling and d12 decay reference."""
    counts = model.num_scaling_params()
    scaling_params = counts["transformer_matrices"] + counts["lm_head"]
    with torch.device("meta"):
        reference = GPT(GPTConfig(
            vocab_size=model.config.vocab_size, n_layer=12,
            n_head=6, n_kv_head=6, n_embd=768, window_pattern="L",
        ))
    ref_counts = reference.num_scaling_params()
    ref_params = ref_counts["transformer_matrices"] + ref_counts["lm_head"]
    batch_scale = math.sqrt(config.total_batch_size / 524_288)
    decay = config.weight_decay * batch_scale * ref_params / scaling_params
    optimizer = model.setup_optimizer(
        embedding_lr=config.embedding_lr * batch_scale,
        unembedding_lr=config.unembedding_lr * batch_scale,
        matrix_lr=config.matrix_lr * batch_scale,
        scalar_lr=config.scalar_lr * batch_scale,
        weight_decay=decay,
    )
    return optimizer, decay


def apply_schedule(optimizer, config, step, initial_decay):
    multiplier = config.lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * multiplier
        if group["kind"] == "muon":
            group["momentum"] = config.muon_momentum(step)
            group["weight_decay"] = initial_decay * 0.5 * (
                1 + math.cos(math.pi * step / config.num_iterations)
            )
    return multiplier


def apply_input_embedding_schedule(model, config, step):
    frozen = step < config.input_embedding_frozen_updates
    if config.input_embedding_freeze is not InputEmbeddingFreeze.NONE:
        # Fixed groups preserve parameter identities on resume. Adam state is
        # created only when the table first receives a trainable gradient.
        model.transformer.wte.weight.requires_grad_(not frozen)
    return frozen


def episode_loss(forward_cycle, inputs, targets, text_length, cycles, timings=None):
    """Unroll one stable episode buffer with full gradients across every handoff."""
    if inputs.ndim != 2 or inputs.shape != targets.shape:
        raise ValueError("Episode inputs and targets must have matching [batch, text] shapes")
    if inputs.shape[1] != cycles * text_length:
        raise ValueError("Episode length does not match cycles * text_length")
    memory = None
    losses = []
    counts = []
    for cycle in range(cycles):
        start = cycle * text_length
        stop = start + text_length
        chunk_targets = targets[:, start:stop]
        if timings is not None:
            synchronize(inputs.device)
            started = time.monotonic()
            graphs_before = dict(torch._dynamo.utils.counters["stats"])
        result = forward_cycle(
            inputs[:, start:stop], memory=memory,
            targets=chunk_targets, loss_reduction="sum",
            write_memory=cycle < cycles - 1,
        )
        if timings is not None:
            synchronize(inputs.device)
            timings.append({
                "cycle": cycle + 1, "duration_seconds": time.monotonic() - started,
                "compiler_stats_before": graphs_before,
                "compiler_stats_after": dict(torch._dynamo.utils.counters["stats"]),
            })
        losses.append(result.loss)
        # EOM predicts each chunk's first input, even when shifted text labels
        # are masked. The previous chunk's final shifted label stays intact.
        counts.append(chunk_targets.ne(-1).sum() + inputs.shape[0])
        memory = result.memory
    cycle_sums = torch.stack(losses)
    cycle_counts = torch.stack(counts)
    count = cycle_counts.sum()
    return cycle_sums.sum() / count, cycle_sums.detach(), cycle_counts


def require_finite_gradients(model, step, *, frozen_parameter=None):
    checks = []
    for name, parameter in model.named_parameters():
        if parameter is frozen_parameter:
            if parameter.requires_grad or parameter.grad is not None:
                raise RuntimeError(f"Frozen input embedding has trainable state or a gradient at optimizer step {step}; update aborted")
            continue
        if parameter.grad is None:
            raise RuntimeError(f"Missing gradient for {name} at optimizer step {step}; update aborted")
        checks.append(torch.isfinite(parameter.grad).all())
    if not checks or not torch.stack(checks).all().item():
        raise FloatingPointError(f"Non-finite gradients at optimizer step {step}; update aborted")


def require_finite_state(model, optimizer, step):
    tensors = list(model.parameters())
    for state in optimizer.state.values():
        tensors.extend(value for value in state.values() if isinstance(value, torch.Tensor))
    checks = [torch.isfinite(tensor.detach()).all() for tensor in tensors]
    if not torch.stack(checks).all().item():
        raise FloatingPointError(
            f"Non-finite model or optimizer state after update {step + 1}; checkpoint refused"
        )


def continuation_counts(inputs, text_length, cycles, bos_token_id):
    """Count within-document handoffs and continuation targets before the next BOS."""
    later_chunks = inputs[:, text_length:].reshape(inputs.shape[0], cycles - 1, text_length)
    continuation = later_chunks.ne(bos_token_id)
    handoffs = continuation[:, :, 0].sum()
    useful_positions = continuation.to(torch.int64).cumprod(dim=-1).sum()
    return handoffs, useful_positions


def _accumulate_gradients(model, forward_cycle, optimizer, loader, config, step, bos_token_id,
                          memory_monitor, world_size):
    """Finish local forward/backward before any optimizer communication."""
    text_length = model.config.text_length
    accumulation = config.accumulation_steps(text_length, world_size)
    input_embedding = model.transformer.wte.weight
    input_embedding_frozen = apply_input_embedding_schedule(model, config, step)
    optimizer.zero_grad(set_to_none=True)
    cycle_sums = torch.zeros(config.cycles, device=model.get_device())
    cycle_counts = torch.zeros(config.cycles, dtype=torch.int64, device=model.get_device())
    handoffs = torch.zeros((), dtype=torch.int64, device=model.get_device())
    useful_positions = torch.zeros_like(handoffs)
    loader_state = None
    first_episode_timings = []
    for microstep in range(accumulation):
        if memory_monitor is not None:
            memory_monitor.require_available(step)
        inputs, targets, loader_state = next(loader)
        if inputs.shape != (config.device_batch_size, config.cycles * text_length):
            raise ValueError("Loader batch does not match the configured episode microbatch")
        if targets.ne(-1).sum().item() != inputs.numel():
            raise ValueError("Packed pretraining requires one valid shifted target per text input")
        loss, sums, counts = episode_loss(
            forward_cycle, inputs, targets, text_length, config.cycles,
            timings=first_episode_timings if microstep == 0 else None,
        )
        # The complete local phase is coordinated by train_update. An inline
        # reduction here would strand peers when another forward has failed.
        require_finite_loss(loss, step, synchronize_workers=False)
        last_batch_loss = loss.detach()
        if microstep == 0:
            synchronize(inputs.device)
            backward_started = time.monotonic()
        (loss / accumulation).backward()
        if microstep == 0:
            synchronize(inputs.device)
            first_backward_seconds = time.monotonic() - backward_started
        cycle_sums += sums
        cycle_counts += counts
        within_doc, continuation = continuation_counts(inputs, text_length, config.cycles, bos_token_id)
        handoffs += within_doc
        useful_positions += continuation
        # No loader prefetch can occur before the entire unroll's backward above.
        del loss, inputs, targets
    require_finite_gradients(
        model, step, frozen_parameter=input_embedding if input_embedding_frozen else None,
    )
    possible_handoffs = accumulation * config.device_batch_size * (config.cycles - 1)
    cycle_losses = (cycle_sums / cycle_counts).tolist()
    return {
        "input_embedding_frozen": input_embedding_frozen,
        "loss": (cycle_sums.sum() / cycle_counts.sum()).item(),
        "last_batch_loss": last_batch_loss.item(),
        "cycle_losses": cycle_losses,
        **{f"cycle_{cycle}_loss": loss for cycle, loss in enumerate(cycle_losses, start=1)},
        "within_document_handoff_fraction": handoffs.item() / possible_handoffs,
        "continuation_text_fraction": useful_positions.item() / (possible_handoffs * text_length),
        "finite_loss_and_gradients": True,
        "first_episode_cycle_forwards": first_episode_timings,
        "first_episode_backward_seconds": first_backward_seconds,
    }, dict(loader_state)


def train_update(model, forward_cycle, optimizer, loader, config, step, bos_token_id,
                 memory_monitor=None, before_optimizer_step=None, runtime=None):
    """Coordinate worker failures before the optimizer synchronizes gradients."""
    from nanochat.rlm_distributed import DistributedRuntime

    runtime = runtime or DistributedRuntime.single(model.get_device())
    result, loader_state = runtime.collective(
        lambda: _accumulate_gradients(model, forward_cycle, optimizer, loader, config, step,
                                      bos_token_id, memory_monitor, runtime.world_size),
        "forward and backward",
    )

    def admit_optimizer():
        started = time.monotonic()
        if before_optimizer_step is not None:
            before_optimizer_step()
        checkpoint_wait = time.monotonic() - started if before_optimizer_step is not None else 0.0
        if memory_monitor is not None:
            memory_monitor.require_available(step)
        # Allocate lazy moments before workers enter gradient collectives.
        if runtime.world_size > 1:
            optimizer.initialize_state()
        return checkpoint_wait

    result["checkpoint_wait_seconds"] = runtime.collective(admit_optimizer, "prepare optimizer update")
    if runtime.world_size > 1:
        optimizer.step(coordinate=runtime.collective)
    else:
        optimizer.step()

    def finish_update():
        require_finite_state(model, optimizer, step)
        optimizer.zero_grad(set_to_none=True)
        if memory_monitor is not None:
            memory_monitor.require_available(step)

    runtime.collective(finish_update, "validate optimizer update")
    result["finite_model_and_optimizer_state"] = True
    return result, loader_state


class MemoryMonitor:
    """Sample shared-system headroom while CUDA counters measure allocator peaks."""

    def __init__(self, minimum_available_gb=None, interval=0.05, runtime=None):
        self.minimum = (
            positive_gb(minimum_available_gb, "NANOCHAT_MIN_SYSTEM_AVAILABLE_GB")
            if minimum_available_gb is not None else 0
        )
        self.interval = interval
        self.runtime = runtime
        self.minimum_observed = psutil.virtual_memory().available
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        try:
            while not self._stop.wait(self.interval):
                with self._lock:
                    self.minimum_observed = min(self.minimum_observed, psutil.virtual_memory().available)
        except Exception as exc:
            self._error = exc

    def __enter__(self):
        try:
            if self.runtime is None:
                self._thread.start()
            else:
                self.runtime.collective(self._thread.start, "start memory monitor")
        except BaseException:
            self._stop.set()
            if self._thread.is_alive():
                self._thread.join()
            raise
        return self

    def require_available(self, step):
        if self._error is not None:
            raise RuntimeError("System memory sampling failed") from self._error
        with self._lock:
            self.minimum_observed = min(self.minimum_observed, psutil.virtual_memory().available)
        if self.minimum_observed < self.minimum:
            raise RuntimeError(
                f"Insufficient system memory at optimizer step {step}: minimum observed "
                f"{self.minimum_observed / 1e9:.3f} GB, required {self.minimum / 1e9:.3f} GB"
            )

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cuda_metrics(device):
    if device.type != "cuda":
        return {name: 0.0 for name in (
            "cuda_allocated_gb", "cuda_reserved_gb", "cuda_peak_allocated_gb", "cuda_peak_reserved_gb",
        )}
    return {
        "cuda_allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        "cuda_reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "cuda_peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
    }


def measured_update(model, forward_cycle, optimizer, loader, config, step, bos_token_id,
                    minimum_available_gb=None, failure_path: Path | None = None,
                    before_optimizer_step=None, runtime=None):
    from nanochat.rlm_distributed import DistributedRuntime

    runtime = runtime or DistributedRuntime.single(model.get_device())

    def prepare_measurement():
        synchronize(model.get_device())
        if model.get_device().type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.get_device())

    runtime.collective(prepare_measurement, "prepare update measurement")
    graphs_before = dict(torch._dynamo.utils.counters["stats"])
    start = time.monotonic()
    monitor = MemoryMonitor(minimum_available_gb, runtime=runtime)
    try:
        with monitor:
            result, loader_state = train_update(
                model, forward_cycle, optimizer, loader, config, step, bos_token_id, monitor,
                before_optimizer_step=before_optimizer_step, runtime=runtime,
            )
            def finish_measurement():
                synchronize(model.get_device())
                monitor.require_available(step)

            runtime.collective(finish_measurement, "finish update measurement")
    except BaseException:
        if failure_path is not None:
            write_json(failure_path, {
                "attempted_update": step + 1,
                "duration_seconds": time.monotonic() - start,
                "minimum_system_available_gb": monitor.minimum_observed / 1e9,
                "compiler_stats_before": graphs_before,
                "compiler_stats_after": dict(torch._dynamo.utils.counters["stats"]),
                **cuda_metrics(model.get_device()),
            })
        raise
    elapsed = time.monotonic() - start
    positions = config.positions_per_episode(model.config.text_length, model.config.memory_tokens)
    positions *= config.device_batch_size * config.accumulation_steps(model.config.text_length, runtime.world_size) * runtime.world_size
    result.update({
        "completed_updates": step + 1,
        "duration_seconds": elapsed,
        "text_targets": config.total_batch_size,
        "prediction_targets": config.prediction_targets_per_update(model.config.text_length),
        "internal_positions": positions,
        "text_tokens_per_second": config.total_batch_size / elapsed,
        "internal_positions_per_second": positions / elapsed,
        "minimum_system_available_gb": monitor.minimum_observed / 1e9,
        "compiler_stats_before": graphs_before,
        "compiler_stats_after": dict(torch._dynamo.utils.counters["stats"]),
        **cuda_metrics(model.get_device()),
    })
    return runtime.aggregate_metrics(result), loader_state


def write_json(path: Path, value):
    _atomic_write(path, lambda name: Path(name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n"))


def validate_run_directory(root: Path):
    """Managed resume artifacts must not redirect reads or writes through symlinks."""
    paths = [root, *(root / name for name in (
        "run.json", ".run.lock", "result.json", "attempts", "checkpoints",
    ))]
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Managed RLM artifact must not be a symlink: {path}")
    for name in ("attempts", "checkpoints"):
        path = root / name
        if path.exists() and not path.is_dir():
            raise ValueError(f"Managed RLM artifact must be a directory: {path}")
    checkpoints = root / "checkpoints"
    if checkpoints.exists():
        for path in checkpoints.iterdir():
            if path.is_symlink():
                raise ValueError(f"Managed RLM checkpoint must not be a symlink: {path}")


def require_latest_resume_checkpoint(checkpoint_dir: Path, step: int):
    """Admit resume only when it cannot replace an already published later step."""
    from nanochat.rlm_latest import LatestCheckpoint

    try:
        latest = find_last_step(checkpoint_dir)
    except FileNotFoundError:
        latest = 0
    rolling = LatestCheckpoint(checkpoint_dir)
    if rolling.exists():
        latest = max(latest, rolling.metadata()["step"])
    if latest == 0:
        raise FileNotFoundError(f"No complete checkpoints found in {checkpoint_dir}")
    if step != latest:
        raise ValueError(
            f"Resume requires the latest published checkpoint, step {latest}; "
            f"requested step {step}. Published run history must be preserved"
        )


def file_digest(path: Path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def source_manifest(root: Path):
    paths = sorted((root / "nanochat").glob("*.py"))
    paths += [root / "scripts" / "rlm_train.py", root / "pyproject.toml", root / "uv.lock"]
    entries = {str(path.relative_to(root)): file_digest(path) for path in paths}
    fingerprint = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return {"sha256": fingerprint, "files": entries}


def rng_state(device):
    return {
        "cpu": torch.get_rng_state().tolist(),
        "cuda": torch.cuda.get_rng_state(device).tolist() if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    torch.set_rng_state(torch.tensor(state["cpu"], dtype=torch.uint8))
    if device.type == "cuda":
        if state["cuda"] is None:
            raise ValueError("CUDA training checkpoint has no CUDA RNG state")
        torch.cuda.set_rng_state(torch.tensor(state["cuda"], dtype=torch.uint8), device)


def checkpoint_metadata(model, config, step, loader_state, contract, token_counts):
    return {
        "model_type": "rlm",
        "model_config": asdict(model.config),
        "training_config": asdict(config),
        "step": step,
        **token_counts,
        "dataloader_state_dict": loader_state,
        "rng_state": rng_state(model.get_device()),
        "tokenizer_sha256": contract["tokenizer_sha256"],
        "run_contract": contract,
        "memory_contract": {
            "initial": "empty", "reset": "episode", "summary_input": "shared_learned_vector",
            "memory_output": "post_backout_final_rmsnorm", "attention": "full_causal",
            "positions": "reset_each_cycle", "loss": "eom_first_token_and_text_next_token",
            "eom_input": "shared_learned_vector_every_cycle",
            "terminal_summary": "omitted_no_consumer",
            "additional_value_embeddings": "text_only", "gradient_horizon": config.cycles,
        },
        "data_resume": "approximate_row_group",
    }

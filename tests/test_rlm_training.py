"""Behavioral checks for recurrent episode accumulation and update admission."""

import copy
import json

import pytest
import torch
import torch.nn.functional as F

from nanochat.checkpoint_manager import find_last_step, load_checkpoint, save_checkpoint
from nanochat.rlm import RecurrentGPT, RLMConfig
from nanochat.rlm_training import (
    TrainingConfig, episode_loss, measured_update, require_latest_resume_checkpoint,
    train_update, validate_run_directory,
)


def small_model():
    torch.manual_seed(812)
    model = RecurrentGPT(RLMConfig(
        sequence_len=9, text_length=4, memory_tokens=2,
        vocab_size=32, n_layer=2, n_head=2, n_kv_head=2,
        n_embd=32, window_pattern="L",
    ))
    model.init_weights()
    # Contextual gradients must be active to detect a detached memory handoff.
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.05)
            block.mlp.c_proj.weight.normal_(std=0.05)
    return model


def episode_batches(number, batch_size=2):
    generator = torch.Generator().manual_seed(901)
    result = []
    for _ in range(number):
        tokens = torch.randint(1, 32, (batch_size, 9), generator=generator)
        tokens[:, 0] = 0
        result.append((tokens[:, :-1].clone(), tokens[:, 1:].clone()))
    return result


class RecordingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.03)
        self.steps = 0

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)


class ReusingLoader:
    """Match the production loader's in-place reuse of input and target storage."""

    def __init__(self, batches):
        self.batches = batches
        self.inputs = torch.empty_like(batches[0][0])
        self.targets = torch.empty_like(batches[0][1])
        self.state = {}
        self.calls = 0

    def __next__(self):
        if self.calls >= len(self.batches):
            raise AssertionError("Training fetched beyond the accumulation group's last consumed episode")
        inputs, targets = self.batches[self.calls]
        self.inputs.copy_(inputs)
        self.targets.copy_(targets)
        self.calls += 1
        self.state["consumed"] = self.calls
        return self.inputs, self.targets, self.state


def manual_episode_loss(model, inputs, targets):
    first = model.forward_cycle(inputs[:, :4])
    second = model.forward_cycle(inputs[:, 4:], memory=first.memory)
    logits = torch.cat((first.logits, second.logits), dim=1)
    labels = torch.cat((inputs[:, :1], targets[:, :4], inputs[:, 4:5], targets[:, 4:]), dim=1)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-1)


def test_episode_loss_weights_valid_targets_and_preserves_handoff_gradients():
    model = small_model()
    reference = copy.deepcopy(model)
    inputs, targets = episode_batches(1)[0]
    targets[:, :3] = -1  # Four supervised predictions in cycle 1, ten in cycle 2, including EOM.
    loss, cycle_sums, cycle_counts = episode_loss(model.forward_cycle, inputs, targets, 4, 2)
    expected = manual_episode_loss(reference, inputs, targets)
    torch.testing.assert_close(loss, expected)
    assert cycle_counts.tolist() == [4, 10]
    torch.testing.assert_close(loss.detach(), cycle_sums.sum() / 14)
    loss.backward()
    expected.backward()
    assert model.summary_embedding.grad.abs().sum().item() > 0
    for (name, parameter), (_, expected_parameter) in zip(model.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(parameter.grad, expected_parameter.grad, msg=lambda message: f"{name}: {message}")


def test_train_update_finishes_backward_before_reusing_loader_and_consumes_exact_batch():
    model = small_model()
    reference = copy.deepcopy(model)
    batches = episode_batches(3)
    loader = ReusingLoader(batches)
    optimizer = RecordingSGD(model.parameters())
    reference_optimizer = RecordingSGD(reference.parameters())
    config = TrainingConfig(device_batch_size=2, total_batch_size=48, compile=False)

    expected_losses = []
    for inputs, targets in batches:
        loss = manual_episode_loss(reference, inputs, targets)
        expected_losses.append(loss.detach())
        (loss / 3).backward()
    reference_optimizer.step()
    metrics, loader_state = train_update(model, model.forward_cycle, optimizer, loader, config, 0, 0)

    assert loader.calls == 3
    assert optimizer.steps == 1
    assert loader_state == {"consumed": 3}
    loader.state["consumed"] = 99
    assert loader_state["consumed"] == 3  # Returned progress owns its snapshot.
    assert metrics["loss"] == pytest.approx(torch.stack(expected_losses).mean().item(), rel=1e-6)
    for (name, parameter), (_, expected_parameter) in zip(model.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(parameter, expected_parameter, msg=lambda message: f"{name}: {message}")
        assert parameter.grad is None


def test_update_reports_text_and_eom_predictions_separately():
    model = small_model()
    optimizer = RecordingSGD(model.parameters())
    config = TrainingConfig(device_batch_size=2, total_batch_size=16, compile=False)
    metrics, _ = measured_update(
        model, model.forward_cycle, optimizer, ReusingLoader(episode_batches(1)), config, 0, 0,
    )
    # Two rows, two cycles each: 16 text targets plus four EOM predictions.
    assert metrics["text_targets"] == 16
    assert metrics["prediction_targets"] == 20
    # First cycle is [EOM][text4][summary2]; final is [memory2][EOM][text4].
    assert metrics["internal_positions"] == 2 * (7 + 7)
    assert metrics["text_tokens_per_second"] == pytest.approx(16 / metrics["duration_seconds"])


@pytest.mark.parametrize("failure", ["missing_gradient", "nonfinite_gradient", "nonfinite_loss"])
def test_invalid_episode_never_updates_parameters(failure):
    model = small_model()
    if failure == "missing_gradient":
        model.register_parameter("unused_parameter", torch.nn.Parameter(torch.ones(1)))
    elif failure == "nonfinite_gradient":
        model.summary_embedding.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    original_forward = model.forward_cycle

    def forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        if failure == "nonfinite_loss":
            output.loss = output.loss * float("inf")
        return output

    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = RecordingSGD(model.parameters())
    config = TrainingConfig(device_batch_size=2, total_batch_size=16, compile=False)
    error = RuntimeError if failure == "missing_gradient" else FloatingPointError
    with pytest.raises(error, match="[Mm]issing gradient|[Nn]on-finite"):
        train_update(model, forward, optimizer, ReusingLoader(episode_batches(1)), config, 0, 0)
    assert optimizer.steps == 0
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name])


def test_memory_threshold_crossed_during_backward_aborts_before_optimizer():
    class BackwardMemoryMonitor:
        def __init__(self):
            self.backward_finished = False

        def require_available(self, step):
            if self.backward_finished:
                raise RuntimeError("Insufficient system memory after backward")

    model = small_model()
    monitor = BackwardMemoryMonitor()

    def after_backward(gradient):
        monitor.backward_finished = True
        return gradient

    model.summary_embedding.register_hook(after_backward)
    optimizer = RecordingSGD(model.parameters())
    config = TrainingConfig(device_batch_size=2, total_batch_size=16, compile=False)
    with pytest.raises(RuntimeError, match="Insufficient system memory"):
        train_update(model, model.forward_cycle, optimizer, ReusingLoader(episode_batches(1)), config, 0, 0, monitor)
    assert optimizer.steps == 0


def test_nondivisible_global_text_budget_is_rejected_before_loader_consumption():
    model = small_model()
    optimizer = RecordingSGD(model.parameters())
    loader = ReusingLoader(episode_batches(1))
    config = TrainingConfig(device_batch_size=2, total_batch_size=17, compile=False)
    with pytest.raises(ValueError, match="must be divisible"):
        train_update(model, model.forward_cycle, optimizer, loader, config, 0, 0)
    assert loader.calls == 0
    assert optimizer.steps == 0


@pytest.mark.parametrize("managed", ["attempts", "checkpoints", "run.json", ".run.lock", "checkpoints/model_000001.pt"])
def test_resume_rejects_managed_symlinks_before_artifact_writes(tmp_path, managed):
    root = tmp_path / "run"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    redirect = root / managed
    redirect.parent.mkdir(parents=True, exist_ok=True)
    redirect.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="must not be a symlink"):
        validate_run_directory(root)
    assert list(external.iterdir()) == []


def test_failed_update_preserves_measurements_without_applying_an_update(tmp_path):
    model = small_model()
    model.summary_embedding.register_hook(lambda gradient: gradient * float("nan"))
    optimizer = RecordingSGD(model.parameters())
    failure_path = tmp_path / "failed-update.json"
    config = TrainingConfig(device_batch_size=2, total_batch_size=16, compile=False)
    with pytest.raises(FloatingPointError, match="Non-finite gradients"):
        measured_update(
            model, model.forward_cycle, optimizer, ReusingLoader(episode_batches(1)),
            config, 0, 0, failure_path=failure_path,
        )
    measurement = json.loads(failure_path.read_text())
    assert measurement["attempted_update"] == 1
    assert measurement["duration_seconds"] > 0
    assert measurement["minimum_system_available_gb"] > 0
    assert optimizer.steps == 0


def test_optimizer_corruption_is_rejected_before_checkpoint_publication():
    class CorruptingOptimizer(RecordingSGD):
        def step(self, closure=None):
            result = super().step(closure)
            parameter = self.param_groups[0]["params"][0]
            self.state[parameter]["momentum_buffer"] = torch.full_like(parameter, float("nan"))
            return result

    model = small_model()
    optimizer = CorruptingOptimizer(model.parameters())
    config = TrainingConfig(device_batch_size=2, total_batch_size=16, compile=False)
    with pytest.raises(FloatingPointError, match="optimizer state after update 1; checkpoint refused"):
        train_update(model, model.forward_cycle, optimizer, ReusingLoader(episode_batches(1)), config, 0, 0)
    assert optimizer.steps == 1


@pytest.mark.parametrize("later_has_optimizer", [False, True])
def test_resume_preserves_later_published_checkpoint_even_without_optimizer(tmp_path, monkeypatch, later_has_optimizer):
    monkeypatch.delenv("NANOCHAT_CHECKPOINT_RESERVE_GB", raising=False)
    root = tmp_path / "checkpoints"
    model = {"weight": torch.tensor([1.0])}
    optimizer = {"momentum": torch.tensor([2.0])}
    save_checkpoint(root, 2, model, optimizer, {"step": 2})
    save_checkpoint(root, 4, model, optimizer if later_has_optimizer else None, {"step": 4})
    original_files = {path.name: path.read_bytes() for path in root.iterdir()}

    with pytest.raises(ValueError, match="latest published checkpoint, step 4"):
        require_latest_resume_checkpoint(root, 2)
    assert {path.name: path.read_bytes() for path in root.iterdir()} == original_files

    require_latest_resume_checkpoint(root, 4)
    if later_has_optimizer:
        _, loaded_optimizer, metadata = load_checkpoint(root, 4, "cpu", load_optimizer=True)
        torch.testing.assert_close(loaded_optimizer["momentum"], optimizer["momentum"])
        assert metadata["step"] == 4
    else:
        # Admission protects the published model; the normal loader still
        # rejects its incomplete optimizer state instead of rewinding the run.
        with pytest.raises(FileNotFoundError, match="Incomplete checkpoint at step 4"):
            load_checkpoint(root, 4, "cpu", load_optimizer=True)


def test_resume_can_retry_unpublished_later_checkpoint_without_replacing_history(tmp_path, monkeypatch):
    monkeypatch.delenv("NANOCHAT_CHECKPOINT_RESERVE_GB", raising=False)
    root = tmp_path / "checkpoints"
    model = {"weight": torch.tensor([1.0])}
    optimizer = {"momentum": torch.tensor([2.0])}
    save_checkpoint(root, 2, model, optimizer, {"step": 2})
    published_files = {path.name: path.read_bytes() for path in root.iterdir()}
    # An interrupted save left tensors but never published the metadata record.
    torch.save({"partial_weight": torch.tensor([9.0])}, root / "model_000004.pt")
    torch.save({"partial_momentum": torch.tensor([9.0])}, root / "optim_000004_rank0.pt")

    require_latest_resume_checkpoint(root, 2)
    loaded_model, loaded_optimizer, _ = load_checkpoint(root, 2, "cpu", load_optimizer=True)
    save_checkpoint(root, 4, loaded_model, loaded_optimizer, {"step": 4})
    assert find_last_step(root) == 4
    for name, contents in published_files.items():
        assert (root / name).read_bytes() == contents
    completed_model, completed_optimizer, _ = load_checkpoint(root, 4, "cpu", load_optimizer=True)
    torch.testing.assert_close(completed_model["weight"], model["weight"])
    torch.testing.assert_close(completed_optimizer["momentum"], optimizer["momentum"])

"""Capacity admission must measure a complete update with resident optimizer state."""

from copy import deepcopy
import json

import pytest
import torch

import nanochat.common as common_module
import nanochat.gpt as gpt_module
import nanochat.optim as optim_module
import nanochat.rlm as rlm_module
import nanochat.rlm_capacity as capacity_module
from nanochat.rlm import RecurrentGPT, RLMConfig
from nanochat.rlm_capacity import capacity_probe, is_cuda_out_of_memory
from nanochat.rlm_policy import InputEmbeddingFreeze
from nanochat.rlm_training import TrainingConfig
from test_rlm_train_cli import (
    arguments, assert_state_equal, run_attempt, source_arguments, training_environment,
)


@pytest.fixture(autouse=True)
def cpu_execution(monkeypatch):
    for module in (common_module, gpt_module, optim_module, rlm_module):
        monkeypatch.setattr(module, "COMPUTE_DTYPE", torch.float32)
    monkeypatch.delenv("NANOCHAT_MIN_SYSTEM_AVAILABLE_GB", raising=False)
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def small_model():
    torch.manual_seed(73)
    model = RecurrentGPT(RLMConfig(
        sequence_len=9, text_length=4, memory_tokens=2, vocab_size=32,
        n_layer=2, n_head=2, n_kv_head=2, n_embd=32,
        activation_checkpointing=True,
    ))
    model.init_weights()
    return model


class Episodes:
    def __init__(self):
        self.calls = 0

    def __next__(self):
        tokens = (torch.arange(18).reshape(2, 9) + self.calls) % 32
        self.calls += 1
        return tokens[:, :-1], tokens[:, 1:], {"consumed": self.calls}


@pytest.mark.parametrize("initially_frozen", [False, True])
def test_preallocated_state_matches_lazy_optimizer_through_two_updates(initially_frozen):
    eager_model = small_model()
    lazy_model = small_model()
    eager_model.transformer.wte.weight.requires_grad_(not initially_frozen)
    lazy_model.transformer.wte.weight.requires_grad_(not initially_frozen)
    eager = eager_model.setup_optimizer()
    lazy = lazy_model.setup_optimizer()
    before_weights = deepcopy(eager_model.state_dict())
    before_rng = torch.get_rng_state().clone()
    eager.initialize_state()
    assert_state_equal(eager_model.state_dict(), before_weights)
    assert torch.equal(torch.get_rng_state(), before_rng)
    if initially_frozen:
        assert eager_model.transformer.wte.weight not in eager.state
    assert all(parameter.grad is None for parameter in eager_model.parameters())

    for update in range(2):
        if update == 1:
            # Existing moments must survive repeated preallocation, while an
            # embedding unfreeze gets new zero moments without advancing a step.
            existing_state = deepcopy(eager.state_dict()["state"])
            eager_model.transformer.wte.weight.requires_grad_(True)
            lazy_model.transformer.wte.weight.requires_grad_(True)
            eager.initialize_state()
            for key, state in existing_state.items():
                assert_state_equal(eager.state_dict()["state"][key], state)
            if initially_frozen:
                embedding_state = eager.state[eager_model.transformer.wte.weight]
                assert embedding_state["step"] == 0
                assert torch.count_nonzero(embedding_state["exp_avg"]) == 0
                assert torch.count_nonzero(embedding_state["exp_avg_sq"]) == 0

        tokens = (torch.arange(18).reshape(2, 9) + update) % 32
        for model, optimizer in ((eager_model, eager), (lazy_model, lazy)):
            model(tokens[:, :-1], tokens[:, 1:]).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        assert_state_equal(eager_model.state_dict(), lazy_model.state_dict())
        assert_state_equal(eager.state_dict(), lazy.state_dict())

    expected_embedding_steps = 1 if initially_frozen else 2
    assert eager.state[eager_model.transformer.wte.weight]["step"] == expected_embedding_steps
    assert eager.state[eager_model.summary_embedding]["step"] == 2


@pytest.mark.parametrize("policy", [InputEmbeddingFreeze.NONE, InputEmbeddingFreeze.PERMANENT])
def test_capacity_probe_runs_one_full_update_with_state_present_at_first_forward(tmp_path, monkeypatch, policy):
    model = small_model()
    optimizer = model.setup_optimizer()
    config = TrainingConfig(
        device_batch_size=2, total_batch_size=32, num_iterations=10,
        warmup_steps=0, compile=False, input_embedding_freeze=policy,
    )
    loader = Episodes()
    before_weights = deepcopy(model.state_dict())
    embedding = model.transformer.wte.weight
    forward_calls = 0
    optimizer_steps = 0
    ordinary_step = optimizer.step

    def checked_step():
        nonlocal optimizer_steps
        optimizer_steps += 1
        return ordinary_step()

    def checked_forward(*args, **kwargs):
        nonlocal forward_calls
        if forward_calls == 0:
            assert_state_equal(model.state_dict(), before_weights)
            for group in optimizer.param_groups:
                if group["kind"] == "muon":
                    assert "momentum_buffer" in optimizer.state[group["params"][0]]
                    assert "second_momentum_buffer" in optimizer.state[group["params"][0]]
                else:
                    for parameter in group["params"]:
                        if parameter.requires_grad:
                            state = optimizer.state[parameter]
                            assert state["step"] == 0
                            assert "exp_avg" in state and "exp_avg_sq" in state
                        else:
                            assert parameter not in optimizer.state
        forward_calls += 1
        return model.forward_cycle(*args, **kwargs)

    def no_cpu_staging(*args, **kwargs):
        pytest.fail("CPU capacity probes must not allocate CUDA checkpoint staging")

    monkeypatch.setattr(optimizer, "step", checked_step)
    monkeypatch.setattr(capacity_module, "DefaultStager", no_cpu_staging)
    report = capacity_probe(model, checked_forward, optimizer, loader, config, 0.0, 0, tmp_path)
    assert optimizer_steps == 1
    assert loader.calls == config.accumulation_steps(model.config.text_length) == 2
    assert forward_calls == loader.calls * config.cycles
    assert report["completed_updates"] == report["attempt_updates"] == 1
    assert len(report["updates"]) == 1
    assert report["updates"][0]["text_targets"] == 32
    assert report["updates"][0]["prediction_targets"] == 40
    assert report["updates"][0]["finite_model_and_optimizer_state"] is True
    assert report["capacity_probe"]["optimizer_state_preallocated"] is True
    assert report["capacity_probe"]["checkpoint_staging_preallocated"] is False
    frozen = policy is InputEmbeddingFreeze.PERMANENT
    assert report["capacity_probe"]["input_embedding_frozen"] is frozen
    assert report["updates"][0]["input_embedding_frozen"] is frozen
    assert torch.equal(embedding, before_weights["transformer.wte.weight"]) is frozen
    assert not torch.equal(model.summary_embedding, before_weights["summary_embedding"])
    assert not torch.equal(model.eom_embedding, before_weights["eom_embedding"])
    assert all(parameter.grad is None for parameter in model.parameters())
    if frozen:
        assert embedding not in optimizer.state
    else:
        assert optimizer.state[embedding]["step"] == 1


def test_capacity_probe_rejects_scheduled_freezing_before_allocating_state(tmp_path):
    model = small_model()
    optimizer = model.setup_optimizer()
    loader = Episodes()
    config = TrainingConfig(input_embedding_freeze=InputEmbeddingFreeze.SCHEDULED)
    with pytest.raises(ValueError, match="none or permanent"):
        capacity_probe(model, model.forward_cycle, optimizer, loader, config, 0.0, 0, tmp_path)
    assert not optimizer.state
    assert loader.calls == 0


@pytest.mark.parametrize("policy", ["none", "permanent"])
def test_capacity_cli_runs_one_update_without_publishing_checkpoints(training_environment, tmp_path, monkeypatch, policy):
    source_dir, source_tokenizer, _ = training_environment
    output = tmp_path / f"capacity-{policy}"
    args = arguments(
        monkeypatch, output, "--mode", "capacity-probe", "--num-iterations", "100",
        "--total-batch-size", "32", "--input-embedding-freeze", policy,
        *source_arguments(source_dir, source_tokenizer),
    )
    attempt = run_attempt(args, "probe")
    report = json.loads((attempt / "result.json").read_text())
    assert report == json.loads((output / "result.json").read_text())
    updates = [json.loads(line) for line in (attempt / "updates.jsonl").read_text().splitlines()]
    assert len(updates) == 1
    assert updates[0]["completed_updates"] == 1
    assert updates[0]["text_targets"] == 32
    assert report["capacity_probe"]["input_embedding_frozen"] is (policy == "permanent")
    assert not (output / "checkpoints").exists()
    assert not list(output.rglob("*.pt"))


@pytest.mark.parametrize("chain", ["cause", "context"])
def test_capacity_failure_classifies_wrapped_cuda_oom(chain):
    try:
        try:
            raise torch.cuda.OutOfMemoryError("allocation failed")
        except torch.cuda.OutOfMemoryError as inner:
            if chain == "cause":
                raise RuntimeError("compiled backend failed") from inner
            raise RuntimeError("compiled backend failed")
    except RuntimeError as error:
        assert is_cuda_out_of_memory(error)


def test_capacity_failure_does_not_misclassify_other_errors_or_loop_on_cycles():
    error = RuntimeError("CUDA out of memory text is not a typed allocation failure")
    assert not is_cuda_out_of_memory(error)
    error.__context__ = error
    assert not is_cuda_out_of_memory(error)
    assert not is_cuda_out_of_memory(FloatingPointError("non-finite gradient"))

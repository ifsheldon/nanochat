"""The H100 freeze policies preserve the 90% boundary before any unfreezing."""

import json

import pytest
import torch

import nanochat.rlm_training as training_module
from nanochat.rlm import RecurrentGPT
from nanochat.rlm_checkpointing import MilestoneCheckpointError
from nanochat.rlm_latest import LatestCheckpoint
from test_rlm_train_cli import (
    arguments,
    assert_state_equal,
    published_steps,
    run_attempt,
    source_arguments,
    training_environment,
)


@pytest.fixture
def optimizer_initialization(monkeypatch):
    captures = []
    original = training_module.build_optimizer

    def capture(model, config):
        optimizer, decay = original(model, config)
        ids = {
            id(parameter): saved_id
            for group, saved_group in zip(optimizer.param_groups, optimizer.state_dict()["param_groups"])
            for parameter, saved_id in zip(group["params"], saved_group["params"])
        }
        captures.append({
            "model": model,
            "weights": {name: value.detach().clone() for name, value in model.state_dict().items()},
            "parameter_ids": {name: ids[id(parameter)] for name, parameter in model.named_parameters()},
        })
        return optimizer, decay

    monkeypatch.setattr(training_module, "build_optimizer", capture)
    return captures


def updates(attempt):
    return [json.loads(line) for line in (attempt / "updates.jsonl").read_text().splitlines()]




def test_permanent_freeze_keeps_source_table_without_adam_state_while_other_weights_train(
    training_environment, optimizer_initialization, tmp_path, monkeypatch,
):
    source_dir, source_tokenizer, source_embedding = training_environment
    output = tmp_path / "permanent"
    attempt = run_attempt(arguments(
        monkeypatch, output, *source_arguments(source_dir, source_tokenizer),
        "--input-embedding-freeze", "permanent", "--save-every", "1", "--keep-checkpoints", "1",
    ), "complete")
    store = LatestCheckpoint(output / "checkpoints")
    model, optimizer, metadata = store.load("cpu")
    initialization = optimizer_initialization[-1]
    assert_state_equal(model["transformer.wte.weight"], source_embedding)
    assert initialization["parameter_ids"]["transformer.wte.weight"] not in optimizer["state"]
    for name in ("eom_embedding", "summary_embedding", "lm_head.weight"):
        assert not torch.equal(model[name], initialization["weights"][name])
        assert optimizer["state"][initialization["parameter_ids"][name]]["step"] == 4
    assert [item["input_embedding_frozen"] for item in updates(attempt)] == [True] * 4
    assert metadata["training_config"]["input_embedding_freeze"] == "permanent"
    assert store.verify_milestone(3)["step"] == 3
    assert published_steps(output) == [4]


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_scheduled_milestone_keeps_frozen_table_after_unfreezing_and_numbered_pruning(
    training_environment, optimizer_initialization, tmp_path, monkeypatch, mode,
):
    source_dir, source_tokenizer, source_embedding = training_environment
    output = tmp_path / "scheduled"
    store = LatestCheckpoint(output / "checkpoints")
    observed_forward_states = []
    original = RecurrentGPT.forward_cycle

    def record_forward(model, *args, **kwargs):
        trainable = model.transformer.wte.weight.requires_grad
        observed_forward_states.append(trainable)
        if trainable:
            # The barrier must finish before the first trainable forward, not
            # merely before the subsequent optimizer mutation.
            assert store.verify_milestone(3)["milestone_checkpoint"] is True
        return original(model, *args, **kwargs)

    monkeypatch.setattr(RecurrentGPT, "forward_cycle", record_forward)
    attempt = run_attempt(arguments(
        monkeypatch, output, *source_arguments(source_dir, source_tokenizer),
        "--input-embedding-freeze", "scheduled", "--save-every", "1",
        "--keep-checkpoints", "1", "--checkpoint-mode", mode,
    ), "complete")
    final_model, final_optimizer, final_metadata = store.load("cpu")
    milestone = torch.load(store.milestone_path, weights_only=True, map_location="cpu")
    wte_id = optimizer_initialization[-1]["parameter_ids"]["transformer.wte.weight"]
    assert_state_equal(milestone["model"]["transformer.wte.weight"], source_embedding)
    assert wte_id not in milestone["optimizer"]["state"]
    assert milestone["metadata"]["step"] == 3
    assert milestone["metadata"]["milestone_checkpoint"] is True
    assert milestone["metadata"]["training_config"]["input_embedding_freeze"] == "scheduled"
    assert milestone["metadata"]["run_contract"]["checkpoint_policy"]["milestone_step"] == 3
    assert milestone["metadata"]["dataloader_state_dict"] == {"consumed": 3}
    assert not torch.equal(final_model["transformer.wte.weight"], source_embedding)
    assert final_optimizer["state"][wte_id]["step"] == 1
    assert final_metadata["step"] == 4
    assert published_steps(output) == [4]
    assert observed_forward_states == [False] * 6 + [True] * 2
    assert [item["input_embedding_frozen"] for item in updates(attempt)] == [True, True, True, False]


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_failed_milestone_stops_before_unfreezing_and_resume_repairs_exactly(
    training_environment, optimizer_initialization, tmp_path, monkeypatch, mode,
):
    source_dir, source_tokenizer, source_embedding = training_environment
    initialization_args = source_arguments(source_dir, source_tokenizer)
    policy_args = (
        "--input-embedding-freeze", "scheduled", "--save-every", "1",
        "--keep-checkpoints", "1", "--checkpoint-mode", mode,
    )
    reference = tmp_path / "reference"
    run_attempt(arguments(monkeypatch, reference, *initialization_args, *policy_args), "complete")
    expected_model, expected_optimizer, expected_metadata = LatestCheckpoint(reference / "checkpoints").load("cpu")

    output = tmp_path / "interrupted"
    store = LatestCheckpoint(output / "checkpoints")
    forwards = []
    original_forward = RecurrentGPT.forward_cycle

    def record_forward(model, *args, **kwargs):
        forwards.append(model.transformer.wte.weight.requires_grad)
        return original_forward(model, *args, **kwargs)

    def fail_milestone(*args, **kwargs):
        raise OSError("injected retained checkpoint disk failure")

    monkeypatch.setattr(RecurrentGPT, "forward_cycle", record_forward)
    with monkeypatch.context() as patch:
        patch.setattr(LatestCheckpoint, "preserve_milestone", fail_milestone)
        with pytest.raises(MilestoneCheckpointError, match="training stopped before the next update"):
            run_attempt(arguments(monkeypatch, output, *initialization_args, *policy_args), "failed")
    assert forwards == [False] * 6
    failed_model, failed_optimizer, failed_metadata = store.load("cpu")
    assert failed_metadata["step"] == 3
    assert failed_metadata["milestone_checkpoint"] is True
    assert_state_equal(failed_model["transformer.wte.weight"], source_embedding)
    wte_id = optimizer_initialization[-1]["parameter_ids"]["transformer.wte.weight"]
    assert wte_id not in failed_optimizer["state"]
    assert not store.milestone_path.exists()

    # Resume must publish the already-completed boundary before constructing a
    # trainable-table forward, without replaying update 3 or consuming its RNG.
    original_preserve = LatestCheckpoint.preserve_milestone
    repaired = []

    def preserve_before_forward(current, step):
        assert forwards == [False] * 6
        result = original_preserve(current, step)
        repaired.append(step)
        return result

    def resumed_forward(model, *args, **kwargs):
        assert repaired == [3]
        assert store.verify_milestone(3)["step"] == 3
        assert model.transformer.wte.weight.requires_grad
        return record_forward(model, *args, **kwargs)

    monkeypatch.setattr(LatestCheckpoint, "preserve_milestone", preserve_before_forward)
    monkeypatch.setattr(RecurrentGPT, "forward_cycle", resumed_forward)
    resumed = run_attempt(arguments(monkeypatch, output, *policy_args, "--resume-latest"), "resume")
    actual_model, actual_optimizer, actual_metadata = store.load("cpu")
    assert_state_equal(actual_model, expected_model)
    assert_state_equal(actual_optimizer, expected_optimizer)
    assert actual_metadata["rng_state"] == expected_metadata["rng_state"]
    assert actual_metadata["dataloader_state_dict"] == expected_metadata["dataloader_state_dict"] == {"consumed": 4}
    assert repaired == [3]
    assert forwards == [False] * 6 + [True] * 2
    assert len(updates(resumed)) == 1
    assert published_steps(output) == [4]

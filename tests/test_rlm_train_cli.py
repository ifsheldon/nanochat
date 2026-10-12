"""Exercise fresh initialization, durable per-update checkpoints, and resume."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import sys
from threading import Event

import pytest
import torch

import nanochat.checkpoint_manager as checkpoint_module
import nanochat.common as common_module
import nanochat.dataloader as dataloader_module
import nanochat.gpt as gpt_module
import nanochat.rlm as rlm_module
import nanochat.rlm_training as training_module
import nanochat.tokenizer as tokenizer_module
from nanochat.checkpoint_manager import load_checkpoint
from nanochat.gpt import GPT, GPTConfig
from scripts import rlm_train


class TinyTokenizer:
    def get_vocab_size(self):
        return 32

    def get_bos_token_id(self):
        return 0


def stochastic_loader(tokenizer, batch_size, sequence_length, *, split, device, resume_state_dict):
    """Global RNG makes exact continuation depend on restoring the saved state."""
    assert split == "train" and device == "cpu"
    consumed = 0 if resume_state_dict is None else resume_state_dict["consumed"]
    while True:
        tokens = torch.randint(1, tokenizer.get_vocab_size(), (batch_size, sequence_length + 1))
        tokens[:, 0] = tokenizer.get_bos_token_id()
        consumed += 1
        yield tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous(), {"consumed": consumed}


@pytest.fixture
def training_environment(tmp_path, monkeypatch):
    for module in (common_module, gpt_module, rlm_module):
        monkeypatch.setattr(module, "COMPUTE_DTYPE", torch.float32)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("NANOCHAT_MIN_SYSTEM_AVAILABLE_GB", raising=False)
    monkeypatch.delenv("NANOCHAT_CHECKPOINT_RESERVE_GB", raising=False)
    tokenizer_path = tmp_path / "cache" / "tokenizer" / "tokenizer.pkl"
    tokenizer_path.parent.mkdir(parents=True)
    tokenizer_path.write_bytes(b"tiny tokenizer fixture")
    monkeypatch.setattr(tokenizer_module, "get_tokenizer", TinyTokenizer)
    monkeypatch.setattr(dataloader_module, "tokenizing_distributed_data_loader_with_state_bos_bestfit", stochastic_loader)
    # Hold external inventories fixed while exercising the actual trainer,
    # model, optimizer, checkpoint publication, pruning, and resume contracts.
    monkeypatch.setattr(rlm_train, "dataset_inventory", lambda: {"sha256": "fixture-data", "files": []})
    monkeypatch.setattr(training_module, "source_manifest", lambda root: {"sha256": "fixture-source", "files": {}})

    torch.manual_seed(813)
    source = GPT(GPTConfig(
        sequence_len=8, vocab_size=32, n_layer=1, n_head=2,
        n_kv_head=2, n_embd=32, window_pattern="L",
    ))
    source.init_weights()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    torch.save(source.state_dict(), source_dir / "model_000007.pt")
    (source_dir / "meta_000007.json").write_text(json.dumps({
        "step": 7, "model_config": asdict(source.config),
    }))
    source_tokenizer = tmp_path / "source-tokenizer.pkl"
    source_tokenizer.write_bytes(tokenizer_path.read_bytes())
    return source_dir, source_tokenizer, source.transformer.wte.weight.detach().clone()


def arguments(monkeypatch, output_dir, *extra):
    argv = [
        "rlm_train", "--output-dir", str(output_dir), "--mode", "train",
        "--device-type", "cpu", "--depth", "2", "--width", "32", "--head-dim", "16",
        "--text-length", "4", "--memory-tokens", "2", "--cycles", "2",
        "--device-batch-size", "2", "--total-batch-size", "16", "--num-iterations", "4",
        "--warmup-steps", "0", "--no-compile", *map(str, extra),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return rlm_train.parse_args()


def source_arguments(source_dir, source_tokenizer):
    return (
        "--init-embedding-checkpoint-dir", source_dir,
        "--init-embedding-step", "7", "--init-embedding-tokenizer", source_tokenizer,
    )


def run_attempt(args, name):
    attempt = args.output_dir / "attempts" / name
    attempt.mkdir(parents=True)
    rlm_train.run(args, attempt)
    return attempt


def published_steps(output_dir):
    return sorted(int(path.stem.split("_")[1]) for path in (output_dir / "checkpoints").glob("meta_*.json"))


def assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for item, reference in zip(actual, expected):
            assert_state_equal(item, reference)
    else:
        assert actual == expected


def test_fresh_transfer_retention_and_resume_match_uninterrupted_training(training_environment, tmp_path, monkeypatch):
    source_dir, source_tokenizer, source_embedding = training_environment
    init_args = source_arguments(source_dir, source_tokenizer)
    policy_args = ("--save-every", "1", "--keep-checkpoints", "2")

    reference_dir = tmp_path / "reference"
    run_attempt(arguments(monkeypatch, reference_dir, *init_args, *policy_args), "complete")
    expected_model, expected_optimizer, expected_metadata = load_checkpoint(
        reference_dir / "checkpoints", 4, torch.device("cpu"), load_optimizer=True,
    )

    resumed_dir = tmp_path / "resumed"
    first = run_attempt(arguments(
        monkeypatch, resumed_dir, *init_args, *policy_args, "--stop-after-step", "3",
    ), "first")
    assert published_steps(resumed_dir) == [2, 3]
    for prefix in ("model_000001", "optim_000001_rank0", "meta_000001"):
        assert not list((resumed_dir / "checkpoints").glob(f"{prefix}.*"))
    proof = json.loads((first / "embedding-initialization.json").read_text())
    assert proof["source_sha256"] == proof["copied_sha256"]
    saved_contract = json.loads((resumed_dir / "run.json").read_text())["contract"]
    assert saved_contract["initialization"]["embedding_sha256"] == proof["source_sha256"]
    assert proof["trainable"] is True
    paused_model, _, paused_metadata = load_checkpoint(resumed_dir / "checkpoints", 3, torch.device("cpu"))
    assert not torch.equal(paused_model["transformer.wte.weight"], source_embedding)
    assert paused_metadata["dataloader_state_dict"] == {"consumed": 3}

    # Resume must require neither the original checkpoint nor its tokenizer.
    for path in source_dir.iterdir():
        path.unlink()
    source_dir.rmdir()
    source_tokenizer.unlink()
    resumed = run_attempt(arguments(
        monkeypatch, resumed_dir, *policy_args, "--resume-from-step", "3",
    ), "resume")
    assert not (resumed / "embedding-initialization.json").exists()
    assert published_steps(resumed_dir) == [3, 4]
    actual_model, actual_optimizer, actual_metadata = load_checkpoint(
        resumed_dir / "checkpoints", 4, torch.device("cpu"), load_optimizer=True,
    )
    assert_state_equal(actual_model, expected_model)
    assert_state_equal(actual_optimizer, expected_optimizer)
    assert actual_metadata["rng_state"] == expected_metadata["rng_state"]
    assert actual_metadata["dataloader_state_dict"] == {"consumed": 4}
    assert actual_metadata["run_contract"] == saved_contract
    assert not torch.equal(actual_model["transformer.wte.weight"], paused_model["transformer.wte.weight"])
    result = json.loads((resumed / "result.json").read_text())
    assert result["status"] == "completed" and result["attempt_updates"] == 1


def test_frozen_pretrained_embedding_resumes_exactly_before_and_after_unfreezing(
    training_environment, tmp_path, monkeypatch, caplog,
):
    from nanochat.rlm_latest import LatestCheckpoint

    caplog.set_level(logging.INFO, logger=rlm_train.logger.name)
    source_dir, source_tokenizer, source_embedding = training_environment
    init_args = source_arguments(source_dir, source_tokenizer)
    policy_args = ("--input-embedding-freeze", "scheduled", "--num-iterations", "20", "--total-batch-size", "32")
    initialized_models = []
    parameter_ids = []
    build_optimizer = training_module.build_optimizer

    def capture_initialization(model, config):
        optimizer, decay = build_optimizer(model, config)
        initialized_models.append({name: value.detach().clone() for name, value in model.state_dict().items()})
        saved_groups = optimizer.state_dict()["param_groups"]
        ids = {
            id(parameter): saved_id
            for group, saved_group in zip(optimizer.param_groups, saved_groups)
            for parameter, saved_id in zip(group["params"], saved_group["params"])
        }
        parameter_ids.append({name: ids[id(parameter)] for name, parameter in model.named_parameters()})
        return optimizer, decay

    def assert_logged_updates(attempt):
        records = [record for record in caplog.records if record.name == rlm_train.logger.name]
        events = [json.loads(record.getMessage()) for record in records if record.getMessage().startswith("{")]
        logged_updates = [item for item in events if "completed_updates" in item and "loss" in item]
        progress = [record for record in records if record.getMessage().startswith("Update ")]
        updates = [json.loads(line) for line in (attempt / "updates.jsonl").read_text().splitlines()]
        report = json.loads((attempt / "result.json").read_text())
        assert logged_updates == updates == report["updates"]
        assert len(progress) == len(updates)
        assert all(record.levelno == logging.INFO for record in records)
        assert [record.getMessage() for record in progress] == [
            f"Update {update['completed_updates']}/20: {update['duration_seconds']:.2f} s"
            for update in updates
        ]
        for update in updates:
            assert update["duration_seconds"] > 0
            assert isinstance(update["last_batch_loss"], float)
            assert update["cycle_1_loss"] == update["cycle_losses"][0]
            assert update["cycle_2_loss"] == update["cycle_losses"][1]
            assert update["loss"] == pytest.approx(sum(update["cycle_losses"]) / 2, rel=1e-6)
            assert update["input_embedding_frozen"] is (update["completed_updates"] <= 18)
        caplog.clear()
        return updates

    monkeypatch.setattr(training_module, "build_optimizer", capture_initialization)
    reference_dir = tmp_path / "frozen-reference"
    reference_attempt = run_attempt(arguments(monkeypatch, reference_dir, *init_args, *policy_args), "complete")
    reference_updates = assert_logged_updates(reference_attempt)
    assert any(abs(item["last_batch_loss"] - item["loss"]) > 1e-5 for item in reference_updates)
    expected_model, expected_optimizer, expected_metadata = LatestCheckpoint(
        reference_dir / "checkpoints",
    ).load(torch.device("cpu"))

    resumed_dir = tmp_path / "frozen-resumed"
    latest = LatestCheckpoint(resumed_dir / "checkpoints")
    first = run_attempt(arguments(
        monkeypatch, resumed_dir, *init_args, *policy_args, "--stop-after-step", "18",
    ), "frozen")
    frozen_updates = assert_logged_updates(first)
    assert len(frozen_updates) == 18 and all(item["input_embedding_frozen"] for item in frozen_updates)
    frozen_model, frozen_optimizer, frozen_metadata = latest.load(torch.device("cpu"))
    assert_state_equal(frozen_model["transformer.wte.weight"], source_embedding)
    wte_id = parameter_ids[-1]["transformer.wte.weight"]
    assert wte_id not in frozen_optimizer["state"]
    for name in ("eom_embedding", "summary_embedding"):
        assert not torch.equal(frozen_model[name], initialized_models[-1][name])
        assert frozen_optimizer["state"][parameter_ids[-1][name]]["step"] == 18
    saved_contract = json.loads((resumed_dir / "run.json").read_text())["contract"]
    assert saved_contract["training_config"]["input_embedding_freeze"] == "scheduled"
    assert saved_contract["training_config"]["num_iterations"] == 20
    assert frozen_metadata["run_contract"] == saved_contract
    assert frozen_metadata["dataloader_state_dict"] == {"consumed": 36}

    # Continuation restores checkpoint provenance and the full-horizon schedule.
    for path in source_dir.iterdir():
        path.unlink()
    source_dir.rmdir()
    source_tokenizer.unlink()
    crossing = run_attempt(arguments(
        monkeypatch, resumed_dir, *policy_args, "--resume-latest", "--stop-after-step", "19",
    ), "cross-boundary")
    assert len(assert_logged_updates(crossing)) == 1
    unfrozen_model, unfrozen_optimizer, _ = latest.load(torch.device("cpu"))
    assert not torch.equal(unfrozen_model["transformer.wte.weight"], source_embedding)
    assert unfrozen_optimizer["state"][wte_id]["step"] == 1
    assert not (crossing / "embedding-initialization.json").exists()

    final = run_attempt(arguments(
        monkeypatch, resumed_dir, *policy_args, "--resume-latest",
    ), "after-boundary")
    assert len(assert_logged_updates(final)) == 1
    actual_model, actual_optimizer, actual_metadata = latest.load(torch.device("cpu"))
    assert_state_equal(actual_model, expected_model)
    assert_state_equal(actual_optimizer, expected_optimizer)
    assert actual_optimizer["state"][wte_id]["step"] == 2
    assert actual_metadata["rng_state"] == expected_metadata["rng_state"]
    assert actual_metadata["dataloader_state_dict"] == {"consumed": 40}
    assert actual_metadata["run_contract"] == saved_contract


def test_resume_rejects_freezing_random_initialization(training_environment, tmp_path, monkeypatch):
    from nanochat.rlm_latest import LatestCheckpoint

    output_dir = tmp_path / "random-freeze"
    run_attempt(arguments(monkeypatch, output_dir, "--stop-after-step", "1"), "random")
    latest = LatestCheckpoint(output_dir / "checkpoints")
    previous_metadata = latest.metadata()
    with pytest.raises(ValueError, match="pretrained"):
        run_attempt(arguments(
            monkeypatch, output_dir, "--resume-latest", "--input-embedding-freeze", "scheduled",
        ), "reject-freeze")
    assert latest.metadata() == previous_metadata


@pytest.mark.parametrize("initially_frozen", [False, True])
def test_resume_requires_matching_embedding_freeze_schedule(
    training_environment, tmp_path, monkeypatch, initially_frozen,
):
    from nanochat.rlm_latest import LatestCheckpoint

    source_dir, source_tokenizer, _ = training_environment
    output_dir = tmp_path / "freeze-mismatch"
    freeze_args = ("--input-embedding-freeze", "scheduled",) if initially_frozen else ()
    run_attempt(arguments(
        monkeypatch, output_dir, *source_arguments(source_dir, source_tokenizer), *freeze_args,
        "--num-iterations", "10", "--stop-after-step", "1",
    ), "original")
    latest = LatestCheckpoint(output_dir / "checkpoints")
    previous_metadata = latest.metadata()
    changed_args = () if initially_frozen else ("--input-embedding-freeze", "scheduled",)
    with pytest.raises(ValueError, match="matching|training|freeze"):
        run_attempt(arguments(
            monkeypatch, output_dir, *changed_args, "--num-iterations", "10", "--resume-latest",
        ), "reject-changed-schedule")
    assert latest.metadata() == previous_metadata


def test_training_saves_update_two_before_final_without_periodic_saves(training_environment, tmp_path, monkeypatch):
    output_dir = tmp_path / "early-checkpoint"
    attempt = run_attempt(arguments(monkeypatch, output_dir, "--save-every", "0"), "complete")
    assert published_steps(output_dir) == [2, 4]
    assert not (attempt / "embedding-initialization.json").exists()
    run_contract = json.loads((output_dir / "run.json").read_text())["contract"]
    assert run_contract["initialization"] == {"kind": "random"}
    measurements = json.loads((attempt / "checkpoint-measurements.json").read_text())
    assert [item["step"] for item in measurements] == [1, 2, 3, 4]
    assert all(item["latest_duration_seconds"] > 0 for item in measurements)
    assert [item["step"] for item in measurements if item["numbered_duration_seconds"] > 0] == [2, 4]


@pytest.mark.parametrize("checkpoint_mode", ["sync", "async"])
@pytest.mark.parametrize("interrupted_after", [1, 3])
def test_latest_saves_every_update_and_resumes_interruption_exactly(training_environment, tmp_path, monkeypatch, interrupted_after, checkpoint_mode):
    from nanochat.rlm_latest import LatestCheckpoint

    policy_args = ("--keep-checkpoints", "1", "--checkpoint-mode", checkpoint_mode)
    reference_dir = tmp_path / "reference"
    run_attempt(arguments(monkeypatch, reference_dir, *policy_args), "complete")
    expected_model, expected_optimizer, expected_metadata = LatestCheckpoint(
        reference_dir / "checkpoints",
    ).load(torch.device("cpu"))

    resumed_dir = tmp_path / "resumed"
    latest = LatestCheckpoint(resumed_dir / "checkpoints")
    saved_steps = []
    save = LatestCheckpoint.save
    measured_update = training_module.measured_update

    def record_save(self, model, optimizer, metadata):
        published = save(self, model, optimizer, metadata)
        saved_steps.append(self.metadata()["step"])
        return published

    def interrupt_next_update(model, forward_cycle, optimizer, loader, config, step, *args, **kwargs):
        # Sync publication finishes before the next forward. Async publication
        # may still be in flight and is drained on the interruption path.
        if checkpoint_mode == "sync":
            assert saved_steps == list(range(1, step + 1))
        if step and checkpoint_mode == "sync":
            saved_model, _, metadata = latest.load(torch.device("cpu"))
            assert metadata["step"] == step
            assert metadata["dataloader_state_dict"] == {"consumed": step}
            assert_state_equal(saved_model, model.state_dict())
        if step == interrupted_after:
            raise KeyboardInterrupt("simulated interruption before next update")
        return measured_update(model, forward_cycle, optimizer, loader, config, step, *args, **kwargs)

    with monkeypatch.context() as interrupted:
        interrupted.setattr(LatestCheckpoint, "save", record_save)
        interrupted.setattr(training_module, "measured_update", interrupt_next_update)
        with pytest.raises(KeyboardInterrupt, match="before next update"):
            run_attempt(arguments(monkeypatch, resumed_dir, *policy_args), "interrupted")

    assert saved_steps == list(range(1, interrupted_after + 1))
    assert latest.metadata()["step"] == interrupted_after
    assert published_steps(resumed_dir) == ([2] if interrupted_after == 3 else [])

    # A numbered checkpoint cannot rewind the newer completed update saved
    # only in latest, even though it is the newest numbered checkpoint.
    if interrupted_after == 3:
        with pytest.raises(ValueError, match="latest|published"):
            run_attempt(arguments(
                monkeypatch, resumed_dir, *policy_args, "--resume-from-step", "2",
            ), "reject-rewind")

    resumed = run_attempt(arguments(
        monkeypatch, resumed_dir, *policy_args, "--resume-latest",
    ), "resume")
    actual_model, actual_optimizer, actual_metadata = latest.load(torch.device("cpu"))
    assert_state_equal(actual_model, expected_model)
    assert_state_equal(actual_optimizer, expected_optimizer)
    assert actual_metadata["step"] == 4
    assert actual_metadata["rng_state"] == expected_metadata["rng_state"]
    assert actual_metadata["dataloader_state_dict"] == {"consumed": 4}
    assert actual_metadata["run_contract"] == expected_metadata["run_contract"]
    assert published_steps(resumed_dir) == [4]
    result = json.loads((resumed / "result.json").read_text())
    assert result["status"] == "completed" and result["attempt_updates"] == 4 - interrupted_after


@pytest.mark.parametrize("checkpoint_mode", ["sync", "async"])
def test_latest_save_failure_warns_and_next_update_recovers(training_environment, tmp_path, monkeypatch, caplog, checkpoint_mode):
    from nanochat.rlm_latest import LatestCheckpoint

    caplog.set_level(logging.INFO, logger="nanochat.rlm_checkpointing")
    reference_dir = tmp_path / "reference"
    run_attempt(arguments(
        monkeypatch, reference_dir, "--checkpoint-mode", checkpoint_mode, "--num-iterations", "3",
    ), "complete")
    expected_model, expected_optimizer, expected_metadata = LatestCheckpoint(
        reference_dir / "checkpoints",
    ).load(torch.device("cpu"))
    caplog.clear()
    output_dir = tmp_path / "save-failure"
    latest = LatestCheckpoint(output_dir / "checkpoints")
    save = LatestCheckpoint.save
    measured_update = training_module.measured_update
    attempted_updates = []

    def fail_second_save(self, model, optimizer, metadata):
        if metadata["step"] == 2:
            raise OSError("simulated checkpoint write failure")
        return save(self, model, optimizer, metadata)

    def record_update(model, forward_cycle, optimizer, loader, config, step, *args, **kwargs):
        attempted_updates.append(step + 1)
        return measured_update(model, forward_cycle, optimizer, loader, config, step, *args, **kwargs)

    monkeypatch.setattr(LatestCheckpoint, "save", fail_second_save)
    monkeypatch.setattr(training_module, "measured_update", record_update)
    attempt = run_attempt(arguments(
        monkeypatch, output_dir, "--checkpoint-mode", checkpoint_mode, "--num-iterations", "3",
    ), "complete")
    assert attempted_updates == [1, 2, 3]
    actual_model, actual_optimizer, actual_metadata = latest.load(torch.device("cpu"))
    assert actual_metadata["step"] == 3
    assert actual_metadata["dataloader_state_dict"] == {"consumed": 3}
    assert_state_equal(actual_model, expected_model)
    assert_state_equal(actual_optimizer, expected_optimizer)
    assert actual_metadata["rng_state"] == expected_metadata["rng_state"]
    result = json.loads((attempt / "result.json").read_text())
    assert result["status"] == "completed" and result["attempt_updates"] == 3
    measurements = json.loads((attempt / "checkpoint-measurements.json").read_text())
    assert [item["step"] for item in measurements] == [1, 2, 3]
    assert measurements[1]["latest_published"] is False
    assert measurements[1]["latest_step"] == 1
    assert measurements[2]["latest_published"] is True
    assert measurements[2]["latest_step"] == 3
    warnings = [record for record in caplog.records if record.getMessage().startswith('{"checkpoint_warning":')]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING
    assert "simulated checkpoint write failure" in warnings[0].getMessage()
    published = [json.loads(record.getMessage())["checkpoint_latest"] for record in caplog.records
                 if record.getMessage().startswith('{"checkpoint_latest":')]
    assert [item["step"] for item in published] == [1, 3]


@pytest.mark.parametrize("checkpoint_mode", ["sync", "async"])
@pytest.mark.parametrize("first_failed_step", [1, 2])
def test_repeated_checkpoint_failures_report_missing_or_stale_latest(training_environment, tmp_path, monkeypatch, caplog, checkpoint_mode, first_failed_step):
    from nanochat.rlm_latest import LatestCheckpoint

    output_dir = tmp_path / "stale-latest"
    save = LatestCheckpoint.save

    def fail_after_first(self, model, optimizer, metadata):
        if metadata["step"] >= first_failed_step:
            raise OSError("simulated persistent checkpoint failure")
        return save(self, model, optimizer, metadata)

    monkeypatch.setattr(LatestCheckpoint, "save", fail_after_first)
    attempt = run_attempt(arguments(
        monkeypatch, output_dir, "--checkpoint-mode", checkpoint_mode, "--num-iterations", "3",
    ), "complete")
    latest = LatestCheckpoint(output_dir / "checkpoints")
    if first_failed_step == 1:
        assert not latest.exists()
    else:
        assert latest.metadata()["step"] == 1
    result = json.loads((attempt / "result.json").read_text())
    assert result["status"] == "completed" and result["completed_updates"] == 3
    expected_latest_step = None if first_failed_step == 1 else 1
    expected_lag = None if first_failed_step == 1 else 2
    for status in (result["checkpoint_measurements"][-1], result["checkpoint_status"]):
        assert status["latest_step"] == expected_latest_step
        assert status["latest_lag_completed_updates"] == expected_lag
    warnings = [record for record in caplog.records if record.getMessage().startswith('{"checkpoint_warning":')]
    assert len(warnings) == 4 - first_failed_step
    assert all(record.levelno == logging.WARNING for record in warnings)


@pytest.mark.parametrize("checkpoint_mode", ["sync", "async"])
def test_numbered_save_failure_preserves_latest_and_recovers(training_environment, tmp_path, monkeypatch, caplog, checkpoint_mode):
    import nanochat.rlm_checkpoints as retention_module
    from nanochat.rlm_latest import LatestCheckpoint

    output_dir = tmp_path / "numbered-failure"
    replace = retention_module.os.replace
    failure_injected = False

    def fail_second_optimizer_publication(source, target, *args, **kwargs):
        nonlocal failure_injected
        if Path(target).name == "optim_000002_rank0.pt" and not failure_injected:
            failure_injected = True
            # The older numbered set survives while the new publication is incomplete.
            assert checkpoint_module.checkpoint_is_complete(output_dir / "checkpoints", 1, load_optimizer=True)
            raise OSError("simulated numbered checkpoint write failure")
        return replace(source, target, *args, **kwargs)

    monkeypatch.setattr(retention_module.os, "replace", fail_second_optimizer_publication)
    attempt = run_attempt(arguments(
        monkeypatch, output_dir, "--checkpoint-mode", checkpoint_mode,
        "--save-every", "1", "--keep-checkpoints", "1", "--num-iterations", "3",
    ), "complete")
    assert LatestCheckpoint(output_dir / "checkpoints").metadata()["step"] == 3
    assert published_steps(output_dir) == [3]
    assert checkpoint_module.checkpoint_is_complete(output_dir / "checkpoints", 3, load_optimizer=True)
    result = json.loads((attempt / "result.json").read_text())
    assert result["completed_updates"] == 3
    assert result["checkpoint_measurements"][1]["latest_published"] is True
    assert result["checkpoint_measurements"][1]["numbered_published"] is False
    assert any(record.levelno == logging.WARNING and "simulated numbered checkpoint write failure" in record.getMessage()
               for record in caplog.records)


def test_latest_resume_repairs_interrupted_numbered_checkpoint(training_environment, tmp_path, monkeypatch):
    import nanochat.rlm_checkpoints as retention_module
    from nanochat.rlm_latest import LatestCheckpoint

    output_dir = tmp_path / "numbered-interruption"
    checkpoint_dir = output_dir / "checkpoints"
    latest = LatestCheckpoint(checkpoint_dir)
    replace = retention_module.os.replace

    def interrupt_optimizer_publication(source, target, *args, **kwargs):
        if Path(target).name == "optim_000002_rank0.pt":
            raise KeyboardInterrupt("simulated interruption during numbered checkpoint")
        return replace(source, target, *args, **kwargs)

    with monkeypatch.context() as interrupted:
        interrupted.setattr(retention_module.os, "replace", interrupt_optimizer_publication)
        with pytest.raises(KeyboardInterrupt, match="during numbered checkpoint"):
            run_attempt(arguments(monkeypatch, output_dir), "interrupted")

    paused_model, paused_optimizer, paused_metadata = latest.load(torch.device("cpu"))
    assert paused_metadata["step"] == 2
    assert (checkpoint_dir / "model_000002.pt").is_file()
    assert not checkpoint_module.checkpoint_is_complete(checkpoint_dir, 2, load_optimizer=True)
    measured_update = training_module.measured_update
    resumed_steps = []

    def require_repaired_checkpoint(model, forward_cycle, optimizer, loader, config, step, *args, **kwargs):
        if not resumed_steps:
            assert step == 2
            repaired_model, repaired_optimizer, repaired_metadata = load_checkpoint(
                checkpoint_dir, 2, torch.device("cpu"), load_optimizer=True,
            )
            assert_state_equal(repaired_model, paused_model)
            assert_state_equal(repaired_optimizer, paused_optimizer)
            assert repaired_metadata["rng_state"] == paused_metadata["rng_state"]
        resumed_steps.append(step + 1)
        return measured_update(model, forward_cycle, optimizer, loader, config, step, *args, **kwargs)

    monkeypatch.setattr(training_module, "measured_update", require_repaired_checkpoint)
    run_attempt(arguments(monkeypatch, output_dir, "--resume-latest"), "resume")
    assert resumed_steps == [3, 4]
    assert published_steps(output_dir) == [2, 4]
    assert latest.metadata()["step"] == 4


@pytest.mark.parametrize("interrupt_stage", ["before_pruning", "after_metadata_unlink"])
def test_latest_resume_finishes_retention_before_next_update(training_environment, tmp_path, monkeypatch, interrupt_stage):
    import nanochat.rlm_checkpointing as writer_module
    import nanochat.rlm_checkpoints as retention_module
    from nanochat.rlm_latest import LatestCheckpoint

    output_dir = tmp_path / "retention-interruption"
    checkpoint_dir = output_dir / "checkpoints"
    latest = LatestCheckpoint(checkpoint_dir)
    policy_args = ("--save-every", "1", "--keep-checkpoints", "1")
    prune_checkpoints = retention_module.prune_checkpoints
    unlink = retention_module.os.unlink

    def interrupt_second_prune(directory, keep_last, published_step):
        if published_step == 2:
            assert checkpoint_module.checkpoint_is_complete(directory, 2, load_optimizer=True)
            raise KeyboardInterrupt("simulated interruption before pruning")
        return prune_checkpoints(directory, keep_last, published_step)

    def interrupt_after_metadata_unlink(path, *args, **kwargs):
        result = unlink(path, *args, **kwargs)
        if Path(path).name == "meta_000001.json":
            raise KeyboardInterrupt("simulated interruption after metadata unlink")
        return result

    with monkeypatch.context() as interrupted:
        if interrupt_stage == "before_pruning":
            interrupted.setattr(writer_module, "prune_checkpoints", interrupt_second_prune)
        else:
            interrupted.setattr(retention_module.os, "unlink", interrupt_after_metadata_unlink)
        with pytest.raises(KeyboardInterrupt, match="simulated interruption"):
            run_attempt(arguments(monkeypatch, output_dir, *policy_args), "interrupted")

    assert latest.metadata()["step"] == 2
    assert checkpoint_module.checkpoint_is_complete(checkpoint_dir, 2, load_optimizer=True)
    assert published_steps(output_dir) == ([1, 2] if interrupt_stage == "before_pruning" else [2])
    if interrupt_stage == "after_metadata_unlink":
        assert (checkpoint_dir / ".retention.json").is_file()
        assert (checkpoint_dir / "model_000001.pt").is_file()
        assert (checkpoint_dir / "optim_000001_rank0.pt").is_file()
    measured_update = training_module.measured_update
    resumed_steps = []

    def require_completed_retention(model, forward_cycle, optimizer, loader, config, step, *args, **kwargs):
        if not resumed_steps:
            assert step == 2
            assert published_steps(output_dir) == [2]
            assert latest.metadata()["step"] == 2
            assert not (checkpoint_dir / ".retention.json").exists()
            assert not (checkpoint_dir / "model_000001.pt").exists()
            assert not (checkpoint_dir / "optim_000001_rank0.pt").exists()
        resumed_steps.append(step + 1)
        return measured_update(model, forward_cycle, optimizer, loader, config, step, *args, **kwargs)

    monkeypatch.setattr(training_module, "measured_update", require_completed_retention)
    run_attempt(arguments(monkeypatch, output_dir, *policy_args, "--resume-latest"), "resume")
    assert resumed_steps == [3, 4]
    assert published_steps(output_dir) == [4]
    assert latest.metadata()["step"] == 4


def test_reviewed_source_transition_preserves_training_and_allows_strict_resume(training_environment, tmp_path, monkeypatch):
    from nanochat.rlm_latest import LatestCheckpoint
    from nanochat.rlm_resume import ResumeTransition

    def manifest(names):
        root = Path(rlm_train.__file__).resolve().parents[1]
        files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
        return {"sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), "files": files}

    old_source = manifest(("scripts/rlm_train.py",))
    new_source = manifest(("scripts/rlm_train.py", "nanochat/rlm_latest.py"))
    output_dir = tmp_path / "transition"
    latest = LatestCheckpoint(output_dir / "checkpoints")
    monkeypatch.setattr(training_module, "source_manifest", lambda root: old_source)
    run_attempt(arguments(
        monkeypatch, output_dir, "--keep-checkpoints", "2", "--stop-after-step", "2",
    ), "original")
    origin_record = (output_dir / "run.json").read_bytes()
    original_metadata = latest.metadata()
    old_contract = original_metadata["run_contract"]
    new_contract = {
        **old_contract,
        "source_sha256": new_source["sha256"],
        "checkpoint_policy": {**old_contract["checkpoint_policy"], "keep_last": 1},
    }
    receipt = ResumeTransition.prepare(
        output_dir, original_metadata, new_contract, old_source, new_source,
        "Reviewed checkpoint workflow change; model, optimizer, data, and schedule are unchanged.",
    )
    receipt_path = tmp_path / "resume-transition.json"
    receipt_path.write_text(json.dumps(receipt.to_dict()))

    monkeypatch.setattr(training_module, "source_manifest", lambda root: new_source)
    with pytest.raises(ValueError, match="--resume-transition"):
        run_attempt(arguments(
            monkeypatch, output_dir, "--keep-checkpoints", "1", "--resume-latest",
        ), "reject-unreviewed-source")
    assert latest.metadata() == original_metadata

    reference_dir = tmp_path / "reference-new-contract"
    run_attempt(arguments(monkeypatch, reference_dir, "--keep-checkpoints", "1"), "complete")
    expected_model, expected_optimizer, expected_metadata = LatestCheckpoint(
        reference_dir / "checkpoints",
    ).load(torch.device("cpu"))
    assert expected_metadata["run_contract"] == new_contract

    transitioned = run_attempt(arguments(
        monkeypatch, output_dir, "--keep-checkpoints", "1", "--resume-latest",
        "--resume-transition", receipt_path, "--stop-after-step", "3",
    ), "reviewed-transition")
    transition_metadata = latest.metadata()
    assert transition_metadata["step"] == 3
    assert transition_metadata["run_contract"] == new_contract
    evidence = receipt.validate(output_dir, original_metadata, new_contract)
    assert transition_metadata["resume_transition"] == evidence
    assert json.loads((transitioned / "resume-transition.json").read_text()) == evidence
    assert (output_dir / "run.json").read_bytes() == origin_record

    # Once the new contract is checkpointed, subsequent resumes are strict and
    # retain the original review evidence without requiring its source file.
    receipt_path.unlink()
    resumed = run_attempt(arguments(
        monkeypatch, output_dir, "--keep-checkpoints", "1", "--resume-latest",
    ), "strict-resume")
    actual_model, actual_optimizer, actual_metadata = latest.load(torch.device("cpu"))
    assert_state_equal(actual_model, expected_model)
    assert_state_equal(actual_optimizer, expected_optimizer)
    assert actual_metadata["rng_state"] == expected_metadata["rng_state"]
    assert actual_metadata["dataloader_state_dict"] == {"consumed": 4}
    assert actual_metadata["run_contract"] == new_contract
    assert actual_metadata["resume_transition"] == evidence
    assert (output_dir / "run.json").read_bytes() == origin_record
    result = json.loads((resumed / "result.json").read_text())
    assert result["status"] == "completed" and result["attempt_updates"] == 1


def test_sync_and_async_save_identical_training_state_for_each_update(training_environment, tmp_path, monkeypatch):
    from nanochat.rlm_latest import LatestCheckpoint

    snapshots = {"sync": {}, "async": {}}
    save = LatestCheckpoint.save

    def capture_saved_state(self, model, optimizer, metadata):
        result = save(self, model, optimizer, metadata)
        snapshots[self.checkpoint_dir.parent.name][metadata["step"]] = self.load(torch.device("cpu"))
        return result

    monkeypatch.setattr(LatestCheckpoint, "save", capture_saved_state)
    reports = {}
    for checkpoint_mode in snapshots:
        attempt = run_attempt(arguments(
            monkeypatch, tmp_path / checkpoint_mode, "--checkpoint-mode", checkpoint_mode,
            "--num-iterations", "3", "--warmup-steps", "2", "--save-every", "1",
        ), "complete")
        reports[checkpoint_mode] = json.loads((attempt / "result.json").read_text())
        # Returning from run must include a completed final save in both modes.
        assert sorted(snapshots[checkpoint_mode]) == [1, 2, 3]
        assert reports[checkpoint_mode]["checkpoint_status"]["latest_step"] == 3
        assert reports[checkpoint_mode]["checkpoint_status"]["pending_step"] is None

    for step in (1, 2, 3):
        sync_model, sync_optimizer, sync_metadata = snapshots["sync"][step]
        async_model, async_optimizer, async_metadata = snapshots["async"][step]
        assert_state_equal(async_model, sync_model)
        assert_state_equal(async_optimizer, sync_optimizer)
        for name in ("rng_state", "dataloader_state_dict", "training_config", "model_config",
                     "step", "text_tokens", "supervised_tokens", "internal_positions"):
            assert async_metadata[name] == sync_metadata[name]
        assert async_metadata["dataloader_state_dict"] == {"consumed": step}
    for actual, expected in zip(reports["async"]["updates"], reports["sync"]["updates"]):
        assert actual["loss"] == expected["loss"]
        assert actual["cycle_losses"] == expected["cycle_losses"]
        assert actual["lr_multiplier"] == expected["lr_multiplier"]


def test_async_save_overlaps_backward_and_finishes_before_optimizer_mutation(training_environment, tmp_path, monkeypatch):
    from nanochat.rlm_latest import LatestCheckpoint

    saving = Event()
    second_gradients_ready = Event()
    first_save_finished = Event()
    save = LatestCheckpoint.save
    require_gradients = training_module.require_finite_gradients
    build_optimizer = training_module.build_optimizer
    optimizer_updates = []

    def save_during_backward(self, model, optimizer, metadata):
        if metadata["step"] == 1:
            model_before = deepcopy(model)
            optimizer_before = deepcopy(optimizer)
            saving.set()
            assert second_gradients_ready.wait(timeout=10), "the second backward never overlapped the first save"
            assert_state_equal(model, model_before)
            assert_state_equal(optimizer, optimizer_before)
        result = save(self, model, optimizer, metadata)
        if metadata["step"] == 1:
            first_save_finished.set()
        return result

    def mark_second_backward(model, step, *, frozen_parameter=None):
        require_gradients(model, step, frozen_parameter=frozen_parameter)
        if step == 1:
            assert saving.wait(timeout=10), "the checkpoint worker never started"
            assert not first_save_finished.is_set()
            second_gradients_ready.set()

    def instrument_optimizer(model, config):
        optimizer, decay = build_optimizer(model, config)
        optimizer_step = optimizer.step

        def require_saved_previous_update():
            if len(optimizer_updates) == 1:
                assert first_save_finished.is_set(), "optimizer mutated tensors before their save finished"
            optimizer_step()
            optimizer_updates.append(len(optimizer_updates) + 1)

        monkeypatch.setattr(optimizer, "step", require_saved_previous_update)
        return optimizer, decay

    monkeypatch.setattr(LatestCheckpoint, "save", save_during_backward)
    monkeypatch.setattr(training_module, "require_finite_gradients", mark_second_backward)
    monkeypatch.setattr(training_module, "build_optimizer", instrument_optimizer)
    attempt = run_attempt(arguments(
        monkeypatch, tmp_path / "overlap", "--checkpoint-mode", "async",
        "--num-iterations", "3", "--warmup-steps", "2",
    ), "complete")
    assert optimizer_updates == [1, 2, 3]
    assert second_gradients_ready.is_set() and first_save_finished.is_set()
    result = json.loads((attempt / "result.json").read_text())
    assert all(item["latest_published"] for item in result["checkpoint_measurements"])
    assert result["checkpoint_status"]["latest_step"] == 3


def test_default_ten_update_archive_interval_and_retirement(training_environment, tmp_path, monkeypatch):
    from nanochat.rlm_latest import LatestCheckpoint

    output_dir = tmp_path / "default-interval"
    attempt = run_attempt(arguments(
        monkeypatch, output_dir, "--num-iterations", "21", "--keep-checkpoints", "2",
    ), "complete")
    measurements = json.loads((attempt / "checkpoint-measurements.json").read_text())
    assert [item["step"] for item in measurements if item["numbered_published"]] == [2, 10, 20, 21]
    assert published_steps(output_dir) == [20, 21]
    assert LatestCheckpoint(output_dir / "checkpoints").metadata()["step"] == 21
    for step in (2, 10):
        assert not (output_dir / "checkpoints" / f"model_{step:06d}.pt").exists()
        assert not (output_dir / "checkpoints" / f"optim_{step:06d}_rank0.pt").exists()


def test_probe_does_not_publish_checkpoints(training_environment, tmp_path, monkeypatch):
    from nanochat.rlm_latest import LatestCheckpoint

    def unexpected_save(*args, **kwargs):
        pytest.fail("a probe must not save a rolling checkpoint")

    monkeypatch.setattr(LatestCheckpoint, "save", unexpected_save)
    output_dir = tmp_path / "probe"
    attempt = run_attempt(arguments(
        monkeypatch, output_dir, "--mode", "probe", "--num-iterations", "2",
    ), "complete")
    assert not (output_dir / "checkpoints" / "latest.pt").exists()
    assert published_steps(output_dir) == []
    result = json.loads((attempt / "result.json").read_text())
    assert result["completed_updates"] == 2
    assert result["checkpoint_overhead_measured"] is False


@pytest.mark.parametrize("extra,match", [
    (("--input-embedding-freeze", "scheduled",), "embedding"),
    (("--init-embedding-checkpoint-dir", "source"), "requires checkpoint directory, step, and source tokenizer together"),
    (("--init-embedding-step", "7", "--init-embedding-tokenizer", "tokenizer.pkl"), "requires checkpoint directory, step, and source tokenizer together"),
    (("--init-embedding-checkpoint-dir", "source", "--init-embedding-step", "7", "--init-embedding-tokenizer", "tokenizer.pkl", "--resume-from-step", "3"), "fresh-run only"),
    (("--init-embedding-checkpoint-dir", "source", "--init-embedding-step", "7", "--init-embedding-tokenizer", "tokenizer.pkl", "--resume-latest"), "fresh-run only"),
    (("--resume-from-step", "3", "--resume-latest"), "not allowed with argument"),
    (("--mode", "probe", "--resume-latest"), "no resume"),
])
def test_cli_rejects_invalid_initialization_and_resume_options(monkeypatch, tmp_path, capsys, extra, match):
    with pytest.raises(SystemExit) as error:
        arguments(monkeypatch, tmp_path / "unused", *extra)
    assert error.value.code == 2
    assert match in capsys.readouterr().err

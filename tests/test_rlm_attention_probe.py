"""Attention comparisons must exclude compilation and preserve production memory."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

import nanochat.rlm_attention_probe as probe_module
import nanochat.rlm_capacity as capacity_module
from nanochat.rlm_attention_probe import attention_probe
from nanochat.rlm_policy import InputEmbeddingFreeze
from nanochat.rlm_training import TrainingConfig
from test_rlm_capacity import cpu_execution, small_model, Episodes
from test_rlm_train_cli import arguments, run_attempt, source_arguments, training_environment


def probe_config(policy=InputEmbeddingFreeze.NONE):
    return TrainingConfig(
        device_batch_size=2, total_batch_size=32, num_iterations=100,
        warmup_steps=0, compile=False, input_embedding_freeze=policy,
    )


def mock_updates(monkeypatch, compiled_updates):
    calls = []
    snapshot = {"stats": {"unique_graphs": 1}, "timing_events": {"backward": 1}}

    def update(model, forward, optimizer, loader, config, step, bos, minimum, failure_path):
        assert optimizer.state, "Optimizer state must be resident before forward/backward"
        compiled = compiled_updates[len(calls)]
        if compiled:
            # A backward compilation or autotune can occur without a new Dynamo graph.
            snapshot["timing_events"]["backward"] += 1
        metrics = {
            "duration_seconds": 100 if compiled else len(calls) + 1,
            "cuda_peak_reserved_gb": 12,
            "cuda_peak_allocated_gb": 10,
            "finite_loss_and_gradients": True,
            "finite_model_and_optimizer_state": True,
        }
        calls.append(step)
        return metrics, {}

    monkeypatch.setattr(probe_module, "compiler_snapshot", lambda: deepcopy(snapshot))
    monkeypatch.setattr(probe_module, "measured_update", update)
    return calls


def test_timed_update_is_retried_if_lazy_backward_compilation_occurs(tmp_path, monkeypatch):
    calls = mock_updates(monkeypatch, [True, False, True, False, False])
    model = small_model()
    report = attention_probe(
        model, model.forward_cycle, model.setup_optimizer(), Episodes(), probe_config(), 0.0, 0, tmp_path,
    )
    assert calls == list(range(5))
    assert report["attention_probe"]["warmup_updates"] == 4
    assert report["attention_probe"]["duration_seconds"] == 5
    assert [item["duration_seconds"] for item in report["warmup_updates"]] == [100, 2, 100, 4]
    assert len(report["updates"]) == 1
    measured = report["updates"][0]
    assert measured["compilation_before"] == measured["compilation_after"]
    assert report["attention_probe"]["compilation_excluded"] is True


def test_continuous_recompilation_fails_without_publishing_a_timing(tmp_path, monkeypatch):
    calls = mock_updates(monkeypatch, [True] * (probe_module.MAX_WARMUP_UPDATES + 1))
    model = small_model()
    with pytest.raises(RuntimeError, match="compilation-free timed update"):
        attention_probe(
            model, model.forward_cycle, model.setup_optimizer(), Episodes(), probe_config(), 0.0, 0, tmp_path,
        )
    assert len(calls) == probe_module.MAX_WARMUP_UPDATES + 1
    assert not list(tmp_path.iterdir())


def test_scheduled_embedding_phase_is_rejected_before_allocations(tmp_path):
    model = small_model()
    optimizer = model.setup_optimizer()
    loader = Episodes()
    with pytest.raises(ValueError, match="none or permanent"):
        attention_probe(
            model, model.forward_cycle, optimizer, loader,
            probe_config(InputEmbeddingFreeze.SCHEDULED), 0.0, 0, tmp_path,
        )
    assert not optimizer.state
    assert loader.calls == 0


def test_failure_releases_pinned_staging_and_propagates_original_error(tmp_path, monkeypatch):
    model = small_model()
    optimizer = model.setup_optimizer()
    staged = {"resident": True}
    events = []

    class Stager:
        def __init__(self, options):
            assert options.use_pinned_memory and options.use_async_staging

        def stage(self, state):
            assert state["optimizer"]["state"]
            events.append("stage")
            return SimpleNamespace(result=lambda: staged)

        def close(self):
            events.append("close")

    def failed_update(*args, **kwargs):
        assert events == ["stage"]
        raise RuntimeError("FA3 backward failed")

    monkeypatch.setattr(model, "get_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(capacity_module, "DefaultStager", Stager)
    monkeypatch.setattr(capacity_module, "synchronize", lambda device: None)
    monkeypatch.setattr(capacity_module, "cuda_metrics", lambda device: {})
    monkeypatch.setattr(probe_module, "measured_update", failed_update)
    with pytest.raises(RuntimeError, match="FA3 backward failed"):
        attention_probe(model, model.forward_cycle, optimizer, Episodes(), probe_config(), 0.0, 0, tmp_path)
    assert events == ["stage", "close"]


@pytest.mark.parametrize("policy,compile_model", [
    pytest.param("none", False, id="trainable"),
    pytest.param("permanent", False, id="frozen"),
    pytest.param("none", True, id="compiled_model"),
])
def test_cli_times_complete_update_without_checkpoint_writes(
    training_environment, tmp_path, monkeypatch, policy, compile_model,
):
    source_dir, source_tokenizer, _ = training_environment
    output = tmp_path / f"attention-{policy}"
    args = arguments(
        monkeypatch, output, "--mode", "attention-probe", "--num-iterations", "100",
        "--total-batch-size", "32", "--input-embedding-freeze", policy,
        "--compile" if compile_model else "--no-compile",
        *source_arguments(source_dir, source_tokenizer),
    )
    attempt = run_attempt(args, "probe")
    report = json.loads((attempt / "result.json").read_text())
    assert report == json.loads((output / "result.json").read_text())
    updates = [json.loads(line) for line in (attempt / "updates.jsonl").read_text().splitlines()]
    assert updates == report["updates"] and len(updates) == 1
    timed = updates[0]
    assert timed["text_targets"] == 32 and timed["prediction_targets"] == 40
    assert timed["finite_loss_and_gradients"] and timed["finite_model_and_optimizer_state"]
    assert timed["compilation_before"] == timed["compilation_after"]
    assert report["attention_probe"]["input_embedding_frozen"] is (policy == "permanent")
    assert timed["input_embedding_frozen"] is (policy == "permanent")
    assert report["attention_probe"]["duration_seconds"] == timed["duration_seconds"] > 0
    assert report["attention_probe"]["warmup_updates"] >= 1
    assert report["attention_impl"] == "sdpa"
    contract = json.loads((output / "run.json").read_text())["contract"]
    assert contract["attention_impl"] == "sdpa"
    assert contract["training_config"]["num_iterations"] == 100
    assert not (output / "checkpoints").exists()
    assert not list(output.rglob("*.pt"))

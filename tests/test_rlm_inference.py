from contextlib import nullcontext
from dataclasses import asdict
import importlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from nanochat.checkpoint_manager import save_checkpoint
from nanochat.rlm import CycleOutput, RLMConfig, RecurrentGPT
from nanochat.rlm_inference import (
    RLMGenerator,
    evaluate_memory_controls,
    load_rlm_checkpoint,
    tokenizer_fingerprint,
)


class TraceModel(torch.nn.Module):
    """An exact recurrence that exposes accidental old-context reuse."""

    def __init__(self, text_length=4):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(text_length=text_length, vocab_size=64)
        self.calls = []

    def get_device(self):
        return self.anchor.device

    def forward_cycle(self, text_ids, memory=None, targets=None, loss_reduction="mean"):
        self.calls.append((text_ids.clone(), None if memory is None else memory.clone()))
        carried = torch.zeros(text_ids.size(0)) if memory is None else memory[:, 0, 0]
        predicted = (text_ids + carried[:, None].long() + 1) % self.config.vocab_size
        eom_predicted = (carried[:, None].long() + 17) % self.config.vocab_size
        predicted = torch.cat((eom_predicted, predicted), dim=1)
        logits = 8 * F.one_hot(predicted, self.config.vocab_size).float()
        outgoing = (carried + text_ids.sum(1)).reshape(-1, 1, 1).repeat(1, 2, 1)
        if text_ids.size(1) == 0:
            # Prefix readout summaries must never replace the completed chunk's
            # memory, even if the model computes and returns them.
            outgoing.fill_(-1000)
        if targets is None:
            return CycleOutput(memory=outgoing, logits=logits)
        joint_targets = torch.cat((text_ids[:, :1], targets), dim=1)
        loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), joint_targets.reshape(-1),
            ignore_index=-1, reduction=loss_reduction,
        )
        return CycleOutput(memory=outgoing, loss=loss)


def tiny_model():
    config = RLMConfig(
        sequence_len=11, vocab_size=64, n_layer=2, n_head=2,
        n_kv_head=2, n_embd=32, text_length=4, memory_tokens=3,
    )
    model = RecurrentGPT(config)
    model.init_weights()
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.04)
            block.mlp.c_proj.weight.normal_(std=0.04)
    return model


def test_generation_predicts_boundary_token_from_new_memory_eom_readout():
    model = TraceModel().eval()
    tokens = list(RLMGenerator(model).generate([1, 2, 3, 4], 3, temperature=0))
    assert tokens == [27, 38, 49]
    assert [text.tolist() for text, _ in model.calls] == [
        [[1, 2, 3, 4]], [[]], [[27]], [[27, 38]],
    ]
    assert model.calls[0][1] is None
    torch.testing.assert_close(model.calls[1][1], torch.full((1, 2, 1), 10.0))
    for _, memory in model.calls[2:]:
        torch.testing.assert_close(memory, model.calls[1][1])


def test_long_prompt_is_segmented_and_old_raw_text_never_reenters_model():
    model = TraceModel().eval()
    generated = list(RLMGenerator(model).generate(list(range(1, 11)), 4, temperature=0))
    assert generated == [47, 20, 11, 6]
    assert [text.tolist() for text, _ in model.calls] == [
        [[1, 2, 3, 4]], [[5, 6, 7, 8]], [[9, 10]],
        [[9, 10, 47]], [[9, 10, 47, 20]], [[]], [[11]],
    ]
    torch.testing.assert_close(model.calls[-1][1], torch.full((1, 2, 1), 122.0))


def test_generation_resets_between_requests_and_stops_without_consuming_stop_token():
    model = TraceModel().eval()
    generator = RLMGenerator(model)
    assert list(generator.generate([1, 2, 3, 4], 8, temperature=0, stop_token_ids=frozenset({27}))) == []
    assert len(model.calls) == 2
    assert list(generator.generate([1, 2, 3, 4], 1, temperature=0)) == [27]
    assert model.calls[-2][1] is None


def test_generation_budget_can_end_when_a_chunk_becomes_full():
    model = TraceModel().eval()
    assert list(RLMGenerator(model).generate([1, 2, 3], 1, temperature=0)) == [4]
    assert [text.tolist() for text, _ in model.calls] == [[[1, 2, 3]]]
    assert list(RLMGenerator(model).generate([1, 2, 3, 4], 0, temperature=0)) == []
    assert len(model.calls) == 1


def test_generation_matches_real_model_manual_rollover():
    torch.manual_seed(12)
    model = tiny_model().eval()
    prompt = [1, 2, 3, 4]
    with torch.inference_mode():
        first = model.forward_cycle(torch.tensor([prompt]))
        readout = model.forward_cycle(torch.empty((1, 0), dtype=torch.long), memory=first.memory)
        boundary = readout.logits[:, 0].argmax(-1).item()
        second = model.forward_cycle(torch.tensor([[boundary]]), memory=first.memory)
        after_boundary = second.logits[:, -1].argmax(-1).item()
    assert list(RLMGenerator(model).generate(prompt, 2, temperature=0)) == [boundary, after_boundary]


def test_memory_controls_use_matched_rows_and_intact_history():
    model = TraceModel(text_length=2)
    inputs = torch.tensor([[0, 0, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6], [7, 7, 8, 8, 9, 9]])
    targets = torch.tensor([[1, 1, 3, 3, 8, 8], [5, 5, 14, 14, 25, 25], [8, 8, 23, 23, 40, 40]])
    targets[0, 0] = -1
    result = evaluate_memory_controls(model, [(inputs, targets)], num_batches=1, cycles=3, seed=123)
    assert model.training  # Training callers recover their original mode.
    assert result.episodes == 3
    assert [cycle.targets for cycle in result.cycles] == [5, 6, 6]
    assert result.cycles[0].intact == result.cycles[0].zero == result.cycles[0].shuffled
    assert result.cycles[1].intact < result.cycles[1].zero
    assert result.cycles[1].intact < result.cycles[1].shuffled
    paired = result.cycles[1].zero_minus_intact
    assert paired.episodes == 3
    assert paired.mean == pytest.approx(16 / 3, abs=1e-6)
    assert paired.standard_error == pytest.approx(8 / 3, abs=1e-6)
    intact_memory = model.calls[1][1]
    torch.testing.assert_close(model.calls[2][1], torch.zeros_like(intact_memory))
    shuffled_memory = model.calls[3][1]
    assert bool((shuffled_memory[:, 0, 0] != intact_memory[:, 0, 0]).all())
    torch.testing.assert_close(shuffled_memory[:, 0, 0].sort().values, intact_memory[:, 0, 0].sort().values)
    for index in (1, 2, 3):
        torch.testing.assert_close(model.calls[index][0], inputs[:, 2:4])
    torch.testing.assert_close(model.calls[4][1][:, 0, 0], torch.tensor([4.0, 18.0, 30.0]))


def test_evaluation_aggregates_valid_targets_and_restores_mode_on_error():
    torch.manual_seed(21)
    model = tiny_model()
    inputs = torch.randint(0, 64, (2, 8))
    targets = torch.randint(0, 64, (2, 8))
    targets[0, :3] = -1
    result = evaluate_memory_controls(model, [(inputs, targets)], 1, 2)
    with torch.inference_mode():
        joint = model(inputs, targets, loss_reduction="none").reshape(2, 2, 5)
        expected = joint[:, :, 1:].sum().item()
    measured = sum(cycle.intact * cycle.targets for cycle in result.cycles)
    assert measured == pytest.approx(expected, abs=1e-5)
    assert measured < joint.sum().item()
    with pytest.raises(ValueError, match="at least two batch lanes"):
        evaluate_memory_controls(model, [(inputs[:1], targets[:1])], 1, 2)
    assert model.training


def test_paired_uncertainty_excludes_unsupervised_rows_without_inventing_one_sample_error():
    model = TraceModel(text_length=2)
    inputs = torch.tensor([[1, 1, 2, 2], [4, 4, 5, 5]])
    targets = torch.tensor([[2, 2, 5, 5], [5, 5, -1, -1]])
    result = evaluate_memory_controls(model, [(inputs, targets)], 1, 2)
    assert result.cycles[1].zero_minus_intact.episodes == 1
    assert result.cycles[1].zero_minus_intact.mean == pytest.approx(8, abs=1e-6)
    assert result.cycles[1].zero_minus_intact.standard_error is None


def install_test_tokenizer(tmp_path, monkeypatch):
    tokenizer_path = tmp_path / "tokenizer" / "tokenizer.pkl"
    tokenizer_path.parent.mkdir()
    tokenizer_path.write_bytes(b"test tokenizer identity")
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    tokenizer = SimpleNamespace(get_vocab_size=lambda: 64)
    monkeypatch.setattr("nanochat.rlm_inference.get_tokenizer", lambda: tokenizer)
    return tokenizer_path


def checkpoint_metadata(model, step=3):
    return {
        "step": step, "model_type": "rlm", "model_config": asdict(model.config),
        "tokenizer_sha256": tokenizer_fingerprint(),
    }


def test_checkpoint_roundtrip_preserves_master_precision_and_recurrent_outputs(tmp_path, monkeypatch):
    install_test_tokenizer(tmp_path, monkeypatch)
    torch.manual_seed(31)
    model = tiny_model().eval()
    with torch.no_grad():
        model.summary_embedding.add_(0.000137)
        model.eom_embedding.add_(0.000173)
    expected_summary = model.summary_embedding.detach().clone()
    expected_eom = model.eom_embedding.detach().clone()
    checkpoint_dir = tmp_path / "checkpoint"
    save_checkpoint(checkpoint_dir, 3, model.state_dict(), None, checkpoint_metadata(model))

    original_init = RecurrentGPT.init_weights
    def lower_precision_initialization(instance):
        original_init(instance)
        instance.summary_embedding.data = instance.summary_embedding.data.bfloat16()
        instance.eom_embedding.data = instance.eom_embedding.data.bfloat16()
    monkeypatch.setattr(RecurrentGPT, "init_weights", lower_precision_initialization)
    restored, _, metadata = load_rlm_checkpoint(checkpoint_dir, None, torch.device("cpu"))
    assert metadata["step"] == 3
    assert not restored.training
    assert restored.summary_embedding.dtype == torch.float32
    torch.testing.assert_close(restored.summary_embedding, expected_summary, rtol=0, atol=0)
    assert restored.eom_embedding.dtype == torch.float32
    torch.testing.assert_close(restored.eom_embedding, expected_eom, rtol=0, atol=0)
    text = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    with torch.inference_mode():
        torch.testing.assert_close(restored(text), model(text), rtol=0, atol=0)


@pytest.mark.parametrize("missing_parameter", ["summary_embedding", "eom_embedding"])
def test_checkpoint_rejects_wrong_tokenizer_and_missing_recurrent_parameter(tmp_path, monkeypatch, missing_parameter):
    tokenizer_path = install_test_tokenizer(tmp_path, monkeypatch)
    model = tiny_model()
    checkpoint_dir = tmp_path / "checkpoint"
    metadata = checkpoint_metadata(model)
    state = model.state_dict()
    del state[missing_parameter]
    save_checkpoint(checkpoint_dir, 3, state, None, metadata)
    with pytest.raises(RuntimeError, match=missing_parameter):
        load_rlm_checkpoint(checkpoint_dir, 3, torch.device("cpu"))
    tokenizer_path.write_bytes(b"another tokenizer with the same vocabulary size")
    with pytest.raises(ValueError, match="fingerprint"):
        load_rlm_checkpoint(checkpoint_dir, 3, torch.device("cpu"))


@pytest.mark.parametrize("module_name", ["scripts.rlm_eval", "scripts.rlm_sample"])
def test_cli_import_does_not_freeze_compute_precision(module_name):
    script = (
        f"import {module_name}; import sys; "
        "assert 'nanochat.common' not in sys.modules; assert 'torch' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )


@pytest.mark.parametrize("module_name", ["scripts.rlm_eval", "scripts.rlm_sample"])
@pytest.mark.parametrize("explicit_environment", [False, True])
def test_cli_configures_precision_and_checks_memory_before_loading(
    module_name, explicit_environment, monkeypatch, tmp_path,
):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(sys, "argv", [module_name, "--checkpoint-dir", str(tmp_path), "--device", "cpu"])
    expected = {
        "NANOCHAT_DTYPE": "bfloat16" if explicit_environment else "float32",
        "NANOCHAT_CUDA_MEMORY_LIMIT_GB": "12" if explicit_environment else "80",
        "NANOCHAT_MIN_SYSTEM_AVAILABLE_GB": "1" if explicit_environment else "16",
    }
    for key, value in expected.items():
        if explicit_environment:
            monkeypatch.setenv(key, value)
        else:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WORLD_SIZE", "1")
    events = []

    def initialize(device_type):
        assert device_type == "cpu"
        assert all(os.environ[key] == value for key, value in expected.items())
        events.append("initialize")
        return False, 0, 0, 1, torch.device("cpu")

    def check_memory(device, step):
        events.append("guard")

    def load_model(*args):
        events.append("load")
        raise RuntimeError("stop before loading checkpoint")

    monkeypatch.setattr("nanochat.common.compute_init", initialize)
    monkeypatch.setattr("nanochat.common.compute_cleanup", lambda: events.append("cleanup"))
    monkeypatch.setattr("nanochat.training_runtime.check_training_memory", check_memory)
    monkeypatch.setattr("nanochat.rlm_inference.load_rlm_checkpoint", load_model)
    monkeypatch.setattr("nanochat.rlm_training.MemoryMonitor", lambda minimum: nullcontext(
        SimpleNamespace(require_available=lambda step: None),
    ))
    with pytest.raises(RuntimeError, match="stop before loading"):
        module.main()
    assert events == ["initialize", "guard", "load", "cleanup"]


@pytest.mark.parametrize("raced_output", [False, True])
def test_eval_records_loading_and_evaluation_resources_and_publishes_exclusively(
    tmp_path, monkeypatch, capsys, raced_output,
):
    from scripts import rlm_eval

    output_path = tmp_path / "evaluation.json"
    monkeypatch.setattr(sys, "argv", [
        "rlm_eval", "--checkpoint-dir", str(tmp_path), "--device", "cpu",
        "--batches", "2", "--output", str(output_path),
    ])
    monkeypatch.setenv("WORLD_SIZE", "1")
    events = []

    class Monitor:
        active = False
        minimum_observed = 64e9

        def __enter__(self):
            self.active = True
            events.append("monitor_start")
            return self

        def require_available(self, step):
            assert self.active

        def __exit__(self, *args):
            self.active = False
            events.append("monitor_stop")

    monitor = Monitor()
    clock = iter([10.0, 12.0, 13.0, 17.0, 18.0])
    snapshots = iter([
        {"cuda_allocated_gb": 1, "cuda_reserved_gb": 2, "cuda_peak_allocated_gb": 9, "cuda_peak_reserved_gb": 12},
        {"cuda_allocated_gb": 3, "cuda_reserved_gb": 4, "cuda_peak_allocated_gb": 7, "cuda_peak_reserved_gb": 15},
    ])

    def load_model(*args):
        assert monitor.active
        monitor.minimum_observed = 32e9
        events.append("load")
        return SimpleNamespace(config=SimpleNamespace(text_length=4)), object(), {"step": 6}

    def evaluate(model, batches, num_batches, cycles, seed):
        assert monitor.active
        assert len(list(batches)) == num_batches == 2
        events.append("evaluate")
        if raced_output:
            output_path.write_text("another process wrote this")
        return SimpleNamespace(to_dict=lambda: {"episodes": 4, "cycles": []})

    monkeypatch.setattr(rlm_eval.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr("nanochat.common.compute_init", lambda device: (False, 0, 0, 1, torch.device("cpu")))
    monkeypatch.setattr("nanochat.common.compute_cleanup", lambda: events.append("cleanup"))
    monkeypatch.setattr("nanochat.training_runtime.check_training_memory", lambda device, step: None)
    monkeypatch.setattr("nanochat.rlm_training.MemoryMonitor", lambda minimum: monitor)
    monkeypatch.setattr("nanochat.rlm_training.cuda_metrics", lambda device: next(snapshots))
    monkeypatch.setattr("nanochat.rlm_training.synchronize", lambda device: None)
    monkeypatch.setattr("nanochat.rlm_inference.load_rlm_checkpoint", load_model)
    monkeypatch.setattr("nanochat.rlm_inference.evaluate_memory_controls", evaluate)
    monkeypatch.setattr("nanochat.dataloader.tokenizing_distributed_data_loader_bos_bestfit", lambda *args, **kwargs: [(None, None)] * 2)
    if raced_output:
        with pytest.raises(FileExistsError):
            rlm_eval.main()
        assert output_path.read_text() == "another process wrote this"
    else:
        rlm_eval.main()
        persisted = output_path.read_text()
        assert capsys.readouterr().out == persisted
        result = json.loads(persisted)
        resources = result["resources"]
        assert result["step"] == 6
        assert resources["load_seconds"] == 2
        assert resources["evaluation_seconds"] == 4
        assert resources["total_seconds"] == 8
        assert resources["minimum_system_available_gb"] == 32
        assert resources["peak_cuda_allocated_gb"] == 9
        assert resources["peak_cuda_reserved_gb"] == 15
    assert events == ["monitor_start", "load", "evaluate", "monitor_stop", "cleanup"]


@pytest.mark.parametrize("symlink", [False, True])
def test_eval_rejects_existing_output_before_loading(tmp_path, monkeypatch, symlink):
    from scripts import rlm_eval

    output_path = tmp_path / "evaluation.json"
    if symlink:
        output_path.symlink_to(tmp_path / "missing-target")
    else:
        output_path.write_text("original")
    monkeypatch.setattr(sys, "argv", [
        "rlm_eval", "--checkpoint-dir", str(tmp_path), "--output", str(output_path),
    ])
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(SystemExit) as error:
        rlm_eval.main()
    assert error.value.code == 2
    if symlink:
        assert output_path.is_symlink()
        assert not output_path.exists()
    else:
        assert output_path.read_text() == "original"

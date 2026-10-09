"""The diagnostic profile must contain a real warmed recurrent backward/update."""

import json
from dataclasses import replace

import pytest
import torch

from nanochat.rlm import RecurrentGPT, RLMConfig
from nanochat.rlm_training import TrainingConfig
from scripts.rlm_profile import export_device_events, profile_workload


class ReusingEpisodes:
    def __init__(self):
        self.calls = 0
        self.generator = torch.Generator().manual_seed(91)
        self.inputs = torch.empty((2, 8), dtype=torch.long)
        self.targets = torch.empty_like(self.inputs)

    def __next__(self):
        tokens = torch.randint(0, 32, (2, 9), generator=self.generator)
        self.inputs.copy_(tokens[:, :-1])
        self.targets.copy_(tokens[:, 1:])
        self.calls += 1
        return self.inputs, self.targets, {"consumed": self.calls}


class RecordingAdamW(torch.optim.AdamW):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.001)
        self.steps = 0
        for group in self.param_groups:
            group.update(initial_lr=group["lr"], kind="adamw")

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)


def fixture():
    torch.manual_seed(19)
    model = RecurrentGPT(RLMConfig(
        sequence_len=9, text_length=4, memory_tokens=2, vocab_size=32,
        n_embd=32, n_layer=2, n_head=2, n_kv_head=2,
        activation_checkpointing=True,
    ))
    model.init_weights()
    optimizer = RecordingAdamW(model.parameters())
    config = TrainingConfig(device_batch_size=2, total_batch_size=32, compile=False)
    return model, optimizer, config, ReusingEpisodes()


@pytest.mark.parametrize("freeze_input_embedding", [False, True])
def test_profile_warms_resident_state_and_exports_real_backward_and_optimizer_events(tmp_path, freeze_input_embedding):
    model, optimizer, config, loader = fixture()
    config = replace(config, num_iterations=10, freeze_input_embedding=freeze_input_embedding)
    initial_embedding = model.transformer.wte.weight.detach().clone()
    initial_eom = model.eom_embedding.detach().clone()
    initial_summary = model.summary_embedding.detach().clone()
    result = profile_workload(
        model, model.forward_cycle, optimizer, loader, config, 0.0, tmp_path,
        warmup_episodes=2,
    )
    assert loader.calls == 3
    assert optimizer.steps == 2
    assert optimizer.state
    assert result["real_text_tokens"] == 16
    assert result["supervised_predictions"] == 20
    assert result["gradient_scale"] == 0.5
    assert result["dataloader_state"] == {"consumed": 3}
    operators = json.loads((tmp_path / "operators.json").read_text())
    by_name = {event["name"]: event for event in operators}
    for name in ("rlm.forward.cycle_1", "rlm.forward.cycle_2", "rlm.episode.backward", "rlm.optimizer.step"):
        assert by_name[name]["calls"] == 1
        assert by_name[name]["total_cpu_us"] > 0
    assert any("Backward" in name for name in by_name)
    trace = json.loads((tmp_path / "trace.json").read_text())
    assert any(event.get("name") == "rlm.episode.backward" for event in trace["traceEvents"])
    assert all(parameter.grad is None for parameter in model.parameters())
    assert result["input_embedding_frozen"] is freeze_input_embedding
    assert not torch.equal(model.eom_embedding, initial_eom)
    assert not torch.equal(model.summary_embedding, initial_summary)
    if freeze_input_embedding:
        assert torch.equal(model.transformer.wte.weight, initial_embedding)
        assert model.transformer.wte.weight not in optimizer.state
    else:
        assert not torch.equal(model.transformer.wte.weight, initial_embedding)
        assert model.transformer.wte.weight in optimizer.state


def test_profile_rejects_nonfinite_warmup_without_creating_optimizer_state(tmp_path):
    model, optimizer, config, loader = fixture()
    model.summary_embedding.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
    with pytest.raises(FloatingPointError, match="Non-finite gradients"):
        profile_workload(model, model.forward_cycle, optimizer, loader, config, 0.0, tmp_path, warmup_episodes=2)
    assert optimizer.steps == 0
    assert not optimizer.state
    assert loader.calls == 1
    failure = json.loads((tmp_path / "failure.json").read_text())
    assert failure["completed_warmup_episodes"] == 0
    assert failure["error_type"] == "FloatingPointError"


def test_device_export_excludes_nested_annotations_and_cpu_attribution(tmp_path):
    trace = {"traceEvents": [
        {"cat": "gpu_user_annotation", "name": "episode", "ph": "X", "dur": 1000},
        {"cat": "gpu_user_annotation", "name": "cycle", "ph": "X", "dur": 900},
        {"cat": "cpu_op", "name": "matmul", "ph": "X", "dur": 800},
        {"cat": "kernel", "name": "matmul", "ph": "X", "dur": 400},
        {"cat": "kernel", "name": "matmul", "ph": "X", "dur": 300},
        {"cat": "kernel", "name": "matmul", "ph": "B", "dur": 9999},
        {"cat": "gpu_memcpy", "name": "copy", "ph": "X", "dur": 20},
        {"cat": "gpu_memset", "name": "zero", "ph": "X", "dur": 10},
    ]}
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace))

    result = export_device_events(trace_path, tmp_path)

    assert result["cuda_kernel_total_us"] == 700
    assert result["cuda_device_event_total_us"] == 730
    events = json.loads((tmp_path / "device-events.json").read_text())
    assert events[0] == {"category": "kernel", "name": "matmul", "calls": 2, "total_us": 700.0}
    assert len(events) == 3
    annotations = json.loads((tmp_path / "device-annotations.json").read_text())
    assert {event["name"] for event in annotations} == {"episode", "cycle"}

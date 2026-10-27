"""Exercise profile dimensions and the checkpoint-owned profile document."""

import json

import pytest

import nanochat.rlm as rlm_module
from nanochat.rlm_latest import LatestCheckpoint
from test_rlm_train_cli import arguments, run_attempt, training_environment


def write_profile(path, **changes):
    fields = {
        "n_layer": 3, "n_embd": 48, "n_head": 4, "n_kv_head": 2,
        "text_length": 8, "memory_tokens": 3, "eom_tokens": 1,
        **changes,
    }
    text = "// Keep this experiment's notes, including Unicode: 中文.\n" + json.dumps(fields, indent=2) + "\n"
    path.write_text(text)
    return text


@pytest.mark.parametrize("source_change", ["delete", "replace"])
def test_gqa_profile_is_copied_exactly_and_explicit_resume_uses_saved_document(
    training_environment, tmp_path, monkeypatch, source_change,
):
    profile_path = tmp_path / "custom.jsonc"
    original = write_profile(profile_path)
    output = tmp_path / "train"
    first = run_attempt(arguments(
        monkeypatch, output, "--total-batch-token-size", "32", "--model-profile", profile_path, "--stop-after-step", "1",
    ), "first")
    saved_path = output / "checkpoints" / "model-profile.jsonc"
    assert saved_path.read_bytes() == original.encode("utf-8")
    model, _, metadata = LatestCheckpoint(output / "checkpoints").load_cpu_mapped()
    config = metadata["model_config"]
    assert {name: config[name] for name in (
        "n_layer", "n_embd", "n_head", "n_kv_head", "text_length", "memory_tokens", "sequence_len",
    )} == {
        "n_layer": 3, "n_embd": 48, "n_head": 4, "n_kv_head": 2,
        "text_length": 8, "memory_tokens": 3, "sequence_len": 15,
    }
    assert model["transformer.h.0.attn.c_k.weight"].shape == (24, 48)
    report = json.loads((first / "result.json").read_text())
    assert report["text_tokens"] == 32
    assert report["supervised_tokens"] == 36
    if source_change == "delete":
        profile_path.unlink()
    else:
        write_profile(profile_path, n_kv_head=4)
    run_attempt(arguments(
        monkeypatch, output, "--total-batch-token-size", "32", "--model-profile", saved_path,
        "--resume-latest", "--stop-after-step", "2",
    ), "resume")
    _, _, resumed = LatestCheckpoint(output / "checkpoints").load_cpu_mapped()
    assert resumed["step"] == 2
    assert resumed["model_config"] == config
    assert saved_path.read_bytes() == original.encode("utf-8")


def test_resume_rejects_different_profile_before_model_allocation(
    training_environment, tmp_path, monkeypatch,
):
    profile_path = tmp_path / "custom.jsonc"
    original = write_profile(profile_path)
    output = tmp_path / "train"
    run_attempt(arguments(
        monkeypatch, output, "--total-batch-token-size", "32", "--model-profile", profile_path, "--stop-after-step", "1",
    ), "first")
    write_profile(profile_path, n_kv_head=4)

    def forbidden_allocation(*args, **kwargs):
        pytest.fail("profile mismatch reached model allocation")

    monkeypatch.setattr(rlm_module, "RecurrentGPT", forbidden_allocation)
    with pytest.raises(ValueError, match="profile"):
        run_attempt(arguments(
            monkeypatch, output, "--total-batch-token-size", "32", "--model-profile", profile_path, "--resume-latest",
        ), "resume")
    assert (output / "checkpoints" / "model-profile.jsonc").read_bytes() == original.encode("utf-8")


@pytest.mark.parametrize("saved_profile_exists", [True, False])
def test_explicit_resume_profile_preserves_equivalent_saved_notes_or_restores_missing_copy(
    training_environment, tmp_path, monkeypatch, saved_profile_exists,
):
    profile_path = tmp_path / "custom.jsonc"
    original = write_profile(profile_path)
    output = tmp_path / "train"
    run_attempt(arguments(
        monkeypatch, output, "--total-batch-token-size", "32", "--model-profile", profile_path, "--stop-after-step", "1",
    ), "first")
    saved_path = output / "checkpoints" / "model-profile.jsonc"
    if not saved_profile_exists:
        saved_path.unlink()
    replacement = "// New notes for the same architecture\n" + original
    profile_path.write_text(replacement)
    run_attempt(arguments(
        monkeypatch, output, "--total-batch-token-size", "32", "--model-profile", profile_path, "--resume-latest", "--stop-after-step", "2",
    ), "resume")
    assert saved_path.read_bytes() == (original if saved_profile_exists else replacement).encode("utf-8")


def test_checkpoint_copy_uses_document_loaded_before_model_allocation(
    training_environment, tmp_path, monkeypatch,
):
    import nanochat.rlm_training as training_module

    profile_path = tmp_path / "custom.jsonc"
    original = write_profile(profile_path)
    build_optimizer = training_module.build_optimizer

    def remove_source_after_allocation(model, config):
        profile_path.unlink()
        return build_optimizer(model, config)

    monkeypatch.setattr(training_module, "build_optimizer", remove_source_after_allocation)
    output = tmp_path / "train"
    run_attempt(arguments(
        monkeypatch, output, "--total-batch-token-size", "32", "--model-profile", profile_path, "--stop-after-step", "1",
    ), "first")
    assert (output / "checkpoints" / "model-profile.jsonc").read_bytes() == original.encode("utf-8")

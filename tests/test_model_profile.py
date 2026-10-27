"""Model profiles drive architecture and survive checkpoint publication intact."""

import json

import pytest


PROFILE = {
    "n_layer": 3, "n_embd": 32, "n_head": 2, "n_kv_head": 1,
    "text_length": 5, "memory_tokens": 3, "eom_tokens": 1,
}


def test_commented_profile_builds_recurrent_model_with_independent_kv_heads(monkeypatch):
    import torch
    import nanochat.gpt as gpt
    import nanochat.rlm as rlm
    from nanochat.model_profile import RLMModelProfile

    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", torch.float32)
    monkeypatch.setattr(rlm, "COMPUTE_DTYPE", torch.float32)
    text = "// A small GQA model\n" + json.dumps(PROFILE)[:-1] + ", /* EOM is explicit */}"
    profile = RLMModelProfile.from_jsonc(text)
    model = rlm.RecurrentGPT(profile.model_config(vocab_size=32))
    model.init_weights()
    first = model.forward_cycle(torch.ones((1, 5), dtype=torch.long))
    second = model.forward_cycle(torch.ones((1, 5), dtype=torch.long), first.memory)
    assert model.config.sequence_len == 12
    assert model.transformer.h[0].attn.c_k.weight.shape == (16, 32)
    assert second.memory.shape == (1, 3, 32)
    assert second.logits.shape == (1, 6, 32)


@pytest.mark.parametrize("value", [0, 2, True, 1.0, "1", None])
def test_extra_context_token_must_be_integer_one(value):
    from nanochat.model_profile import RLMModelProfile

    with pytest.raises(ValueError, match="eom_tokens"):
        RLMModelProfile.from_jsonc(json.dumps({**PROFILE, "eom_tokens": value}))


@pytest.mark.parametrize("change", [
    {"n_embd": 31}, {"n_embd": 28, "n_head": 4}, {"n_kv_head": 3},
    {"text_length": 0}, {"memory_tokens": True}, {"n_layer": -1},
])
def test_incompatible_dimensions_fail_during_profile_loading(change):
    from nanochat.model_profile import RLMModelProfile

    with pytest.raises(ValueError):
        RLMModelProfile.from_jsonc(json.dumps({**PROFILE, **change}))


@pytest.mark.parametrize("text", [
    "[]", "{", json.dumps({**PROFILE, "cycles": 4}),
    json.dumps({k: v for k, v in PROFILE.items() if k != "eom_tokens"}),
    json.dumps(PROFILE)[:-1] + ', "eom_tokens": 2}',
])
def test_malformed_or_ambiguous_schema_is_rejected(text):
    from nanochat.model_profile import RLMModelProfile

    with pytest.raises(ValueError):
        RLMModelProfile.from_jsonc(text)


def test_copy_preserves_loaded_text_and_reuses_existing_profile(tmp_path):
    from nanochat.model_profile import ModelProfileDocument

    source = tmp_path / "source.jsonc"
    original = ("// Preserve my comments\r\n" + json.dumps(PROFILE) + "\r\n").encode()
    source.write_bytes(original)
    document = ModelProfileDocument.read(source)
    source.unlink()
    destination = tmp_path / "saved.jsonc"
    document.copy_once(destination)
    before = destination.stat()
    equivalent = tmp_path / "equivalent.jsonc"
    equivalent.write_text(json.dumps(PROFILE))
    ModelProfileDocument.read(equivalent).copy_once(destination)
    assert destination.read_bytes() == original
    assert destination.stat().st_ino == before.st_ino
    assert destination.stat().st_mtime_ns == before.st_mtime_ns


def test_copy_never_replaces_a_different_or_symlinked_profile(tmp_path):
    from nanochat.model_profile import ModelProfileDocument

    source = tmp_path / "source.jsonc"
    source.write_text(json.dumps(PROFILE))
    document = ModelProfileDocument.read(source)
    destination = tmp_path / "saved.jsonc"
    other = json.dumps({**PROFILE, "memory_tokens": 2})
    destination.write_text(other)
    with pytest.raises(ValueError, match="different"):
        document.copy_once(destination)
    assert destination.read_text() == other
    destination.unlink()
    destination.symlink_to(source)
    with pytest.raises(ValueError):
        document.copy_once(destination)
    assert destination.is_symlink()


def test_failed_publication_leaves_no_partial_profile(tmp_path, monkeypatch):
    import os
    from nanochat.model_profile import ModelProfileDocument

    source = tmp_path / "source.jsonc"
    source.write_text(json.dumps(PROFILE))
    document = ModelProfileDocument.read(source)
    destination = tmp_path / "saved.jsonc"

    def failed_link(*args, **kwargs):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(os, "link", failed_link)
    with pytest.raises(ValueError, match="publication failure"):
        document.copy_once(destination)
    assert not destination.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["source.jsonc"]

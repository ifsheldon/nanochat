"""Input-table transfer must not import the source model's other learned state."""

from dataclasses import asdict
import hashlib
import json
import shutil

import pytest
import torch

import nanochat.gpt as gpt_module
import nanochat.rlm as rlm_module
from nanochat.gpt import GPT, GPTConfig
from nanochat.rlm import RecurrentGPT, RLMConfig
from nanochat.rlm_initialization import TokenEmbeddingInitialization


@pytest.fixture(autouse=True)
def cpu_float32(monkeypatch):
    monkeypatch.setattr(gpt_module, "COMPUTE_DTYPE", torch.float32)
    monkeypatch.setattr(rlm_module, "COMPUTE_DTYPE", torch.float32)


@pytest.fixture
def isolated_compiler_cache():
    # Real optimizer kernels share a process-wide specialization cache. Keep
    # unrelated tests' model shapes out of these FP32/BF16 transfer checks.
    torch.compiler.reset()
    try:
        yield
    finally:
        torch.compiler.reset()


def target_model(*, vocab_size=32, n_embd=32):
    torch.manual_seed(61)
    model = RecurrentGPT(RLMConfig(
        sequence_len=9, text_length=4, memory_tokens=2, vocab_size=vocab_size,
        n_layer=2, n_head=2, n_kv_head=2, n_embd=n_embd,
    ))
    model.init_weights()
    return model


@pytest.fixture
def source_files(tmp_path):
    torch.manual_seed(19)
    # A different depth must be allowed when vocabulary and embedding width match.
    model = GPT(GPTConfig(
        sequence_len=8, vocab_size=32, n_layer=1,
        n_head=2, n_kv_head=2, n_embd=32, window_pattern="L",
    ))
    model.init_weights()
    checkpoint = tmp_path / "base"
    checkpoint.mkdir()
    torch.save(model.state_dict(), checkpoint / "model_000007.pt")
    metadata = {"step": 7, "model_config": asdict(model.config)}
    (checkpoint / "meta_000007.json").write_text(json.dumps(metadata))
    source_tokenizer = tmp_path / "source-tokenizer.pkl"
    target_tokenizer = tmp_path / "target-tokenizer.pkl"
    # The initializer verifies byte identity without unpickling either artifact.
    source_tokenizer.write_bytes(b"same tokenizer identity, not executable pickle")
    target_tokenizer.write_bytes(source_tokenizer.read_bytes())
    return checkpoint, source_tokenizer, target_tokenizer


def initializer(source_files):
    checkpoint, source_tokenizer, target_tokenizer = source_files
    return TokenEmbeddingInitialization.from_checkpoint(
        checkpoint, 7, source_tokenizer, target_tokenizer,
    )


def rewrite_embedding(source_files, replacement):
    checkpoint = source_files[0] / "model_000007.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state["transformer.wte.weight"] = replacement(state["transformer.wte.weight"])
    torch.save(state, checkpoint)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.usefixtures("isolated_compiler_cache")
def test_transfer_changes_only_wte_preserves_rng_and_trains_every_parameter(source_files, monkeypatch, dtype):
    rewrite_embedding(source_files, lambda tensor: tensor.to(dtype))
    model = target_model()
    model.transformer.wte.to(dtype=dtype)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    buffers_before = {name: tensor.clone() for name, tensor in model.named_buffers()}
    parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}
    rng_before = torch.get_rng_state().clone()
    original_load = torch.load
    calls = []

    def checked_load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    source = initializer(source_files)
    contract = source.contract()
    assert json.loads(json.dumps(contract)) == contract
    proof = source.initialize(model)
    assert calls == [{"map_location": "cpu", "weights_only": True, "mmap": True}]
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert {name: id(parameter) for name, parameter in model.named_parameters()} == parameter_ids
    changed = [name for name, tensor in model.state_dict().items() if not torch.equal(tensor, before[name])]
    assert changed == ["transformer.wte.weight"]
    for name, tensor in model.named_buffers():
        assert torch.equal(tensor, buffers_before[name]), name
    assert all(parameter.requires_grad for parameter in model.parameters())
    copied = model.transformer.wte.weight.detach().view(torch.uint8).numpy().tobytes()
    expected_sha256 = hashlib.sha256(copied).hexdigest()
    assert proof["source_sha256"] == proof["copied_sha256"] == expected_sha256
    assert contract["embedding_sha256"] == expected_sha256
    for key, path in (
        ("checkpoint", source_files[0] / "model_000007.pt"),
        ("metadata", source_files[0] / "meta_000007.json"),
        ("source_tokenizer", source_files[1]),
        ("target_tokenizer", source_files[2]),
    ):
        assert contract[key] == {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    # A real recurrent update must still include the transferred table and both
    # independently initialized EOM/summary parameters in the optimizer.
    optimizer = model.setup_optimizer()
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimized == {id(parameter) for parameter in model.parameters()}
    tokens = torch.arange(18).reshape(2, 9) % model.config.vocab_size
    initial_embedding = model.transformer.wte.weight.detach().clone()
    model(tokens[:, :-1], tokens[:, 1:]).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    optimizer.step()
    assert not torch.equal(model.transformer.wte.weight, initial_embedding)
    assert model.transformer.wte.weight in optimizer.state
    assert model.summary_embedding in optimizer.state
    assert model.eom_embedding in optimizer.state


def test_tokenizer_mismatch_is_rejected_before_checkpoint_loading(source_files, monkeypatch):
    source_files[2].write_bytes(b"different tokenizer, same vocabulary size is insufficient")

    def no_load(*args, **kwargs):
        pytest.fail("Mismatched tokenizers must be rejected before loading the checkpoint")

    monkeypatch.setattr(torch, "load", no_load)
    with pytest.raises(ValueError, match="tokenizer SHA-256 identities"):
        initializer(source_files)


@pytest.mark.parametrize("field,value", [("vocab_size", 31), ("n_embd", 48)])
def test_target_metadata_mismatch_does_not_change_any_state(source_files, field, value):
    source = initializer(source_files)
    model = target_model(**{field: value})
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with pytest.raises(ValueError, match="vocabulary size and embedding width"):
        source.initialize(model)
    assert all(torch.equal(tensor, before[name]) for name, tensor in model.state_dict().items())


@pytest.mark.parametrize("kind", ["padding", "dtype", "nonfinite", "width", "rows", "missing"])
def test_invalid_source_embedding_is_rejected_without_changing_target(source_files, kind):
    if kind == "padding":
        replacement = lambda tensor: torch.cat((tensor, tensor[:1]), dim=0)
        match = "padded token embedding shapes"
    elif kind == "dtype":
        replacement = lambda tensor: tensor.to(torch.bfloat16)
        match = "dtypes must match"
    elif kind == "nonfinite":
        replacement = lambda tensor: tensor.fill_(float("nan"))
        match = "non-finite"
    elif kind == "width":
        replacement = lambda tensor: tensor[:, :-1].contiguous()
        match = "shape does not match source model metadata"
    elif kind == "rows":
        replacement = lambda tensor: tensor[:1].contiguous()
        match = "shape does not match source model metadata"
    else:
        replacement = lambda tensor: None
        match = "lacks transformer.wte.weight"
    rewrite_embedding(source_files, replacement)
    model = target_model()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with pytest.raises(ValueError, match=match):
        initializer(source_files).initialize(model)
    assert all(torch.equal(tensor, before[name]) for name, tensor in model.state_dict().items())


@pytest.mark.parametrize("field,value", [("step", 8), ("vocab_size", 0), ("n_embd", "32")])
def test_source_metadata_must_publish_valid_dimensions_and_step(source_files, field, value):
    path = source_files[0] / "meta_000007.json"
    metadata = json.loads(path.read_text())
    target = metadata if field == "step" else metadata["model_config"]
    target[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="requested checkpoint step|positive integer"):
        initializer(source_files)


@pytest.mark.parametrize("source_index", [0, 1, 2])
def test_changed_source_is_rejected_before_transfer(source_files, source_index):
    source = initializer(source_files)
    path = source_files[source_index]
    if source_index == 0:
        path = path / "meta_000007.json"
    path.write_bytes(path.read_bytes() + b" ")
    model = target_model()
    before = model.transformer.wte.weight.detach().clone()
    with pytest.raises(ValueError, match="changed after validation"):
        source.initialize(model)
    assert torch.equal(model.transformer.wte.weight, before)


def test_symlink_source_is_rejected(source_files):
    path = source_files[1]
    moved = path.with_suffix(".original")
    path.rename(moved)
    path.symlink_to(moved)
    with pytest.raises(ValueError, match="cannot contain symlinks"):
        initializer(source_files)


def test_frozen_target_is_rejected_without_overriding_training_policy(source_files):
    model = target_model()
    model.lm_head.requires_grad_(False)
    before = model.transformer.wte.weight.detach().clone()
    with pytest.raises(ValueError, match="every model parameter to remain trainable"):
        initializer(source_files).initialize(model)
    assert torch.equal(model.transformer.wte.weight, before)
    assert not model.lm_head.weight.requires_grad


@pytest.fixture
def bundle_inputs(source_files, tmp_path):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.pkl").write_bytes(source_files[1].read_bytes())
    torch.save(torch.ones(32, dtype=torch.int32), tokenizer_dir / "token_bytes.pt")
    return source_files[0], tokenizer_dir, tmp_path / "bundle"


def export_bundle(bundle_inputs):
    checkpoint, tokenizer_dir, destination = bundle_inputs
    return TokenEmbeddingInitialization.export_from_checkpoint(checkpoint, 7, tokenizer_dir, destination)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_full_and_embedding_only_checkpoints_use_the_same_loader(source_files, bundle_inputs, dtype):
    # A view with a much larger backing storage must not transfer the unused tail.
    def as_view(tensor):
        storage = torch.full((tensor.numel() * 10,), 99.0, dtype=dtype)
        view = storage[:tensor.numel()].view(tensor.shape)
        view.copy_(tensor)
        return view

    rewrite_embedding(source_files, as_view)
    original = initializer(source_files)
    output = export_bundle(bundle_inputs)
    compact_path = output / "model_000007.pt"
    compact = torch.load(compact_path, map_location="cpu", weights_only=True)
    assert set(compact) == {"transformer.wte.weight"}
    table = compact["transformer.wte.weight"]
    assert table.untyped_storage().nbytes() == table.numel() * table.element_size()
    assert compact_path.stat().st_size < (source_files[0] / compact_path.name).stat().st_size
    assert {path.name for path in output.iterdir()} == {
        "model_000007.pt", "meta_000007.json", "manifest.json", "tokenizer.pkl", "token_bytes.pt",
    }
    expected = original.embedding_sha256
    original_checkpoint_sha256 = original.contract()["checkpoint"]["sha256"]
    # The exported checkpoint must stay usable after removing the entire source.
    del original
    shutil.rmtree(source_files[0])
    for source in (
        TokenEmbeddingInitialization.from_checkpoint(output, 7, output / "tokenizer.pkl", source_files[2]),
        TokenEmbeddingInitialization.from_bundle(output, source_files[2]),
    ):
        model = target_model()
        model.transformer.wte.to(dtype=dtype)
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        proof = source.initialize(model)
        assert proof["copied_sha256"] == expected
        changed = [name for name, tensor in model.state_dict().items() if not torch.equal(tensor, before[name])]
        assert changed == ["transformer.wte.weight"]
    assert source.contract()["source_checkpoint"]["sha256"] == original_checkpoint_sha256


@pytest.mark.parametrize("asset", ["model_000007.pt", "meta_000007.json", "tokenizer.pkl", "token_bytes.pt"])
def test_bundle_rejects_modified_files_before_loading(bundle_inputs, source_files, monkeypatch, asset):
    output = export_bundle(bundle_inputs)
    path = output / asset
    path.write_bytes(path.read_bytes() + b"tampered")

    def no_load(*args, **kwargs):
        pytest.fail("Bundle checksum failures must be rejected before loading tensors")

    monkeypatch.setattr(torch, "load", no_load)
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        TokenEmbeddingInitialization.from_bundle(output, source_files[2])


@pytest.mark.parametrize("field,replacement,match", [
    ("embedding_sha256", "0" * 64, "table SHA-256 mismatch"),
    ("embedding_shape", [65, 32], "shape or dtype"),
    ("embedding_dtype", "torch.bfloat16", "shape or dtype"),
    ("vocab_size", 31, "dimensions do not match"),
    ("version", True, "version"),
    ("files", {"../model_000007.pt": "0" * 64}, "canonical filenames"),
])
def test_bundle_rejects_invalid_manifest(bundle_inputs, source_files, field, replacement, match):
    output = export_bundle(bundle_inputs)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = replacement
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        TokenEmbeddingInitialization.from_bundle(output, source_files[2])


def test_bundle_rejects_target_tokenizer_mismatch(bundle_inputs, source_files, monkeypatch):
    output = export_bundle(bundle_inputs)
    source_files[2].write_bytes(b"another vocabulary")

    def no_load(*args, **kwargs):
        pytest.fail("Tokenizer mismatch must be rejected before loading tensors")

    monkeypatch.setattr(torch, "load", no_load)
    with pytest.raises(ValueError, match="tokenizer SHA-256 identities"):
        TokenEmbeddingInitialization.from_bundle(output, source_files[2])


def test_bundle_rechecks_changed_assets_before_transfer(bundle_inputs, source_files):
    output = export_bundle(bundle_inputs)
    source = TokenEmbeddingInitialization.from_bundle(output, source_files[2])
    (output / "token_bytes.pt").write_bytes(b"changed after validation")
    model = target_model()
    before = model.transformer.wte.weight.detach().clone()
    with pytest.raises(ValueError, match="changed after validation"):
        source.initialize(model)
    assert torch.equal(model.transformer.wte.weight, before)


@pytest.mark.parametrize("filename", ["manifest.json", "model_000007.pt", "tokenizer.pkl", "token_bytes.pt"])
def test_bundle_rejects_symlinks(bundle_inputs, source_files, filename):
    output = export_bundle(bundle_inputs)
    path = output / filename
    original = path.with_suffix(".original")
    path.rename(original)
    path.symlink_to(original)
    with pytest.raises(ValueError, match="cannot contain symlinks"):
        TokenEmbeddingInitialization.from_bundle(output, source_files[2])


def test_export_refuses_existing_directory_and_cleans_failed_export(bundle_inputs, monkeypatch):
    destination = bundle_inputs[2]
    destination.mkdir()
    keep = destination / "keep"
    keep.write_text("existing data")
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        export_bundle(bundle_inputs)
    assert keep.read_text() == "existing data"
    keep.unlink()
    destination.rmdir()

    def fail_save(*args, **kwargs):
        assert not (destination / "manifest.json").exists()
        raise OSError("simulated storage failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="simulated storage failure"):
        export_bundle(bundle_inputs)
    assert not destination.exists()

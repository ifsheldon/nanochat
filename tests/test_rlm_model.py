"""CPU checks for recurrent causality, gradients, losses, and model integration."""

import copy

import pytest
import torch
import torch.nn.functional as F

import nanochat.flash_attention as attention_module
import nanochat.gpt as gpt_module
import nanochat.rlm as rlm_module
from nanochat.gpt import GPT, GPTConfig
from nanochat.rlm import RLMConfig, RecurrentGPT


@pytest.fixture(autouse=True)
def cpu_float32(monkeypatch):
    monkeypatch.setattr(attention_module, "USE_FA3", False)
    monkeypatch.setattr(gpt_module, "COMPUTE_DTYPE", torch.float32)
    monkeypatch.setattr(rlm_module, "COMPUTE_DTYPE", torch.float32)


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model(*, activation_checkpointing=False, contextual=True):
    torch.manual_seed(41)
    config = RLMConfig(
        sequence_len=13, vocab_size=32, n_layer=2,
        n_head=2, n_kv_head=2, n_embd=32,
        text_length=6, memory_tokens=3,
        activation_checkpointing=activation_checkpointing,
    )
    with torch.device("meta"):
        model = RecurrentGPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    if contextual:
        # At exact initialization every block's zero output projection prevents
        # contextual reading. Exercise the trained regime without a warmup run.
        with torch.no_grad():
            for block in model.transformer.h:
                block.attn.c_proj.weight.normal_(std=0.05)
                block.mlp.c_proj.weight.normal_(std=0.05)
            model.lm_head.weight.normal_(std=0.1)
            model.smear_lambda.fill_(0.2)
    return model


def episode_tokens(batch=2):
    raw = torch.arange(batch * 13).reshape(batch, 13) % 32
    return raw[:, :-1], raw[:, 1:]


def test_causal_text_and_summary_suffix_cannot_reveal_future():
    model = make_model()
    text = torch.tensor([[1, 2, 3, 4, 5, 6]])
    initial = model.forward_cycle(text)
    changed = text.clone()
    changed[:, 4:] = torch.tensor([20, 21])
    future_changed = model.forward_cycle(changed)
    torch.testing.assert_close(initial.logits[:, :5], future_changed.logits[:, :5], rtol=0, atol=0)
    assert not torch.equal(initial.logits[:, 5:], future_changed.logits[:, 5:])

    with torch.no_grad():
        model.summary_embedding.normal_(std=2)
    summary_changed = model.forward_cycle(text)
    torch.testing.assert_close(initial.logits, summary_changed.logits, rtol=0, atol=0)
    assert not torch.equal(initial.memory, summary_changed.memory)


def test_second_cycle_loss_trains_the_first_summary_through_memory():
    model = make_model()
    x, y = episode_tokens()
    first = model.forward_cycle(x[:, :6])
    first.memory.retain_grad()
    second = model.forward_cycle(x[:, 6:], first.memory, y[:, 6:])
    second.loss.backward()

    assert first.memory.grad is not None
    assert torch.isfinite(first.memory.grad).all()
    assert torch.count_nonzero(first.memory.grad) > 0
    assert model.summary_embedding.grad is not None
    assert torch.count_nonzero(model.summary_embedding.grad) > 0
    intact = second.loss.detach()
    disrupted = model.forward_cycle(x[:, 6:], torch.zeros_like(first.memory), y[:, 6:]).loss
    assert not torch.isclose(intact, disrupted, rtol=1e-5, atol=1e-6)


def test_episode_preserves_both_boundary_targets_and_weights_valid_targets_equally():
    model = make_model()
    x, y = episode_tokens()
    first = model.forward_cycle(x[:, :6])
    second = model.forward_cycle(x[:, 6:], first.memory)
    logits = torch.cat((first.logits, second.logits), dim=1)
    targets = y.clone()
    targets[0, 1:4] = -1  # The cycles now have unequal valid-target counts.
    joint_targets = torch.cat((
        torch.cat((x[:, :1], targets[:, :6]), dim=1),
        torch.cat((x[:, 6:7], targets[:, 6:]), dim=1),
    ), dim=1)
    expected = F.cross_entropy(logits.reshape(-1, 32), joint_targets.reshape(-1), ignore_index=-1, reduction='none')

    actual = model(x, targets, loss_reduction='none')
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model(x, targets), expected.sum() / (joint_targets != -1).sum())
    torch.testing.assert_close(model(x, targets, loss_reduction='sum'), expected.sum())
    # Both the preceding text and the new EOM predict the boundary token.
    old_context_loss = F.cross_entropy(first.logits[:, -1], x[:, 6], reduction='none')
    memory_only_loss = F.cross_entropy(second.logits[:, 0], x[:, 6], reduction='none')
    torch.testing.assert_close(actual.reshape(2, 14)[:, 6], old_context_loss)
    torch.testing.assert_close(actual.reshape(2, 14)[:, 7], memory_only_loss)
    # Initial EOM predicts the first input, including a document's BOS token.
    initial_loss = F.cross_entropy(first.logits[:, 0], x[:, 0], reduction='none')
    torch.testing.assert_close(actual.reshape(2, 14)[:, 0], initial_loss)


def test_eom_supervision_is_retained_when_shifted_text_targets_are_masked():
    model = make_model()
    text = torch.tensor([[0, 1, 2, 3, 4, 5]])  # BOS is 0 in this test vocabulary.
    masked_targets = torch.full_like(text, -1)
    output = model.forward_cycle(text)
    expected = F.cross_entropy(output.logits[:, 0], text[:, 0])
    loss = model.forward_cycle(text, targets=masked_targets).loss
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(model(text, masked_targets), expected)
    loss.backward()
    assert torch.isfinite(model.eom_embedding.grad).all()
    assert torch.count_nonzero(model.eom_embedding.grad) > 0


def test_eom_only_future_loss_trains_the_memory_writer():
    model = make_model()
    x, _ = episode_tokens()
    first = model.forward_cycle(x[:, :6])
    first.memory.retain_grad()
    second = model.forward_cycle(x[:, 6:], first.memory, torch.full_like(x[:, 6:], -1))
    second.loss.backward()
    assert torch.count_nonzero(first.memory.grad) > 0
    assert torch.isfinite(first.memory.grad).all()
    assert torch.count_nonzero(model.summary_embedding.grad) > 0
    assert torch.isfinite(model.summary_embedding.grad).all()


@pytest.mark.parametrize("with_memory", [False, True])
def test_empty_eom_readout_matches_full_chunk_without_seeing_its_text(with_memory):
    model = make_model()
    text = torch.tensor([[1, 2, 3, 4, 5, 6]])
    memory = model.forward_cycle(torch.flip(text, dims=(1,))).memory if with_memory else None
    expected = model.forward_cycle(text, memory).logits[:, :1]
    empty = torch.empty((1, 0), dtype=torch.long)
    actual = model.forward_cycle(empty, memory)
    torch.testing.assert_close(actual.logits, expected)
    changed = model.forward_cycle((text + 7) % 32, memory)
    torch.testing.assert_close(changed.logits[:, :1], expected, rtol=0, atol=0)
    if with_memory:
        read_only = model.forward_cycle(empty, memory, write_memory=False)
        torch.testing.assert_close(read_only.logits, expected)
        assert read_only.memory is None
    else:
        with pytest.raises(ValueError, match="empty initial EOM readout requires write_memory=True"):
            model.forward_cycle(empty, write_memory=False)
    with pytest.raises(ValueError, match="empty RLM text chunk"):
        model.forward_cycle(empty, memory, targets=empty)
    with pytest.raises(ValueError, match="empty RLM text chunk"):
        model.cycle_targets(empty, empty)


def test_eom_and_text_use_one_joint_vocabulary_projection():
    model = make_model()
    x, y = episode_tokens()
    shapes = []
    handle = model.lm_head.register_forward_pre_hook(lambda module, inputs: shapes.append(inputs[0].shape))
    try:
        output = model.forward_cycle(x[:, :6], targets=y[:, :6], loss_reduction='none')
    finally:
        handle.remove()
    assert shapes == [torch.Size((2, 7, 32))]
    assert output.loss.numel() == 14


def test_complete_episode_resets_memory_and_matches_explicit_handoffs():
    model = make_model()
    x, _ = episode_tokens()
    first = model.forward_cycle(x[:, :6])
    second = model.forward_cycle(x[:, 6:], first.memory)
    expected = torch.cat((first.logits, second.logits), dim=1)
    torch.testing.assert_close(model(x), expected)
    model(torch.flip(x, dims=(1,)))
    torch.testing.assert_close(model(x), expected)


def test_eom_cycle_matches_explicit_gpt_prefix_and_retains_standard_cached_decoding():
    from nanochat.engine import KVCache

    recurrent = make_model()
    config = GPTConfig(
        sequence_len=13, vocab_size=32, n_layer=2,
        n_head=2, n_kv_head=2, n_embd=32, window_pattern="L",
    )
    ordinary = GPT(config)
    ordinary.init_weights()
    ordinary.load_state_dict({
        key: value for key, value in recurrent.state_dict().items()
        if key not in ("summary_embedding", "eom_embedding")
    })
    with torch.no_grad():
        # Surrogate token 0 independently represents the model-owned EOM query.
        ordinary.transformer.wte.weight[0].copy_(recurrent.eom_embedding)
        for embedding in ordinary.value_embeds.values():
            embedding.weight[0].zero_()
    text = torch.tensor([[1, 2, 3, 4, 5, 6]])
    prefixed = torch.cat((torch.zeros((1, 1), dtype=torch.long), text), dim=1)
    expected = ordinary(prefixed)
    torch.testing.assert_close(recurrent.forward_cycle(text).logits, expected)

    cache = KVCache(batch_size=1, num_heads=2, seq_len=13, head_dim=16, num_layers=2, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        prefix = ordinary(prefixed[:, :3], kv_cache=cache)
        suffix = [ordinary(prefixed[:, index:index + 1], kv_cache=cache) for index in range(3, 7)]
    torch.testing.assert_close(torch.cat([prefix, *suffix], dim=1), expected)


def test_activation_checkpointing_preserves_cross_cycle_loss_and_gradients():
    ordinary = make_model()
    checkpointed = copy.deepcopy(ordinary)
    checkpointed.config.activation_checkpointing = True
    x, y = episode_tokens()
    ordinary_loss = ordinary(x, y)
    checkpointed_loss = checkpointed(x, y)
    ordinary_loss.backward()
    checkpointed_loss.backward()
    torch.testing.assert_close(ordinary_loss, checkpointed_loss)
    for (name, parameter), (other_name, other_parameter) in zip(ordinary.named_parameters(), checkpointed.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(parameter.grad, other_parameter.grad)


@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_terminal_summary_elision_preserves_logits_loss_and_all_gradients(
    monkeypatch, compute_dtype, activation_checkpointing,
):
    monkeypatch.setattr(gpt_module, "COMPUTE_DTYPE", compute_dtype)
    monkeypatch.setattr(rlm_module, "COMPUTE_DTYPE", compute_dtype)
    full = make_model(activation_checkpointing=activation_checkpointing)
    elided = copy.deepcopy(full)
    x, y = episode_tokens()
    full_first = full.forward_cycle(x[:, :6], targets=y[:, :6], loss_reduction='sum')
    full_second = full.forward_cycle(x[:, 6:], full_first.memory)
    elided_first = elided.forward_cycle(x[:, :6], targets=y[:, :6], loss_reduction='sum')
    elided_first.memory.retain_grad()
    elided_second = elided.forward_cycle(x[:, 6:], elided_first.memory, write_memory=False)
    labels = full.cycle_targets(x[:, 6:], y[:, 6:])
    full_loss = full_first.loss + F.cross_entropy(full_second.logits.reshape(-1, 32), labels.reshape(-1), reduction='sum')
    elided_loss = elided_first.loss + F.cross_entropy(elided_second.logits.reshape(-1, 32), labels.reshape(-1), reduction='sum')
    tolerance = {'atol': 3e-3, 'rtol': 3e-2} if compute_dtype == torch.bfloat16 else {'atol': 1e-6, 'rtol': 1e-5}
    torch.testing.assert_close(elided_second.logits, full_second.logits, **tolerance)
    torch.testing.assert_close(elided_loss, full_loss, **tolerance)
    assert full_second.memory is not None
    assert elided_second.memory is None

    full_loss.backward()
    elided_loss.backward()
    assert torch.count_nonzero(elided_first.memory.grad) > 0
    assert torch.isfinite(elided_first.memory.grad).all()
    for (name, parameter), (other_name, other_parameter) in zip(full.named_parameters(), elided.named_parameters()):
        assert name == other_name
        assert parameter.grad is not None, name
        assert other_parameter.grad is not None, name
        torch.testing.assert_close(other_parameter.grad, parameter.grad, **tolerance)


def test_native_episode_omits_only_the_terminal_summary():
    model = make_model()
    raw = torch.arange(38).reshape(2, 19) % 32
    inputs, targets = raw[:, :-1], raw[:, 1:]
    lengths = []
    handle = model.transformer.h[0].register_forward_pre_hook(lambda module, args: lengths.append(args[0].size(1)))
    try:
        actual = model(inputs, targets)
    finally:
        handle.remove()
    assert lengths == [10, 13, 10]
    memory = None
    full_loss = 0
    for start in range(0, 18, 6):
        output = model.forward_cycle(inputs[:, start:start + 6], memory, targets[:, start:start + 6], loss_reduction='sum')
        full_loss = full_loss + output.loss
        memory = output.memory
    torch.testing.assert_close(actual, full_loss / 42)


@pytest.mark.parametrize("activation_checkpointing", [False, True])
@pytest.mark.parametrize("write_memory", [False, True])
def test_compiled_cycle_preserves_memory_handoff_gradients(activation_checkpointing, write_memory):
    model = make_model(activation_checkpointing=activation_checkpointing)
    reference = copy.deepcopy(model)
    compiled_cycle = torch.compile(model.forward_cycle, backend="eager", fullgraph=True)
    x, y = episode_tokens()
    first = compiled_cycle(x[:, :6])
    second = compiled_cycle(x[:, 6:], first.memory, y[:, 6:], write_memory=write_memory)
    second.loss.backward()
    reference_first = reference.forward_cycle(x[:, :6])
    reference_second = reference.forward_cycle(x[:, 6:], reference_first.memory, y[:, 6:], write_memory=write_memory)
    reference_second.loss.backward()
    torch.testing.assert_close(second.loss, reference_second.loss)
    torch.testing.assert_close(model.summary_embedding.grad, reference.summary_embedding.grad)
    assert (second.memory is None) == (not write_memory)


def test_bfloat16_memory_handoff_keeps_the_gradient_path(monkeypatch):
    monkeypatch.setattr(gpt_module, "COMPUTE_DTYPE", torch.bfloat16)
    monkeypatch.setattr(rlm_module, "COMPUTE_DTYPE", torch.bfloat16)
    model = make_model()
    x, y = episode_tokens()
    first = model.forward_cycle(x[:, :6])
    assert first.memory.dtype == torch.bfloat16
    second = model.forward_cycle(x[:, 6:], first.memory, y[:, 6:])
    second.loss.backward()
    assert torch.isfinite(second.loss)
    assert torch.isfinite(model.summary_embedding.grad).all()
    assert torch.count_nonzero(model.summary_embedding.grad) > 0


def test_shared_summary_and_eom_parameters_are_optimized_and_checkpointed(tmp_path):
    model = make_model()
    optimizer = model.setup_optimizer()
    parameters = [parameter for group in optimizer.param_groups for parameter in group['params']]
    assert len(parameters) == len({id(parameter) for parameter in parameters})
    assert {id(parameter) for parameter in parameters} == {id(parameter) for parameter in model.parameters()}
    counts = model.num_scaling_params()
    assert counts['total'] == sum(parameter.numel() for parameter in model.parameters())

    x, y = episode_tokens()
    model(x, y).backward()
    initial_summary = model.summary_embedding.detach().clone()
    initial_eom = model.eom_embedding.detach().clone()
    optimizer.step()
    assert not torch.equal(model.summary_embedding, initial_summary)
    assert not torch.equal(model.eom_embedding, initial_eom)
    assert optimizer.state[model.summary_embedding]['step'] == 1
    assert optimizer.state[model.eom_embedding]['step'] == 1

    checkpoint_path = tmp_path / "model.pt"
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint_path)
    restored = make_model()
    restored_optimizer = restored.setup_optimizer()
    saved = torch.load(checkpoint_path, weights_only=True)
    restored.load_state_dict(saved['model'])
    restored_optimizer.load_state_dict(saved['optimizer'])
    torch.testing.assert_close(restored(x), model(x))
    torch.testing.assert_close(
        restored_optimizer.state[restored.summary_embedding]['exp_avg'],
        optimizer.state[model.summary_embedding]['exp_avg'],
    )
    torch.testing.assert_close(
        restored_optimizer.state[restored.eom_embedding]['exp_avg'],
        optimizer.state[model.eom_embedding]['exp_avg'],
    )


def test_cycle_accepts_partial_text_but_rejects_invalid_memory_and_episode():
    model = make_model()
    text = torch.tensor([[1]])
    first = model.forward_cycle(text)
    assert first.logits.shape == (1, 2, 32)
    model.forward_cycle(text, first.memory)
    with pytest.raises(ValueError, match="memory must have shape"):
        model.forward_cycle(text, first.memory[:, :2])
    with pytest.raises(ValueError, match="compute dtype"):
        model.forward_cycle(text, first.memory.double())
    with pytest.raises(ValueError, match="complete text_length chunks"):
        model(text)
    with pytest.raises(ValueError, match="targets must match"):
        model.forward_cycle(text, targets=torch.tensor([[2, 3]]))


def test_config_rejects_truncated_attention_and_insufficient_cycle_capacity():
    with pytest.raises(ValueError, match="full causal attention"):
        RLMConfig(window_pattern="SSSL")
    with pytest.raises(ValueError, match="cover EOM, text, and both memory blocks"):
        RLMConfig(sequence_len=1024)
    with pytest.raises(ValueError, match="2048 < 2049"):
        RLMConfig(text_length=1024)


def test_native_generation_crosses_boundaries_and_resets_between_calls():
    from nanochat.rlm_inference import RLMGenerator

    model = make_model().eval()
    prompt = [1, 2, 3]  # A partial first chunk, followed by two memory handoffs.
    expected = list(RLMGenerator(model).generate(prompt, 15, temperature=0.7, top_k=8, seed=17))
    generated = list(model.generate(prompt, 15, temperature=0.7, top_k=8, seed=17))
    assert len(generated) == 15
    assert generated == expected
    list(model.generate([8, 9, 10, 11, 12], 9, temperature=0, seed=23))
    assert list(model.generate(prompt, 15, temperature=0.7, top_k=8, seed=17)) == expected


def test_native_generation_requires_explicit_eval_mode():
    model = make_model()
    with pytest.raises(ValueError, match=r"requires model\.eval\(\)"):
        list(model.generate([1, 2, 3], 2))
    assert model.training
    model.eval()
    assert len(list(model.generate([1, 2, 3], 2, temperature=0))) == 2
    assert not model.training


@pytest.mark.parametrize("name,value", [
    ("sequence_len", 0), ("vocab_size", 0), ("n_layer", 0),
    ("n_head", 0), ("n_kv_head", 0), ("n_embd", 0),
    ("text_length", -1), ("memory_tokens", 0),
    ("n_layer", 2.5), ("memory_tokens", True),
])
def test_config_rejects_nonpositive_or_noninteger_dimensions(name, value):
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        RLMConfig(**{name: value})


@pytest.mark.parametrize("dimensions,message", [
    ({"n_embd": 16, "n_head": 1, "n_kv_head": 1}, "at least 24"),
    ({"n_embd": 32, "n_head": 3, "n_kv_head": 1}, "divisible by n_head"),
    ({"n_embd": 32, "n_head": 4, "n_kv_head": 3}, "must divide n_head"),
    ({"n_embd": 32, "n_head": 2, "n_kv_head": 4}, "cannot exceed it"),
    ({"n_embd": 33, "n_head": 1, "n_kv_head": 1}, "must be even"),
])
def test_config_rejects_incompatible_attention_and_smear_dimensions(dimensions, message):
    with pytest.raises(ValueError, match=message):
        RLMConfig(**dimensions)


def test_even_sdpa_head_dimensions_need_no_flash_attention_alignment():
    # Head dimension 10 is valid for SDPA despite not being a multiple of eight.
    model = RecurrentGPT(RLMConfig(
        sequence_len=9, vocab_size=32, n_layer=1,
        n_embd=30, n_head=3, n_kv_head=1, text_length=4, memory_tokens=2,
    ))
    model.init_weights()
    text = torch.tensor([[1, 2, 3, 4]])
    first = model.forward_cycle(text)
    second = model.forward_cycle(text, first.memory, targets=torch.tensor([[2, 3, 4, 5]]))
    second.loss.backward()
    assert torch.isfinite(second.loss)
    assert torch.isfinite(model.transformer.h[0].attn.c_proj.weight.grad).all()

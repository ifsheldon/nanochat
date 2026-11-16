"""Assistant-only episode supervision for recurrent SFT."""

import copy

import pytest
import torch
import torch.nn.functional as F

from nanochat.rlm import RecurrentGPT, RLMConfig
from nanochat.sft_dataloader import parse_episode_mix
from nanochat.rlm_sft import eom_supervision, evaluate_sft_loss, sft_episode_loss, sft_update


TEXT = 4
CYCLES = 3


@pytest.fixture(autouse=True)
def cpu_float32(monkeypatch):
    # The reference gradients are compared at FP32 precision on every host.
    monkeypatch.setattr("nanochat.gpt.COMPUTE_DTYPE", torch.float32)
    monkeypatch.setattr("nanochat.rlm.COMPUTE_DTYPE", torch.float32)
    monkeypatch.setattr("nanochat.flash_attention.USE_FA3", False)


def small_model():
    torch.manual_seed(812)
    model = RecurrentGPT(RLMConfig(
        sequence_len=9, text_length=TEXT, memory_tokens=2,
        vocab_size=32, n_layer=2, n_head=2, n_kv_head=2,
        n_embd=32, window_pattern="L",
    ))
    model.init_weights()
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.05)
            block.mlp.c_proj.weight.normal_(std=0.05)
    return model


def episode():
    # Chunks start at inputs 0, 4, and 8. Row 0's assistant span covers input 8
    # only; row 1's covers inputs 4 and 8. Supervised text targets: 6 and 8.
    inputs = torch.tensor([
        [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    ], dtype=torch.int32)
    mask = torch.tensor([
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    ])
    shifted = torch.tensor([
        [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    ])
    next_mask = torch.cat((mask[:, 1:], torch.zeros(2, 1, dtype=mask.dtype)), dim=1)
    targets = shifted.masked_fill(next_mask == 0, -1)
    return inputs, targets, mask


def test_eom_supervised_only_where_chunk_starts_with_assistant_token():
    _, targets, mask = episode()
    valid = eom_supervision(targets, TEXT, CYCLES)
    assert valid.tolist() == [[False, False, True], [False, True, True]]
    # The rule equals the supervision mask of each chunk's first input token.
    assert valid[:, 1:].tolist() == (mask[:, TEXT::TEXT] == 1).tolist()


def test_episode_loss_matches_manual_assistant_only_average():
    model = small_model()
    inputs, targets, _ = episode()
    loss, sums, counts = sft_episode_loss(model.forward_cycle, inputs.long(), targets, TEXT, CYCLES)
    memory = None
    manual_sum = 0.0
    manual_count = 0
    valid = eom_supervision(targets, TEXT, CYCLES)
    for cycle in range(CYCLES):
        chunk = inputs[:, cycle * TEXT:(cycle + 1) * TEXT].long()
        chunk_targets = targets[:, cycle * TEXT:(cycle + 1) * TEXT]
        output = model.forward_cycle(chunk, memory=memory, write_memory=cycle < CYCLES - 1)
        memory = output.memory
        eom_targets = chunk[:, 0].masked_fill(~valid[:, cycle], -1)
        joint = torch.cat((eom_targets[:, None], chunk_targets), dim=1)
        manual_sum = manual_sum + F.cross_entropy(
            output.logits.reshape(-1, output.logits.size(-1)), joint.reshape(-1), ignore_index=-1, reduction="sum",
        )
        manual_count += int(joint.ne(-1).sum())
        assert counts[cycle].item() == int(joint.ne(-1).sum())
    assert manual_count == 6 + 8 + 3
    assert counts.sum().item() == manual_count
    assert torch.allclose(loss, manual_sum / manual_count, atol=1e-5)
    assert torch.allclose(sums.sum(), manual_sum, atol=1e-4)


def test_update_steps_optimizer_and_reports_progress():
    model = small_model()
    inputs, targets, _ = episode()
    state = {"consumed": 0}

    def loader():
        while True:
            state["consumed"] += 2
            yield inputs.clone(), targets.clone(), dict(state)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    metrics, data_state = sft_update(
        model, model.forward_cycle, optimizer, loader(), 0,
        episode_mix=parse_episode_mix('3x2:2'), loss_config=configured_loss('microbatch', cycles=(3,)),
    )
    assert data_state == {"consumed": 4}
    assert metrics["completed_updates"] == 1
    assert metrics["supervised_targets"] == 2 * 17
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert all(parameter.grad is None for parameter in model.parameters())


def test_evaluation_averages_over_all_supervised_targets():
    model = small_model()
    inputs, targets, _ = episode()
    loader = iter([(inputs, targets, None)] * 3)
    value = evaluate_sft_loss(model.forward_cycle, loader, 3, TEXT)
    loss, _, _ = sft_episode_loss(model.forward_cycle, inputs.long(), targets, TEXT, CYCLES)
    assert abs(value - loss.item()) < 1e-5


def mixed_batches():
    """Three batches with 7, 18, and 6 supervised predictions, including EOM."""
    short_inputs = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [0, 8, 9, 10, 11, 12, 13, 14]], dtype=torch.int32)
    sparse_targets = torch.tensor([[-1, -1, -1, 4, 5, 6, 7, 8], [-1, -1, -1, -1, -1, -1, -1, 15]])
    dense_targets = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 9, 10, 11, 12, 13, 14, 15]])
    long_inputs = torch.arange(12, dtype=torch.int32)[None, :]
    long_targets = torch.tensor([[-1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, 12]])
    return [(short_inputs, sparse_targets, 'first'), (short_inputs, dense_targets, 'second'), (long_inputs, long_targets, 'last')]


def manual_target_sums(model, batches):
    """Independent cross-entropy reduction over text and eligible boundary labels."""
    sums = []
    for inputs, targets, _ in batches:
        total = torch.zeros(())
        memory = None
        cycles = inputs.size(1) // TEXT
        for cycle in range(cycles):
            start, stop = cycle * TEXT, (cycle + 1) * TEXT
            chunk = inputs[:, start:stop].long()
            result = model.forward_cycle(chunk, memory=memory, write_memory=cycle < cycles - 1)
            memory = result.memory
            boundary = chunk[:, :1].clone()
            if cycle == 0:
                boundary.fill_(-1)
            else:
                boundary[targets[:, start - 1] == -1] = -1
            labels = torch.cat((boundary, targets[:, start:stop]), dim=1)
            total = total + F.cross_entropy(result.logits.flatten(0, 1), labels.flatten(), ignore_index=-1, reduction='sum')
        sums.append(total)
    return sums


def configured_loss(mode, weights=None, cycles=(2, 3)):
    from nanochat.sft_training import SFTLossConfig
    return SFTLossConfig.parse(mode, weights, cycles)


@pytest.mark.parametrize('mode', ['microbatch', 'group-token'])
def test_mixed_update_gradient_and_metrics_match_the_selected_objective(mode):
    actual = small_model()
    reference = copy.deepcopy(actual)
    batches = mixed_batches()
    sums = manual_target_sums(reference, batches)
    expected_token_loss = sum(sums) / 31
    if mode == 'microbatch':
        objective = (sums[0] / 7 + sums[1] / 18 + sums[2] / 6) / 3
    else:
        objective = 0.4 * (sums[0] + sums[1]) / 25 + 0.6 * sums[2] / 6
    expected_group_losses = [(sums[0] + sums[1]) / 25, sums[2] / 6]
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    objective.backward()
    ref_optimizer.step()
    optimizer = torch.optim.SGD(actual.parameters(), lr=0.05)
    metrics, state = sft_update(
        actual, actual.forward_cycle, optimizer, iter(batches), 9,
        episode_mix=parse_episode_mix('2x2:2,3x1:1'),
        loss_config=configured_loss(mode, '2:0.4,3:0.6' if mode == 'group-token' else None),
    )
    for parameter, expected in zip(actual.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter, expected, atol=1e-6, rtol=1e-5)
    assert state == 'last'
    assert metrics['completed_updates'] == 10
    assert metrics['loss'] == pytest.approx(objective.item(), abs=1e-6)
    assert metrics['token_loss'] == pytest.approx(expected_token_loss.item(), abs=1e-6)
    assert metrics['supervised_targets'] == 31
    assert metrics['episode_groups']['2']['supervised_targets'] == 25
    assert metrics['episode_groups']['3']['supervised_targets'] == 6
    for cycles, expected_loss in zip((2, 3), expected_group_losses):
        assert metrics['episode_groups'][str(cycles)]['loss'] == pytest.approx(expected_loss.item(), abs=1e-6)
        assert len(metrics['episode_groups'][str(cycles)]['cycle_losses']) == cycles
    assert all(parameter.grad is None for parameter in actual.parameters())


@pytest.mark.parametrize('mode,unchanged', [('microbatch', False), ('group-token', True)])
def test_repartitioning_microbatches_preserves_only_the_group_normalized_objective(mode, unchanged):
    whole = small_model()
    split = copy.deepcopy(whole)
    batches = mixed_batches()
    split_batches = [(x[i:i+1], y[i:i+1], state) for x, y, state in batches[:2] for i in range(2)] + batches[2:]
    config = configured_loss(mode, '2:0.4,3:0.6' if mode == 'group-token' else None)
    metrics = []
    for model, data, mix in [(whole, batches, '2x2:2,3x1:1'), (split, split_batches, '2x1:4,3x1:1')]:
        result, _ = sft_update(model, model.forward_cycle, torch.optim.SGD(model.parameters(), lr=0.05),
                               iter(data), 0, episode_mix=parse_episode_mix(mix), loss_config=config)
        metrics.append(result)
    if unchanged:
        for parameter, expected in zip(split.parameters(), whole.parameters()):
            torch.testing.assert_close(parameter, expected, atol=1e-6, rtol=1e-5)
        assert metrics[0]['loss'] == pytest.approx(metrics[1]['loss'], abs=1e-6)
    else:
        assert any(not torch.allclose(a, b, atol=1e-6, rtol=1e-5) for a, b in zip(split.parameters(), whole.parameters()))
        assert abs(metrics[0]['loss'] - metrics[1]['loss']) > 1e-5


def test_mixed_evaluation_uses_all_targets_independently_of_microbatch_partition():
    model = small_model()
    batches = mixed_batches()
    with torch.no_grad():
        expected = sum(manual_target_sums(model, batches)).item() / 31
    actual = evaluate_sft_loss(model.forward_cycle, iter(batches), 3, TEXT)
    assert actual == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize('invalid', ['pattern', 'zero-targets'])
def test_invalid_update_fails_before_forward_or_optimizer_step(invalid):
    batches = mixed_batches()
    if invalid == 'pattern':
        batches[2] = batches[0]
        error = 'episode mix'
    else:
        x, y, state = batches[-1]
        batches[-1] = (x, torch.full_like(y, -1), state)
        error = 'supervised targets'
    model = small_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    def forbidden_forward(*args, **kwargs):
        pytest.fail('Invalid update reached the model')
    with pytest.raises(ValueError, match=error):
        sft_update(model, forbidden_forward, optimizer, iter(batches), 0,
                   episode_mix=parse_episode_mix('2x2:2,3x1:1'), loss_config=configured_loss('microbatch'))

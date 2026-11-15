"""Assistant-only episode supervision for recurrent SFT."""

import torch
import torch.nn.functional as F

from nanochat.rlm import RecurrentGPT, RLMConfig
from nanochat.rlm_sft import eom_supervision, evaluate_sft_loss, sft_episode_loss, sft_update


TEXT = 4
CYCLES = 3


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
        model, model.forward_cycle, optimizer, loader(), 2, 0, device_batch_size=2, cycles=CYCLES,
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
    value = evaluate_sft_loss(model.forward_cycle, loader, 3, TEXT, CYCLES)
    loss, _, _ = sft_episode_loss(model.forward_cycle, inputs.long(), targets, TEXT, CYCLES)
    assert abs(value - loss.item()) < 1e-5

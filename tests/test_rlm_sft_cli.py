"""CLI contracts and a local one-update mixed-episode SFT workflow."""

import json

import pytest
import torch

from scripts import rlm_sft


MIX = "1x3:1,2x2:1,3x1:1"


@pytest.mark.parametrize("provided,missing", [
    (["--episode-mix", MIX], "--loss-weighting"),
    (["--loss-weighting", "microbatch"], "--episode-mix"),
])
def test_both_selectors_are_required(provided, missing, capsys):
    with pytest.raises(SystemExit) as exc:
        rlm_sft.parse_args(provided)
    assert exc.value.code == 2
    assert missing in capsys.readouterr().err


@pytest.mark.parametrize("extra", [
    ["--loss-weighting", "group-token"],
    ["--loss-weighting", "microbatch", "--group-loss-weights", "1:0.1,2:0.2,3:0.7"],
    ["--loss-weighting", "group-token", "--group-loss-weights", "1:0.1,2:0.9"],
    ["--loss-weighting", "group-token", "--group-loss-weights", "1:0.1,2:0.2,4:0.7"],
    ["--loss-weighting", "group-token", "--group-loss-weights", "1:0.1,2:0.2,3:0.6"],
    ["--loss-weighting", "group-token", "--group-loss-weights", "1:0.1,2:-0.2,3:1.1"],
    ["--loss-weighting", "group-token", "--group-loss-weights", "1:nan,2:0.2,3:0.8"],
    ["--loss-weighting", "group-token", "--group-loss-weights", "1:inf,2:0.2,3:0.8"],
    ["--loss-weighting", "group-token", "--group-loss-weights", "1:0.1,1:0.2,3:0.7"],
    ["--loss-weighting", "group-token", "--group-loss-weights", "0:0.1,2:0.2,3:0.7"],
    ["--loss-weighting", "group-token", "--group-loss-weights", "1=0.1,2=0.2,3=0.7"],
])
def test_invalid_weighting_is_an_argument_error(extra, capsys):
    with pytest.raises(SystemExit) as exc:
        rlm_sft.parse_args(["--episode-mix", MIX, *extra])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_parser_resolves_typed_inputs_without_fixed_shape_arguments():
    from nanochat.sft_dataloader import EpisodeShape
    from nanochat.sft_training import LossWeighting, SFTLossConfig

    args = rlm_sft.parse_args([
        "--episode-mix", "3x1:1,1x3:1,2x2:1", "--loss-weighting", "group-token",
        "--group-loss-weights", "3:0.7,1:0.1,2:0.2",
    ])
    assert args.episode_mix == (EpisodeShape(1, 3, 1), EpisodeShape(2, 2, 1), EpisodeShape(3, 1, 1))
    assert args.loss_config == SFTLossConfig(LossWeighting.GROUP_TOKEN, {1: 0.1, 2: 0.2, 3: 0.7})
    assert not {"cycles", "device_batch_size", "loss_weighting", "group_loss_weights"} & vars(args).keys()


class LocalConversations:
    """Avoid remote task downloads while preserving actual conversation packing."""

    def __init__(self, **kwargs):
        self.conversations = [
            ([0, 1, 2, 3, 4], [0, 0, 1, 1, 1]),
            (list(range(9)), [0, 0, 1, 1, 1, 1, 1, 1, 1]),
            (list(range(13)), [0, *([1] * 12)]),
        ]

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, index):
        return self.conversations[index]


class LocalTokenizer:
    def get_bos_token_id(self):
        return 0

    def render_conversation(self, conversation, max_tokens=None):
        ids, mask = conversation
        return ids[:max_tokens], mask[:max_tokens]


@pytest.fixture
def local_pretraining(monkeypatch):
    from nanochat.rlm import RecurrentGPT, RLMConfig
    from nanochat import rlm_inference
    from tasks import gsm8k, mmlu, smoltalk

    torch.manual_seed(29)
    model = RecurrentGPT(RLMConfig(
        sequence_len=9, text_length=4, memory_tokens=2, vocab_size=32,
        n_layer=2, n_head=2, n_kv_head=2, n_embd=24, window_pattern="L",
    ))
    model.init_weights()
    with torch.no_grad():
        model.lm_head.weight.normal_(std=0.2)
    tokenizer = LocalTokenizer()
    metadata = {
        "training_config": {
            "total_batch_size": 40, "embedding_lr": 0.2, "unembedding_lr": 0.004,
            "matrix_lr": 0.02, "scalar_lr": 0.5,
        },
        "memory_contract": {"gradient_horizon": 4, "layout": "memory_eom_text_summary"},
    }
    monkeypatch.setattr(rlm_inference, "load_rlm_checkpoint", lambda *_: (model, tokenizer, metadata))
    monkeypatch.setattr(rlm_inference, "tokenizer_fingerprint", lambda: "local-tokenizer")
    monkeypatch.setattr(smoltalk, "SmolTalk", LocalConversations)
    monkeypatch.setattr(mmlu, "MMLU", LocalConversations)
    monkeypatch.setattr(gsm8k, "GSM8K", LocalConversations)
    monkeypatch.delenv("NANOCHAT_MIN_SYSTEM_AVAILABLE_GB", raising=False)
    monkeypatch.delenv("NANOCHAT_CHECKPOINT_RESERVE_GB", raising=False)
    # Preserve the optimizer's compiled fullgraph functions without building
    # machine code for this small CPU integration fixture.
    with torch._dynamo.config.patch(debug_backend_override=">=0:eager", disable=False):
        yield model, tokenizer, metadata


def local_args(tmp_path, mode="microbatch", *extra):
    weights = ["--group-loss-weights", "1:0.1,2:0.2,3:0.7"] if mode == "group-token" else []
    args = rlm_sft.parse_args([
        "--checkpoint-dir", str(tmp_path / "source"), "--step", "7",
        "--output-dir", str(tmp_path / "sft"), "--device-type", "cpu",
        "--episode-mix", MIX, "--loss-weighting", mode, *weights,
        "--num-iterations", "1", "--load-optimizer", "0", "--no-compile",
        "--eval-every", "-1", "--eval-batches", "3", "--save-every", "-1", *extra,
    ])
    args.output_dir.mkdir()
    return args


@pytest.mark.parametrize("override", [None, 20, 80])
def test_budget_must_equal_one_complete_pattern(local_pretraining, tmp_path, override):
    _, _, metadata = local_pretraining
    extra = [] if override is None else ["--total-batch-size", str(override)]
    if override is None:
        metadata["training_config"]["total_batch_size"] = 80
    args = local_args(tmp_path, "microbatch", *extra)
    with pytest.raises(ValueError, match="must equal.*40"):
        rlm_sft.run(args)
    assert not (args.output_dir / "run.json").exists()


@pytest.mark.parametrize("mode", ["microbatch", "group-token"])
def test_one_update_logs_selected_objective_and_checkpoint_contract(local_pretraining, tmp_path, mode):
    from nanochat.rlm_sft import sft_episode_loss
    from nanochat.sft_dataloader import sft_mixed_episode_loader

    model, tokenizer, _ = local_pretraining
    args = local_args(tmp_path, mode, "--eval-batches", "2")
    expected_loader = sft_mixed_episode_loader(
        LocalConversations(), tokenizer, 4, args.episode_mix, torch.device("cpu"),
    )
    losses, targets = [], []
    with torch.no_grad():
        for shape in args.episode_mix:
            inputs, batch_targets, _ = next(expected_loader)
            loss, _, counts = sft_episode_loss(model.forward_cycle, inputs.long(), batch_targets, 4, shape.cycles)
            losses.append(loss.item())
            targets.append(counts.sum().item())
    assert targets == [9, 16, 14]
    expected_objective = sum(losses) / 3 if mode == "microbatch" else sum(
        loss * weight for loss, weight in zip(losses, (0.1, 0.2, 0.7))
    )
    expected_token_loss = sum(loss * count for loss, count in zip(losses, targets)) / 39
    assert abs(expected_objective - expected_token_loss) > 1e-3
    before = model.lm_head.weight.detach().clone()

    rlm_sft.run(args)

    [metrics] = [json.loads(line) for line in (args.output_dir / "updates.jsonl").read_text().splitlines()]
    assert metrics["loss"] == pytest.approx(expected_objective, abs=1e-5)
    assert metrics["token_loss"] == pytest.approx(expected_token_loss, abs=1e-5)
    assert metrics["supervised_targets"] == 39
    assert metrics["completed_updates"] == 1
    assert not torch.equal(before, model.lm_head.weight)
    for cycles, loss, count in zip((1, 2, 3), losses, targets):
        group = metrics["episode_groups"][str(cycles)]
        assert group["loss"] == pytest.approx(loss, abs=1e-5)
        assert group["supervised_targets"] == count
        assert len(group["cycle_losses"]) == cycles
    run = json.loads((args.output_dir / "run.json").read_text())
    assert run["accumulation_steps"] == 3
    assert run["sft_config"]["total_batch_size"] == 40
    assert run["sft_config"]["episode_mix"] == [
        {"cycles": 1, "batch_size": 3, "microbatches": 1},
        {"cycles": 2, "batch_size": 2, "microbatches": 1},
        {"cycles": 3, "batch_size": 1, "microbatches": 1},
    ]
    assert run["sft_config"]["loss_weighting"] == mode
    assert run["sft_config"]["group_loss_weights"] == (
        {"1": 0.1, "2": 0.2, "3": 0.7} if mode == "group-token" else None
    )
    checkpoint = json.loads((args.output_dir / "checkpoints" / "meta_000001.json").read_text())
    assert checkpoint["run_contract"] == run
    assert checkpoint["memory_contract"]["gradient_horizons"] == [1, 2, 3]
    assert "gradient_horizon" not in checkpoint["memory_contract"]
    # eval-batches counts microbatches, even when it stops mid-pattern.
    validation = sft_mixed_episode_loader(
        LocalConversations(), tokenizer, 4, args.episode_mix, torch.device("cpu"),
    )
    validation_sum = 0.0
    with torch.no_grad():
        for cycles in (1, 2):
            inputs, batch_targets, _ = next(validation)
            _, sums, _ = sft_episode_loss(model.forward_cycle, inputs.long(), batch_targets, 4, cycles)
            validation_sum += sums.sum().item()
    assert checkpoint["val_loss"] == pytest.approx(validation_sum / 25, abs=1e-5)
    assert (args.output_dir / "checkpoints" / "optim_000001_rank0.pt").is_file()
    result = json.loads((args.output_dir / "result.json").read_text())
    assert result["status"] == "completed" and result["completed_updates"] == 1


@pytest.mark.parametrize("initial_limit", [8, 64])
@pytest.mark.parametrize("mix,budget", [(MIX, 40), ("4x8:1,12x2:1,24x1:1", 320)])
def test_compiled_training_and_validation_keep_all_episode_variants(local_pretraining, tmp_path, initial_limit, mix, budget):
    args = local_args(tmp_path, "microbatch", "--compile", "--eval-every", "1",
                      "--episode-mix", mix, "--total-batch-size", str(budget))
    torch._dynamo.reset()
    # Dynamo still traces every graph; its built-in eager backend avoids a costly
    # machine-code build while enforcing the same specialization cache limits.
    with torch._dynamo.config.patch(
        recompile_limit=initial_limit, fail_on_recompile_limit_hit=True,
        debug_backend_override=">=0:eager", disable=False,
    ):
        rlm_sft.run(args)
        assert torch._dynamo.config.recompile_limit >= initial_limit
    result = json.loads((args.output_dir / "result.json").read_text())
    assert result["completed_updates"] == 1 and result["val_loss"] > 0
    torch._dynamo.reset()

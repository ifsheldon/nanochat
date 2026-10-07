"""Exercise the real pretraining loop without corpus files or GPU work."""

import os
from pathlib import Path
import subprocess
import sys


def test_benchmark_runs_two_accumulated_updates_without_production_side_effects(tmp_path):
    # A subprocess isolates base_train's argument parsing, RNG, and GC changes.
    # Keep the model and optimizer real; replace only external input/output.
    program = r"""
import json
import math
import runpy
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


def eager_compile(function=None, **kwargs):
    return (lambda function: function) if function is None else function


torch.compile = eager_compile

from nanochat.gpt import GPT
from nanochat.optim import MuonAdamW

torch.set_num_threads(1)
output = Path(sys.argv[1])
trained_batches = []
losses = []
learning_rates = []
prefetched_batches = []
original_forward = GPT.forward
original_step = MuonAdamW.step


def observe_forward(self, inputs, targets, *args, **kwargs):
    trained_batches.append(int(inputs[0, 0]))
    loss = original_forward(self, inputs, targets, *args, **kwargs)
    losses.append(float(loss.detach()))
    return loss


def observe_step(self, *args, **kwargs):
    learning_rates.append([
        group['lr'] / group['initial_lr'] for group in self.param_groups
    ])
    return original_step(self, *args, **kwargs)


def training_batches(tokenizer, batch_size, sequence_length, **kwargs):
    assert kwargs['split'] == 'train'
    assert kwargs['resume_state_dict'] is None
    for batch_id in range(5):
        prefetched_batches.append(batch_id)
        inputs = torch.full((batch_size, sequence_length), batch_id, dtype=torch.long)
        targets = torch.full_like(inputs, batch_id + 1)
        yield inputs, targets, {'pq_idx': 0, 'rg_idx': batch_id, 'epoch': 1}
    raise AssertionError('benchmark consumed more than two accumulated updates')


def unexpected(*args, **kwargs):
    raise AssertionError('benchmark performed evaluation, sampling, or checkpoint/W&B I/O')


sys.argv = [
    'scripts.base_train', '--device-type=cpu', '--depth=1', '--aspect-ratio=32',
    '--head-dim=16', '--max-seq-len=8', '--window-pattern=L',
    '--device-batch-size=2', '--total-batch-size=32', '--num-iterations=5568',
    '--warmup-steps=40', '--eval-every=1', '--core-metric-every=1',
    '--sample-every=1', '--save-every=1', f'--benchmark-output={output}',
]
patches = {
    'nanochat.tokenizer.get_tokenizer': lambda: SimpleNamespace(get_vocab_size=lambda: 256),
    'nanochat.tokenizer.get_token_bytes': lambda device: torch.ones(256, device=device),
    'nanochat.dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit': training_batches,
    'nanochat.dataloader.tokenizing_distributed_data_loader_bos_bestfit': unexpected,
    'nanochat.checkpoint_manager.save_checkpoint': unexpected,
    'nanochat.checkpoint_manager.load_checkpoint': unexpected,
    'nanochat.loss_eval.evaluate_bpb': unexpected,
    'scripts.base_eval.evaluate_core': unexpected,
    'nanochat.engine.Engine': unexpected,
    'wandb.init': unexpected,
}
with ExitStack() as stack:
    for target, replacement in patches.items():
        stack.enter_context(patch(target, replacement))
    stack.enter_context(patch.object(GPT, 'forward', observe_forward))
    stack.enter_context(patch.object(MuonAdamW, 'step', observe_step))
    state = runpy.run_module('scripts.base_train', run_name='__main__')

assert trained_batches == [0, 1, 2, 3], trained_batches
assert prefetched_batches == [0, 1, 2, 3, 4], prefetched_batches
assert len(learning_rates) == 2, learning_rates
for actual, expected in zip(learning_rates, (0.025, 0.05)):
    assert all(math.isclose(rate, expected) for rate in actual), actual
optimizer = state['optimizer']
adam_steps = [item['step'] for item in optimizer.state.values() if 'step' in item]
assert adam_steps and all(step == 2 for step in adam_steps), adam_steps
assert torch.count_nonzero(state['orig_model'].lm_head.weight) > 0
assert all(math.isfinite(loss) for loss in losses)

result = json.loads(output.read_text())
assert result['completed_updates'] == 2
assert result['timed_update'] == 2
assert result['training_horizon_updates'] == 5568
assert result['tokens_per_update'] == 32
assert result['gradient_accumulation_steps'] == 2
assert math.isclose(result['learning_rate_multiplier'], 0.05)
assert math.isclose(result['loss'], losses[-1])
assert result['warmup_update_seconds'] > 0
assert result['measured_update_seconds'] > 0
assert not any(result['compile_stats_delta'].values())
assert not (output.parent / 'base_checkpoints').exists()
"""
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "NANOCHAT_ATTENTION_IMPL": "sdpa",
        "NANOCHAT_DTYPE": "float32",
        "NANOCHAT_BASE_DIR": str(tmp_path),
        "OMP_NUM_THREADS": "1",
    }
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "NANOCHAT_MIN_SYSTEM_AVAILABLE_GB"):
        env.pop(name, None)
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path / "benchmark.json")],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr

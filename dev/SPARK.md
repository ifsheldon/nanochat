# DGX Spark small experiments

Run the existing six-layer demo from `runs/runcpu.sh` on the Spark's single CUDA GPU with `runs/runspark.sh`.
The runner uses PyTorch 2.14.0 from the official stable CUDA 13.2 wheel index, with Triton 3.8.0 and locked dependencies.
It requires `uv`, a C compiler for PyTorch compilation, and a working NVIDIA driver.
The CUDA runtime libraries are installed inside the Python environment by uv; the system CUDA toolkit is separate.
The runner uses Triton's bundled assembler, with no `TRITON_PTXAS_PATH` override.
Run all commands from the repository root.

```bash
bash runs/runspark.sh all
```

For a terminal-independent run with tmux:

```bash
tmux new-session -d -s nanochat-spark 'bash runs/runspark.sh all'
tmux attach -t nanochat-spark
```

Detach with Ctrl-b followed by d.
Each invocation writes a timestamped log under `~/.cache/nanochat/spark/logs`.
Set `NANOCHAT_BASE_DIR` to use another artifact directory.

## Stages

| Command | Work |
| --- | --- |
| `bash runs/runspark.sh prepare` | Install locked GPU and development dependencies, verify CUDA, download eight ClimbMix training shards plus validation, train the 32,768-token tokenizer on up to two billion characters, evaluate tokenization. |
| `bash runs/runspark.sh smoke` | Pretrain for 20 steps, save and reload, evaluate bits per byte and a small CORE sample, fine-tune for 20 optimizer updates, then exercise chat inference. |
| `bash runs/runspark.sh demo` | Pretrain for 5,000 steps, evaluate, fine-tune for 1,500 optimizer updates, then run a chat prompt. |
| `bash runs/runspark.sh sft` | Reuse the demo's pretrained checkpoint and rerun fine-tuning, final ChatCORE sampling, and chat inference. |
| `bash runs/runspark.sh all` | Run all three stages sequentially, stopping on the first failure. |

Run `prepare` before `smoke` or `demo` on a fresh installation.
Re-running `prepare` retrains the tokenizer; use the existing tokenizer for any checkpoints you intend to reuse.
The smoke and demo checkpoints use separate tags, `spark-smoke-d6-torch214` and `spark-d6-torch214`.
Re-running training with the same tag writes to the same checkpoint directory; use a separate `NANOCHAT_BASE_DIR` to retain independent experiments.

The demo preserves the upstream six-layer architecture, 512-token context, 32 sequences per batch, 16,384 tokens per optimizer step, and training horizons.
It uses BF16 as detected by nanochat and full-context attention (`--window-pattern=L`), which supports the existing PyTorch SDPA path when Flash Attention 3 is unavailable.
No matching FA3 hub build is available for the installed PyTorch 2.14/cu132 stack, so the original attention detection selects SDPA and passes GPU forward/backward checks.
The extra FA3 architecture guard used during the PyTorch 2.9.1 experiments has been removed.
It adds pretraining checkpoints every 1,000 steps and limits ChatCORE to the final SFT step with 16 examples per categorical task and four per generative task.
SFT packing preserves full tokenized conversation lengths and selects the longest complete conversation that fits the remaining row space.
If nothing fits a partially filled row, the remainder is padded; if nothing fits an empty row, only the shortest oversized conversation is truncated to the context length plus one target token.
This follows the packing-time approach discussed in [upstream PR #486](https://github.com/karpathy/nanochat/pull/486), avoiding the bias toward truncated conversations introduced by truncating every conversation before packing.
Conversations with no supervised target in that usable prefix are excluded and counted as consumed for dataset progress.
If a rank's entire dataset cycle has no usable targets, the loader raises an error instead of emitting empty training batches or looping indefinitely.
The standalone packing loader and its state are in `nanochat/sft_dataloader.py`; optimizer-update progress and learning-rate scheduling are handled by `nanochat/sft_training.py`.
Truncation can still cut an assistant response, but it is used only when no complete buffered conversation fits an empty row.
SFT now counts completed optimizer updates independently of prefetching and gradient accumulation, so `--num-iterations=1500` saves at optimizer step 1,500.
The historical measurements below predate this progress fix and therefore record 1,499 updates for the earlier demo.
These small evaluation samples validate the workflow; they are not comparable to full benchmark scores.
Weights & Biases logging is disabled with `--run=dummy`.
Initial PyTorch/Triton compilation can take substantially longer than subsequent steps.
The code does not define peak FLOPS for GB10, so its reported zero MFU is not a measure of GPU utilization; use tokens per second and `nvidia-smi`.

The runner uses `uv sync --locked`; `uv.lock` is regenerated alongside dependency changes in `pyproject.toml`.
To upgrade an existing environment without retraining its tokenizer, run `uv sync --extra gpu --group dev --locked`, then run the `smoke` and `demo` stages.

## Use the trained model

```bash
source .venv/bin/activate
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat/spark"
python -m scripts.chat_cli --device-type=cuda --model-tag=spark-d6-torch214
```

The base checkpoint is in `$NANOCHAT_BASE_DIR/base_checkpoints/spark-d6-torch214` and the conversational checkpoint is in `$NANOCHAT_BASE_DIR/chatsft_checkpoints/spark-d6-torch214`.
This is a small educational model; the next research scale in the upstream README is depth 12.
Measure the six-layer run before allocating time to larger experiments.

## PyTorch 2.14 validation on September 5, 2026

The upgraded environment uses Python 3.10.21, PyTorch 2.14.0+cu132, and Triton 3.8.0 on the same GB10 and driver 580.173.02.
Triton automatically selects its bundled Blackwell assembler, CUDA 13.3.33, for this GPU.
No system toolkit or driver installation was needed, and all checks ran with `TRITON_PTXAS_PATH` unset.
The attention, optimizer, inference engine, and tokenizer tests reported 31 passed and 10 FA3-specific checks skipped, with no warnings.

The complete `smoke` stage passed: 20 pretraining steps, checkpoint reload, BPB and small CORE evaluation, 19 SFT optimizer steps, and chat inference.
Pretraining validation bits per byte fell from 3.116971 to 2.210339; SFT finished with validation bits per byte of 1.8433 and finite training losses.
Peak allocated GPU memory during pretraining was 3,047.59 MiB.
Median throughput over logged pretraining steps 10 through 19 was 164,058 tokens/second versus 54,254 on the 2.9.1 baseline, about 3.02 times faster for this short experiment with the same model and batch settings.
This comparison covers steady steps in the smoke test, not total pipeline runtime.
The upgraded smoke log is `~/.cache/nanochat/spark/logs/smoke-20260905-205020.log`.

After reviewing the earlier workarounds, the FA3 architecture guard and its four test cases were reverted because the original attention detection works with the upgraded environment.
The targeted tests after this cleanup reported 27 passed and 10 FA3-specific checks skipped.
The assembler override had already been removed during the upgrade.
The SFT bug remains independent of PyTorch: using the real training mixture, the original loader produced supervised-token counts of `[1570, 2345, 2559, 0, 0, 0, 0, 0]` for its first eight batches.
The smoke measurements above used the initial eager-truncation fix, which has since been replaced by selective packing-time truncation and filtering of prefixes without supervised targets.

The selective-packing revision passed 29 targeted loader, tokenizer, and inference tests, including oversized buffers, complete-conversation priority, loss-mask boundaries, excluded-example progress, per-rank exhaustion, and contexts longer than 2,048 tokens.
A GPU SFT smoke run completed 19 optimizer steps with finite losses and validation bits per byte of 2.0515; its log is `~/.cache/nanochat/spark/logs/sft-selective-smoke.log`.
Validation data packing differs from the eager-truncation experiment, so their BPB values are not directly comparable.
The earlier complete demo's SFT checkpoint is preserved under `~/.cache/nanochat/spark/archive/spark-d6-torch214-eager-sft`.
The `sft` stage can retrain from the existing 5,000-step base checkpoint without repeating pretraining.
The selective-packing `sft` rerun completed all 1,499 optimizer steps with finite logged losses, saved its checkpoint, and completed the small ChatCORE evaluation and chat inference.
Final validation bits per byte were 0.726387; the run log is `~/.cache/nanochat/spark/logs/sft-20260905-211411.log`.

## PyTorch 2.9.1 baseline on September 5, 2026

The initial run used NVIDIA GB10 SM 12.1, driver 580.173.02, Python 3.10.21, PyTorch 2.9.1+cu128, Triton 3.5.1, and the system CUDA 13.0.88 assembler.
The 20-step smoke test completed with approximately 54,000 tokens/second after compilation and 3,002.84 MiB peak allocated GPU memory.
Validation bits per byte fell from 3.116971 to 2.210130, and a separate process successfully reloaded the checkpoint, generated samples, and measured 2.210151 validation bits per byte.
The checkpoint-reload evaluation uses batch size 1; the training evaluation uses batch size 32.
The attention and optimizer tests passed with 13 passed and 10 FA3-specific checks skipped; the tokenizer and inference engine tests passed with 18 passed.
The small CORE evaluation completed, and the SFT smoke run completed 19 optimizer steps with finite losses after the context-length fix, saving a checkpoint with validation bits per byte of 1.8428.

The baseline demo was stopped at approximately step 3,600 for the PyTorch upgrade; its original `spark-d6` checkpoints through step 3,000 are retained.
The upgraded experiments use the new tags above to avoid mixing results.
The system CUDA 13.0 toolkit was already installed before this work.
The baseline required `TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas` because Triton 3.5.1's bundled assembler did not support GB10.

To run the relevant tests with the current environment:

```bash
OMP_NUM_THREADS=1 \
    .venv/bin/python -m pytest tests/test_attention_fallback.py tests/test_optim.py tests/test_engine.py tests/test_tokenizer.py tests/test_sft_dataloader.py -q
```

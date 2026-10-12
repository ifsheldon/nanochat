# DGX Spark small experiments

## Current status — October 5, 2026

The six-layer experiments below are complete, and the subsequent [d24 training pipeline](SPARK_D24.md) has completed pretraining and SFT.
The d24 run remains intentionally paused at final chat evaluation following the user's September 15 request to free the GPU.
No nanochat training or evaluation process was running when this status was checked.
Do not resume this production run's training or final evaluation until the user requests it.
The separately authorized October 8 [FA4 training comparison](SPARK_FA4.md) is complete; it used fresh models and left these checkpoints and evaluation results untouched.

| Result | Saved value |
| --- | --- |
| Base checkpoint | Step 5,568; 5,838,471,168 training tokens; validation BPB 0.690965 |
| SFT checkpoint | Step 574; validation BPB 0.271158 |
| Full base CORE | 0.253972, below the GPT-2 target of 0.256525; target not met |
| ARC-Easy | 64.18% |
| ARC-Challenge | 48.12% |
| MMLU | 37.09% |
| GSM8K | 12.28% |
| HumanEval | Interrupted; no completed score |

The base and SFT BPB values measure different datasets and are not directly comparable.
Both final model and optimizer checkpoints are present under `~/.cache/nanochat/spark-d24`, using tag `spark-d24-bf16`.
Training does not need to be repeated to finish evaluation.
`report.json` is not present because HumanEval and the aggregate final report remain unfinished.

## Handoff to another agent

Work in `/home/maplespark/code/nanochat` on branch `spark` and read [the d24 workflow](SPARK_D24.md) before acting.
Use the locked uv environment, BF16, SDPA, and the 80,000,000,000-byte CUDA allocator limit; CPU and external CUDA allocations are additional overhead.
The active artifact root is `/home/maplespark/.cache/nanochat/spark-d24`; `/home/maplespark/.cache/nanochat/spark` contains the earlier small experiments.
Artifacts and checkpoints are local cache files, not part of Git, and must be preserved separately when moving to another machine.

Read these files to recover the authoritative run state:

- `status.json` and `pause.json`: intentional pause, completed stages, and remaining task.
- `base-results.json` and `base_eval/base_model_005568.csv`: full CORE results and the unmet capability target.
- `chat-results.json`: the four completed chat-task scores listed above.
- `base_checkpoints/spark-d24-bf16/meta_005568.json` and `chatsft_checkpoints/spark-d24-bf16/meta_000574.json`: final training metadata, alongside model and optimizer files.
- `production-console.log` and `logs/20260915-203649-615938/05-scripts.chat_eval.log`: pipeline output and the interrupted HumanEval invocation.
- `validated.json` and `data-inventory.json`: batch 16, 32-way accumulation, measured CUDA peak of 63.145 GB, source fingerprint, and tokenizer/data checksums.

Once the user requests resumption, restart only HumanEval from SFT step 574.
HumanEval has no per-example resume, so restart that task from its beginning; its interrupted partial count is not a final score.
The generic `runs/runspark_d24.sh chat-eval` stage starts all five tasks again and rewrites `chat-results.json` as it goes, so do not use it to resume this partial evaluation.
Use a new timestamped log for this command:

```bash
NANOCHAT_BASE_DIR="$HOME/.cache/nanochat/spark-d24" \
NANOCHAT_DATASETS_DIR="$HOME/.cache/nanochat/spark-d24" \
NANOCHAT_CUDA_MEMORY_LIMIT_GB=80 NANOCHAT_DTYPE=bfloat16 \
NANOCHAT_ATTENTION_IMPL=sdpa OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
uv run --no-sync python -m scripts.chat_eval \
    --device-type=cuda --source=sft --model-tag=spark-d24-bf16 \
    --step=574 --task-name=HumanEval --batch-size=8 \
    --temperature=0 --num-samples=1 --max-new-tokens=512
```

After successful completion, preserve the four existing scores and merge the final HumanEval accuracy into `chat-results.json`.
Assemble `report.json` using the fields and calculation in `Pipeline.chat_eval` in `scripts/spark_d24.py`: average `(accuracy - 0.25) / 0.75` for the three categorical tasks together with raw GSM8K and HumanEval accuracy.
Retain `gpt2_target_met=false`, add the base/SFT checkpoint paths and steps, and include the interactive chat command.
Mark `status.json` completed only after all results and the final report are saved; retain the historical pause record.
Do not automatically extend training to chase the CORE target.

The runner rejects fresh production overwrites.
Pretraining resume restores a complete checkpoint without requiring source, dependency, or saved-configuration fingerprints to match.
Pretraining uses approximate row-group data resumption, and exact SFT mid-epoch resumption is not implemented.
Verify changes with `OMP_NUM_THREADS=1 uv run --no-sync python -m pytest -q`, `uv lock --check`, shell syntax checks, and `git diff --check` before committing.

## Small experiment workflow

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
New invocations keep checkpoints, tokenizer files, logs, metrics, and runtime caches under `training_run/spark/` in this project.
Set `NANOCHAT_RUN_NAME` to choose another name under `training_run/`, or `NANOCHAT_BASE_DIR` to select an explicit artifact directory.
Raw ClimbMix, task, and CORE evaluation datasets are shared under the project's `datasets/`; `NANOCHAT_DATASETS_DIR` selects an explicit shared data root.
The runner selects the first eight ClimbMix training shards plus the fixed validation shard, even when other runs have downloaded additional shards.
Every invocation captures stdout and stderr in a timestamped file under its run's `logs/` directory.
Existing artifacts under `~/.cache/nanochat` stay in place and are not copied automatically.

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
During the September 5 validation, no matching FA3 hub build was available for the installed PyTorch 2.14/cu132 stack, so the original attention detection selected SDPA and passed GPU forward/backward checks.
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
export NANOCHAT_RUN_NAME=spark
python -m scripts.chat_cli --device-type=cuda --model-tag=spark-d6-torch214
```

The selected run contains `base_checkpoints/spark-d6-torch214` for the base model and `chatsft_checkpoints/spark-d6-torch214` for the conversational model.
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

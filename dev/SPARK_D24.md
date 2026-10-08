# d24 training on DGX Spark

For the October 5, 2026 run status and instructions to finish only the paused HumanEval evaluation, see [current status and agent handoff](SPARK.md#current-status--october-5-2026).

This workflow trains a fresh 1,384,122,122-parameter nanochat model on one GB10 GPU through pretraining, base evaluation, SFT, and chat evaluation.
The quality target is the repository's GPT-2 reference CORE score of 0.256525; completing training does not by itself establish that capability.
Run commands from the repository root on the `spark` branch.

## Configuration

The locked environment uses PyTorch 2.14.0+cu132 and Triton 3.8.0.
Training uses BF16 compute, existing parameter/optimizer precision, 24 layers, width 1536, 12 heads, 2048-token context, and full-context SDPA.
The runner sets `NANOCHAT_ATTENTION_IMPL=sdpa` to bypass remote FA3 kernel discovery throughout the run; the default `auto` retains upstream detection for other workflows.
The tokenizer is copied from the validated small Spark experiment, without retraining.
The global batch is 1,048,576 tokens, the target tokens-to-scaling-parameters ratio is 8, and the production horizon is 5,568 optimizer updates, or 5,838,471,168 tokens.
Initial device batch 16 requires 32 accumulation microbatches per update.
Validation halves the device batch if necessary while preserving the global batch and context.
These BF16/full-context choices differ from the upstream FP8/sliding-window speedrun, so final capability must be measured.

`NANOCHAT_CUDA_MEMORY_LIMIT_GB=80` limits the PyTorch CUDA allocator to 80,000,000,000 bytes through shared compute initialization.
Invalid, non-positive, non-finite, and above-device-capacity values raise errors.
CPU memory and CUDA allocations outside the PyTorch allocator are additional overhead.
`NANOCHAT_MIN_SYSTEM_AVAILABLE_GB=16` stops training at an update boundary if system available memory is too low.
Memory logs report CUDA allocated/reserved/peak-reserved memory and system available memory separately, without adding RSS and CUDA usage together.
`NANOCHAT_CHECKPOINT_RESERVE_GB=100` requires space for the checkpoint plus 100 GB free before saving.
All these quantities use decimal GB.

## Run the pipeline

```bash
tmux new-session -d -s nanochat-d24 'bash runs/runspark_d24.sh all'
tmux attach -t nanochat-d24
```

Detach with Ctrl-b followed by d.
The artifact root defaults to `~/.cache/nanochat/spark-d24`; override `NANOCHAT_BASE_DIR` to select a different run directory.
The model tag is `spark-d24-bf16`, and validation uses separate tags.
The runner takes an exclusive file lock, records each command and source revision/diff/untracked source, and writes timestamped command logs and JSON memory/error results.
`status.json` identifies the current stage and its log directory.
W&B is disabled.

| Stage | Behavior |
| --- | --- |
| `prepare` | Copy the existing tokenizer and cached data, download 170 training shards plus validation, preload SFT/evaluation data, verify Parquet metadata, and record SHA-256 checksums. |
| `validate` | Run two accumulated d24 updates, pause, resume to update four with the same horizon, inspect finite model/optimizer tensors, reload for inference, and run two accumulated SFT updates. |
| `pretrain` | Start fresh production pretraining, or explicitly resume a complete checkpoint. |
| `base-eval` | Run full CORE, 20,971,520-token BPB on each split, and standard generated samples. |
| `sft` | Fine-tune the completed base checkpoint through one consumption-based epoch of SmolTalk, MMLU x3, and GSM8K x4. |
| `chat-eval` | Run complete ARC-Easy, ARC-Challenge, MMLU, GSM8K, and HumanEval evaluations and write the final report. |
| `all` | Run the six stages in order, stopping at the first failure. |

Each stage can be run as `bash runs/runspark_d24.sh STAGE`.
After running preparation and validation separately, start the remaining stages without repeating validation:

```bash
tmux new-session -d -s nanochat-d24 'bash runs/runspark_d24.sh pretrain && bash runs/runspark_d24.sh base-eval && bash runs/runspark_d24.sh sft && bash runs/runspark_d24.sh chat-eval'
```

The validation horizon uses the full production global batch but a 32,768-token validation sample to check memory and reload behavior.
Peak CUDA reserved memory must be at most 72 GB through base training, resume, inference, and SFT.
On validation OOM or excess peak memory, retry in a fresh process at device batches 8, 4, 2, and 1.
Other failures stop immediately.
If no batch passes, the workflow stops without changing the model, context, or token budget.
`validated.json` records the selected batch and projected checkpoint storage.

## Progress and recovery

Production pretraining evaluates validation BPB and saves checkpoints every 100 updates and at completion.
Its logged step line uses the upstream zero-based index; checkpoint steps count completed optimizer updates.
Full CORE runs after pretraining, and full chat evaluation runs after the final SFT checkpoint is saved.
SFT evaluates BPB every 100 updates and at completion.
Training aborts on non-finite loss before applying the update, or when the configured system-memory threshold fails.
There is no automatic production restart loop or automatic extension of the training horizon.

SFT `--num-iterations=N` now means exactly N optimizer updates, regardless of accumulation.
Prefetched batches do not count toward completed progress.
With the default `-1`, SFT finishes the accumulation group that crosses one dataset-consumption epoch, including excluded conversations in consumption accounting.
The overrun is bounded by one optimizer batch of consumption; best-fit buffering means this is not an exact once-only traversal of every conversation.
The learning-rate schedule uses progress before the current update and is bounded in [0, 1].

Checkpoint tensors are written to temporary files and renamed, with metadata published last after all ranks finish their tensor files.
Automatic selection requires published metadata and the required tensor files; the d24 runner additionally requires the optimizer checkpoint and passes explicit steps to every load.
An interrupted save does not publish its step.
The runner retains every checkpoint during training, and validation verifies their projected disk footprint plus the free-space reserve.
The approved manual cleanup on October 8, 2026 retained complete d24 base checkpoints 5500 and 5568 and final SFT checkpoint 574, removing base checkpoints 100 through 5400 and the obsolete validation checkpoint fixtures.
Combined with the old smoke, profiling, and RLM CPU-test checkpoints, this removed 68 checkpoint sets and reclaimed 573.15 GB; the six-update RLM pilot and D6 research/demo checkpoints remain.
Run logs, evaluation results, source snapshots, and dataset/tokenizer caches were preserved, with removed checkpoint metadata archived in `~/.cache/nanochat/checkpoint-cleanup-20261008T143350Z/deleted-checkpoint-metadata.json`.
The exact deletion manifest, retained-file verification, and disk-space measurements are in that directory's `plan.json` and `result.json`.

To resume an interrupted pretraining run, substitute a retained complete checkpoint step:

```bash
bash runs/runspark_d24.sh pretrain --resume-from-step=5500
```

Resume requires matching architecture, batch configuration, horizon, tokenizer, and ordered data inventory.
The runner also verifies the training-source and dependency fingerprint recorded by validation.
Model, optimizer, and training-loop state are restored; data position retains upstream approximate row-group resumption, so interrupted training is not claimed to be bit-for-bit identical.
Fresh production invocations refuse an existing production marker or checkpoint directory.
If a run fails before any complete checkpoint, preserve its artifact directory and use a new root for a fresh run.
Interrupted SFT restarts from the final base checkpoint; preserve any existing SFT output before restarting that stage.
After resuming pretraining, run `base-eval`, `sft`, and `chat-eval` sequentially.

## Results and checks

`base-results.json` records full CORE and whether it exceeds 0.256525.
`chat-results.json` records the complete per-task chat results.
Chat evaluation uses temperature 0, one sample, and 512 maximum new tokens, starting categorical tasks at batch 8 and reducing batch on OOM without reducing task coverage.
`report.json` records operational completion, capability outcome, task scores, checkpoint paths, and the interactive chat command.
A score below the target is reported explicitly; SFT and chat evaluation still complete.

```bash
NANOCHAT_BASE_DIR="$HOME/.cache/nanochat/spark-d24" \
NANOCHAT_CUDA_MEMORY_LIMIT_GB=80 \
NANOCHAT_ATTENTION_IMPL=sdpa \
uv run --no-sync python -m scripts.chat_cli --device-type=cuda --model-tag=spark-d24-bf16
```

Run the repository checks before production:

```bash
OMP_NUM_THREADS=1 uv run --no-sync python -m pytest -q
uv lock --check
bash -n runs/runspark_d24.sh runs/runspark.sh
git diff --check
```

## Local validation on September 5, 2026

The full test suite passed with 88 tests passed and 10 FA3-specific tests skipped.
The d24 GPU validation completed two updates, saved, resumed to update four, verified finite model/optimizer tensors and restored optimizer counters/learning rates, and reloaded for inference.
The SFT validation completed exactly two optimizer updates with 32 microbatches per update and finite losses.
Batch 16 passed with peak reserved CUDA memory of 63.145 GB, below the 72 GB acceptance threshold.
Each base checkpoint occupies approximately 9.968 GB; the conservative production checkpoint projection is 568.150 GB.
Validation artifacts and command metrics are under `~/.cache/nanochat/spark-d24/logs/20260905-215904-429689`.
The shell launcher was edited during that validation invocation and reported a trailing shell error after the Python validation had successfully completed; a fresh preparation/inventory-check invocation of the finalized launcher subsequently exited successfully.
Steady validation updates took approximately 175–180 seconds while another GPU workload was active, implying roughly 11–12 days for pretraining alone if those conditions persist.
This is an ETA from a short run, not a capability result or a total-pipeline runtime guarantee.

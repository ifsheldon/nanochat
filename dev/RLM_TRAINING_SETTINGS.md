# RLM training settings

RLM splits each training episode into consecutive text chunks and passes learned memory between them.
Choose the architecture in a [JSONC model profile](../model_profiles/README.md), then choose how many episodes contribute to each optimizer update with the training settings below.
Run commands from the repository root after preparing the tokenizer and dataset using the [Spark workflow](spark-training/SPARK_RLM.md) or the [H100 launcher](h100-training/H100_RLM.md).

The direct pretraining trainer requires both `--model-profile PATH` and `--total-batch-token-size T` for every mode, including probes and resume.
The H100 launcher requires both for a fresh run; on resume the profile remains required, while omitting the token budget restores the saved recipe's budget.
An explicit H100 resume budget must match the saved value.
The pretraining option `--total-batch-size` has been removed without an alias; base GPT and chat entry points keep their own existing options.

The pretraining settings below describe `scripts.rlm_train` and the H100 launcher.
The separate [recurrent SFT settings](#recurrent-sft) describe `scripts.rlm_sft`, which loads architecture from a pretraining checkpoint and requires an episode mix and loss-weighting choice.

## Episodes, microbatches, and optimizer updates

An **episode** contains `cycles * text_length` packed text tokens in one row of the data loader.
It begins with empty memory, processes every cycle on the same worker, and discards memory at its end.
Memory remains attached to autograd across all cycles, so a later cycle's loss can train the earlier memory writer.
Memory does not carry across episodes or accumulation steps.
An episode may contain several packed documents; it is not necessarily one document.

A **chunk** is a fixed `text_length`-token slice of an episode, and a **cycle** is the one Transformer call that processes a chunk together with its memory, EOM, and summary positions.
An episode with `C` cycles has exactly `C` chunks, and cycle `k` processes chunk `k`.
Chunk boundaries are fixed token positions and ignore document boundaries, so only the first chunk is guaranteed to begin at a document start.
"Chunk" is used only for this fixed slice; the [RLM design](RLM-architecture/RLM_DESIGN.md#terminology) defines these and the related terms.

A **device microbatch** contains `device_batch_size` independent episodes on one GPU.
Each worker completes `accumulation_steps` microbatches, accumulating gradients before one **optimizer update**.
Workers average gradients for that global update; adding GPUs does not split an episode's recurrent path across devices.

The global text-token budget is:

$$
T = W \times B \times A \times C \times L
$$

| Symbol | Setting | Unit and meaning |
| --- | --- | --- |
| `T` | `--total-batch-token-size` | Real-text prediction targets across all workers per optimizer update |
| `W` | Worker/GPU count | `--num-gpus` on H100; the distributed launcher's world size for the direct trainer; 1 for a single process |
| `B` | `--device-batch-size` | Episodes per worker per accumulation step |
| `A` | Derived accumulation steps | Microbatches per worker per optimizer update; no separate CLI option |
| `C` | `--cycles` | Cycles per episode, one per consecutive chunk |
| `L` | Profile `text_length` | Text tokens per cycle |
| `M` | Profile `memory_tokens` | Incoming memory slots and outgoing summary slots where present; outside the text budget |

The trainer calculates `A = T / (W * B * C * L)` and requires a positive integer result.
Choose a compatible budget explicitly; the trainer does not round it or adjust device batch size.
At fixed `T`, increasing the GPU count or device batch size decreases accumulation.
Changing cycle count or text length also changes accumulation and the episode's recurrent computation.
Changing memory length affects internal positions and resource use without changing this equation.

`T` counts the ordinary shifted targets for packed text positions, including BOS tokens in the packed text stream.
It excludes learned EOM inputs, the additional chunk-start prediction made at each EOM position, incoming memory, and outgoing summary positions.
The loss includes both ordinary text targets and EOM predictions; EOM predicts an existing text token, not a new vocabulary token.

For each complete update:

| Count | Formula |
| --- | --- |
| Global episodes | `E = W * B * A = T / (C * L)` |
| Ordinary text targets | `T` |
| Extra EOM predictions | `E * C = T / L` |
| Total supervised predictions | `E * C * (L + 1) = T + T / L` |
| Transformer input positions | `E * (C * (L + 1) + 2 * (C - 1) * M)` |

The first cycle has `[EOM][text][summary]` and uses `L + 1 + M` input positions.
Each intermediate cycle has `[memory][EOM][text][summary]` and uses `L + 1 + 2*M` positions.
The final cycle has `[memory][EOM][text]` and uses `L + 1 + M` positions, because its outgoing summary has no consumer.
With two cycles there are no intermediate cycles.
Internal-position counts describe model input work; they are neither extra corpus text nor an RLM FLOP estimate.

## Worked four-cycle budgets

These examples use d24, four cycles, and device batch 32.
Both budgets produce 1,048,576 supervised predictions per update, but they use different text-target and episode counts.

| Setting or count | `rlm-d24.jsonc` | `rlm-d24-c511-m128.jsonc` |
| --- | ---: | ---: |
| Text length `L` | 1023 | 511 |
| Memory slots `M` | 512 | 128 |
| Required chosen text budget `T` | 1,047,552 | 1,046,528 |
| Text targets per episode | 4092 | 2044 |
| Predictions per episode | 4096 | 2048 |
| Global episodes per update | 256 | 512 |
| Accumulation on 1 / 2 / 4 GPUs | 8 / 4 / 2 | 16 / 8 / 4 |
| Cycle input lengths | 1536, 2048, 2048, 1536 | 640, 768, 768, 640 |
| Input positions per episode | 7168 | 2816 |
| Input positions per update | 1,835,008 | 1,441,792 |

The 1,047,552 budget does not divide into complete four-cycle episodes of 511 text tokens; select 1,046,528 for the comparison above.
The H100 launcher supports both profiles with their explicit budgets.
These counts do not establish measured throughput, memory capacity, or model quality.

For a bounded direct-trainer probe with the smaller text and memory profile:

```bash
NANOCHAT_RUN_NAME=rlm-c511-probe NANOCHAT_TRAIN_SHARDS=170 \
uv run --no-sync python -m scripts.rlm_train \
    --model-profile model_profiles/rlm-d24-c511-m128.jsonc \
    --total-batch-token-size 1046528 --cycles 4 \
    --device-batch-size 32 --activation-checkpointing --mode probe
```

## Direct trainer options

The following options belong to `python -m scripts.rlm_train`; `--help` prints the current CLI without launching training.
Defaults apply only where shown, and schedule/checkpoint step numbers refer to completed optimizer updates.

### Model, workload, and execution

| Option | Default | Meaning and constraints |
| --- | --- | --- |
| `--model-profile PATH` | Required | Seven-field architecture document; required for fresh runs and resume, with no automatic selection |
| `--total-batch-token-size T` | Required | Positive global text-target budget per optimizer update; must satisfy the divisibility equation above in every mode and on resume |
| `--cycles C` | `2` | Cycles per episode, one per chunk; integer at least 2 so later losses supervise the memory writer |
| `--device-batch-size B` | `16` | Positive number of episodes on each worker per accumulation step |
| `--mode` | `probe` | `probe`, `capacity-probe`, `attention-probe`, or `train`; limits described below |
| `--num-iterations N` | Required for `train`; `2` otherwise | Positive full schedule horizon; ordinary probes run this many updates and require at least 2; capacity and attention probes use their own execution limits |
| `--device-type` | `cuda` | `cuda` or `cpu`; distributed GPU workers are assigned by the process launcher |
| `--seed` | `42` | Integer seed for fresh initialization; resume restores saved RNG state |
| `--compile` / `--no-compile` | Enabled | Compile the model's cycle forward; optimizer kernels can still compile with `--no-compile` |
| `--activation-checkpointing` | Disabled | Recompute block activations during backward to reduce retained activations; preserves gradients through recurrent memory |
| `--output-dir PATH` | `<run>/train` | Fresh training/probe destination, or the existing destination when resuming; does not relocate the tokenizer |

`probe` runs at least two full optimizer updates, including compilation in the first update, and saves measurements without checkpoints.
`capacity-probe` runs exactly one update with optimizer state and checkpoint staging allocations already resident.
`attention-probe` allows at most five warmup updates, then measures one complete update without new compilation; it fails if that condition is not reached.
For capacity and attention probes, retain the intended training `--num-iterations` because it controls the schedule even though the probe executes fewer updates.
All probe modes reject resume, early stopping, nonzero numbered-save intervals, and retention settings; they never save checkpoints.
Capacity and attention probes also reject scheduled embedding freezing; select `none` or `permanent` explicitly for the phase being tested.

### Learning rates and schedule

| Option | Default | Meaning and constraints |
| --- | ---: | --- |
| `--warmup-steps` | `40` | Nonnegative warmup updates; `0` disables warmup |
| `--warmdown-ratio` | `0.65` | Fraction of the full horizon reserved for linear learning-rate decay, from 0 to 1 |
| `--final-lr-frac` | `0.05` | End-of-horizon learning-rate multiplier, from 0 to 1 |
| `--embedding-lr` | `0.3` | Base learning rate for the input table, value embeddings, EOM, and summary inputs; finite and nonnegative |
| `--unembedding-lr` | `0.008` | Base learning rate for the untied output vocabulary projection; finite and nonnegative |
| `--matrix-lr` | `0.02` | Base Muon learning rate for Transformer matrices; finite and nonnegative |
| `--scalar-lr` | `0.5` | Base learning rate for residual and initial-input scaling parameters; finite and nonnegative |
| `--weight-decay` | `0.28` | Base Muon matrix-decay coefficient; finite and nonnegative; AdamW groups retain their own coefficients |

These learning-rate inputs are base values, not the final per-parameter rates.
Fresh optimizer construction scales the four CLI learning rates by `sqrt(T / 524288)`.
Input/output embedding groups also use width scaling `(n_embd / 768)^(-1/2)`; value embeddings use half the scaled embedding rate, residual scalars use `0.01 * scalar_lr`, and initial-input scalars use `scalar_lr`.
Smear/backout parameters have their own fixed base rate in the optimizer.
The schedule multiplies each group's initial rate, warms linearly, then decays over `round(warmdown_ratio * num_iterations)` updates toward `final_lr_frac`.
A warmup longer than the run keeps the entire run in warmup.
Muon decay additionally scales by the batch factor and the ratio of d12 reference scaling parameters to the selected model's scaling parameters, then follows a cosine decay over the horizon.
Muon momentum follows the implementation's fixed warmup and warmdown schedule.
Consequently, changing `T` for a fresh run changes optimization as well as accumulation.

### Initialization and input-embedding freezing

| Option | Default | Meaning and constraints |
| --- | --- | --- |
| `--init-embedding-checkpoint-dir PATH` | None | Fresh-run source of `transformer.wte.weight`; supply together with source step and tokenizer |
| `--init-embedding-step` | None | Nonnegative published source checkpoint step |
| `--init-embedding-tokenizer PATH` | None | Source `tokenizer.pkl`; must exactly match the training tokenizer |
| `--init-embedding-bundle PATH` | None | Verified embedding-only export; alternative to the three source-checkpoint options |
| `--input-embedding-freeze` | `none` | `none`, `scheduled`, or `permanent`; freezing requires pretrained input-table initialization or its saved provenance |

Without initialization options, weights start fresh.
Transfer copies only the main input embedding table; Transformer, value embeddings, output head, EOM, and summary parameters retain fresh initialization.
The source must match the tokenizer, vocabulary, embedding width, padded table shape, and dtype.
Initialization options are rejected on resume because trained weights and initialization provenance come from the checkpoint.

`scheduled` freezes the input table for the first `floor(9 * N / 10)` updates and trains it for the remaining updates.
For `N = 5574`, updates 1–5016 are frozen and update 5017 is the first trainable update.
`permanent` freezes the table throughout the horizon; `none` trains it throughout.
Freezing removes that table's gradient, weight decay, and optimizer updates, while EOM, summary, value embeddings, and the output head remain trainable.
Its Adam state is created when it first becomes trainable.

### Checkpoints, stopping, and resume

| Option | Default | Meaning and constraints |
| --- | --- | --- |
| `--save-every` | `10` in training | Nonnegative interval for numbered checkpoints; `0` disables periodic saves, while update 2 and final/paused saves remain |
| `--keep-checkpoints` | `10` in training | Positive number of newest numbered checkpoints to retain, in addition to rolling latest and any milestone |
| `--checkpoint-mode` | `async` | `async` overlaps writing with the next forward/backward; `sync` finishes writing before continuing |
| `--resume-latest` | Disabled | Restore the complete rolling `checkpoints/latest.pt`; mutually exclusive with `--resume-from-step` |
| `--resume-from-step` | None | Positive numbered checkpoint step; only the newest published update may be resumed |
| `--stop-after-step` | None | Stop and save at this completed update, within the full horizon and beyond the resumed step when resuming; does not shorten the learning-rate or freeze schedule |

Rolling latest is saved after every completed training update, independently of numbered-checkpoint frequency.
Checkpoints contain model and optimizer state, completed progress, loader position, RNG state, and cumulative token/position counts; episode memory has already been discarded at that boundary.
The coordinator copies the original model profile once to `OUTPUT/checkpoints/model-profile.jsonc`, preserving comments and line endings.
Later saves retain an equivalent existing document.
Both pretrained-table freeze policies also preserve `pretrain-90pct.pt` at the positive 90% boundary; a required milestone must finish before training continues.
Ordinary checkpoint errors warn and retry on later updates, while a required milestone failure stops training.

Activation checkpointing controls recomputation during backward; checkpoint saving controls recoverable files on disk.
Asynchronous disk saving still waits before the next optimizer mutation, so it does not race with changes to saved weights.
GPU/CPU staging allocations and the disk reserve remain separate resource costs.

When resuming the direct trainer, explicitly pass the model profile, token budget, full horizon, and desired training/checkpoint options again.
Select the saved profile path or a source with the same seven field values; omit initialization options.
The direct trainer can change a compatible batch budget or cycle count on resume, and records new counts without rewriting prior cumulative totals.
Saved optimizer parameter groups, including their initial learning rates, are restored; changing CLI base rates or the batch budget does not rebuild those saved base rates.
The current full horizon and freeze policy determine the remaining schedule.
Loader continuation is approximate at row-group granularity; changing worker count repartitions data and does not preserve exact example order.

## H100 launcher options and fixed recipe

Use `bash runs/runh100_rlm.sh` so the locked environment is prepared before invoking `scripts.h100_rlm`.
The launcher exposes the following options and chooses the remaining recipe settings itself.

| Option | Default | Meaning and constraints |
| --- | --- | --- |
| `--num-gpus` | Required | `1`, `2`, or `4` visible full H100 GPUs on one host; required on resume too |
| `--model-profile PATH` | Required | Explicit architecture profile for fresh runs and resume |
| `--total-batch-token-size T` | Required for a fresh run | Positive global text targets per update; resume restores the saved budget when omitted and rejects a different explicit value |
| `--assets PATH` | Required for a fresh run | Verified embedding-only seed bundle; omit on resume |
| `--run-name NAME` | `NANOCHAT_RUN_NAME` or `pretrain-1` | One directory name under repository `training_run/`; mutually exclusive with `--output-dir` |
| `--output-dir PATH` | Selected run directory | Explicit run root; trainer outputs go under its `train/` directory; without destination CLI options, `NANOCHAT_BASE_DIR` can override the named-run root |
| `--datasets-dir PATH` | Saved path on resume; otherwise `NANOCHAT_DATASETS_DIR` or repository `datasets/` | Shared dataset root |
| `--resume` | Disabled | Restore the saved recipe/backend and latest checkpoint without dataset download, probes, or initialization |

The launcher fixes four cycles, 5574 optimizer updates, activation checkpointing, model compilation, BF16, seed 42, and the learning-rate/schedule defaults in the direct-trainer tables above.
It probes device batch 32 with a trainable input table, then 32 with a frozen table, then 16 with a trainable table, advancing only on CUDA out-of-memory failures.
Trainable-table capacity recipes use scheduled freezing in production; the frozen-table recipe remains permanently frozen.
Accumulation is derived separately for each recipe and GPU count from the explicit text budget.
Before GPU setup, a fresh launch requires that budget to be divisible by `num_gpus * 32 * 4 * text_length`, so the first batch-32 recipe is valid and the batch-16 fallback is also compatible.
Resume validates divisibility for its saved device-batch recipe at the requested GPU count.
The same budget is passed to capacity probing, attention benchmarking, and production training.
Keeping `N = 5574` with a different `T` changes the total text processed over the run; it does not automatically preserve a prior token total.
Production uses asynchronous checkpoints, numbered saves every 10 updates, and retention of 10 numbered checkpoints.
The launcher compares SDPA with FA3, chooses the faster successful backend, and uses SDPA on ties or failed FA3 probes.

H100 resume restores the saved token budget, unlike the direct trainer's required explicit budget.
Changing GPU count recomputes accumulation while keeping that budget and the selected recipe fixed.
Current `selected-recipe.json` records use schema version 4 and store the text budget; per-launch `logs/TIMESTAMP/invocation.json` also records it with GPU count and accumulation.
The direct trainer's serialized `training_config.total_batch_size` remains the internal field name for the value supplied by `--total-batch-token-size`; it counts text targets, never episodes.
This field name in run records and checkpoints is not a CLI alias.

## Recurrent SFT

[`scripts.rlm_sft`](../scripts/rlm_sft.py) fine-tunes a pretraining checkpoint in one process using the [mixed-episode conversation loader](RLM-architecture/RLM_LONG_DOCUMENTS.md#implementation-the-mixed-episode-loader).
It loads the architecture from the checkpoint, so it does not take `--model-profile`.
`NANOCHAT_RUN_NAME` selects the tokenizer under `training_run/<name>/tokenizer`; it must match the source checkpoint.
By default the source is that run's `train/checkpoints`, and the new output directory is its `sft` directory.
Run `uv run --no-sync python -m scripts.rlm_sft --help` for all options.

| Option | Default | Meaning and constraints |
| --- | --- | --- |
| `--episode-mix CYCLESxBATCH:MICROBATCHES,...` | Required | Shapes in one complete optimizer update; each entry specifies cycles per episode, episodes per microbatch, and microbatch count |
| `--loss-weighting {microbatch,group-token}` | Required | Equal microbatch mean losses, or explicit weights applied to each cycle group's supervised-target mean |
| `--group-loss-weights CYCLES:WEIGHT,...` | Required only for `group-token` | Unique positive cycle keys must exactly match the episode mix; weights must be finite, nonnegative, and sum to one; forbidden with `microbatch` |
| `--total-batch-size T` | Inherit from pretraining | Text positions per optimizer update; must equal `sum(cycles * batch_size * microbatches) * text_length` for exactly one complete mix |
| `--eval-batches N` | `16` | Microbatches per held-out evaluation, not optimizer updates; validation always reports a global supervised-target mean |
| `--checkpoint-dir PATH` | `<run>/train/checkpoints` | Source pretraining checkpoint directory; `--step` selects a step, otherwise the latest complete checkpoint is loaded |
| `--output-dir PATH` | `<run>/sft` | New SFT output directory; it must not already exist |

The SFT options `--cycles` and `--device-batch-size` have been removed.
For fixed-length episodes, pass a one-entry mix such as `--episode-mix 4x8:32`.
The loader still crops or skips over-long conversations on arrival with this single entry; it does not reproduce the original packer's buffer behavior.

With `text_length = 1023`, both `4x8:32` and `4x8:29,12x2:2,24x1:2` contain 1,024 chunks and require `--total-batch-size 1047552`.
They accumulate 32 and 33 microbatches respectively.
The budget counts input text positions, including ignored user and padding positions, rather than the number of supervised targets.
Loss denominators count assistant text targets and eligible EOM readouts that predict assistant tokens.

`microbatch` optimizes the average of the microbatch mean losses.
`group-token` pools supervised losses and target counts within each cycle group, then sums those group means using the supplied weights.
The [loss-weighting investigation](RLM-architecture/RLM_SFT_LOSS_WEIGHTING.md) recommends equal-microbatch averaging as the first comparison baseline; neither objective has demonstrated better model quality.
The example group weights below are an explicit experimental choice, not a default or validated optimum.

These examples assume a source checkpoint with `text_length = 1023`; replace `<name>` with its run name and choose fresh output directories:

```bash
NANOCHAT_RUN_NAME='<name>' uv run --no-sync python -m scripts.rlm_sft \
    --checkpoint-dir 'training_run/<name>/train/checkpoints' \
    --output-dir 'training_run/<name>/sft-microbatch' \
    --episode-mix 4x8:29,12x2:2,24x1:2 \
    --loss-weighting microbatch \
    --total-batch-size 1047552 --eval-batches 33

NANOCHAT_RUN_NAME='<name>' uv run --no-sync python -m scripts.rlm_sft \
    --checkpoint-dir 'training_run/<name>/train/checkpoints' \
    --output-dir 'training_run/<name>/sft-group-token' \
    --episode-mix 4x8:29,12x2:2,24x1:2 \
    --loss-weighting group-token --group-loss-weights 4:0.88,12:0.06,24:0.06 \
    --total-batch-size 1047552 --eval-batches 33
```

Use `--eval-batches 33` or a multiple of 33 to evaluate whole patterns of this example mix.
Other positive counts are allowed but evaluate only that many microbatches, which may be a partial pattern.
Validation uses the same episode mix and always divides total supervised loss by total supervised-target count, independently of the selected training objective.

Training stages one complete mix as CPU input and target tensors and counts supervision before running the model.
It transfers and backpropagates one microbatch at a time, retaining one microbatch's recurrent graph on the GPU at a time.
When compilation is enabled, the trainer raises `torch._dynamo.config.recompile_limit` to at least `8 * number_of_shapes`, preserving any larger existing value, to cover initial, first recurrent, later recurrent, and final cycles in training and validation.
The first memory handoff can have a different tensor stride from later handoffs, requiring a separate compiled variant.

`updates.jsonl` records the actual optimized scalar as `loss`, the global supervised-target mean as `token_loss`, and `supervised_targets` for the whole update.
`episode_groups` is keyed by cycle count and contains each group's token-mean `loss`, `supervised_targets`, and per-cycle `cycle_losses`.
`run.json` and checkpoint metadata serialize the resolved `episode_mix`, `loss_weighting`, and `group_loss_weights` under `sft_config`, with checkpoints storing the run record as `run_contract`.
Checkpoint `memory_contract.gradient_horizons` lists the selected cycle counts and replaces the old scalar `gradient_horizon`.

Evaluate a completed RLM SFT checkpoint on the five ChatCORE tasks (`ARC-Easy`, `ARC-Challenge`, `MMLU`, `GSM8K`, and `HumanEval`) with [`scripts.rlm_chat_eval`](../scripts/rlm_chat_eval.py), which defaults to `<run>/sft/checkpoints`:

```bash
NANOCHAT_RUN_NAME='<name>' uv run --no-sync python -m scripts.rlm_chat_eval \
    --checkpoint-dir 'training_run/<name>/sft/checkpoints' \
    --batch-size 32 \
    --output 'training_run/<name>/chatcore.json'
```

Categorical tasks group prompts by complete chunk count (`len // text_length`), consolidate full chunks without padding, and read choice-letter logits at each prompt's terminal position (or `[memory][<EOM>]` on an exact chunk boundary).
Generative tasks use `RLMEngine`'s intra-cycle `KVCache`, calculator tool state (`<|python_start|>...<|python_end|>`), and recurrent `text_length` rollover.

## Runtime environment and storage

The direct trainer uses the defaults below when the environment does not already supply values.
All memory and disk limits are decimal GB, where 1 GB is 1,000,000,000 bytes, and must be finite and positive.
The H100 launcher sets its own runtime values as shown; pass GPU visibility to the launcher, and launch it outside an existing distributed worker environment.

| Environment variable | Direct-trainer default | H100 behavior and meaning |
| --- | --- | --- |
| `NANOCHAT_RUN_NAME` | `pretrain-1` | Selects the run name unless overridden by launcher CLI |
| `NANOCHAT_BASE_DIR` | Repository `training_run/<name>/` | H100 sets its selected output root; tokenizer is under `<run>/tokenizer` |
| `NANOCHAT_DATASETS_DIR` | Repository `datasets/` | H100 sets the selected shared dataset directory, separately from run outputs |
| `NANOCHAT_TRAIN_SHARDS` | Unset, use discovered parquet files | H100 sets `170`; explicit values 1–6542 choose the first N training shards plus validation shard 6542 |
| `NANOCHAT_DTYPE` | `bfloat16` on CUDA, `float32` on CPU | H100 sets `bfloat16`; RLM accepts BF16 and FP32, not FP16 |
| `NANOCHAT_ATTENTION_IMPL` | `sdpa` | Direct values: `sdpa`, `fa3`, `auto`; explicit FA3 fails if unavailable; H100 probes SDPA/FA3 and sets or restores its selected backend |
| `NANOCHAT_CUDA_MEMORY_LIMIT_GB` | `80` | CUDA allocator ceiling, not total process memory; H100 uses the smallest free/total capacity among selected GPUs minus 2 GB |
| `NANOCHAT_MIN_SYSTEM_AVAILABLE_GB` | `16` | Minimum available host/shared memory during execution; H100 sets `12` after requiring 32 GB available before launch |
| `NANOCHAT_CHECKPOINT_RESERVE_GB` | `100` | Free disk reserve required by checkpoint saving; H100 also checks setup/storage needs before launch |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` | H100 sets the same CUDA allocator configuration |
| `OMP_NUM_THREADS` | `1` | H100 sets `1`; CPU thread count |
| `CUDA_VISIBLE_DEVICES` | Inherited | Limits visible GPUs and their order; H100 selects the first requested count |
| `PYTHONUNBUFFERED` | Inherited | H100 sets `1` for prompt log output |
| `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` | Set by distributed launcher | Worker identity, count, and rendezvous; H100 creates its own workers through `torch.distributed.run` |

The run's library-cache defaults are `UV_CACHE_DIR`, `TORCH_HOME`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `CUDA_CACHE_PATH`, `HF_HUB_CACHE`, `HF_XET_CACHE`, and `TIKTOKEN_CACHE_DIR` beneath `<run>/cache/`.
`WANDB_DIR` defaults to `<run>`, `WANDB_CACHE_DIR` to `<run>/cache/wandb`, and `WANDB_DATA_DIR` to `<run>/wandb/artifacts`; setting these paths does not enable a tracker in the RLM trainer.
Explicit library-cache environment overrides remain honored.
See [project storage](PROJECT_STORAGE.md) for directory layouts, and the hardware workflow guides for setup, monitoring, and dated measurements.

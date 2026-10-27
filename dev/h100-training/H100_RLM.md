# RLM pretraining on 1, 2, or 4 H100 GPUs

Use the supplied files on the merged `rlm` branch to start a fresh pretraining run.
The initialization embedding and matching tokenizer are provided in `assets/h100-seed/` and downloaded through Git LFS during setup.
The launcher installs the Python environment, downloads the dataset, checks capacity, benchmarks attention, and starts training with the faster supported backend.
Select the number of GPUs explicitly with `--num-gpus 1`, `--num-gpus 2`, or `--num-gpus 4` for every launch, including resume.
Every launch also requires `--model-profile PATH`; there is no default profile, including on resume.
Fresh launches require `--total-batch-token-size`, the number of real-text targets across all GPUs per optimizer update; resume restores the saved budget when this option is omitted.
The [training settings reference](../RLM_TRAINING_SETTINGS.md) explains episodes, accumulation, token counts, and every trainer/launcher option.
The [H100 implementation plan](H100_MULTI_GPU_PLAN.md) records the distributed execution and checkpoint design, completed tasks, and validation results.
Before the first multi-GPU pretraining run on a server, follow the [server validation handoff](H100_SERVER_HANDOFF.md).

## Requirements

- Linux x86-64 with at least the requested 1, 2, or 4 unpartitioned H100 GPUs visible on one host, including H100 80 GB and H100 NVL 94 GB.
- A working NVIDIA driver compatible with the locked PyTorch 2.14.1 CUDA 12.6 build.
- At least 32 GB of available host RAM before launch.
- Approximately 280 GB of free disk space for a new run, plus space for the supplied bundle and Python environment.
- Git, Git LFS, `uv`, and Python 3.10 or newer available as `python3`.
- Internet access for dependencies, the supplied files, and the dataset.

Each selected GPU must have at least 75 GB (75 × 10^9 bytes) of total GPU memory; there is no upper memory limit.
The allocator budget uses the least free memory among the selected GPUs.
Capacity probes also measure host memory with local checkpoint staging and the coordinator's complete optimizer snapshot resident, so multi-GPU runs can require more than the minimum host RAM.
Use a persistent terminal session or your normal job scheduler so disconnecting SSH does not terminate training.
The launcher checks the GPU, available memory, and disk space before training; it does not install or upgrade the NVIDIA driver.

## Start pretraining

Initialize Git LFS, clone the branch, download the supplied files, and run the launcher from the repository root:

```bash
git lfs install
git clone --branch rlm https://github.com/ifsheldon/nanochat.git
cd nanochat
git lfs pull
bash runs/runh100_rlm.sh \
    --num-gpus 1 \
    --model-profile model_profiles/rlm-d24.jsonc \
    --total-batch-token-size 1047552 \
    --assets assets/h100-seed \
    --run-name pretrain-1
```

For an existing checkout, run `git lfs install`, `git fetch origin`, `git switch rlm`, `git pull --ff-only`, and `git lfs pull` before launching.
The launcher selects the first requested number of visible GPUs, in visibility order.
Use `CUDA_VISIBLE_DEVICES` to choose their physical device indices or GPU UUIDs; for example:

```bash
CUDA_VISIBLE_DEVICES=2,3 bash runs/runh100_rlm.sh \
    --num-gpus 2 --model-profile model_profiles/rlm-d24.jsonc \
    --total-batch-token-size 1047552 \
    --assets assets/h100-seed --run-name pretrain-2gpu

CUDA_VISIBLE_DEVICES=0,1,2,3 bash runs/runh100_rlm.sh \
    --num-gpus 4 --model-profile model_profiles/rlm-d24.jsonc \
    --total-batch-token-size 1047552 \
    --assets assets/h100-seed --run-name pretrain-4gpu
```

Run the script directly, outside `torchrun` or another distributed worker environment.
It starts one worker per selected GPU through `torch.distributed.run` for two or four GPUs; dataset preparation runs once.
A selected GPU partition or a different accelerator is rejected.
Every recurrent episode and its four-cycle gradient path stays on its assigned GPU.
The existing distributed optimizer averages gradients and partitions optimizer state across workers.

The launcher uses `uv.lock` and Python 3.10.21 to install the required environment automatically.
The GPU extra selects PyTorch 2.14.1+cu126 on this Linux x86-64 host; the same lockfile selects PyTorch 2.14.1+cu132 on Linux ARM64 for DGX Spark.
It verifies the supplied files and installs their tokenizer, then downloads 170 training shards and one validation shard, approximately 15.7 GB in total.
Existing dataset files are reused and verified.
No separate dataset transfer is needed.

The default locations are:

| Contents | Location |
| --- | --- |
| Run outputs, tokenizer, and caches | `training_run/pretrain-1/` |
| Shared dataset | `datasets/` |

Choose a new `--run-name` for each fresh run.
Use `--output-dir /path/to/run` or `--datasets-dir /path/to/datasets` if those files need to live on another disk.
Explicit cache-directory environment settings remain honored.
Use `bash runs/runh100_rlm.sh --help` for the available options.

The model architecture comes from a seven-field [JSONC profile](../../model_profiles/README.md), selected explicitly with required `--model-profile PATH`.
The bundled reference `model_profiles/rlm-d24.jsonc` specifies 24 layers, width 1536, 12 query and KV heads, 1023 text tokens, 512 memory tokens, and exactly one EOM token.
The launcher preserves the original document as `OUTPUT/model-profile.jsonc` before probing and passes that same copy through capacity selection, attention selection, and training.
The supplied initialization embedding requires width 1536 and its matching tokenizer; a different width requires a compatible asset bundle.
The bundled profile with 511 text tokens and 128 memory slots uses the same backbone and can launch with a compatible 1,046,528-text-target budget:

```bash
bash runs/runh100_rlm.sh \
    --num-gpus 2 --model-profile model_profiles/rlm-d24-c511-m128.jsonc \
    --total-batch-token-size 1046528 \
    --assets assets/h100-seed --run-name pretrain-c511-m128-2gpu
```

## Automatic capacity selection

The launcher tries the following recipes in order with PyTorch SDPA, using one complete optimizer update for each capacity probe.
The accumulation counts below apply to the reference d24 profile's 1023 text tokens with `--total-batch-token-size 1047552`:

| Priority | Batch per GPU | Accumulation with 1 / 2 / 4 GPUs | Input embedding during probe | Selected training recipe |
| --- | --- | --- | --- | --- |
| 1 | 32 | 8 / 4 / 2 | Trainable | Batch 32; frozen for updates 1–5016, trainable for 5017–5574 |
| 2 | 32 | 8 / 4 / 2 | Frozen | Batch 32; input embedding frozen for all 5574 updates |
| 3 | 16 | 16 / 8 / 4 | Trainable | Batch 16; frozen for updates 1–5016, trainable for 5017–5574 |

After a successful capacity probe, the launcher benchmarks attention for the selected recipe as described below.
Production then starts fresh from the supplied initialization; probe updates are discarded.
Every recipe uses 5574 optimizer updates and the explicitly selected global text-target budget.
The 1,047,552 budget above gives 256 complete four-cycle episodes and 1,048,576 supervised predictions per update with the reference d24 profile.
Increasing GPU count reduces accumulation per worker and keeps this global batch unchanged.
Accumulation is `total_batch_token_size / (device_batch_size * 4 * text_length * num_gpus)` and must be a positive integer.
Fresh launches validate the batch-32 recipe before GPU setup, so the budget must be divisible by `32 * 4 * text_length * num_gpus`; the batch-16 fallback then also divides evenly.
With `rlm-d24-c511-m128.jsonc` and a 1,046,528 budget, each update contains 512 episodes and 1,048,576 predictions; device batch 32 accumulates 16 / 8 / 4 microbatches on 1 / 2 / 4 GPUs, and device batch 16 uses twice those counts.
The text budget excludes the extra EOM predictions and memory/summary positions, and is passed unchanged to capacity probes, attention probes, and training.
Changing that budget changes total training text across the fixed 5574-update horizon.
The remaining training settings are configured by the launcher.

Only a CUDA out-of-memory result advances to the next recipe.
Other failures stop the launcher for diagnosis.
If all three recipes run out of memory, it prints a warning and exits without starting training.

## Automatic attention selection

For the selected recipe, the launcher compares PyTorch SDPA with FlashAttention 3 loaded through Hugging Face `kernels`.
Each backend runs in its own subprocess with the same model, data, batch size, accumulation, activation checkpointing, and optimizer settings.
Scheduled-freeze recipes use trainable input embeddings during the comparison, matching the capacity probe's more demanding phase; the permanently frozen recipe keeps them frozen.

Each backend warms up for at most five full optimizer updates until all workers complete an update without compilation, then measures exactly one further update without compilation.
Workers make the warmup decision together, and the reported update duration is the slowest worker's duration.
The measurement includes forward, backward, optimizer work, and training health checks, with CUDA synchronization around the timed update.
Compilation and checkpoint disk writes are excluded; optimizer state, local checkpoint staging buffers, and the coordinator's complete optimizer staging buffers are already resident.
The launcher chooses the faster successful backend, using SDPA for a tie, and records the measurements and selection in `selected-recipe.json`.
The comparison is one steady update for the selected recipe, not a full-run throughput measurement.

If FlashAttention cannot load or fails during compilation or execution, runs out of memory, or its subprocess crashes, the launcher records the failure and proceeds with SDPA.
A failed SDPA probe stops the launcher for diagnosis.
SDPA can itself use PyTorch's built-in flash attention kernels, so this comparison selects between PyTorch's attention implementation and the separate Hugging Face kernel.

To probe a backend separately for the 1,047,552-budget example, run `uv run --no-sync python -m scripts.rlm_train --model-profile OUTPUT/model-profile.jsonc --total-batch-token-size 1047552 --mode attention-probe` with the same remaining training arguments and a fresh output directory, setting `NANOCHAT_ATTENTION_IMPL=sdpa` or `NANOCHAT_ATTENTION_IMPL=fa3` explicitly.
For another recipe, pass its chosen text budget instead; the direct trainer requires this option even when the H100 launcher would restore it on resume.
Use `--input-embedding-freeze none` or `--input-embedding-freeze permanent`; scheduled freezing is not supported by this probe.
Retain the full training horizon in `--num-iterations` because it controls the learning-rate schedule, not the number of compilation warmup updates.

## Monitor the run

`OUTPUT` below means the selected run directory, normally `training_run/pretrain-1`.

| File or directory | What to check |
| --- | --- |
| `OUTPUT/logs/TIMESTAMP/` | Dataset preparation, capacity probe, attention probe, and training logs |
| `OUTPUT/selected-recipe.json` | Schema 4: original GPU count, global text-token budget, per-worker accumulation, selected recipe, attention measurements, and training arguments |
| `OUTPUT/model-profile.jsonc` | Original profile used for fresh-launch probes and training |
| `OUTPUT/logs/TIMESTAMP/invocation.json` | Actual GPU count, global text-token budget, and accumulation for the current launch or resume |
| `OUTPUT/probes/` | Capacity probe results and any failures |
| `OUTPUT/probes/attention-sdpa/`, `OUTPUT/probes/attention-fa3/` | Attention measurements and failures when a result could be written |
| `OUTPUT/train/attempts/` | Training measurements and attempt records |
| `OUTPUT/train/result.json` | Latest completed attempt or failure |
| `OUTPUT/train/checkpoints/` | Published training checkpoints |
| `OUTPUT/train/checkpoints/model-profile.jsonc` | Original model profile copied once alongside checkpoints; pass this path explicitly when resuming |

Confirm production reaches update 2 and publishes both `latest.pt` and a numbered checkpoint before leaving it unattended.
The capacity probe checks one update; continue watching for later memory, driver, or storage failures.

`latest.pt` is saved after every optimizer update for interruption recovery.
The coordinator owns shared run artifacts and writes one model plus the complete optimizer state, so checkpoints can resume with any supported GPU count.
Each worker stages its local optimizer state on the CPU; the coordinator gathers those tensors and writes the snapshot asynchronously while the next forward/backward runs.
All workers wait for that snapshot before updating parameters, and a required milestone must finish before the input embedding unfreezes.
Worker diagnostics use separate files; update losses average across workers, CUDA memory reports the largest per-worker usage, and throughput uses the global text batch and slowest-worker duration.
Numbered checkpoints are saved at update 2, every 10 updates, and the final or explicitly paused update; the latest 10 numbered checkpoints are retained.
`pretrain-90pct.pt` preserves the complete update-5016 state separately from numbered-checkpoint retention.
For recipes that unfreeze the input embedding, this checkpoint is saved before update 5017 begins.
It is also saved for the permanently frozen recipe.
Keep these files in the run directory so the launcher can resume from them.

## Resume after interruption

When the run has a complete `latest.pt`, resume it with:

```bash
bash runs/runh100_rlm.sh \
    --num-gpus 4 --resume --run-name pretrain-1 \
    --model-profile training_run/pretrain-1/train/checkpoints/model-profile.jsonc
```

If the original launch used `--output-dir`, use that same option and path when resuming.
The command omits `--total-batch-token-size` to restore the saved recipe's budget; supplying the same value explicitly is allowed, and a different value is rejected.
The launcher uses the saved dataset directory by default; pass `--datasets-dir` to select a different location.
Resume restores the selected recipe and attention backend, model, complete optimizer state, data-loader position, and random state, then continues the original 5574-update budget.
The requested GPU count may differ from the saved count: transitions between 1, 2, and 4 GPUs repartition optimizer moments and recompute accumulation while preserving the global batch.
For example, the command above resumes the one-GPU example on four GPUs.
At the same count, each worker restores its own saved loader cursor and RNG state.
At a changed count, workers repartition data from the earliest saved row-group cursor and reuse saved RNG states deterministically.
Data continuation remains approximate at row-group granularity and does not preserve exact example order when the count changes.
It skips dataset downloading, capacity and attention probes, and embedding initialization, so `--assets` is not needed.

The required `--model-profile PATH` selects the architecture, and the saved recipe selects the training settings.
Pass `OUTPUT/train/checkpoints/model-profile.jsonc` explicitly to use the saved document, as above; the original source file is then no longer needed, and editing it does not change resume.
You may instead pass a source profile with the same seven field values as the saved checkpoint profile, whose original comments are retained.
If the checkpoint directory has no saved profile, pass a profile matching the checkpoint's architecture so the trainer can copy it on its next save.
Resume does not require identical source files, a matching saved command line, or matching tokenizer and dataset hashes.
Checkpoint state is loaded normally, so model tensors and optimizer state must still be loadable by the selected model.

If preparation or a probe failed, or production stopped before publishing a checkpoint, inspect the logs and structured results, fix the cause, and start with a new run name.
Retain the failed run's files for diagnosis.
For help on the H100 machine, provide the selected recipe when available, the latest log directory, and the relevant probe or training result.

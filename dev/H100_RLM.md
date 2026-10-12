# Single-H100 RLM handoff

This branch packages a fresh RLM pretraining run for one H100 80 GB GPU.
The launcher keeps the current Spark model, data selection, token budget, optimizer settings, and learning-rate schedule, and chooses among exactly three authorized batch/freezing recipes using one-update capacity probes.
It does not need the current Spark run's checkpoints or optimizer state.

The verified seed bundle is included at `assets/h100-seed/`.
Git LFS stores its three binary files: `model_005568.pt`, `tokenizer.pkl`, and `token_bytes.pt`.
The accompanying `meta_005568.json` and `manifest.json` are ordinary Git files.
The model file contains only the pretrained input embedding; the full source checkpoint is not included.

## Start on the H100 machine

Install Git LFS and `uv` on the H100 machine if they are not already available.
Initialize Git LFS before cloning so the binary seed files are downloaded correctly.
The machine needs Linux, a working NVIDIA driver compatible with the locked CUDA 13.2 PyTorch build, internet access for dependencies and the original dataset, and at least 32 GB of host RAM currently available before launch.
The launcher does not install or upgrade the NVIDIA driver.
Use a persistent terminal session or your machine's normal job scheduler so disconnecting SSH does not terminate training.

```bash
git lfs install
git clone --branch h100-rlm git@github.com:ifsheldon/nanochat.git
cd nanochat
git lfs pull
bash runs/runh100_rlm.sh \
    --assets assets/h100-seed \
    --run-name pretrain-1
```

For an existing checkout, run `git lfs install`, `git fetch origin`, `git switch h100-rlm`, `git pull --ff-only`, and `git lfs pull`, then use the same launch command.
The launcher verifies the manifest checksums and rejects missing payloads or unexpanded LFS pointer files.

The script uses `uv.lock`, Python 3.10.21, and the repository's GPU extra to create or synchronize the local environment.
Before invoking uv, its standard-library bootstrap sets run-local defaults for package and runtime download caches, including uv and Hugging Face Xet.
Explicit cache-directory environment settings remain honored.
It then verifies the GPU and resource budget, verifies the embedding bundle, installs its matching tokenizer, prepares the dataset, probes the recipes in order, and starts a fresh production run.
Use `bash runs/runh100_rlm.sh --help` for available options.
If the host exposes multiple GPUs, select the H100 explicitly with `CUDA_VISIBLE_DEVICES=0` or the appropriate device index before this command.
Run the script directly, outside `torchrun` or another distributed worker environment.
A GPU partition or a different accelerator is rejected.

By default, the run is written to `training_run/pretrain-1` in the repository, and its tokenizer is stored in `training_run/pretrain-1/tokenizer`.
Choose another `--run-name` for a separate run, or use `--output-dir /path/to/run` as an explicit destination override.
If neither option is given, `NANOCHAT_BASE_DIR` can override the run destination and `NANOCHAT_RUN_NAME` can select its name.
Shared datasets default to the repository's `datasets/` directory; `--datasets-dir /path/to/datasets` or `NANOCHAT_DATASETS_DIR` selects a different shared root.
A dataset lock serializes preparation and verification across concurrent H100 launchers.
The run lock separately protects its tokenizer and outputs.
Compiler, CUDA, kernel-download, and tracking caches default to this run's `cache/` and tracking directories; explicit environment settings remain honored.
The launcher refuses a tokenizer mismatch within a run; choose a new run directory instead of changing its initialization.
These are defaults for new runs; existing Spark files and running jobs remain at their current locations.

NanoChat already defines the download URLs and validation shard in `nanochat/dataset.py`.
This launcher calls the existing downloader with `python -m nanochat.dataset -n 170`, which downloads training shards `00000` through `00169` and automatically adds validation shard `06542`.
Existing files are reused by that downloader.
The launcher checks that all 171 required Parquet files exist and contain text rows, then records their sizes and SHA-256 checksums.
Additional shards belonging to other runs can remain in the shared directory.
`NANOCHAT_TRAIN_SHARDS=170` constrains the training loader to the same 170 training shards and validation shard `06542`, so additional shared files do not change this run's data selection.
It also checks the final inventory because the existing downloader can report individual download failures without returning a failed process status.
The dataset is approximately 15.7 GB; no separate dataset transfer is required.

## Optional: rebuild the seed bundle

The repository already includes the seed needed for the launch above.
To reproduce or intentionally replace that seed, export the original pretrained input embedding and its matching tokenizer:

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m scripts.rlm_export_embedding \
    --checkpoint-dir /path/to/base_checkpoints/spark-d24-bf16 \
    --step 5568 \
    --tokenizer-dir /path/to/tokenizer \
    --output-dir /path/to/h100-seed
```

The destination must be new and its parent must exist.
This produces a separate candidate bundle; normal H100 setup obtains the published bundle through Git LFS.
The five files are `model_005568.pt`, `meta_005568.json`, `tokenizer.pkl`, `token_bytes.pt`, and `manifest.json`.
The model file contains only `transformer.wte.weight`; it retains the checkpoint naming convention so the existing initialization loader can read it.
The `--init-embedding-checkpoint-dir`, `--init-embedding-step`, and `--init-embedding-tokenizer` options accept both original full checkpoints and compact checkpoints through the same loader.
The H100 launcher uses `--init-embedding-bundle` to additionally verify the exported manifest and tokenizer assets.
The tensor is cloned before saving, preventing its serialization from retaining the full source checkpoint's storage.
The bundle preserves the source model's provenance and records SHA-256 checksums for every payload file and for the embedding tensor.
Only the input table is copied into the new RLM; the remaining model weights, `<EOM>`, summary inputs, and optimizer start fresh.

## Capacity decision

Each candidate runs in its own subprocess with a fresh model and independent output directory.
A probe performs exactly one complete optimizer update with the same 1,047,552 real-text targets as production, including all accumulation microbatches.
Optimizer state and reusable checkpoint staging are allocated before the measured update to include the persistent allocations that would otherwise first appear after update one.
The probe retains the full 5,574-update learning-rate schedule; its one-update stop does not shorten that schedule.
Probe weights and optimizer state are discarded before the production run starts from the seed again.

| Priority | Probe batch × accumulation | Input embedding during probe | Selected production recipe |
| --- | --- | --- | --- |
| 1 | 32 × 8 | Trainable | Batch 32; frozen for updates 1–5016, trainable for 5017–5574 |
| 2 | 32 × 8 | Frozen | Batch 32; input embedding frozen for all 5574 updates |
| 3 | 16 × 16 | Trainable | Batch 16; frozen for updates 1–5016, trainable for 5017–5574 |

Only a structured CUDA out-of-memory failure advances to the next candidate.
A driver, compiler, tokenizer, dataset, filesystem, nonfinite-value, or unclassified process failure stops the launcher for diagnosis.
If all three candidates run out of memory, it prints a warning and exits without starting production.
It does not choose a fourth recipe, lower precision, or a smaller effective batch.
A successful trainable-table probe selects the original scheduled freezing policy for production, rather than making the table trainable from the first production update.

The allocator limit is computed from the H100's measured total and currently free GPU memory, leaving 2 decimal GB for external CUDA allocations.
The same limit applies to all probes and production in that launch.
The Spark-specific 100 GB allocator limit is never inherited.
The host-memory guard remains at 12 decimal GB available throughout training.
Reusable checkpoint staging needs roughly 10 GB of pinned host RAM, in addition to loading buffers and other host allocations.
The preflight requires 32 GB available host RAM to leave room for these allocations and the guard.

The disk preflight budgets up to 156 GB for retained and in-flight checkpoints plus the configured 100 GB free-space reserve, and adds space for missing dataset shards when the datasets share that filesystem.
Allow approximately 280 GB of free disk space for a new run with no downloaded datasets, plus the seed bundle and Python environment.
On resume, existing checkpoint bytes count toward the checkpoint budget.
Actual checkpoint sizes depend on whether the input table has Adam state.

A one-update probe checks the intended resident-memory workload; it cannot guarantee that a different driver, compiler, later recompilation, another process, or later host-memory pressure will never cause a failure.
The probe reserves checkpoint staging memory but does not validate actual disk checkpoint serialization or reload.
Confirm production reaches update 2 and publishes both rolling and numbered checkpoints before leaving it unattended.

## Fixed training recipe

The model has 24 layers, width 1536, head dimension 128, four recurrent cycles, 1023 text tokens per chunk, and 512 memory vectors.
It uses BF16, full causal SDPA, compiled execution, activation checkpointing, and seed 42.
The total real-text batch is 1,047,552 tokens across either 32 × 8 or 16 × 16 microbatches.
Training runs for 5574 optimizer updates, with 40 warmup updates, a 0.65 warmdown ratio, and a final learning-rate fraction of 0.05.
The configured embedding, unembedding, matrix, and scalar learning rates are 0.3, 0.008, 0.02, and 0.5; configured weight decay is 0.28.
The existing optimizer applies the same width/batch scaling and schedules as the Spark recipe.

Async checkpointing publishes rolling `train/checkpoints/latest.pt` after every optimizer update.
Numbered checkpoints are saved at update 2, every 10 updates, and the final or explicitly paused update, with the latest 10 numbered checkpoints retained.
`train/checkpoints/pretrain-90pct.pt` preserves the complete update-5016 state for both the scheduled and permanently frozen policies.
For scheduled freezing, it must be durable before the table can receive gradients at update 5017.
The milestone is outside normal numbered-checkpoint retention and preserves the model, optimizer, loader state, RNG state, and run contract.
Rolling `latest.pt` remains available for interruption recovery.
Failure to preserve the required milestone stops training before update 5017, including for the permanently frozen policy.

## Monitor and resume

`OUTPUT` below means the selected run directory, which defaults to `training_run/pretrain-1`.
`OUTPUT/selected-recipe.json` records the selected option, exact production arguments, tokenizer and dataset checksums, resource settings, shared dataset location, and GPU/runtime information.
`OUTPUT/logs/TIMESTAMP/` contains the download, probe, and production logs together with the invocation and resource report.
`OUTPUT/probes/` keeps each attempted probe's measurements and structured result.
`OUTPUT/train/attempts/` contains the trainer's update measurements and source/configuration records; `OUTPUT/train/result.json` summarizes the latest completed attempt or failure.
The selected recipe is written before production starts and is reused for recovery.

After an interruption with a complete rolling checkpoint:

```bash
bash runs/runh100_rlm.sh --resume --run-name pretrain-1
```

Resume skips downloading, capacity probes, and embedding initialization.
It verifies the original tokenizer and dataset checksums, restores the selected batch/freezing policy and the full 5574-update budget, and resumes the saved model, optimizer, RNG, and data-loader state.
The original seed bundle is not required for resume.
Keep the checkout, run directory, and selected dataset files unchanged because the trainer verifies its source and run contract; do not silently switch commits or edit the recorded recipe to bypass a mismatch.
If production failed before any checkpoint was published, retain the diagnostics and start again with a new output directory after fixing the cause.
Similarly, a partially completed probe run requires a new output directory rather than reusing previous probe results.

For an agent helping on the H100 machine, begin with this guide, `OUTPUT/selected-recipe.json` when present, the latest log directory, and the probe or trainer `result.json`.
Use the structured failure to distinguish a capacity fallback from an environment or data problem.
Do not modify architecture, total token batch, schedule, dataset selection, or the three-option order to make a run fit without an explicit change to the experiment.

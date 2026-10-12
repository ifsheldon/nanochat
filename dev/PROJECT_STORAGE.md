# Project storage

New runs use two directories in the repository: `training_run/` holds artifacts for individual runs, and `datasets/` holds downloaded data shared across runs.
The default run is `training_run/pretrain-1`.
These defaults are resolved relative to the repository, independently of the shell's current directory.
They do not move existing files or change already running jobs.
The shell launchers require Python 3.10 or newer to resolve paths before setting up the training environment.

A typical layout is:

```text
training_run/
  .gitkeep
  pretrain-1/
    tokenizer/
    cache/
    ... checkpoints, logs, reports, and other run artifacts ...
  pretrain-2/
    ... artifacts for another run ...
datasets/
  .gitkeep
  base_data_climbmix/
  task_data/
  eval_bundle/
```

The exact checkpoint and log subdirectories depend on the training workflow.
Tokenizers, model and optimizer checkpoints, measurements, reports, and library caches belong to the selected run.
ClimbMix pretraining shards, downloaded task datasets, and the CORE evaluation bundle belong to the shared datasets directory.
Additional shared data does not need to be copied into each run.
Concurrent runs serialize shared downloads, and an interrupted CORE extraction can retry from its cached archive.
Git tracks the two `.gitkeep` placeholders and ignores other contents beneath these storage roots.

## Select a run and shared datasets

Set `NANOCHAT_RUN_NAME` before launching a workflow to select a named run:

```bash
export NANOCHAT_RUN_NAME=pretrain-2
```

Without an explicit destination, this selects `training_run/pretrain-2` within the repository.
Run names must be one nonempty directory name, without path separators, parent-directory traversal, or surrounding whitespace.
A workflow that supports a `--run-name` option can select the same named-run location explicitly.
An explicit run name takes precedence over the run-related environment variables.

`NANOCHAT_BASE_DIR` selects an explicit artifact directory and takes precedence over `NANOCHAT_RUN_NAME` when the workflow has not selected a run name or output path explicitly:

```bash
export NANOCHAT_BASE_DIR=/path/to/experiment-output
export NANOCHAT_DATASETS_DIR=/path/to/shared-datasets
```

`NANOCHAT_DATASETS_DIR` overrides only the shared dataset location; changing the run name or artifact directory does not move the datasets.
When it is unset, datasets remain under the repository's `datasets/` directory.
Workflow-specific output arguments remain explicit overrides where available.
Use absolute paths for explicit directories when launching from different working directories.
Storage directory paths reject parent-directory traversal and symlink components.
Resolving these paths does not create directories; the downloader, trainer, or other owning workflow creates them when needed.

## Keep a fixed dataset selection

The ClimbMix downloader still downloads the first requested number of training shards and automatically includes validation shard `06542`:

```bash
uv run --locked --extra gpu python -m nanochat.dataset -n 170
```

A shared dataset directory can contain more shards than a particular run needs.
Set `NANOCHAT_TRAIN_SHARDS` to keep that run's selection fixed:

```bash
export NANOCHAT_TRAIN_SHARDS=170
```

This selects training shards `00000` through `00169` and validation shard `06542`, regardless of additional cached shards.
A missing selected file is an error; the loader does not silently choose a smaller or different dataset.
The value must be an integer from 1 through 6542.
Without this variable, the loader lists the available Parquet files in sorted order and uses the last one for validation.
Keep the selection unchanged when resuming an experiment.

The scaling-laws sweep uses the existing shared shards and never downloads them automatically.
Before starting an expensive sweep, it verifies that training and validation data are available and trains a missing run tokenizer with the standard 2-billion-character, 32768-token settings.
A complete tokenizer is reused; a partial tokenizer is an error and is not overwritten.

## Library caches and existing runs

uv package, compiler, CUDA, Torch, kernel-download, Hugging Face Xet, tokenizer-download, and tracking caches default to the selected run's `cache/` or tracking directories.
The startup helper validates all default destinations before updating the environment, rejecting symlinks or files in their directory paths.
It configures these defaults before the relevant libraries are imported; the shell runners also configure the uv cache before installing dependencies.
Explicit settings such as `UV_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `CUDA_CACHE_PATH`, `TORCH_HOME`, `HF_HUB_CACHE`, `HF_XET_CACHE`, `TIKTOKEN_CACHE_DIR`, and the `WANDB_*` directory variables remain honored.
Changing directory defaults does not change precision, model parameters, batch sizes, schedules, or checkpoint policies.

Existing runs and downloaded files remain at their original paths.
To reuse an existing cache with current code, set both `NANOCHAT_BASE_DIR` to its tokenizer/run root and `NANOCHAT_DATASETS_DIR` to the directory containing its datasets.
For example, a run previously using `~/.cache/nanochat/spark-d24` for both requires both variables to point there.
Keep its explicit output and checkpoint paths, shard selection, and other recorded arguments when continuing the same experiment.
RLM resume restores checkpoint state without requiring source or configuration fingerprints to match.
Do not move existing files while training is active.
These defaults apply to future launches, and no automatic migration or copying from an old cache directory is performed.

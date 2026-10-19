# Single-H100 RLM pretraining

Use the supplied files on the `h100-rlm` branch to start a fresh pretraining run.
The initialization embedding and matching tokenizer are provided in `assets/h100-seed/` and downloaded through Git LFS during setup.
The launcher installs the Python environment, downloads the dataset, checks capacity, and starts training.

## Requirements

- Linux with exactly one unpartitioned H100 GPU visible to the run, including H100 80 GB and H100 NVL 94 GB.
- A working NVIDIA driver compatible with the locked CUDA 13.2 PyTorch build.
- At least 32 GB of available host RAM before launch.
- Approximately 280 GB of free disk space for a new run, plus space for the supplied bundle and Python environment.
- Git, Git LFS, `uv`, and Python 3.10 or newer available as `python3`.
- Internet access for dependencies, the supplied files, and the dataset.

The GPU capacity check requires at least 75 GB (75 × 10^9 bytes) of total GPU memory and has no upper memory limit.
Use a persistent terminal session or your normal job scheduler so disconnecting SSH does not terminate training.
The launcher checks the GPU, available memory, and disk space before training; it does not install or upgrade the NVIDIA driver.

## Start pretraining

Initialize Git LFS, clone the branch, download the supplied files, and run the launcher from the repository root:

```bash
git lfs install
git clone --branch h100-rlm https://github.com/ifsheldon/nanochat.git
cd nanochat
git lfs pull
bash runs/runh100_rlm.sh \
    --assets assets/h100-seed \
    --run-name pretrain-1
```

For an existing checkout, run `git lfs install`, `git fetch origin`, `git switch h100-rlm`, `git pull --ff-only`, and `git lfs pull` before launching.
If the host exposes multiple GPUs, set `CUDA_VISIBLE_DEVICES` to the H100's device index when running the launcher.
Run the script directly, outside `torchrun` or another distributed worker environment.
A GPU partition or a different accelerator is rejected.

The launcher uses `uv.lock` and Python 3.10.21 to install the required environment automatically.
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

## Automatic capacity selection

The launcher tries the following recipes in order, using one complete optimizer update for each capacity probe:

| Priority | Probe batch × accumulation | Input embedding during probe | Selected training recipe |
| --- | --- | --- | --- |
| 1 | 32 × 8 | Trainable | Batch 32; frozen for updates 1–5016, trainable for 5017–5574 |
| 2 | 32 × 8 | Frozen | Batch 32; input embedding frozen for all 5574 updates |
| 3 | 16 × 16 | Trainable | Batch 16; frozen for updates 1–5016, trainable for 5017–5574 |

After a successful probe, production starts fresh from the supplied initialization with the selected recipe.
Every recipe uses 5574 optimizer updates and 1,047,552 real-text tokens per update.
The remaining model and training settings are already configured by the launcher.

Only a CUDA out-of-memory result advances to the next recipe.
Other failures stop the launcher for diagnosis.
If all three recipes run out of memory, it prints a warning and exits without starting training.

## Monitor the run

`OUTPUT` below means the selected run directory, normally `training_run/pretrain-1`.

| File or directory | What to check |
| --- | --- |
| `OUTPUT/logs/TIMESTAMP/` | Dataset preparation, capacity probe, and training logs |
| `OUTPUT/selected-recipe.json` | Selected recipe and exact training arguments |
| `OUTPUT/probes/` | Capacity probe results and any failures |
| `OUTPUT/train/attempts/` | Training measurements and attempt records |
| `OUTPUT/train/result.json` | Latest completed attempt or failure |
| `OUTPUT/train/checkpoints/` | Published training checkpoints |

Confirm production reaches update 2 and publishes both `latest.pt` and a numbered checkpoint before leaving it unattended.
The capacity probe checks one update; continue watching for later memory, driver, or storage failures.

`latest.pt` is saved after every optimizer update for interruption recovery.
Numbered checkpoints are saved at update 2, every 10 updates, and the final or explicitly paused update; the latest 10 numbered checkpoints are retained.
`pretrain-90pct.pt` preserves the complete update-5016 state separately from numbered-checkpoint retention.
For recipes that unfreeze the input embedding, this checkpoint is saved before update 5017 begins.
It is also saved for the permanently frozen recipe.
Keep these files in the run directory so the launcher can resume from them.

## Resume after interruption

When the run has a complete `latest.pt`, resume it with:

```bash
bash runs/runh100_rlm.sh --resume --run-name pretrain-1
```

If the original launch used `--output-dir`, use that same option and path when resuming.
The launcher uses the saved dataset directory by default; pass `--datasets-dir` to select a different location.
Resume restores the selected recipe, model, optimizer, data-loader position, and random state, then continues the original 5574-update budget.
It skips dataset downloading, capacity probes, and embedding initialization, so `--assets` is not needed.

The saved recipe selects the model and training settings.
Resume does not require identical source files, a matching saved command line, or matching tokenizer and dataset hashes.
Checkpoint state is loaded normally, so model tensors and optimizer state must still be loadable by the selected model.

If preparation or a probe failed, or production stopped before publishing a checkpoint, inspect the logs and structured results, fix the cause, and start with a new run name.
Retain the failed run's files for diagnosis.
For help on the H100 machine, provide the selected recipe when available, the latest log directory, and the relevant probe or training result.

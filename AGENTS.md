# Implementation scope

When implementing a feature, do not add unnecessary checks or associated features unless the user explicitly requests them.
If you consider additional checks or related features important, explain them to the user and obtain explicit approval before implementing them.

# Project overview

nanochat is a Python/PyTorch toolkit for language-model tokenization, pretraining, supervised fine-tuning, reinforcement learning, evaluation, and inference.
This fork adds DGX Spark workflows and recurrent latent memory (RLM), including pretraining on 1, 2, or 4 H100 80 GB or H100 NVL 94 GB GPUs on one host.
RLM reuses a GPT backbone across bounded text chunks and passes learned memory between chunks.

## Codebase structure

| Path | Purpose |
| --- | --- |
| `nanochat/` | Core models, tokenizers, data loaders, optimizers, inference, training utilities, and checkpoint handling. |
| `scripts/` | Python CLI entrypoints for training, evaluation, sampling, profiling, and hardware-specific orchestration; run with `python -m scripts.<name>`. |
| `runs/` | Shell launchers for complete workflows, including `runh100_rlm.sh`, Spark pipelines, CPU demos, and scaling experiments. |
| `tasks/` | Task datasets and evaluation adapters, including ARC, GSM8K, HumanEval, MMLU, and SmolTalk. |
| `tests/` | Pytest coverage for models, data loading, training, checkpoints, resource limits, and launchers. |
| `dev/` | Documentation index and shared references; `spark-training/` and `h100-training/` hold operational guides, while `RLM-architecture/` holds design and analysis. |
| `assets/h100-seed/` | Supplied H100 initialization embedding and tokenizer; binaries use Git LFS, with checksums in `manifest.json`. |
| `training_run/` | Ignored run outputs: checkpoints, tokenizers, logs, reports, and caches; the default run is `pretrain-1`. |
| `datasets/` | Ignored shared downloads for pretraining, task datasets, and evaluation. |
| `pyproject.toml`, `uv.lock` | Python dependencies and locked environment, managed with `uv`. |
| `.gitattributes`, `.lfsconfig` | Git LFS tracking and the shared asset repository endpoint. |

## Main implementation paths

- `nanochat/gpt.py` defines the base transformer; `nanochat/rlm.py` defines the recurrent model.
- `nanochat/tokenizer.py`, `dataset.py`, and `dataloader.py` handle tokenization and pretraining data; `sft_dataloader.py` handles conversation packing.
- `nanochat/optim.py` implements optimizers; `training_runtime.py` provides shared training resource checks; `paths.py` resolves run, dataset, and cache locations.
- `nanochat/rlm_distributed.py` coordinates workers; `optim_checkpoint.py` partitions portable optimizer state; `rlm_distributed_checkpoint.py` gathers worker snapshots for coordinator publication.
- `nanochat/rlm_training.py` implements recurrent training; the other `rlm_*.py` modules separate data handling, initialization, freezing policy, capacity probes, inference, and checkpoints.
- `scripts/base_train.py`, `chat_sft.py`, and `chat_rl.py` drive the base-model and chat training stages.
- `scripts/rlm_train.py`, `rlm_eval.py`, and `rlm_sample.py` expose recurrent training, evaluation, and generation; `scripts/h100_rlm.py` orchestrates the H100 workflow with required `--num-gpus {1,2,4}` and portable resume across those counts.

## Workflow guides

- [H100 pretraining](dev/h100-training/H100_RLM.md): run, monitor, and resume using the supplied files.
- [Spark RLM training](dev/spark-training/SPARK_RLM.md): training, evaluation, and measured results.
- [RLM design](dev/RLM-architecture/SPARK_RLM_PLAN.md) and [memory-boundary analysis](dev/RLM-architecture/SPARK_RLM_DOCUMENT_BOUNDARIES.md): architecture and supporting research.
- [Spark d24](dev/spark-training/SPARK_D24.md) and [small Spark experiments](dev/spark-training/SPARK.md): base-model workflows.
- [Project storage](dev/PROJECT_STORAGE.md): artifact locations, dataset sharing, and path overrides.
- [README](README.md): broader project background and the original training pipelines.

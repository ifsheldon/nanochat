# Development documentation

Training guides, implementation plans, and experiment records are grouped by hardware; recurrent latent memory model design and architectural analysis are kept separately.
Run commands from the repository root unless a guide says otherwise.

## Spark training

- [Small experiments and run status](spark-training/SPARK.md)
- [D24 training and evaluation](spark-training/SPARK_D24.md)
- [RLM training, evaluation, and measured results](spark-training/SPARK_RLM.md)
- [RLM implementation plan and training history](spark-training/SPARK_RLM_PLAN.md)

## H100 training

- [Pretraining on 1, 2, or 4 H100 GPUs with the supplied files](h100-training/H100_RLM.md)
- [Server-agent handoff: GPU allocation and checks before pretraining](h100-training/H100_SERVER_HANDOFF.md)
- [H100 GPU selection implementation plan](h100-training/H100_MULTI_GPU_PLAN.md)

## RLM architecture

- [Model design, data contract, and architectural experiments](RLM-architecture/RLM_DESIGN.md)
- [Document-length and memory-boundary analysis](RLM-architecture/SPARK_RLM_DOCUMENT_BOUNDARIES.md)
- [RLM proposal and source repository](https://github.com/ifsheldon/recurrent-latent-memory)
- [Interactive memory-cycle illustration](https://recurrent-latent-memory.reify.ing)

## Shared references

- [RLM training settings: episodes, token budgets, CLI options, and runtime environment](RLM_TRAINING_SETTINGS.md)
- [RLM model profiles: required explicit selection, fields, and checkpoint copies](../model_profiles/README.md)
- [JSONC model profile implementation plan](MODEL_PROFILES_PLAN.md)
- [Plot NanoChat and RLM training curves](../.agents/skills/plot-nanochat-training/SKILL.md), a project-local skill with optional original NanoChat comparisons.
- [Project storage](PROJECT_STORAGE.md)
- [Development log](LOG.md)
- [Leaderboard](LEADERBOARD.md)

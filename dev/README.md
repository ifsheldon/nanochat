# Development documentation

Training guides and experiment records are grouped by hardware; recurrent latent memory design and analysis are kept separately.
Run commands from the repository root unless a guide says otherwise.

## Spark training

- [Small experiments and run status](spark-training/SPARK.md)
- [D24 training and evaluation](spark-training/SPARK_D24.md)
- [RLM training, evaluation, and measured results](spark-training/SPARK_RLM.md)

## RLM architecture

- [Design and implementation plan](RLM-architecture/SPARK_RLM_PLAN.md)
- [Document-length and memory-boundary analysis](RLM-architecture/SPARK_RLM_DOCUMENT_BOUNDARIES.md)

## Shared references

- [Plot NanoChat and RLM training curves](../.agents/skills/plot-nanochat-training/SKILL.md), a project-local skill with optional original NanoChat comparisons.
- [Project storage](PROJECT_STORAGE.md)
- [Development log](LOG.md)
- [Leaderboard](LEADERBOARD.md)

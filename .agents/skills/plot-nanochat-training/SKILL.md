---
name: plot-nanochat-training
description: Use when plotting NanoChat or recurrent NanoChat (RLM) pretraining loss, per-cycle loss, or time per optimizer update, including resumed runs with or without original NanoChat baseline assets.
---

# Plot NanoChat training

Generate standalone Matplotlib PNG and SVG figures from recorded training metrics, with a CSV and provenance file.
Plotting is CPU-only and does not require loading model weights or modifying the training environment.

## Choose the assumption

| Available assets | Mode and interpretation |
| --- | --- |
| **1. Pretrained original NanoChat and its logs are available** | Compare recorded original and RLM curves using `--original-log`; the checkpoint is useful context for initialization but is not loaded. |
| **2. Neither the original model nor its logs are available** | Plot RLM-only progress by omitting `--original-log`; report that no original baseline comparison is available. |

Logs alone suffice for comparison if the original checkpoint is absent.
A checkpoint alone cannot reconstruct historical curves; use RLM-only mode if the original logs are missing.
These assumptions concern plotting assets, not whether RLM was initialized from scratch or from pretrained embeddings; establish initialization from run metadata.

## Find the inputs

Use the run identified in the conversation or repository documentation.
The helper reads `RUN/attempts/*/updates.jsonl` in attempt order and the corresponding `invocation.json` training budget when present.
Original logs must use the `scripts/base_train.py` text format with `step`, `loss`, and `dt` fields.
If several unrelated runs are plausible, clarify which run to plot.
If no RLM metrics exist, explain what is missing; a plotting request does not authorize training or downloading a baseline.

## Run the helper

Resolve `SKILL_DIR` to this skill's installed directory, and use an output directory that does not yet exist.
Use the user's timezone when known.

```bash
uv run --no-project --script "$SKILL_DIR/scripts/plot_training.py" \
  --run-dir /absolute/path/to/rlm-run \
  --original-log /absolute/path/to/01-scripts.base_train.log \
  --output-dir /absolute/path/to/new-plots \
  --timezone Asia/Singapore
```

For assumption 2, omit the entire `--original-log` argument.
Use `--through-update N` for a reproducible RLM historical cutoff; the original log remains available in full.
`--recent-updates N` sets the zoom window.
The isolated `uv` script environment supplies NumPy and Matplotlib without syncing the training project's dependencies.

| Output | Contents |
| --- | --- |
| `loss_curves.png` / `.svg` | Full loss history, shared-range comparison or recent RLM loss, all recurrent cycles, and recent cycle detail. |
| `cycles_1_2.png` / `.svg` | Cycle 1 and 2 over the full and recent RLM history, with original NanoChat overlaid where observations overlap; cycle 1 alone for a one-cycle run. |
| `time_per_update.png` / `.svg` | Recorded seconds per update and trailing 31-update median, with full and shared or recent views. |
| `metrics.csv`, `provenance.json` | Raw and plotted values, input hashes, snapshot time, update ranges, resume boundaries, observed embedding-phase changes, smoothing settings, and comparison caveats. |

## Preserve metric meaning

- Original `step` is zero-based; plot `step + 1` against one-based RLM `completed_updates`.
- Original `loss` is already a bias-corrected last-microbatch EMA with beta 0.9; plot it unchanged.
- Smooth RLM `last_batch_loss`, update-mean `loss`, and each `cycle_losses` series once with the same EMA; cycle means include all accumulation batches, so label them separately from last-batch loss.
- Derive cycle count, update ranges, and embedding-phase markers from observations; never assume four cycles, a fixed budget, or permanent freezing.
- The shared range is the intersection of available update ranges; retain longer tails in full-history panels.
- Continue RLM smoothing across contiguous resume attempts, mark boundaries, and omit downtime from the update axis.
  A partial trailing JSONL record is excluded and counted; conflicting or gapped histories require identifying the authoritative history before plotting, not silently deduplicating or filling values.
- These are observed-run comparisons: objectives, initialization, context layout, hardware, and timer scope can differ.
  Read [metric-notes.md](references/metric-notes.md) when explaining those differences or estimating completion time.

## Deliver

Inspect the generated images for clipping and readable legends.
Embed the relevant PNGs and link the output directory or CSV using absolute paths.
State the mode, snapshot cutoff, and the main comparison limitation.
Keep training loss distinct from held-out validation or generalization claims.

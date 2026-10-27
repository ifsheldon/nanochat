# RLM model profiles

RLM training, capacity probes, attention probes, and the H100 launcher require `--model-profile PATH` on every invocation, including resume.
There is no default or automatic profile selection; [`rlm-d24.jsonc`](rlm-d24.jsonc) is a bundled reference example.
Profiles support `//` and `/* ... */` comments and trailing commas through the `json5` parser.

Each profile requires exactly these seven fields:

| Field | Meaning | Reference d24 value |
| --- | --- | ---: |
| `n_layer` | Shared Transformer layers | 24 |
| `n_embd` | Model width | 1536 |
| `n_head` | Query attention heads | 12 |
| `n_kv_head` | Key/value heads; fewer than query heads enables grouped-query attention | 12 |
| `text_length` | Text tokens per recurrent cycle | 1023 |
| `memory_tokens` | Incoming memory slots and outgoing summary slots | 512 |
| `eom_tokens` | Extra learned EOM input before the text; must be the integer `1` | 1 |

All fields must be positive integers; booleans and fractional values are rejected.
The width must be at least 24 and divisible by `n_head`, the derived head dimension must be even, and `n_kv_head` must divide `n_head` without exceeding it.
Missing, unknown, or duplicate fields are rejected before model allocation.

Head dimension is `n_embd / n_head`, and maximum sequence capacity is `text_length + eom_tokens + 2 * memory_tokens`.
The first cycle has no incoming memory; training's final cycle omits the unused outgoing summary.
Vocabulary size comes from the tokenizer, and full causal attention remains fixed.
Cycles, global and device batch sizes, learning rates, schedules, initialization, embedding freezing, GPU count, precision, activation checkpointing, compilation, and output paths remain training or execution settings outside the profile.

Pass the reference profile to the trainer or H100 launcher, or copy it and edit its fields to define another model:

```bash
uv run --no-sync python -m scripts.rlm_train \
    --model-profile model_profiles/rlm-d24.jsonc \
    --mode probe --device-batch-size 16

bash runs/runh100_rlm.sh \
    --num-gpus 2 --model-profile model_profiles/rlm-d24.jsonc \
    --assets assets/h100-seed --run-name pretrain-2gpu
```

Prepare the tokenizer and dataset as described in the [Spark guide](../dev/spark-training/SPARK_RLM.md) or use the [H100 launcher](../dev/h100-training/H100_RLM.md), which prepares them automatically.
The direct trainer's `--total-batch-size` defaults to 1,047,552 real-text targets per optimizer update; choose another divisible budget for a custom architecture.
The H100 launcher keeps its existing four-cycle training recipe and fixed 1,047,552-text-target batch; accumulation is calculated from the selected profile's `text_length` and GPU count.
The global text batch must be divisible by `device_batch_size * cycles * text_length * num_gpus`.
Changing the model width requires a matching initialization embedding or fresh initialization through the direct trainer; the supplied H100 embedding has width 1536.

The separate RLM `--depth`, `--width`, `--head-dim`, `--text-length`, and `--memory-tokens` options have been removed.
The base GPT trainer's CLI is unchanged.

## Checkpoints and resume

The trainer reads the selected file once before constructing the model.
Before its first checkpoint save, the coordinator copies that original UTF-8 document to `OUTPUT/checkpoints/model-profile.jsonc`, preserving comments and line endings.
Later saves reuse the copy; an existing copy with the same fields is left unchanged, including its comments.
Probe modes do not save checkpoints or create this checkpoint copy.

Resume requires `--model-profile PATH` too.
For the direct trainer, pass `--model-profile OUTPUT/checkpoints/model-profile.jsonc` to select the saved document explicitly; the original source file can then be edited or removed without changing the resumed architecture.
You may instead pass a source profile with the same seven field values as the saved profile; its comments may differ, and the saved document is retained unchanged.
If no checkpoint profile is present, pass a profile matching the checkpoint's architecture; it is copied when the next checkpoint is saved.
Repeat the desired training and checkpoint options when invoking the direct trainer; the H100 launcher restores its saved training recipe.

The H100 launcher also saves `OUTPUT/model-profile.jsonc` before its first probe and passes that same file to capacity probing, attention benchmarking, and fresh training.
Its trainer then saves `OUTPUT/train/checkpoints/model-profile.jsonc`; pass that path explicitly with `--model-profile` when resuming through the H100 launcher, or pass a matching source profile.

# H100 server validation handoff

For the agent on the H100 server: validate the distributed RLM workflow before starting a long pretraining run.
Local verification covered CPU/Gloo workers and one GB10 GPU; multiple H100s, NCCL training, and H100 throughput remain unverified.
Read the [H100 workflow](H100_RLM.md) for setup and resource requirements and the [implementation record](H100_MULTI_GPU_PLAN.md) for existing coverage.
Use the [training settings reference](../RLM_TRAINING_SETTINGS.md) for episode, token-budget, accumulation, and CLI definitions.

## 1. Ask the human before using GPUs

Ask the human server owner explicitly:

> How many H100 GPUs may I use for the multi-GPU tests, 2 or 4, and which device indices or UUIDs may I use?

Wait for the answer before launching GPU tests.
Set `CUDA_VISIBLE_DEVICES` to the approved devices for every launch; use only subsets of that allocation for smaller-count tests.
Do not infer permission from the number of visible GPUs.
If only one GPU is approved, run single-GPU checks and report multi-GPU validation as outstanding.
If two are approved, leave four-GPU behavior explicitly untested.

## 2. Prepare and check the server

- Record the checkout commit, approved devices, `nvidia-smi topo -m`, driver, PyTorch/CUDA versions, free host RAM, and free disk space.
- Install the locked environment with `uv sync --locked --python 3.10.21 --extra gpu --group dev`; use the existing launcher helpers to check H100 eligibility, prepare and verify the seed/tokenizer and dataset, and apply memory limits.
- Use new validation run directories and preserve all logs and failed attempts; keep production outputs separate.
- Run the existing focused regressions before spending time on full-size GPU runs:

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m pytest -q \
    tests/test_model_profile.py tests/test_rlm_profile_training.py \
    tests/test_h100_rlm.py tests/test_optim_checkpoint.py \
    tests/test_optim_distributed_failures.py tests/test_rlm_distributed.py \
    tests/test_rlm_distributed_checkpoint.py tests/test_rlm_distributed_training.py \
    tests/test_rlm_train_distributed_cli.py tests/test_rlm_milestone.py
```

These distributed tests use CPU/Gloo; the following steps must exercise real CUDA/NCCL on the approved H100s.

## 3. Check communication and a small model first

Run a bounded CUDA/NCCL all-reduce and the runtime's Gloo control/checkpoint-group communication with one process per approved GPU.
Confirm distinct device assignment, correct reductions, and clean exit of every worker.
Then compare a tiny deterministic RLM optimizer update on one GPU and the approved multi-GPU count using identical global examples and initial weights.
Check losses, gradients, and model/Adam/Muon updates within appropriate numerical tolerances; adapt the existing small-model regression fixtures to CUDA rather than treating CPU results as hardware evidence.

## 4. Probe the real recipe and save checkpoints

**Do not use `runs/runh100_rlm.sh` as a bounded test: it starts full pretraining after its probes and has no stop-limit passthrough.**
Use the preparation, `choose_recipe`, `choose_attention`, `training_arguments`, and `training_command` helpers in [`scripts/h100_rlm.py`](../../scripts/h100_rlm.py), then invoke the trainer directly for bounded training.
Keep the helper-generated environment, including tokenizer/data paths, GPU mask, dtype, attention backend, and memory reserves.
Select a [JSONC model profile](../../model_profiles/README.md), preserve its original document in the scratch run, and pass the same `model_profile` path to `choose_recipe`, `choose_attention`, and `training_arguments`.
Every trainer or launcher invocation requires `--model-profile PATH`, including resume; no profile is selected automatically.
Use `model_profiles/rlm-d24.jsonc` for the full-size checks and a separate small profile for tiny-model tests; the former architecture CLI flags have been removed.
Pass the same explicit `total_batch_token_size` to `choose_recipe`, `choose_attention`, and `training_arguments`; the generated direct-trainer commands must include `--total-batch-token-size` in every mode, including resume.
For the reference 1023-text/512-memory profile, use 1,047,552 text targets.

- Run capacity and attention probes at each count intended for full-size validation, including the smaller counts needed for resume tests.
  Confirm the reported worker count, finite updates, local staging buffers and the coordinator's complete optimizer snapshot, memory headroom, and collective warmup completion.
  Preserve any FlashAttention failure; SDPA is acceptable when the documented fallback succeeds.
- Run at least three compiled d24 updates with asynchronous checkpoints by adding `--stop-after-step 3` to the trainer's training arguments.
  Keep `--num-iterations 5574`, the selected recipe, and the selected global text-target budget unchanged.
  Verify the update-2 numbered checkpoint, a complete rolling checkpoint at the final completed step, drained writers, and successful exit of all ranks.
  Verify that `train/checkpoints/model-profile.jsonc` preserves the selected JSONC document, including comments.

Capacity and attention modes do not accept early stopping, resume, or checkpoint-save options; their update limits are internal.
They require embedding freeze `none` or `permanent`, not `scheduled`.

## 5. Verify resume and the freeze boundary

- Test `1 → 2 → 1`, or `1 → 2 → 4 → 1` when four GPUs are approved, using a common recipe that fits every tested count.
  Resume the same scratch training directory with `--resume-latest`, omit fresh embedding initialization, and advance `--stop-after-step` beyond the saved step for each bounded attempt.
  Verify loaded model weights, complete Adam/Muon state, completed step, and per-worker loader/RNG metadata before another healthy update.
  Remove the original scratch profile source and pass `--model-profile OUTPUT/train/checkpoints/model-profile.jsonc` explicitly on resume to exercise the saved checkpoint copy, where `OUTPUT` is the scratch run root.
- Keep the global batch fixed and confirm accumulation changes with worker count: batch 32 uses `8 / 4 / 2` microbatches for `1 / 2 / 4` GPUs; batch 16 uses `16 / 8 / 4`.
  Changed-count data continuation is approximate at row-group granularity, so production losses need not match bit-for-bit after resume.
- Confirm schema-4 `selected-recipe.json` and the launch invocation record contain the selected token budget.
  H100 launcher resume must restore that budget when omitted, accept the same explicit value, and reject a different value; direct-trainer resume still requires `--total-batch-token-size` explicitly.
- Cover both frozen and trainable input-embedding optimizer state.
  For scheduled freezing, use a separate short-horizon scratch fixture with `--num-iterations 12`: stop at 9, resume through the mandatory step-10 `pretrain-90pct.pt`, verify unfreezing and Adam-state creation at update 11, then resume that state at another approved count for update 12.
  Keep this altered schedule separate from the 5574-update production recipe; if permanent freezing is selected, verify that the input table stays fixed.

## 6. Report results before pretraining

Report the commit, approved and tested GPU counts, device mask, selected batch/backend/freeze policy, commands, result/checkpoint paths, update timings, memory peaks, and any failures or untested cases.
A stalled rank, non-finite state, incomplete checkpoint, incorrect resumed optimizer state, or exhausted memory headroom blocks pretraining until resolved.
Ask the human to confirm the production GPU count and authorize the long run after reviewing the results, then use a fresh run name and the documented launcher command with explicit `--num-gpus`, `--model-profile`, and `--total-batch-token-size`.

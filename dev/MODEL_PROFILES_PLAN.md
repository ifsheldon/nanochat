# JSONC model profiles implementation plan

> Implementation uses focused tasks, regression tests, and independent review.

**Goal:** Configure RLM architecture through a required, explicitly selected seven-field JSONC profile and preserve the original profile with checkpoints.

**Architecture:** A typed profile supplies the existing RLM model configuration; training and execution settings remain CLI options.
A profile document retains its original text for a single checkpoint-directory copy, and resume requires an explicit profile path like every other invocation.

**Tech stack:** Python, the `json5` parser for commented JSON, PyTorch, uv, and pytest.

**Spec:** The user approved `n_layer`, `n_embd`, `n_head`, `n_kv_head`, `text_length`, and `memory_tokens`, plus an explicit extra-token field fixed to one, and requested copying the JSONC profile when saving checkpoints if it has not already been copied.
The profile argument must be required, with no default for fresh runs or resume.

## Global constraints

- Name the extra-token field `eom_tokens`; require an integer exactly equal to `1`.
- Require the seven architecture fields and derive head dimension and sequence capacity; obtain vocabulary size from the tokenizer.
- Keep cycles, global batch, optimizer/schedule, GPU allocation, precision, compilation, activation checkpointing, initialization, and paths outside the profile.
- Bundle `model_profiles/rlm-d24.jsonc` as a reference example; replace the RLM trainer's separate architecture flags with required `--model-profile PATH` on the trainer and H100 launcher, including resume.
- Save the original UTF-8 profile text as `checkpoints/model-profile.jsonc` before the first checkpoint publication, from the coordinator only; preserve an existing equivalent copy.
- Require callers to select the checkpoint copy or a matching source profile explicitly on resume; the selected profile must agree with an existing saved profile so the saved document describes the checkpoint.
- Keep one-, two-, and four-GPU execution and optimizer portability intact; calculate accumulation from the actual profile's text length.
- No new branches, worktrees, commits, production training, or unrelated architecture options.

## Interfaces

- `nanochat.model_profile.CHECKPOINT_PROFILE_NAME` is `model-profile.jsonc`; the module defines no default profile path.
- `RLMModelProfile` is a frozen dataclass with the seven integer fields, `from_jsonc(text)`, a derived `sequence_len`, and `model_config(vocab_size=..., activation_checkpointing=...)` returning `RLMConfig`.
- `ModelProfileDocument.read(path: Path)` returns the parsed `profile` and original `text`; `copy_once(destination: Path)` atomically publishes that original text without replacing an existing equivalent profile.
- Imports of the profile module remain usable before installing Torch or parser dependencies; parsing/model construction can import their dependencies lazily.

## Task 1: Profile document and reference example

Files: `nanochat/model_profile.py`, `model_profiles/rlm-d24.jsonc`, `pyproject.toml`, `uv.lock`, and `tests/test_model_profile.py`.

- [x] Write failing tests for commented profiles, derived model dimensions, configurable KV heads, malformed schema, invalid EOM count, and exact-text copy/reuse.
- [x] Add the parser dependency without replacing the active Torch environment; implement typed parsing and atomic document copying.
- [x] Verify invalid profiles fail clearly, original comments survive, and an existing different profile is never silently replaced.

## Task 2: Trainer and checkpoint integration

Files: `scripts/rlm_train.py`, `tests/test_rlm_train_cli.py`, `tests/test_rlm_train_distributed_cli.py`, and focused profile-training coverage if needed.

- [x] Write failing trainer tests for custom model dimensions/GQA and profile persistence through checkpoint/resume.
- [x] Replace architecture flags with the required selected document, retaining all existing training options and resolved model metadata.
- [x] Copy the selected document before checkpoint publication, only on the coordinator, and reuse the saved document when explicitly selected on resume.
- [x] Update existing small-model fixtures to supply profiles and verify single-worker plus distributed resume with the explicitly selected checkpoint copy after removing the original profile file.

## Task 3: H100 launcher

Files: `scripts/h100_rlm.py`, `tests/test_h100_rlm.py`, and the shell wrapper only if needed.

- [x] Test profile propagation through capacity, attention, training, and resume commands, including profile-dependent accumulation.
- [x] Require `--model-profile`, preserve one selected profile across fresh-launch stages, and require explicit selection of the saved checkpoint profile or a matching source on resume.
- [x] Retain the three batch/freeze recipes, global token budget, GPU selection, and automatic attention policy.
- [x] Verify shell help and argument validation still work before the Python environment is installed.

## Task 4: Documentation and final verification

Files: profile usage documentation, `AGENTS.md`, `README.md`, documentation indexes, RLM training instructions, and the H100 server handoff.

- [x] Document all seven fields, checkpoint-copy behavior, profile selection, seed compatibility, and the replacement of architecture CLI flags.
- [x] Update runnable examples while preserving dated experiment evidence.
- [x] Independently review the combined change and run targeted tests, the CPU suite, feasible CUDA checks, lockfile, shell syntax, and documentation/whitespace checks.

## Review focus

Missing profile arguments and profile errors must precede model allocation; all launch stages must use the same selected architecture; a checkpoint's profile must remain available for explicit selection after the source file changes or disappears; coordinator-only copying must not leave a partial profile; custom text length must not produce misleading accumulation records.

## Initial feature verification completed October 27, 2026

These checks preceded the follow-up requirement to make the profile argument mandatory.

- Focused model-profile tests: 21 passed, covering commented parsing, invalid dimensions/EOM, a real GQA forward, exact-byte copy/reuse, and failed publication.
- Trainer/profile/distributed integration tests: 49 passed, including source-file deletion and 2 → 4 → 1 CPU-worker resume.
- H100 launcher tests: 127 passed, including custom text length/GQA, profile propagation through every stage, and resume using the checkpoint copy.
- Full CPU suite: `CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m pytest -q --tb=short` passed 720 tests with 22 skipped.
- CUDA optimizer/checkpoint suite: `uv run --no-sync python -m pytest tests/test_optim.py tests/test_rlm_checkpointing.py -q --tb=short` passed 27 tests on the local NVIDIA GB10.
- `uv lock --check`, shell syntax/help, dependency-free launcher argument parsing, local Markdown links, and tracked/new-file whitespace checks passed.
- Independent review found no actionable issues; actual multi-H100/NCCL execution remains covered by the server handoff rather than these local results.

## Required-argument follow-up verification

- Missing `--model-profile` now fails argument parsing for all trainer modes, both trainer resume options, and fresh/resumed H100 launches.
- Trainer, profile-training, distributed CLI, and storage tests: 67 passed; explicit checkpoint-copy selection preserves resume after the source profile is deleted.
- H100 launcher and model-profile tests: 150 passed; launcher helpers also require an explicit profile path.
- Independent review found no remaining implicit profile defaults or broken active callers.
- Shell help/syntax, dependency-free launcher argument parsing, 13 runnable documentation examples, 54 local documentation links, and whitespace checks passed.
- The full CPU and CUDA suites above were not repeated for this CLI-only correction.

## Explicit token-budget extension

The user selected `--total-batch-token-size` as the explicit input and requested its meaning in documentation, code docstrings, and argument help, together with a reference explaining all training settings and the global-batch equation.
This option counts real-text targets per optimizer update across all workers and accumulation steps; it excludes additional EOM predictions and memory/summary positions.
Accumulation remains derived, and no separate accumulation CLI option is added.

- [x] Require `--total-batch-token-size` on every direct RLM trainer invocation, remove the old RLM flag, and document the existing `TrainingConfig.total_batch_size` field and accounting methods.
- [x] Accept the budget explicitly for fresh H100 launches, pass it through all probes and training, record it in recipe/invocation metadata, and reuse it on resume across GPU counts.
- [x] Reject conflicting H100 resume overrides and invalid token-budget divisibility; retain the existing batch/freeze recipe selection and four-cycle setting.
- [x] Add a central RLM training-settings guide with definitions, the batch equation, a worked d24 example, all trainer/launcher options, and runtime settings; update related guides and indexes.
- [x] Verify the renamed CLI, profile-dependent token-budget propagation, changed-count resume, accounting, documentation links, and affected tests; obtain independent review.

The H100 helpers receive a required `total_batch_token_size` argument, and saved recipes record that field explicitly under a new schema version.
The direct trainer maps the explicit CLI name to the existing `TrainingConfig.total_batch_size` field; model configuration and training accounting retain their current meaning.
The base GPT and chat trainer CLIs remain unchanged, and the removed RLM option receives no compatibility alias.
This work stays in the current checkout without commits or production training.

### Token-budget verification

- Trainer CLI, distributed CLI, profile-training, storage, attention-probe, and capacity tests: 91 passed.
- H100 launcher tests: 124 passed, including custom-text-length profiles, budget propagation, and saved-budget resume across GPU counts.
- Independent review found no actionable issues and separately passed 30 focused H100 budget/accumulation tests from the affected suite.
- Verified coverage of all 30 trainer options and eight H100 options in the central guide, 14 runnable examples, 65 local documentation links, and the worked d24 profile calculation.
- Shell syntax/help, dependency-free launcher argument parsing, compilation, and whitespace checks passed.
- The full CPU and CUDA suites recorded above were not repeated for this CLI/configuration extension; no actual H100 training was launched.

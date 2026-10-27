# H100 GPU selection implementation plan

> Implementation uses focused tasks, meaningful regression tests, and independent review.

**Goal:** Let the H100 launcher train and resume on one host using an explicitly selected 1, 2, or 4 H100 GPUs, including changing the count on resume.

**Architecture:** Run one worker per selected GPU with the existing MuonAdamW distributed optimizer and rank-sharded data loader.
Checkpoints retain one model and a complete optimizer state, independent of worker count, so resume can partition moments for the requested count.
The existing atomic rolling, numbered, and milestone checkpoint publication remains the storage protocol.

**Tech stack:** Python, PyTorch 2.14.1, NCCL training collectives, Gloo CPU coordination and checkpoint collectives, uv, and pytest.

**Spec:** The user requests support for 2 and 4 H100s, requires `--num-gpus`, and explicitly requires resume between 1, 2, and 4 GPUs.

## Global constraints

- `--num-gpus {1,2,4}` is compulsory for both fresh runs and resume.
- Use the selected number of visible H100s on one host; respect `CUDA_VISIBLE_DEVICES` when selecting devices.
- Keep 1,047,552 real-text targets per optimizer update, 5574 updates, and the current model, freeze schedule, learning-rate schedule, checkpoint frequency, and attention-selection policy.
- Batch 32 uses accumulation 8, 4, or 2 for 1, 2, or 4 GPUs; batch 16 uses 16, 8, or 4.
- Every recurrent episode stays on one GPU, with full gradients through its four cycles.
- Use optimizer-managed gradient synchronization; do not add a DDP wrapper.
- Resume preserves complete optimizer state, supports a changed worker count, and retains approximate row-group data continuation.
- No new branches, worktrees, commits, pushes, or production training launches are authorized by this implementation request.
- Preserve the pending branch-name documentation edits and the paused Spark run.

## Decisions

Portable complete optimizer checkpoints avoid a worker-count restriction and keep the current checkpoint artifact contract.
Each worker stages only its optimizer shard; rank zero also stages the model.
A dedicated checkpoint process group gathers CPU tensor shards directly, without serializing tensor payloads through object collectives, while the checkpoint worker runs independently of training collectives.
Rank zero writes the complete snapshot with the existing checkpoint writer; all workers learn the publication result before they mutate saved state.
Checkpoint staging and gather allocations must be represented in capacity and attention probes.
CPU mapping on load avoids allocating a complete optimizer copy on every GPU before partitioning it.

Checkpoints record each saved worker's loader cursor and RNG state.
At unchanged worker count, restore each worker's own state.
At changed worker count, use a shared earliest saved row-group cursor and repartition using the loader's existing rank-aware resume logic; restore a deterministic saved RNG state for each new worker.
This may repeat or skip examples within a row group, as in the existing resume contract, and does not promise identical sample order across worker counts.

## Review focus

- A worker failing before an optimizer collective must stop all workers with the original failure category instead of leaving peers blocked.
- A checkpoint failure on any worker must preserve the previous complete checkpoint; milestone failures must stop before unfreezing.
- Optimizer partitioning must preserve lazy frozen state, Adam steps, Muon padding, and every rank's moments across 1/2/4 transitions.
- Attention warmup and backend selection must use collective decisions, and timing must reflect the slowest worker.
- Multiple visible GPUs, explicit GPU masks, required CLI arguments, and changed-count resume must select exactly the requested devices.

## Task 1: Portable optimizer state

Files: new `nanochat/optim_checkpoint.py` and `tests/test_optim_checkpoint.py`.

Interfaces: `OptimizerStateLayout.from_optimizer(optimizer)` captures parameter IDs, tensor shapes, and optimizer group kinds; `layout.gather(local_state, group)` returns a complete CPU optimizer state on rank zero; `layout.shard(full_state, rank, world_size)` returns the appropriate independent state for a target worker; `layout.allocate_full(local_state)` reserves the same root buffers used by gathering for probes.

- [x] Write failing tests for Adam moments, replicated small states, lazy frozen embeddings, Muon group padding, and 1/2/4 repartitioning.
- [x] Implement the layout and CPU tensor gather without DDP or optimizer algorithm changes.
- [x] Verify actual multiprocess Gloo gathers and round-trip repartitioning.
- [x] Review the implementation and test coverage.

## Task 2: Coordinated checkpoint snapshots

Files: new `nanochat/rlm_distributed_checkpoint.py`, mapped loading in `nanochat/rlm_latest.py`, and `tests/test_rlm_distributed_checkpoint.py`.

Interfaces: `DistributedCheckpointWriter(checkpoint_dir, *, mode, keep_last, optimizer, checkpoint_group, control_group, latest_step=None)` provides the existing writer's `submit`, `wait`, `status`, `close`, and `measurements` contract; `abort()` drains local work on a failed run; `LatestCheckpoint.load_cpu_mapped()` reads the complete snapshot without eagerly copying its tensor storage.

- [x] Write tests for complete-state publication, worker failure, asynchronous snapshot immutability, milestone durability, and mapped loading.
- [x] Stage local state, gather complete optimizer tensors on the dedicated checkpoint group, and delegate durable publication to the existing writer on rank zero.
- [x] Record per-worker loader/RNG state in metadata and propagate publication results to every worker.
- [x] Verify resume can reconstruct all optimizer states and retains existing checkpoint retention behavior.

## Task 3: Distributed RLM execution

Files: new `nanochat/rlm_distributed.py`; changes to `scripts/rlm_train.py`, `nanochat/rlm_training.py`, `nanochat/rlm_capacity.py`, `nanochat/rlm_attention_probe.py`, `nanochat/optim.py`, and `nanochat/training_runtime.py`; focused distributed integration tests.

Interfaces: a runtime context owns rank, world size, device, CPU control and checkpoint groups, collective error handling, coordinator-only artifact writes, and scalar/metric reductions.

- [x] Add failing tests for world-size-aware accumulation and equivalent global updates.
- [x] Initialize workers before shared directory/lock handling; let the coordinator own shared artifacts and give workers separate diagnostic files.
- [x] Coordinate forward/backward failures before optimizer collectives, aggregate global metrics, and synchronize probe warmup decisions.
- [x] Restore portable optimizer partitions and appropriate loader/RNG state for the requested count.
- [x] Exercise real 2- and 4-process CPU control flow and cross-count checkpoint resume with small models; retain single-worker tests.

## Task 4: H100 launcher and documentation

Files: `scripts/h100_rlm.py`, `runs/runh100_rlm.sh`, `tests/test_h100_rlm.py`, `dev/h100-training/H100_RLM.md`, `AGENTS.md`, `README.md`, and `dev/README.md`.

- [x] Test missing/invalid GPU counts, visibility masks, 2/4-worker commands, capacity fallback, and changed-count resume.
- [x] Require the GPU-count argument, validate/select the requested visible H100s, and launch probes and training through `torch.distributed.run` when needed.
- [x] Record actual GPU count and accumulation in the selected recipe; resume accepts a different explicit count without changing the training budget.
- [x] Update examples, checkpoint semantics, requirements, and the documentation index.

## Final verification

- [x] Run targeted tests after each task and independent review of the combined change.
- [x] Run the CPU suite, lockfile check, shell syntax checks, and whitespace checks.
- [x] Run feasible small GPU checks on the local GB10; explicitly report that 2/4-H100 throughput and NCCL execution require the H100 host.

## Review resolutions

Real worker failure tests exposed shared health checks that performed inline reductions inside otherwise local forward and preparation phases.
RLM now uses those checks locally and propagates errors through the surrounding CPU coordination phase; other training workflows retain their existing health-check behavior.
The optimizer separates local buffer preparation and computation from reduction and gather phases, allowing a worker allocation or compute error to reach every worker before the next communication phase.
Its update arithmetic remains unchanged; reductions and gathers remain asynchronous within their phases, while computation finishes before parameter gathering begins.
Normal probe and checkpoint-writer shutdown failures are coordinated before any worker publishes a successful result.
Failure cleanup drains admitted checkpoint work without starting a new rendezvous.

## Verification results

- Full CPU suite: `CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m pytest -q --tb=short` passed 656 tests with 22 skipped.
- CUDA-enabled optimizer and checkpoint checks: `uv run --no-sync python -m pytest tests/test_optim.py tests/test_rlm_checkpointing.py -q` passed 27 tests on NVIDIA GB10.
- A separate tiny BF16 CUDA RLM smoke completed two optimizer updates, asynchronous rolling and numbered publication, exact mapped-checkpoint tensor comparison, and pinned staging through the distributed checkpoint helper.
- Real CPU process tests cover 1→2→4→1 update equivalence, 2→4→1 trainer resume, two-worker capacity and attention probes, optimizer allocation and compute OOMs, forward OOMs, checkpoint failures, and shutdown failures.
- `uv lock --check`, `bash -n runs/runh100_rlm.sh`, and tracked plus new-file whitespace checks passed.
- The shared trainer test fixture resets the compiler cache between unrelated model-shape tests; the previously failing ordered reproduction plus CLI tests passed all 96 tests after this isolation fix.

The local host has one GB10 GPU.
Two/four-H100 NCCL execution, capacity, and throughput still require validation on the H100 host; the launcher runs its capacity and attention checks there before fresh training.

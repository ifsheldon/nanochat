# Recurrent latent memory on DGX Spark

This workflow implements the first version of the [RLM plan](../RLM-architecture/SPARK_RLM_PLAN.md).
It trains models with differentiable memory between bounded text chunks and provides separate evaluation and sampling entry points.
Initialization defaults to fresh weights; an optional verified transfer initializes only the main input token embedding from an existing checkpoint.
The same loader accepts a full checkpoint or a checkpoint containing only `transformer.wte.weight`; [H100 RLM handoff](../h100-training/H100_RLM.md) documents pretraining with the supplied asset bundle.
The existing d24 production artifacts remain under `~/.cache/nanochat/spark-d24`.
New runs use the repository's `training_run/<name>/`, selected with `NANOCHAT_RUN_NAME`; the default name is `pretrain-1`.
The tokenizer lives in `<run>/tokenizer`, training and probe outputs default to `<run>/train`, and profiles default to `<run>/profiles/<UTC timestamp>`.
`NANOCHAT_BASE_DIR` overrides the run root; `--output-dir` overrides only the particular command's output directory and does not relocate its tokenizer.
Shared pretraining, task, and evaluation datasets default to the repository's `datasets/`, independently overridden by `NANOCHAT_DATASETS_DIR`.
Set `NANOCHAT_TRAIN_SHARDS=170` for the Spark RLM recipe to select training shards 0–169 and validation shard 6542 even when the shared directory contains more shards.
The dated measurements and historical commands below retain their original paths; existing files are not moved or copied automatically.
To evaluate, profile, or explicitly resume an existing cached run with current code, set both `NANOCHAT_BASE_DIR="$HOME/.cache/nanochat/spark-d24"` and `NANOCHAT_DATASETS_DIR="$HOME/.cache/nanochat/spark-d24"`, retain `NANOCHAT_TRAIN_SHARDS=170`, and pass the run's original `--checkpoint-dir`, `--run-config`, or `--output-dir` as appropriate.
The old run root alone no longer selects its dataset, and changing the defaults does not modify an already running process.

## Model and data contract

The default model has 24 layers, width 1536, 12 heads, full causal SDPA, and BF16 compute with the existing mixed parameter precision.
Each episode starts with empty memory and contains two 1023-token text chunks.
Every cycle places one learned `<EOM>` input between incoming memory and text, including the first cycle with empty memory.
Each memory-writing cycle appends 512 copies of one trainable summary vector; their final normalized hidden states become the next cycle's 512-vector memory prefix.
Training omits the final cycle's summary suffix because no later cycle consumes it.
For two-cycle training, both cycles process 1536 positions: `[EOM][text][summary]` followed by `[memory][EOM][text]`.
Intermediate cycles in longer episodes retain both memory and summary blocks and process 2048 positions.
Memory remains attached to autograd throughout the episode and resets before the next episode.
No earlier text or per-layer KV cache crosses a cycle boundary.

The joint `<EOM>` plus text span receives one vocabulary projection and one cross-entropy operation per cycle.
`<EOM>` predicts the first input token of its chunk, including BOS in the first cycle; all text positions retain their ordinary shifted targets.
Thus a recurrent boundary token is supervised twice with different contexts: from the old chunk's final text position and from the new cycle's memory-only `<EOM>` readout.
`<EOM>` is an input embedding, never a prediction target, and does not change the tokenizer vocabulary.
The existing additional value embeddings apply only to text; memory, `<EOM>`, and summary positions receive ordinary projected attention values.
Input normalization, smear, and the initial residual operate across the combined sequence; memory is extracted after backout and final normalization.
RoPE positions restart at zero each cycle.
The summary and `<EOM>` parameters use AdamW groups with the embedding learning-rate policy.
By default all model parameters remain trainable, including the main input embedding, additional value embeddings, and untied LM head.
For a run initialized from a pretrained input table, `--input-embedding-freeze scheduled` holds only that table fixed until the final 10% of the full training budget.
Use `--input-embedding-freeze permanent` to keep the table frozen for the entire budget, or `none` to train it throughout.
The separate `<EOM>` and summary inputs, additional value embeddings, LM head, and Transformer stay trainable throughout; the tokenizer vocabulary is unchanged.
The other proposed boundary/type tokens and auxiliary classification loss remain deferred.

The existing BOS best-fit loader packs an entire episode before splitting its input and target tensors into chunks.
It inserts BOS per document, adds no EOS, and does not insert a new BOS at a recurrent boundary.
Packed episodes may contain multiple documents and cropped tails.
Metrics report how often a handoff continues a document and the fraction of later text preceding another BOS.
Loader buffers are reused only after the full episode backward completes.

At batch 16, two cycles give 32 text subsequences, 32,736 shifted text targets, and 32 additional `<EOM>` predictions per microbatch.
The default 1,047,552-text-target optimizer batch accumulates 32 episode microbatches, supervising 1,048,576 predictions and processing 1,572,864 internal positions.
Batch 32 uses 16 microbatches for the same update budget.
Internal positions do not count as additional training text or advance the learning-rate schedule.
Unroll lengths that do not divide the global text batch require an explicit batch-budget choice.

## Memory probe

Prepare the shared shards with `uv run --no-sync python -m nanochat.dataset -n 170` and install or train the tokenizer under the selected run's `tokenizer/` directory before starting a probe.
Pretrained input-table initialization requires the original matching tokenizer; do not retrain that tokenizer.
Use a fresh run name or explicit output directory for each new attempt; the trainer refuses to overwrite an existing training/probe directory.
Run from the repository root using the locked uv environment.

```bash
NANOCHAT_RUN_NAME=spark-rlm-probe-b16 NANOCHAT_TRAIN_SHARDS=170 \
uv run --no-sync python -m scripts.rlm_train \
    --mode probe \
    --device-batch-size 16
```

The probe completes two full optimizer updates without evaluation or checkpoint saving.
The separate `--mode capacity-probe` executes exactly one complete optimizer update with optimizer state allocated before its first forward and, on CUDA, reusable pinned checkpoint buffers already resident.
It accepts only explicit `none` or `permanent` input-embedding policies, discards its updated weights, and writes a structured result for the H100 launcher's ordered selection.
This measures capacity with the required persistent allocations; it does not measure warmed throughput or checkpoint disk-writing time.
It starts without activation checkpointing and compiles the cycle forward with separate first-cycle and recurrent shapes.
Optimizer kernels also compile, even with `--no-compile` for the model.
The first update includes compilation and lazy optimizer-state creation; the second includes resident optimizer state throughout forward/backward.

Defaults are an 80 decimal GB CUDA allocator ceiling, a 16 GB minimum system available memory, and expandable allocator segments.
CPU memory and external CUDA allocations add overhead on the Spark's shared memory system.
A background sampler records the minimum observed system availability every 50 ms, while synchronized CUDA counters record allocated and reserved peaks.
The trainer checks finite loss and gradients before applying an optimizer update and checks model/optimizer tensors afterward before publishing a checkpoint.
The workflow retains a 72 GB peak-reserved acceptance target, including measured checkpoint overhead; evaluation runs separately after loading a checkpoint.

If batch 16 fails, keep its output directory and run a separately named case with `--device-batch-size 8`, or explicitly enable `--activation-checkpointing`.
No automatic batch reduction conceals the requested configuration's result.
Reports include per-update time, text throughput, internal-position throughput, cycle losses, compiler counters, memory peaks, and document-continuation statistics.
RLM configuration, progress, and checkpoint events use Python's standard `logging` through NanoChat's existing console handler, which writes timestamps, logger names, and severity to stderr and flushes each message.
Normal events use `INFO`; recoverable checkpoint errors use `WARNING`.
After each completed update, an `INFO` message reports `Update <step>/<full budget>: <seconds> s`, rounded to two decimal places from the synchronized `duration_seconds`.
The full-precision numeric `duration_seconds` is written and flushed to the attempt's `updates.jsonl` after every completed update and retained in the result's per-update records, independently of console log filtering.
The JSONL file contains raw JSON without log prefixes; console JSON messages carry the handler's timestamp and severity prefix.
This includes data loading, forward/backward, optimizer work, health checks, and checkpoint waiting inside that update; current checkpoint submission/writing and final writer draining are measured separately.
Each update keeps `loss`, the mean over all supervised predictions in all accumulation batches, and adds `last_batch_loss`, the unscaled mean for the final accumulation batch across all cycles.
It also logs one scalar per cycle, `cycle_1_loss` through `cycle_C_loss`, averaged over every batch in that update, alongside the existing `cycle_losses` list.
These losses include `<EOM>` predictions and shifted text targets; equal-sized packed batches give each batch equal weight.
The additional loss fields total `1 + C` and appear in the structured console output, `updates.jsonl`, and the result's per-update records.
They do not estimate RLM FLOPs using the ordinary GPT token estimator.

## Operator profiling

Profile a recorded run's model and training configuration using real training data and a fresh model:

```bash
NANOCHAT_RUN_NAME=spark-rlm-probe-b16 NANOCHAT_TRAIN_SHARDS=170 \
uv run --no-sync python -m scripts.rlm_profile \
    --no-record-shapes
```

The default reference is `<run>/train/run.json`; use `--run-config` or `--output-dir` to select another recorded configuration or a fresh profile destination.
The profiler warms all configured cycle shapes, backward, and the optimizer before tracing one episode and one optimizer step with CPU and CUDA activity.
Its first warmup step allocates optimizer state; later warmups and the traced episode run with that state resident.
Each diagnostic step consumes one episode scaled by the production accumulation factor, so this is operator attribution rather than a training-throughput or learning measurement.
Production amortizes the optimizer and update health checks over all accumulation microbatches.
Use ordinary unprofiled multi-update probes for speed comparisons.

Artifacts include a Chrome trace, CPU/device operator tables, structured operator and device-event timings, per-phase scopes, memory measurements, and source/configuration fingerprints.
When the recorded configuration enables scheduled input-embedding freezing, profiling applies the same policy at its first two diagnostic optimizer steps and records `input_embedding_frozen`.
CUDA profiling fails explicitly if no CUDA device events were captured.
The default records input shapes; `--no-record-shapes` avoids the additional tensor-reference retention associated with shape recording.
Resource limits use the same environment variables and defaults as training.
The recorded file supplies the configuration; the profiler runs the current source and records its fingerprint, rather than loading checkpoint weights or replaying archived code.
Recorded configurations must use the current training schema; archived experiments retain their source snapshots for reproduction.

## Bounded training, checkpointing, and resume

Training requires an explicit optimizer-update budget.
For example, after choosing a feasible device batch:

```bash
NANOCHAT_RUN_NAME=spark-rlm-learning-20 NANOCHAT_TRAIN_SHARDS=170 \
uv run --no-sync python -m scripts.rlm_train \
    --mode train --num-iterations 20 --warmup-steps 2 \
    --device-batch-size 16 --activation-checkpointing --save-every 10
```

The short budget and warmup above are experiment choices, not the original production horizon.
The base learning-rate policy preserves batch scaling, model-width scaling, and the d12 reference for Muon weight decay.
Every run records its complete configuration and source fingerprint.

Checkpoints are saved only after a completed optimizer update, with model, summary, and `<EOM>` parameters, optimizer state, PyTorch RNG state, last-consumed loader progress, and separate real-text, prediction, and internal-position counters.
Training saves `checkpoints/latest.pt` after every completed update.
The default `--checkpoint-mode async` overlaps the save with the next update's forward and backward passes, then waits for the writer before the next optimizer step can change any saved tensors.
One writer handles at most one checkpoint request at a time, including its numbered save and retention.
On CUDA, the writer uses PyTorch's `DefaultStager` to copy into reusable pinned CPU buffers on a dedicated stream, then serializes the completed CPU snapshot.
Nonblocking copies move the long transfer wait out of the GIL-holding storage-copy binding; the staging stream's synchronization releases the GIL while the copy finishes.
The stager is initialized on the training thread, preserves shared storage and tensor layouts, and closes after the final save drains.
Metadata and optimizer parameter-group values are captured before the next loader, RNG, or schedule advance; tensor storage stays unchanged until the writer finishes.
The CPU staging buffers add approximately 9.97 GB of retained host memory for this model and optimizer, consuming the Spark's shared memory alongside CUDA allocations.
Latest and numbered writes reuse the same completed snapshot, so a numbered save does not repeat the GPU-to-CPU transfer.
Use `--checkpoint-mode sync` to finish checkpointing before starting the next update's computation.
This single file contains the full model, optimizer, and metadata together; the replacement is written to a temporary file, flushed and fsynced, atomically published, and followed by a directory fsync.
The bundle's `metadata.step` identifies its completed update; publication logs and the result's `checkpoint_status` also report the last confirmed durable update and its lag behind completed training.
While saves succeed, stopping during an update or its save loses at most one completed optimizer update, plus any unfinished computation.
Checkpoint write and retention errors log structured `WARNING` messages and training continues; subsequent updates retry saving, so repeated errors can leave `latest` multiple updates behind.
Staging failures also warn and retry after queued copies have drained; if CUDA itself cannot synchronize after a failed stage, training fails because parameter mutation would be unsafe.
The writer still finishes or fails before the next optimizer step, so warning-and-continue never permits a save to race with parameter mutation.
Normal completion and explicit update-boundary stops drain the final writer before reporting the saved step.
Owned staging files left by an interrupted rolling save are removed during the next save.

Training also saves numbered checkpoints at update 2, each `--save-every` interval, and the final or explicitly paused update.
The defaults are `--save-every 10 --keep-checkpoints 10`, retaining the ten newest numbered checkpoints in addition to the rolling latest checkpoint.
Numbered model, optimizer, and metadata files are staged and fsynced before publication; a durable record permits interrupted publication to finish safely.
Metadata is published after both tensor files.
The ordering is latest publication, numbered publication, then deletion of older numbered checkpoints; saving never deletes an older checkpoint to make space for a new one.
Retention validates every managed checkpoint set before deleting any older set and records removals in checkpoint measurements and structured logs.
A durable retirement record permits an interrupted deletion to finish safely on resume or a later successful numbered save; incomplete or invalid sets without a matching record block deletion and produce a warning.
There is no live episode memory or partial gradient accumulation at this boundary.
The default free-space reserve for checkpoint saving is 100 GB.

Runs using either pretrained input-table freeze policy also preserve `checkpoints/pretrain-90pct.pt` at `floor(9 * num_iterations / 10)` completed updates, provided that boundary is positive.
For 5574 updates, this is update 5016, before the scheduled policy enables gradients at update 5017; a permanent-freeze run retains the same progress milestone without unfreezing.
The writer durably copies the complete published rolling bundle, retaining model, optimizer, RNG, loader progress, and metadata while keeping `latest.pt` available.
This immutable file survives numbered-checkpoint retention and requires space for one additional full checkpoint.
Training waits for its durable publication before beginning the next forward; a failure stops training rather than proceeding without the requested milestone.
Resuming at that boundary repairs an interrupted copy from `latest.pt`; later resumes verify the retained milestone and its training identity.

To pause reproducibly at an update boundary, add `--stop-after-step 10` to the training command.
Stopping the service uses the most recent fully published rolling checkpoint; it does not invoke a signal-time save.
To resume, repeat the original model/training and checkpoint-policy options, keep the original `--num-iterations`, and add `--resume-latest` while removing the earlier stop flag and any initialization flags.
The explicit `--resume-from-step 10` option selects a numbered checkpoint and is mutually exclusive with `--resume-latest`.
Only the newest published update across rolling and numbered checkpoints can be resumed; choosing an older step is rejected to protect newer published history.
If a stop interrupted a numbered checkpoint save after the rolling checkpoint was published, latest resume repairs that exact numbered set from the rolling checkpoint and completes pending retention before training continues.
Resume requires identical model settings, training budget, source/dependency fingerprint, installed Python/PyTorch/CUDA/Triton versions, tokenizer bytes, ordered data paths/sizes/timestamps, device type, and compute precision.
An explicitly reviewed source or checkpoint-policy change requires a `--resume-transition` receipt binding the exact run directory, checkpoint metadata, old and new contracts, and source manifests.
Every other contract field must remain identical; a receipt cannot authorize a model, training, data, tokenizer, initialization, precision, or runtime change.
The old checkpoint and founding `run.json` remain unchanged; the new attempt and checkpoints record the actual source contract and transition provenance.
Subsequent resumes use the new checkpoint's matching contract without the transition flag, and further source changes invalidate the earlier receipt.
Data identity records file metadata rather than rehashing the entire corpus.
The best-fit loader retains approximate row-group resumption, so resumed training is not claimed to replay the exact uninterrupted token stream.

Each invocation writes an `attempts/` subdirectory containing the command, source snapshot including untracked modules, source/data inventories, and update metrics.
Completed measurements go to `result.json`; failures produce `failure.json` and, for failed updates, `failed-update.json` within the attempt, preserving prior attempt artifacts.
Training checkpoints are under `checkpoints/` within the training output directory, which defaults to `<run>/train`.
Checkpoint saving records writer elapsed time and separate durations for CPU staging, rolling saves, and numbered saves.
Update timing includes any `checkpoint_wait_seconds` spent at the optimizer barrier, and `execution_wall_seconds` covers all updates plus final checkpoint draining, excluding model loading and other setup.
The system-memory sampler spans training and checkpointing; async per-update CUDA peaks include overlapping checkpoint activity, and memory values attached to checkpoint results reflect the main thread's observation when it joins the writer.
Writer durations overlap update durations in async mode and must not be added to estimate total runtime.
Numbered saves add another write on their scheduled updates.

### Export a portable input embedding

The initialization loader accepts both full model checkpoints and compact checkpoints containing only `transformer.wte.weight`.
Export the original pretrained table and its matching tokenizer before transferring them to another machine:

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m scripts.rlm_export_embedding \
    --checkpoint-dir /path/to/base_checkpoints/spark-d24-bf16 \
    --step 5568 \
    --tokenizer-dir /path/to/tokenizer \
    --output-dir /path/to/embedding-seed
```

The output directory must be new and its parent must already exist.
The bundle contains `model_005568.pt`, `meta_005568.json`, `tokenizer.pkl`, `token_bytes.pt`, and `manifest.json`; the model file contains a cloned input table, and the manifest records payload checksums and original checkpoint provenance.
Copy the complete directory to the target machine and install its matching tokenizer assets in the selected run's tokenizer directory.
The existing `--init-embedding-checkpoint-dir`, `--init-embedding-step`, and `--init-embedding-tokenizer` options accept the compact checkpoint through the same loader as a full checkpoint.
Only the input table is copied into a fresh RLM; the remaining model weights and optimizer start fresh.
Resume uses the saved RLM checkpoint and omits initialization flags.

### New run with scheduled input-embedding freezing

Use `--input-embedding-freeze scheduled` together with the three `--init-embedding-*` options for a fresh run.
Only the pretrained `transformer.wte.weight` table is frozen; copying happens before the freeze is applied, and the new `<EOM>` and summary parameters retain their fresh, trainable initialization.
The table receives no gradient, weight decay, or Adam state updates while frozen.
Its optimizer group remains in place, and its Adam moments are created on the first trainable update, using the learning rate at that point in the original schedule.
The schedule freezes the first `floor(9 * num_iterations / 10)` completed updates and trains the table for the final `ceil(num_iterations / 10)` updates.
For a 5574-update run, updates 1–5016 are frozen and update 5017 is the first trainable update.
The initial configuration logs this boundary, and each update logs `input_embedding_frozen`.
The compiled forward/backward may compile again when the table becomes trainable.

The freeze policy is part of the saved training configuration.
The current CLI expresses it with `--input-embedding-freeze`; historical Spark invocations used the former scheduled-only `--freeze-input-embedding` flag and retain their original source snapshots.
Resume keeps `--input-embedding-freeze scheduled` and the original full `--num-iterations`, restores trained weights and optimizer state, and omits the three initialization options.
The completed checkpoint step determines the current phase; an early stop or a resumed invocation does not restart or shorten the freeze schedule.
Freezing a run without pretrained initialization provenance or changing its freeze policy on resume is rejected.

The fresh run was launched on October 10, 2026 at 00:57 SGT with committed training source `0bd758d2b56d1d8c9fe07bda80c36eb3be327d8f` and the following configuration:

```bash
NANOCHAT_BASE_DIR="$HOME/.cache/nanochat/spark-d24" \
NANOCHAT_DTYPE=bfloat16 NANOCHAT_ATTENTION_IMPL=sdpa OMP_NUM_THREADS=1 \
NANOCHAT_CUDA_MEMORY_LIMIT_GB=100 \
NANOCHAT_MIN_SYSTEM_AVAILABLE_GB=12 \
NANOCHAT_CHECKPOINT_RESERVE_GB=100 \
uv run --no-sync python -m scripts.rlm_train \
    --mode train --num-iterations 5574 --cycles 4 \
    --device-batch-size 32 --activation-checkpointing \
    --save-every 10 --keep-checkpoints 10 --checkpoint-mode async \
    --init-embedding-checkpoint-dir "$HOME/.cache/nanochat/spark-d24/base_checkpoints/spark-d24-bf16" \
    --init-embedding-step 5568 \
    --init-embedding-tokenizer "$HOME/.cache/nanochat/spark-d24/tokenizer/tokenizer.pkl" \
    --freeze-input-embedding \
    --output-dir "$HOME/.cache/nanochat/spark-rlm/pretrain-u4-b32-frozen-20261010"
```

The persistent user service is `nanochat-rlm-u4-b32-frozen-20261010.service`, using the locked `.venv/bin/python` and recording both output streams in `pretrain-u4-b32-frozen-20261010.log` beside the run directory.
The adjacent `pretrain-u4-b32-frozen-20261010-launch.json` records the exact command, environment, revision, resource checks, and the passing 406-test suite with 10 skipped tests.
Automatic restart is disabled; a failure or reboot requires an explicit resume from the latest complete checkpoint with the original budget and freeze policy.
The older October 8 run remains paused and retains its existing checkpoints.

```bash
systemctl --user status nanochat-rlm-u4-b32-frozen-20261010.service
tail -f "$HOME/.cache/nanochat/spark-rlm/pretrain-u4-b32-frozen-20261010.log"
```

Startup verification passed after two completed updates and publication of both rolling and numbered checkpoint 2.
The updates had losses 10.369809 and 10.255392 and took 293.66 and 196.11 seconds; the first included compilation, while the second compiled no additional graphs.
Both records contain `duration_seconds`, `last_batch_loss`, all four cycle means, `input_embedding_frozen: true`, and passing finite-state checks.
The second update peaked at 77.421 GB allocated and 78.803 GB reserved, with minimum system available memory of 23.139 GB under the configured 100 GB allocator cap and 12 GB system guard.
CPU mapping of the 9.766 GB numbered checkpoint verified that the input table's SHA-256 still matches the initialized pretrained table exactly and that it has no Adam state, while `<EOM>` and summary inputs each have Adam step 2.
The check reconstructed parameter identities with a meta-device model and did not allocate a GPU model or interrupt training.
The run's `startup-verification.json` records these checks, source provenance, and the service's active state at 01:06 SGT.

### Four-cycle pretraining with the step-5568 input embedding

The selected full run uses four cycles, batch 32, activation checkpointing, and terminal-summary omission.
Its 5574-update budget consumes 5,839,054,848 real text targets, with eight accumulation microbatches per update.
Only `transformer.wte.weight` is copied from the completed d24 checkpoint at step 5568.
The Transformer, additional value embeddings, LM head, shared summary input, and `<EOM>` input retain their normal fresh initialization, and every parameter remains trainable.
The new summary and `<EOM>` inputs are separate model parameters, so the existing tokenizer vocabulary is unchanged.

```bash
NANOCHAT_BASE_DIR="$HOME/.cache/nanochat/spark-d24" \
NANOCHAT_DTYPE=bfloat16 NANOCHAT_ATTENTION_IMPL=sdpa OMP_NUM_THREADS=1 \
NANOCHAT_CUDA_MEMORY_LIMIT_GB=100 \
NANOCHAT_MIN_SYSTEM_AVAILABLE_GB=12 \
NANOCHAT_CHECKPOINT_RESERVE_GB=100 \
uv run --no-sync python -m scripts.rlm_train \
    --mode train --num-iterations 5574 --cycles 4 \
    --device-batch-size 32 --activation-checkpointing \
    --save-every 10 --keep-checkpoints 10 --checkpoint-mode async \
    --init-embedding-checkpoint-dir "$HOME/.cache/nanochat/spark-d24/base_checkpoints/spark-d24-bf16" \
    --init-embedding-step 5568 \
    --init-embedding-tokenizer "$HOME/.cache/nanochat/spark-d24/tokenizer/tokenizer.pkl" \
    --output-dir "$HOME/.cache/nanochat/spark-rlm/pretrain-u4-b32-20261008"
```

The initializer requires identical source and target tokenizer bytes, vocabulary, embedding width, padded table shape, and dtype.
It loads the source table through a CPU mapping and records checkpoint, metadata, tokenizer, and embedding SHA-256 identities in the run contract.
`embedding-initialization.json` verifies the copied table's exact digest before optimizer construction.
Resume restores the RLM checkpoint's trained weights and saved initialization provenance without reopening or reapplying the original d24 checkpoint.
For this run, use `--save-every 10 --keep-checkpoints 10 --checkpoint-mode async` when resuming and omit all three `--init-embedding-*` arguments.
The 100 GB allocator cap and 12 GB system guard are explicit settings for this selected configuration; the general defaults remain unchanged.
Ten numbered checkpoints require approximately 100 GB, plus approximately 10 GB for `latest.pt` and another approximately 10 GB temporarily while replacing it or publishing a numbered checkpoint.
The earlier four-cycle probe projects about 12.6 days for training updates alone; checkpointing and other overhead add to this estimate.

The selected run was launched on October 8, 2026 at 14:54 UTC as the user service `nanochat-rlm-u4-b32-20261008.service`.
The service uses the repository's locked `.venv/bin/python`, records both output streams in the adjacent `.log` file, and continues after the interactive session ends.
User lingering is enabled; automatic restart is disabled, so a failure or reboot requires an explicit resume from the latest complete checkpoint.
The adjacent `pretrain-u4-b32-20261008-launch.json` records the exact command, environment, service name, and validation results.

```bash
systemctl --user status nanochat-rlm-u4-b32-20261008.service
tail -f "$HOME/.cache/nanochat/spark-rlm/pretrain-u4-b32-20261008.log"
```

Startup verification completed two updates with finite losses, gradients, model weights, and optimizer state.
Their losses were 10.369809 and 10.255342, with elapsed times of 208.17 and 195.70 seconds; update 2 compiled no additional graphs.
Peak CUDA allocated and reserved memory reached 77.824 GB and 79.558 GB, and the second update retained at least 33.33 GB of system available memory.
The complete step-2 checkpoint contains 9.968 GB of model, optimizer, and metadata files and passed run-contract and retention verification.
Saving took 39.22 seconds, did not increase CUDA reservation, and retained at least 32.80 GB of system available memory.
Training continued after this early save; `startup-verification.json` records the measurements and service status at verification time.
The user stopped this run on October 9, 2026 at 19:46 SGT after update 381; the process exited and released its GPU allocations.
The latest complete checkpoint at that stop was update 300, so resuming will repeat the 81 completed updates that were not checkpointed under the earlier periodic-only policy.
The subsequent rolling-checkpoint change bounds future loss while saves succeed; it cannot recover those already discarded weights, and warning-only save failures can increase the unsaved gap.
`pause.json` records the stopped state, verified checkpoint files, environment, and resume arguments preserving the 5574-update horizon and latest-ten numbered-checkpoint policy.
The first resume from the saved update 300 uses its reviewed source-transition receipt; after a new rolling checkpoint is published, use `--resume-latest` without that receipt.
The prepared receipt and source snapshot for the earlier pinned-staging implementation are under this run's `resume-preparation/`; `pause.json` preserves that source-bound receipt and first-resume command.
The subsequent scheduled-freezing and logging changes alter the source fingerprint, so resuming this older paused run with current sources requires a new explicit review and receipt.
Preparing this change did not restart training or alter checkpoint 300 and the founding run record.

The rolling save path was validated on October 9 using checkpoint 300 loaded on the GPU, without executing training updates.
The initial 9.968 GB bundle write took 37.41 seconds and atomic replacement took 39.72 seconds; a full GPU reload preserved the metadata and passed finite model and optimizer-state checks.
This isolated I/O validation reserved 9.985 GB of CUDA memory and retained at least 102.37 GB of system available memory; it does not measure the combined training-and-save peak.
Relative to the earlier 195.70-second training update, a roughly 40-second save implies about 20% additional update time, before the occasional numbered save.
Measurements are under `~/.cache/nanochat/spark-rlm/latest-io-validation-20261009T120257Z`; the temporary duplicate checkpoint was removed after validation.

### Three-update synchronous and asynchronous checkpoint comparison

On October 9, 2026, both modes trained the full four-cycle, batch-32 model for updates 301 through 303 from independent copies of checkpoint 300, using seed 42 and the same saved CPU/CUDA RNG state.
All 24 actual loader input and target batches had identical SHA-256 hashes across the cases.
Both used the original 5574-update schedule, BF16, activation checkpointing, a 100 GB CUDA allocator cap, a 12 GB system-memory guard, and the existing real training corpus.
Each case wrote `latest` three times and a numbered checkpoint at the final update, with the final writer fully drained before timing ended.

| Measurement | Synchronous | Asynchronous, dedicated CUDA stream |
| --- | ---: | ---: |
| Total for three updates and all checkpoint writes | 753.55 s | 714.71 s |
| Average per update including checkpointing | 251.18 s | 238.24 s |
| Update computation and optimizer-barrier durations | 208.96 / 196.17 / 194.32 s | 208.40 / 217.48 / 217.68 s |
| Latest writer durations | 38.53 / 36.64 / 39.38 s | 72.10 / 69.89 / 35.74 s |
| Final numbered writer duration | 39.53 s | 35.36 s |
| Maximum checkpoint wait inside an overlapped update | Not applicable | 0.00085 s |
| Peak CUDA reserved memory | 79.85 GB | 79.85 GB |
| Minimum system available memory from the invocation sampler | 31.40 GB | 32.26 GB |

Asynchronous checkpointing reduced measured elapsed time by 38.84 seconds, or 5.15%, with no checkpoint warnings.
Concurrent checkpoint writing increased the next update's computation time and the writer's elapsed time, but each overlapping write finished before the next optimizer step.
Writer durations overlap training and must not be summed with update durations.
These totals exclude model loading and final verification, include first-update compilation, and describe one three-update comparison rather than established long-run throughput.
The input-hashing phase includes waits for previously queued GPU work; its recorded duration is not an isolated measure of hashing overhead and is not subtracted from the totals.

An initial async trial using the shared default CUDA stream took 759.42 seconds, slightly longer than sync, because its copies queued behind training kernels and left 28.54 and 30.29 seconds of waiting at optimizer barriers.
The corrected case uses a private copy stream and reuses the unchanged sync control; each case records its own source fingerprint, and the only subsequent trainer-source change was the async stream selection.
The initial comparison is under `~/.cache/nanochat/spark-rlm/checkpoint-comparison-20261009T124713Z`; the corrected comparison, exact commands, input hashes, and state comparison are under `~/.cache/nanochat/spark-rlm/checkpoint-comparison-stream-20261009T131500Z`.

Final model and optimizer checkpoints matched their numbered counterparts exactly in both cases.
The corrected async checkpoint also matched every tensor and scalar in the live model and optimizer exactly after training; that verification ran after the measured interval.
CPU/CUDA RNG state, loader progress, configuration, and token counters matched across cases.
Cross-run model and optimizer values were not bitwise identical, and the largest loss difference was 0.00002503; GPU deterministic algorithms were not enabled, so these tests do not claim exact cross-run numerical reproducibility.
The production run remained paused throughout all nine independent benchmark updates, and its checkpoint 300 and founding `run.json` were preserved.
Its prepared resume command now selects asynchronous checkpointing, numbered saves every ten updates, and ten retained numbered checkpoints, using a new source-bound receipt recorded in `pause.json`.

Update 302 is the only warm, non-final update in this comparison.
Its reconstructed synchronous cycle took 232.81 seconds: 196.17 seconds of computation, 36.64 seconds saving, and 0.0017 seconds preparing the checkpoint state containers.
The asynchronous cycle took 217.48 seconds, including its 0.00075-second optimizer barrier and 0.0014-second container preparation.
That saves 15.32 seconds, or 6.58%, for this sample: computation slowed by 21.31 seconds while the previous checkpoint was being written, consuming part of the avoided save wait.
These are component totals rather than exact update-start timestamps; unaccounted loop overhead across the complete three-update runs was below 0.04 seconds in each mode.
One eligible update does not establish a steady-state average or its variability.

### Checkpoint overlap profiling

A bounded Nsight Systems profile on October 9, 2026 separated computation, checkpoint saving, and their overlap using the unchanged full-size checkpoint-300 model and optimizer.
Each computation condition ran the same three batch-32, four-cycle forward/backward episodes after warmup; no optimizer updates were applied.
The profile captured CUDA, NVTX, Python GIL, and operating-system runtime events, with additional wall and thread-CPU timing around storage copies, serialization, file writes, and `fsync`.

| Checkpoint component | Saving alone | Saving during computation |
| --- | ---: | ---: |
| GPU-to-CPU `storage.cpu()` calls | 29.770 s | 61.009 s |
| File writes | 3.247 s | 6.894 s |
| ZIP/checksum/bookkeeping, excluding file writes | 2.028 s | 3.647 s |
| File and directory `fsync` | 0.293 s | 0.519 s |
| Complete checkpoint | 35.600 s | 73.384 s |

Storage copying accounted for 84% of checkpoint time alone and 83% during overlap.
File-write time is excluded from the ZIP residual to avoid counting nested calls twice; that residual does not isolate CRC32 from other archive work.
The measured copy traffic was 9.96749 GB per checkpoint.

The identical three computation episodes took 71.884 seconds alone and 95.202 seconds during overlap, with equal losses and exactly 23,253 GPU kernels in each condition.
Nsight attributed 11.225 seconds of the additional 23.318 seconds to longer aggregate kernel execution and 12.093 seconds to additional time without a training kernel executing.
Copy operations occupied 12.104 seconds without concurrent training kernels; time with neither kernels nor copies executing remained below 0.05 seconds in each condition.
The same 12.104 seconds overlapped the checkpoint writer holding Python's GIL while the main computation thread or autograd thread waited for it.
The writer held the GIL for 61.500 seconds during overlap, including 60.876 seconds inside `storage.cpu()`; the installed PyTorch storage-copy binding retains the GIL during its blocking copy.
Only 0.00457 seconds of the time without training kernels coincided with the harness's `cudaDeviceSynchronize` calls, so those episode boundaries do not account for the observed launch gaps.
Blocking storage copies therefore inhibit training submission, alongside the separate increase in GPU kernel duration.
This shows interference during concurrent saving rather than a delay at the optimizer barrier or a long disk-flush wait.
It does not establish saturated DRAM bandwidth because hardware bandwidth counters were not collected.
The third overlapped episode returned to 23.956 seconds, close to its 23.977-second control, as checkpoint activity ended.
This profile identified the blocking storage-copy path as the next optimization target, leading to the subsequent pinned-staging implementation described above.
Its measurements describe the earlier direct GPU serialization path, before pinned staging was introduced.

This attribution probe cleared gradients and synchronized the device after each episode, unlike a full eight-episode accumulated update, and tracing adds overhead.
Use the complete non-final update measurement above for throughput, rather than extrapolating these shorter phases.
The probe reserved at most 75.223 GB of CUDA memory and retained at least 37.379 GB of system available memory.
The production run remained paused, and trainer sources and the prepared resume receipt were unchanged.
The harness, raw Nsight report, SQLite export, invocation, timings, and derived analyses are under `~/.cache/nanochat/spark-rlm/checkpoint-profile-20261009T134116Z`.

### Pinned CPU staging comparison

The subsequent CUDA checkpoint writer uses PyTorch's `DefaultStager` to copy the model and optimizer into reusable pinned CPU buffers before serializing them.
The private staging stream queues nonblocking storage copies and releases the GIL while waiting for completion; rolling and numbered saves reuse the same completed CPU snapshot.
On October 9, 2026, this implementation repeated updates 301 through 303 with the same checkpoint, seed, configuration, and all 24 input/target hashes as both earlier cases.
The table reuses the preceding synchronous and asynchronous controls; these are separate invocations with recorded source fingerprints.

| Measurement | Synchronous control | Previous async writer | Async with pinned staging |
| --- | ---: | ---: | ---: |
| Total for three updates and all checkpoint writes | 753.55 s | 714.71 s | 610.68 s |
| Average per update including checkpointing | 251.18 s | 238.24 s | 203.56 s |
| Warm non-final update 302, reconstructed cycle | 232.81 s | 217.48 s | 198.69 s |
| CPU staging durations | Included in saving | Included in saving | 3.49 / 0.18 / 0.18 s |
| Latest serialization and publication durations | 38.53 / 36.64 / 39.38 s | 72.10 / 69.89 / 35.74 s | 6.32 / 6.20 / 3.54 s |
| Final numbered serialization and publication duration | 39.53 s | 35.36 s | 3.68 s |
| Peak CUDA reserved memory | 79.85 GB | 79.85 GB | 79.85 GB |
| Minimum system available memory | 31.40 GB | 32.26 GB | 24.60 GB |

Pinned staging reduced total measured time by 14.56% relative to the previous async writer and 18.96% relative to the synchronous control.
The warm non-final update improved by 8.64% relative to the previous async writer; it includes the optimizer barrier and checkpoint container preparation, with both writers finishing before the next optimizer step.
The first staging operation allocates the cached buffers; subsequent operations reuse them, adding approximately 9.97 GB of retained host storage for this model and optimizer.
The invocation remained above the 12 GB system-memory guard and produced no checkpoint warnings.
Writer work overlaps computation, and the pinned writer's serialization durations exclude its separately reported CPU staging, so these columns are not additive update costs.
As in the preceding comparison, totals include first-update compilation and final writer draining but exclude model loading and verification; one warm non-final sample does not establish long-run throughput.

The rolling and numbered checkpoints matched exactly, and every saved model/optimizer tensor and scalar matched the final live training state exactly.
RNG state, loader progress, configuration, and token counters matched the synchronous control, while cross-run model and optimizer values were not bitwise identical with GPU deterministic algorithms disabled.
The maximum loss difference from that control was 0.00000644.
The comparison, input hashes, exact-state validation, source fingerprints, and commands are under `~/.cache/nanochat/spark-rlm/checkpoint-comparison-pinned-20261009T141227Z`.
The production run remained paused at checkpoint 300 throughout all twelve independent comparison updates; its source-bound resume receipt and `pause.json` were refreshed for the validated pinned-staging implementation without altering the founding run record or checkpoint.

A separate minimal reproducer isolates the blocking storage-copy GIL behavior using a one-byte copy after deliberately queued GPU work, without changing PyTorch's CPU thread count.
Ten fresh-process runs, including five with reversed case order, consistently recorded one heartbeat during blocking `Storage.copy_` versus 18–19 during blocking `Tensor.copy_`, with both calls taking approximately 20 ms.
The storage heartbeat can occur at the return boundary; a separate validation checked both copied bytes outside the timed interval.
This isolates interpreter blocking during a GPU wait rather than measuring copy bandwidth.
The compact script, captured output, additional cases, and an unpublished issue draft are under `~/.cache/nanochat/spark-rlm/storage-copy-gil-issue-20261009`.

The same directory preserves the original 77-line application comparison under `training-benchmark-source/repro_training_checkpoint.py`, with real CUDA forward/backward, AdamW state, and background `torch.save` to a temporary file.
It uses four 4096-wide linear layers with GELU, FP32 parameters, BF16 computation, and 16 gradient-accumulation passes over one fixed batch; the resulting checkpoint contains approximately 805.31 MB of actual model and optimizer state.
The writer uses a separate stream while the next backward pass runs, then the training thread waits for saving before applying the next optimizer update, matching the production snapshot boundary.
It contains no artificial GPU delay, heartbeat thread, or unrelated checkpoint payload.
Three fresh processes per mode ran in rotating order; the following medians use five warm updates per process, excluding each invocation's first save.

| Application example | Forward/backward interval | Save barrier before optimizer | Complete update and save |
| --- | ---: | ---: | ---: |
| No checkpoint | 0.1689 s | 0.0000 s | 0.1778 s |
| Background `torch.save` of CUDA state | 0.7404 s | 1.6120 s | 2.3735 s |
| Background pinned staging, then `torch.save` | 0.2087 s | 0.2352 s | 0.4525 s |

The forward/backward interval starts before writer submission and includes gradient clearing and synchronization of the training stream only, so the explicit save barrier remains separately measured.
Separate reload checks verified all 16 saved tensors and all optimizer metadata exactly for both saving modes, before the next optimizer mutation; those checks are excluded from the benchmark.
This example demonstrates interference in a real training loop, while the one-byte reproducer and earlier Nsight trace identify the GIL mechanism.
Pinned staging also changes allocation and transfer behavior, so its full speedup cannot be attributed solely to the GIL.
The example spends more time checkpointing than computing, uses buffered file writes without `fsync`, and does not predict production throughput; temporary checkpoints are removed on exit.
The exact script fingerprint, raw output, medians, and correctness results are in `results-training-runs.json`, `results-training-summary.json`, and `results-training-validation.json` in the containing artifact directory.

The current `repro_training_checkpoint.py` is a smaller 45-line demonstration with one default 4096-wide linear layer and AdamW, comparing training alone with background saving.
It removes the seed, command-line modes, pinned-staging implementation, mixed precision, fused optimizer selection, and secondary timing metrics; one initial training update still creates the optimizer state, and the dedicated copy stream and pre-optimizer writer barrier preserve the original concurrency contract.
Five fresh processes each ran two alternating pairs without fixing random values; the second-pair medians were 0.066 seconds without saving and 0.558 seconds during saving.
Separate reload checks verified all eight saved tensors and all optimizer metadata for both checkpoints in the validation run.
The new measurements and source fingerprint are in `results-training-minimal.json`, with representative output in `results-training-minimal.txt`; the older three-mode benchmark above remains separate historical evidence.

## Evaluation and generation

Use the dedicated RLM entry points so checkpoint loading preserves the recurrent contract.
Both verify the tokenizer fingerprint and model vocabulary before loading the exact saved parameters.

```bash
NANOCHAT_RUN_NAME=spark-rlm-learning-20 NANOCHAT_TRAIN_SHARDS=170 \
uv run --no-sync python -m scripts.rlm_eval \
    --step 20 --cycles 2 --batch-size 2 --batches 8 \
    --sampler contiguous_documents \
    --output training_run/spark-rlm-learning-20/evaluation.json

NANOCHAT_RUN_NAME=spark-rlm-learning-20 NANOCHAT_TRAIN_SHARDS=170 \
uv run --no-sync python -m scripts.rlm_sample \
    --step 20 --prompt "The meaning of life is" --max-tokens 128
```

Both commands default to `<run>/train/checkpoints`; `--checkpoint-dir` preserves access to an explicitly selected checkpoint directory.
Evaluation compares identical held-out rows using intact, zeroed, and shuffled incoming memory and reports shifted text loss separately by cycle, excluding the additional `<EOM>` predictions from these text-only metrics.
The default `--sampler bos_bestfit` uses the existing packed loader.
Use `--sampler contiguous_documents` for one uninterrupted prefix per sufficiently long validation document, with BOS only at the beginning and one extra token for shifted targets.
This sampler visits validation documents once, skips short documents, drops any final partial batch, and fails if it cannot supply the requested number of full batches.
It also reports paired per-episode loss differences and standard errors, treating episodes as independent observations; overall loss weights text targets, while these paired statistics weight episodes equally.
The resource report measures checkpoint loading and evaluation separately, including CUDA peaks and minimum system available memory across both phases.
The optional output file must be new and its parent directory must already exist.
Each intervention changes the current cycle's incoming memory; the intact trajectory continues to later cycles.
Cycle 1 has no memory intervention and cannot establish memory use.
Natural-language quality claims still require matched ordinary-context training baselines.

Generation retains only incoming memory and the current text chunk and recomputes that bounded context for each token.
At a boundary it replaces the completed chunk with its summary, then predicts the new chunk's first token from `[memory][<EOM>]`.
This implementation has no persistent KV cache, so rollover cannot retain hidden previous-text access or stale smear state.
It favors an explicit generation contract; decoding performance has not been optimized.
For Python callers, `model.eval().generate(prompt_ids, max_tokens=128)` uses this same recurrent sampler and resets memory for each new call.
RLM configuration rejects invalid dimensions, incompatible head counts, and insufficient cycle capacity before allocating the model.
Python callers can request `forward_cycle(..., write_memory=False)` when no outgoing memory is needed; it omits the summary suffix and returns `memory=None`.
The native episode forward and trainer use this option for the final cycle, preserving the supervised logits, loss, and cross-cycle gradient path.
The default remains `write_memory=True` for callers performing a memory handoff.
An empty initial readout without incoming memory requires that default; an empty readout with incoming memory supports either setting.

## Historical controlled recall result

A small controlled task tested whether a later query could recover a payload appearing only in the earlier chunk, using a two-layer model with a separate AdamW setup and independent training and held-out generators.
The run used 300 updates, batch 64, 16-token chunks, four memory slots, and 512 held-out episodes.
On October 8, 2026, the original layout without `<EOM>` reached 100% intact accuracy, 12.89% with zeroed memory, and 13.28% with shuffled memory; chance was 12.5%.
The intact accuracy's Wilson 95% interval was 99.26% to 100%.
This demonstrated learnable information transfer on the controlled task, not natural-language quality or long-horizon retention.
The standalone synthetic diagnostic was removed during cleanup; the model's causal and cross-cycle gradient regression tests remain.
Historical results are retained in `~/.cache/nanochat/spark-rlm/recall-20261008/metrics.json`.

## EOM speed comparison on October 8, 2026

A fresh batch-32 control and a separate `<EOM>` run each completed three optimizer updates with two cycles, 512 memory slots, BF16, full SDPA, and activation checkpointing.
These baseline runs retained the unused final summary suffix and used 16 accumulation microbatches, 1,048,576 supervised predictions, and 1,835,008 internal positions per update.
The comparison excludes the first update and averages updates 2 and 3; neither steady update compiled additional graphs.

| Layout | Real text targets/update | Steady update times | Mean update | Real text tokens/s | Peak CUDA reserved |
| --- | ---: | --- | ---: | ---: | ---: |
| Original, 1024 text/cycle | 1,048,576 | 195.68 s, 194.53 s | 195.11 s | 5,374 | 67.64 GB |
| `<EOM>` + 1023 text/cycle | 1,047,552 | 194.96 s, 194.06 s | 194.51 s | 5,386 | 68.02 GB |
| `<EOM>`, unused terminal summary omitted | 1,047,552 | 165.90 s, 166.08 s | 165.99 s | 6,311 | 55.90 GB |

Adding `<EOM>` while retaining the full summary layout changed measured update time by -0.30% and real-text throughput by +0.21%, which does not establish a speed difference with only two steady samples per case.
Peak allocated memory was 67.4162 GB for the original layout and 67.4164 GB with `<EOM>`.
The `<EOM>` run's minimum system available memory was 41.18 GB; all losses, gradients, model parameters, and optimizer state remained finite.
Both runs used the approved 100 GB allocator limit and 12 GB system guard, while training defaults remain 80 GB and 16 GB.
The first updates took 197.79 seconds for the cached original graph and 275.59 seconds for the freshly compiled `<EOM>` graph; these are not steady speed measurements.

The `<EOM>` run processed 0.098% fewer real text tokens because each cycle reserves one of the 1024 readout positions for `<EOM>`.
The unchanged base learning-rate recipe scales affected learning rates and initial Muon decay by `sqrt(1023/1024)`, approximately 0.049% below the original run.
The shorter packed rows also change the selected text stream, and the joint training loss now includes `<EOM>` predictions, so these probes are a resource comparison rather than matched-data quality evidence.
Artifacts, complete launch source snapshots, and explicit resource settings are under `~/.cache/nanochat/spark-rlm/old-b32-control-20261008-1249` and `~/.cache/nanochat/spark-rlm/eom-b32-baseline-20261008-1258`.

### Profile and terminal-summary optimization

The complete-suffix `<EOM>` profile traced one warmed batch-32 episode and one optimizer step with optimizer state resident and no new compiled graphs.
Its 5,995 actual CUDA kernels occupied 13.176 seconds of a 13.220-second instrumented interval, all on one stream.
Nested GPU annotation ranges and CPU operator attribution are excluded from these totals to avoid double counting.

| Kernel family | GPU time | Share of kernel time |
| --- | ---: | ---: |
| Matrix products | 7.413 s | 56.26% |
| Fused Triton operations | 4.017 s | 30.49% |
| FlashAttention kernels | 1.271 s | 9.65% |
| Other elementwise operations and reductions | 0.398 s | 3.02% |
| Explicit copy kernels | 0.076 s | 0.58% |

These are mutually exclusive classifications by generated kernel name; fused kernels can combine normalization, activation, copy, reduction, or optimizer work.
Cycle forwards cost 1.255 and 1.679 seconds; backward including activation recomputation cost 9.065 seconds.
The optimizer cost 0.835 seconds, gradient checks 0.096 seconds, and model/optimizer-state checks 0.246 seconds, each incurred once per production update rather than once per accumulation microbatch.
The clone/slice kernel identified as a readout-copy candidate cost only 1.747 ms over two calls; this does not support explicit `.contiguous()` calls as a material speed optimization.
The finite checks remain enabled.

The measured matrix-work cost motivates omitting the terminal 512 summary positions, whose outputs have no consumer and cannot affect earlier predictions under causal attention.
This reduces internal positions by 14.29% while preserving the text/EOM loss, first-cycle summary, and differentiable memory handoff.
CPU tests verify joint logits, losses, and every parameter gradient with active contextual weights in FP32/BF16, with and without activation checkpointing.
An additional compiled D24/B32/BF16 check used the same real-data batch and weights, with nonzero attention/MLP output projections and smear to exercise contextual gradients.
Its joint losses matched exactly, all 177 parameter gradients were finite, maximum absolute gradient difference was `1.90735e-6`, and aggregate gradient relative L2 difference was 0.0663%.
Its declared 2% per-tensor relative-L2 criterion failed for the tiny `smear_lambda` scalar alone: `9.09965e-7` versus `8.82960e-7`, an absolute difference of `2.70046e-8` and relative difference of 2.97%.
That failure remains recorded; the other 176 parameter tensors and the incoming-memory gradient passed their bounds, and the summary/EOM gradients remained nonzero.
Changed BF16 reduction shapes are a plausible explanation, while the FP32 equivalence tests and causal suffix structure support the semantic change; the GPU check does not establish bitwise gradient equivalence.
The complete numerical evidence and independent assessment are in `~/.cache/nanochat/spark-rlm/eom-elision-parity-20261008`.
The profile and corrected operator attribution are in `~/.cache/nanochat/spark-rlm/profile-eom-b32-baseline-20261008`, including `profile-analysis.json`.

The optimized batch-32 run then completed three full optimizer updates under the same model, training, tokenizer, data, precision, and runtime contracts as the complete-suffix `<EOM>` run; only the source fingerprint changed.
Updates 2 and 3 took 165.90 and 166.08 seconds, averaging 165.99 seconds or 6,311 real text tokens/second, without compiling new graphs.
Compared with the complete-suffix `<EOM>` baseline, update time fell 14.66%, throughput rose 17.18%, and peak CUDA reservation fell by 12.12 GB to 55.90 GB.
Peak allocated memory was 55.86 GB and minimum system available memory was 57.66 GB.
The first update took 173.87 seconds including cached warmup and optimizer-state creation.
All three updates had finite loss, gradients, model parameters, and optimizer state; joint losses matched the complete-suffix run exactly in updates 1 and 2 and differed by `1.90735e-6` in update 3.
This remains a short training-only speed probe with two steady samples; checkpoint saving and evaluation overhead were not measured for this configuration.
The run, full source snapshot, explicit resource settings, and three-way `comparison.json` are under `~/.cache/nanochat/spark-rlm/eom-b32-elided-20261008-1324`.

### Historical frozen main input embedding probe

The optional freezing policy and its CLI flag were removed after this experiment found no measurable speed benefit.
The recorded results below describe that archived experiment; the October 10 pretraining run above uses the later pretrained-input policy that freezes the table on a schedule.

A subsequent two-update probe froze only `transformer.wte.weight` and kept the optimized `<EOM>` layout, batch 32, two cycles, activation checkpointing, seed, data, tokenizer, runtime, and resource limits above.
The comparison uses update 2 from each run, after compilation and with optimizer state resident.
Both updates process 1,047,552 real text targets and 1,048,576 supervised predictions through 16 accumulation microbatches.

| Metric | Trainable input embedding | Frozen input embedding | Change |
| --- | ---: | ---: | ---: |
| Update 2 time | 165.897 s | 165.941 s | +0.027% |
| Real text tokens/s | 6,314.5 | 6,312.8 | -0.027% |
| Peak CUDA allocated | 55.8631 GB | 55.5611 GB | -0.3020 GB, -0.541% |
| Peak CUDA reserved | 55.8996 GB | 55.6479 GB | -0.2517 GB, -0.450% |
| Allocated after update | 10.0409 GB | 9.8396 GB | -0.2013 GB |

The peak allocated saving is exactly 301,989,888 bytes, or 288 MiB: one BF16 gradient tensor and two BF16 AdamW moments for the 50,331,648-parameter input table.
The 100.66 MB of input weights remain resident, and the 12 additional value-embedding tables remain trainable.
This probe found no measurable speedup; its single steady update also differs by only -0.028% from the earlier run's 165.988-second mean across updates 2 and 3.
Freezing the input table saves little memory because it accounts for only 3.64% of parameters and does not remove the Transformer activations needed by other trainable parameters and recurrent memory.

Both frozen updates passed the finite loss, gradient, model, and optimizer-state checks, and update 2 compiled no new graphs.
The input table's SHA-256 digest remained unchanged, with no gradient, optimizer ownership, or optimizer state.
The first update took 216.78 seconds including compilation; its loss matched the trainable reference exactly at 10.397381.
The second update's loss was 10.348749 versus 10.346804 with trainable input embeddings; two fresh-initialization updates do not establish the effect on model quality.
The frozen run has a two-update horizon versus three in the existing reference: both use the same warmup rates, but the Muon decay scalar applied at the end of update 2 is 0.5 versus 0.75 of initial decay.
This difference cannot affect that update's preceding forward/backward and leaves optimizer operations and shapes unchanged; it is retained in the comparison metadata.
These are training-only measurements without checkpoint saving or evaluation overhead.
The run, launch source snapshot, unchanged-weight proof, and `comparison.json` are under `~/.cache/nanochat/spark-rlm/eom-b32-frozen-input-20261008-1345`.

### Four-cycle batch-32 feasibility and pretraining time

The optimized four-cycle configuration completed two full optimizer updates under a 100 GB CUDA allocator cap and 12 GB system-availability guard.
It retained `<EOM>`, 1023 text tokens per cycle, 512 memory slots, BF16, full SDPA, activation checkpointing, and trainable embeddings.
Only the terminal cycle omits its unused summary suffix, giving cycle lengths of 1536, 2048, 2048, and 1536 positions.

```bash
NANOCHAT_BASE_DIR="$HOME/.cache/nanochat/spark-d24" \
NANOCHAT_CUDA_MEMORY_LIMIT_GB=100 \
NANOCHAT_MIN_SYSTEM_AVAILABLE_GB=12 \
uv run --no-sync python -m scripts.rlm_train \
    --mode probe --num-iterations 2 --cycles 4 \
    --device-batch-size 32 --activation-checkpointing \
    --output-dir "$HOME/.cache/nanochat/spark-rlm/probe-u4-b32"
```

Four cycles use eight accumulation microbatches, while two cycles use sixteen, preserving 1,047,552 real text targets and 1,048,576 supervised predictions per optimizer update.
Internal positions increase from 1,572,864 to 1,835,008 per update, or 16.67%; all intermediate memory handoffs retain their gradient paths.
The table uses the four-cycle second update and the earlier optimized two-cycle mean across updates 2 and 3, excluding compilation.

| Metric | Two cycles, batch 32 | Four cycles, batch 32 |
| --- | ---: | ---: |
| Steady update time | 165.988 s | 195.507 s |
| Real text tokens/s | 6,311 | 5,358 |
| Peak CUDA allocated | 55.863 GB | 77.824 GB |
| Peak CUDA reserved | 55.900 GB | 79.558 GB |
| Projected training-update time for 5,574 updates | 257.0 hours / 10.71 days | 302.7 hours / 12.61 days |

The four-cycle probe fits the requested 100 GB training cap, with 20.44 GB of reserved-memory headroom and a minimum measured system availability of 32.02 GB.
CUDA allocator figures exclude CPU and external CUDA allocations; the system-availability guard remains separate.
It exceeds the unchanged 72 GB acceptance target for the earlier full workflow, and checkpoint/evaluation overhead has not been measured for this four-cycle configuration.
Both updates passed all finite loss, gradient, model, and optimizer-state checks, and the second update compiled no additional graphs: the count remained 14.
The first update took 328.13 seconds including compilation and peaked at 73.70 GB reserved before optimizer state was resident throughout a full update.
Measured steady update time increased by 17.78% over two cycles, adding approximately 1.90 days at the same total text budget.

The existing [d24 plan](SPARK_D24.md#configuration) budgets 5,838,471,168 real text tokens, originally 5,568 updates of 1,048,576 tokens.
With `<EOM>`, preserving that text budget requires `ceil(5,838,471,168 / 1,047,552) = 5574` full updates, consuming 5,839,054,848 real text tokens after rounding.
The projections above multiply that update count by measured steady update time; startup, compilation, checkpoint saving, evaluation, and interruptions add overhead.
Keeping exactly 5,568 updates instead would consume 0.098% less real text and shorten the two-cycle estimate by only 16.6 minutes.
These estimates use one steady four-cycle sample and two steady two-cycle samples, and do not establish a model-quality benefit from longer recurrence.
The model, tokenizer, data inventory, precision, attention implementation, and runtime contracts match; longer packed episodes change the selected training stream.
The two-update probe horizon also changes the Muon decay scalar at the end of update 2 relative to the three-update reference, without changing the preceding forward/backward or optimizer operation shapes.
Artifacts, source snapshots, explicit resource settings, and the full comparison and projection are under `~/.cache/nanochat/spark-rlm/eom-u4-b32-20261008-1359`.

## Original-layout D24 memory measurements on October 8, 2026

These historical cases preceded `<EOM>` and used two recurrent cycles, 1024 text tokens per cycle, and 512 memory/summary positions.
The batch-16 and batch-32 probes used a global batch of 1,048,576 text targets; batch 48 used 1,081,344 targets so each update contained 11 complete microbatches.

| Device batch | Activation checkpointing | Outcome | Peak CUDA reserved | Update time |
| --- | --- | --- | ---: | ---: |
| 16 episodes | Off | Out of memory during the first unroll; no optimizer update completed | 79.91 GB at the 80 GB limit | No completed update |
| 16 episodes | On | Two complete updates, finite losses and gradients | 41.75 GB | 314.02 s first, 207.07 s second |
| 32 episodes | On | Two complete updates, finite losses, gradients, and optimizer state | 67.94 GB | 306.88 s first, 196.29 s second |
| 48 episodes | On | With a 16 GB system guard: first update completed; second stopped at the guard | 94.01 GB under the approved 100 GB limit | 325.67 s first; second incomplete |
| 48 episodes | On | With an approved 12 GB system guard and cached compilation: two complete updates, all finite | 94.01 GB under the approved 100 GB limit | 206.42 s first, 202.24 s second |

The checkpointed batch-16 second update processed 5,064 text tokens/second and retained at least 66.69 GB of system available memory.
Its peak allocated memory was 41.55 GB.
The first cycle and recurrent cycle each compiled once; the second optimizer update added no compiled graphs.
Activation checkpointing retains full gradients across both cycles and recomputes intermediate block activations during backward.
These measurements establish batch-16 training feasibility; the probe excluded checkpoint saving and evaluation.
The failed and successful batch-16 artifacts are respectively `~/.cache/nanochat/spark-rlm/probe-b16-20261008` and `~/.cache/nanochat/spark-rlm/probe-b16-checkpointed-20261008`.

The batch-32 probe retained the same global text batch, model, optimizer settings, 40-update warmup, and data/tokenizer fingerprints, reducing accumulation from 32 to 16 episode microbatches.
It passed under the existing 80 GB allocator ceiling, so the authorized 100 GB retry was unnecessary.
The second update reached 5,342 text tokens/second, with 67.42 GB peak allocated memory and 40.87 GB minimum system available memory; no new graphs compiled in that update.
Compared with batch 16's second update, measured update time fell 5.2% and throughput rose 5.5%, at a cost of 26.19 GB additional peak reserved memory.
These are individual steady-update measurements rather than repeated benchmark averages.
The current trainer also checks finite model and optimizer state after each update, a check added after the original batch-16 probe without changing its training computation.
Batch 32 stays below the 72 GB training-reserve target, but this probe excludes checkpoint saving and evaluation.
Its artifacts are in `~/.cache/nanochat/spark-rlm/probe-b32-checkpointed-20261008-1135`.

The batch-48 probe used the explicitly approved 100 GB CUDA ceiling while retaining the 16 GB minimum system available memory.
Its global text batch was 3.125% larger because 1,048,576 targets does not divide evenly by 48 two-cycle episodes.
Configured base learning rates were unchanged; the existing batch-scaling policy raised affected effective learning rates and initial Muon decay by approximately 1.55%.
The first update completed with finite loss, gradients, and optimizer state, taking 325.67 seconds including compilation and peaking at 88.66 GB reserved; minimum system available memory was 16.13 GB.
During the second update, peak allocated and reserved CUDA memory reached 93.29 GB and 94.01 GB, while minimum system available memory fell to 15.85 GB.
The system-memory guard then stopped the run before its second optimizer step.
That failed update ran for 37.27 seconds, which is only partial work and must not be reported as an update time or extrapolated into a measured speedup.
No additional graphs compiled during the failed second update.
The failed batch-48 artifacts are in `~/.cache/nanochat/spark-rlm/probe-b48-checkpointed-100gb-20261008-1150`.

At the user's request, a separate batch-48 retry lowered the system-memory guard to 12 GB while retaining the 100 GB CUDA ceiling and all model/training settings.
The retry completed both updates with finite loss, gradients, model parameters, and optimizer state.
With compilation cached, its first update took 206.42 seconds; the second took 202.24 seconds and processed 5,347 text tokens/second, without compiling additional graphs.
Peak allocated and reserved CUDA memory were 93.29 GB and 94.01 GB, and minimum system available memory was 18.24 GB.
The observed minimum stayed above even the earlier 16 GB threshold, so the successful retry must not be attributed solely to lowering the guard; cached compilation and different transient memory use also distinguish the attempts.
Accounting for its 3.125% larger text batch, the second update corresponds to 196.11 seconds per 1,048,576 text tokens, compared with batch 32's 196.29 seconds.
The throughput difference is only 0.09% in these single-update measurements, while peak reserved memory increases by 26.07 GB.
Batch 32 therefore remains the preferred measured configuration for throughput per memory footprint; batch 48 passes this training-only retry but exceeds the unchanged 72 GB full-workflow reserve target.
The successful retry is in `~/.cache/nanochat/spark-rlm/probe-b48-checkpointed-100gb-guard12-20261008-1159`, including `probe-settings.json` recording the explicit resource overrides.
The trainer's default system-memory guard remains 16 GB.

A real-data CPU integration run also completed training to update 2, checkpoint publication, resume to update 4, memory-ablation evaluation, and sampling.
Its small-model artifacts are in `~/.cache/nanochat/spark-rlm/cpu-resume-20261008`.

## Original-layout six-update d24 learning pilot

The bounded run used fresh weights, batch 16, activation checkpointing, a two-update warmup, and 6,291,456 supervised text tokens over six optimizer updates.
Training took 1245.61 seconds, with loss falling from 10.3974 to 7.0645 nats per text token.
The peak CUDA reservation remained 41.75 GB, and minimum observed system available memory was 67.92 GB.
Final checkpoint saving took another 37.11 seconds without increasing the CUDA reservation.
Separate checkpoint loading and held-out evaluation at batch 4 peaked at 11.99 GB reserved, leaving the measured workflow below the 72 GB target.

On 128 held-out packed episodes, second-cycle loss was 6.951757 with intact memory, 6.963053 with zeroed memory, and 6.951917 with shuffled memory.
The paired zero-minus-intact difference was 0.011296 ± 0.001021 standard error; shuffled-minus-intact was 0.000160 ± 0.000056.
On 128 uninterrupted validation-document prefixes, second-cycle loss was 7.032266 intact, 7.044650 zeroed, and 7.032714 shuffled.
The corresponding paired differences were 0.012384 ± 0.001117 and 0.000448 ± 0.000083 standard error.
Each evaluation scored 262,144 text targets across both cycles and took approximately 43.15 seconds, excluding checkpoint loading.
These are small early-run effects; the shuffled control is especially relevant because zeroing memory also changes its distribution.
The summary parameter's AdamW first moment had nonzero norm, 0.000929, confirming that this d24 run updated the memory writer.
On the first four packed validation episodes, first-cycle memory slots had mean off-diagonal cosine similarity 0.99878 and mean entropy effective rank 2.57, defined from normalized singular values of the uncentered slot matrix.
The slots remain highly similar after this short run; these diagnostics do not establish useful retention or justify the 512-slot budget.
A CUDA generation check consumed a 1020-token prompt and generated eight tokens across the 1024-token boundary.

Artifacts are in `~/.cache/nanochat/spark-rlm/learning-6-20261008`, including `result.json`, `evaluation.json`, `evaluation-contiguous.json`, `summary-gradient-check.json`, `slot-diagnostics.json`, and `generation-check.json`.
The experiment's step-6 model and optimizer files were removed during the October 9 housekeeping described below.
The invocation's source snapshot is authoritative: later validation, generation, and evaluation-data changes intentionally make the current source fingerprint incompatible with training resume of this pilot.
The pilot also predates the learned `<EOM>` parameter; the current implementation intentionally has no old-layout compatibility path.
Matched ordinary 2048-token and independent 1024-token training baselines, longer-horizon quality comparisons, and learning-efficiency claims remain future experiments.

## Experiment checkpoint housekeeping

On October 9, 2026, cleanup removed 23 checkpoint tensor files from the completed synchronous/asynchronous comparisons, the pinned-staging comparison, the checkpoint profiler, and the six-update learning pilot, totaling 139.55 GB.
It also removed 1,170 abandoned pytest checkpoint files totaling 61.26 MB; symlink and hardlink security-test fixtures were left untouched.
The cleanup reclaimed approximately 139.61 GB while preserving experiment metadata, metrics, input hashes, validation reports, logs, traces, commands, and source snapshots.
Checks that reload the removed experiment tensors require regenerating those checkpoints.
All four production checkpoint generations at updates 2, 100, 200, and 300 remain complete, and the production run remains paused.
The prepared resume source fingerprint and receipt are unchanged by the commits and cleanup.
The exact deletion manifest, per-file deletion log, and verification receipt are under `~/.cache/nanochat/spark-rlm/housekeeping-20261009T154103Z`.

## Development checks

```bash
CUDA_VISIBLE_DEVICES='' NANOCHAT_DTYPE=float32 NANOCHAT_ATTENTION_IMPL=sdpa OMP_NUM_THREADS=1 \
uv run --no-sync python -m pytest tests/test_rlm_model.py tests/test_rlm_training.py tests/test_rlm_initialization.py tests/test_rlm_checkpoints.py tests/test_rlm_latest.py tests/test_rlm_resume.py tests/test_rlm_checkpointing.py tests/test_rlm_train_cli.py tests/test_rlm_inference.py tests/test_rlm_data.py tests/test_rlm_profile.py -q

uv lock --check
git diff --check
```

The targeted tests cover causal isolation, cross-cycle gradients, shifted target accounting, loader buffer lifetime, optimizer/summary state, input-table-only initialization, interrupted rolling saves and retention, reviewed source transitions, exact fixture training/resume, generation rollover, and paired memory interventions.
An additional CPU comparison against the pre-refactor ordinary GPT exactly matched logits, parameter gradients, cached prefill/decode, parameter counts, and optimizer groups in FP32/BF16 with full attention and sliding attention with grouped-query heads.
After removing the freezing option and standalone recall diagnostic on October 8, 2026, `NANOCHAT_ATTENTION_IMPL=sdpa OMP_NUM_THREADS=1 uv run --no-sync python -m pytest -q` completed with 176 passed and 10 skipped, including the available CUDA tests.
The uv lock check, shell syntax checks for the existing Spark runners, and diff whitespace check also passed.
The subsequent pretrained-input initialization, retention, and integrated resume changes passed the same full command with 220 passed and 10 skipped, including the available CUDA tests.
The rolling-checkpoint and interrupted-retention changes passed the full command on October 9 with 329 passed and 10 skipped, including the available CUDA tests, as well as the lock, runner syntax, and diff checks above.
The subsequent async writer, warning-and-continue policy, ten-update defaults, and recoverable numbered publication passed the full command with 388 passed and 10 skipped, including actual CUDA copy-stream and state-round-trip tests.
The pinned-staging follow-up passed with 392 tests and 10 skipped, including actual CUDA snapshot reuse, shared storage and strided views, exact latest/numbered state, failed-copy drainage, and the optimizer mutation barrier.
The scheduled-freezing and additional loss logging changes passed with 406 tests and 10 skipped, including compiled unfreezing with activation checkpointing, exact resume across the freeze boundary, delayed Adam state in pinned checkpoints, profiler freeze behavior, and independent accumulation-loss checks.
The standard-library logging follow-up passed all 52 tests in `tests/test_rlm_train_cli.py` and `tests/test_rlm_checkpointing.py`, verifying per-update duration agreement between logs, JSONL, and results across resume, plus warning severity and checkpoint recovery.
Before the October 10 launch, the complete current tree passed the full suite with 406 tests and 10 skipped, plus the uv lock, runner shell syntax, and diff whitespace checks.
Run the project's full required checks before committing.

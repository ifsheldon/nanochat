# NanoChat plotting semantics

## Recorded losses

In the supported `scripts/base_train.py` logger, `train_loss` is overwritten on each accumulation microstep.
The displayed `loss` is therefore the final microbatch's loss after bias-corrected EMA, not an average over the optimizer update.
The logger uses beta 0.9 and prints six decimal places.
The plotted completed-update number is the printed step plus one.
For a different NanoChat revision or log format, inspect its logging code before applying these semantics.

RLM `nanochat/rlm_training.py` records `last_batch_loss` for the final accumulated episode, `loss` as the loss sum divided by the target count over the whole update, and `cycle_losses` as the corresponding per-cycle means over all accumulation batches.
The helper preserves raw values in CSV and computes bias-corrected EMA once, starting at the first available update and continuing across contiguous attempts.
For partial history, this EMA is initialized at the available starting point; it cannot reproduce the missing prefix's EMA state.

RLM's current objective also supervises predictions at `<EOM>` positions.
Use run contracts and captured source to establish the actual prediction budget, context layout, and initialization before claiming comparability.
Equal update indices alone do not establish equal token exposure, compute, or model quality.

## Resume and embedding phases

Read observed `input_embedding_frozen` values for phase markers.
Older invocations can use `freeze_input_embedding`, while newer invocations use `input_embedding_freeze`; a missing legacy boolean does not mean trainable embeddings.
Scheduled freezing, permanent freezing, and no freezing are different policies.
The helper reads the budget from invocation metadata but labels phase transitions only once they occur in update records.
Do not trust a stale top-level `result.json` as the status of a later attempt.

The helper accepts a contiguous recorded history that can start after update 1.
If overlapping attempts contain replayed updates, inspect the resume/checkpoint provenance to determine which trajectory is authoritative, then prepare an explicitly selected input history with its provenance.
Do not silently choose the last duplicate, join incompatible experiments, interpolate missing losses, or treat partial lines as completed updates.

## Update timing and ETA

Original `dt` is in milliseconds and excludes evaluation and checkpoint saves outside the update timer.
RLM `duration_seconds` includes the measured update, health checks, and waits for the previous asynchronous checkpoint; submission of the current checkpoint and final draining occur outside that timer.
Check captured source if the run uses another revision or checkpoint policy.
The helper displays every measured duration, including compilation spikes; its median line is a trend, not elapsed wall time.

If the user requests an ETA, derive recent completion-to-completion intervals from timestamped training logs within the current uninterrupted attempt.
Exclude pause gaps and identify compilation/startup intervals explicitly.
Use the current budget and remaining updates, report the sample window, and express uncertainty with a range when timing varies.
Re-estimate after an embedding unfreeze or other measured phase change.
If the run is stopped, report remaining active training time conditional on resuming; do not give an unconditional finish timestamp.
When only `duration_seconds` is available, label an estimate as timer-based and acknowledge omitted work rather than presenting it as a wall-clock ETA.

## Scope of conclusions

Without original logs, describe RLM's own loss trend, differences among cycles, and throughput trend.
No synthetic baseline, inferred historical curve from weights, or unsupported claim of improvement over original NanoChat belongs in that mode.
With original logs, compare like-labelled metrics over common updates and explain relevant objective, initialization, hardware, and timing differences.
Validation loss, bits per byte, CORE scores, and other evaluations need their own recorded evidence; this helper does not run evaluation or infer those metrics from training loss.

# Recurrent latent memory pretraining plan

Date: October 8, 2026.
Branch: `spark-rlm`, created from `spark` at `f148fe7`.
Status: the initial model, training, evaluation, and sampling workflow is implemented; d24 batch 16 passed the two-update probe with activation checkpointing, and a six-update natural-language pilot completed with checkpointing and held-out memory controls.
See the [RLM workflow and measured results](SPARK_RLM.md) for commands, implementation details, and validation status.
An additional batch-32 probe passed two complete updates with activation checkpointing, peaking at 67.94 GB reserved and taking 196.29 seconds for the second update at the same global text batch.
Batch 48 was subsequently probed with an approved 100 GB CUDA ceiling and 1,081,344 text targets per update; its second update stopped at the retained 16 GB system-memory guard, so that attempt has no complete steady-update timing.
A separate batch-48 retry with an approved 12 GB system guard and cached compilation completed both updates at 94.01 GB peak reserved; its 202.24-second second update achieved essentially the same text throughput as batch 32 despite the higher memory use.

The current layout adds a learned `<EOM>` input in every cycle and reduces real text to 1023 tokens per cycle, preserving the original maximum cycle capacity.
Training also omits the final cycle's unused summary suffix after measuring the complete-suffix baseline.
At batch 32, the fresh original control averaged 195.11 seconds/update and the full-summary `<EOM>` run averaged 194.51 seconds; terminal-summary omission reduced this to 165.99 seconds and 55.90 GB peak reserved memory.
The optimization preserved the objective and handoff, increased measured text throughput by 17.18% over full-summary `<EOM>`, and passed CPU equivalence tests; the workflow report records the small BF16 GPU gradient differences and the one failed near-zero scalar relative-tolerance check.
Freezing only the main input embedding then saved 0.302 GB of peak allocated memory, or 0.54%, and produced no measurable speedup in a two-update batch-32 probe: update 2 took 165.94 seconds versus 165.90 seconds with trainable input embeddings.
The experimental freezing option was removed after that measurement; the subsequent October 8 run kept all model parameters trainable.
A four-cycle batch-32 probe then completed two updates under the requested 100 GB CUDA cap with 79.56 GB peak reserved memory and a 195.51-second steady update, retaining trainable embeddings and activation checkpointing.
At the existing 5.84-billion-real-text-token budget, rounded to 5574 updates for `<EOM>`, this projects 12.61 days of training updates versus 10.71 days for optimized two-cycle batch-32 training, before startup, checkpointing, and evaluation overhead.
The selected full pretraining run uses four cycles and batch 32, initializes only the main input token embedding from d24 step 5568, and keeps all parameters trainable.
It saves a rolling `latest.pt` after every completed update and numbered checkpoints at update 2, every 10 updates, and the final update, retaining the latest ten numbered checkpoints in addition to the rolling checkpoint.
Checkpoint writing defaults to asynchronous overlap with the next forward/backward, with a writer barrier before the next optimizer step and final draining at an explicit stop.
The CUDA writer now prepares a pinned CPU snapshot through PyTorch's `DefaultStager`, then reuses that snapshot for rolling and numbered writes; retained host buffers add approximately 9.97 GB for the selected model and optimizer.
Save failures warn and continue, logging the completed update, saved latest update, and checkpoint lag; the one-completed-update loss bound applies only while saves succeed.
Both rolling and numbered publication finish before older numbered checkpoints are retired; durable publication and retirement records permit recovery after interrupted filesystem operations.
A full-size three-update comparison reduced total training-and-checkpoint time from 753.55 seconds synchronously to 714.71 seconds with a dedicated async CUDA copy stream, a 5.15% reduction; the workflow report records the identical input/RNG evidence, exact saved-versus-live state check, and cross-run floating-point differences.
Its warm non-final update took 232.81 seconds synchronously versus 217.48 seconds asynchronously, a 6.58% reduction in one sample.
A subsequent bounded Nsight profile found that GPU-to-CPU storage copies consumed 83–84% of checkpoint time and that overlapping saves both prolonged training kernels and blocked training threads on Python's GIL during storage copying; the workflow report records the attribution and measurement limits.
The pinned-staging follow-up reduced the same three-update total to 610.68 seconds, 14.56% below the previous async writer, and the warm non-final update to 198.69 seconds, an 8.64% reduction.
Its checkpoint matched the final live state exactly; CUDA reservation stayed at 79.85 GB and minimum system available memory was 24.60 GB with the additional pinned host buffers.
The production run remains paused at checkpoint 300 with a refreshed source-bound resume receipt; the workflow report includes the short-trial limits.
The fresh October 10 run loaded the same pretrained input table and freezes it until the final 10% of its full update budget, keeping the separate `<EOM>` and summary inputs and all other parameters trainable without changing the tokenizer vocabulary.
The scheduled policy is implemented behind `--freeze-input-embedding`, with its phase reconstructed from the full budget and completed checkpoint step on resume; `pretrain-u4-b32-frozen-20261010` started at 00:57 SGT with 5574 total updates and its first trainable input-table update at 5017.
Its first two updates and checkpoint 2 passed startup verification: the input table remained bit-exact with no Adam state, `<EOM>` and summary inputs each had two Adam updates, and the second update took 196.11 seconds with loss 10.255392.
Update logging adds the final accumulation batch's unscaled loss plus one scalar cycle mean across all accumulation batches, preserving the existing aggregate loss.
RLM progress and checkpoint events use NanoChat's standard Python logger at `INFO`, with recoverable checkpoint errors at `WARNING`; the per-update duration is logged and independently persisted as full-precision `duration_seconds` in `updates.jsonl`.
The older run's saved receipt predates these source changes and requires renewed review before resuming with current code.
The earlier batch-size measurements and six-update pilot describe the original layout without `<EOM>`; the current contract follows below.

## Objective and current decisions

Train a fresh nanochat model that carries information between text chunks through continuous latent memory.
Start with two recurrent cycles and measure whether 16 complete episodes fit in one device microbatch before increasing the recurrence length.
The forward episode length and gradient unroll length must match initially: all memory transitions inside an episode remain differentiable, and memory resets between episodes.

| Item | Initial setting |
| --- | --- |
| Backbone | Shared d24 Transformer: 24 layers, width 1536, 12 heads |
| Initialization | Initial probes use fresh weights; selected full run copies only the main input embedding from d24 step 5568 and reuses its tokenizer |
| Attention and precision | Full causal SDPA, `window_pattern="L"`, BF16 compute |
| Text per cycle, `C` | 1023 tokens |
| Memory slots, `K` | 512 continuous vectors |
| Summary inputs | 512 copies of one shared, trainable `<summarize>` embedding |
| Initial memory | Empty, with no learned prefix in the first cycle |
| Cycles per episode, `U` | 2 initially; evaluate 3 and longer after measuring memory and learning |
| Device microbatch, `B` | Target 16 episodes, containing 32 text subsequences when `U=2` |
| Objective | Shifted text cross-entropy plus `<EOM>` predicting each chunk's first token |
| Memory boundary tokens | Learned `<EOM>` every cycle; defer `<bom>`, `<bosum>`, and `<eosum>` |
| Memory type objective | Defer `<M>` / `<S>` classification; initial auxiliary weight is zero |
| Input embedding policy | October 10 run freezes only the pretrained main input table for the first 90% of the full update budget, then trains it for the final 10%; value embeddings, LM head, `<EOM>`, summary, and Transformer stay trainable |

These choices build on the [d24 configuration](SPARK_D24.md) and the [RLM proposal](../../recurrent-latent-memory/README.md).
The RLM proposal is in a neighboring repository; the relative link assumes both repositories share a parent directory.
The sections below distinguish initial implementation choices from later experiments.

## One episode and its gradient path

Use the same Transformer weights for every cycle.
Two cycles mean two Transformer calls and one memory handoff, rather than two additional Transformer layers.

```text
Cycle 1: [ EOM: 1 ][ text_1: 1023 ][ shared summary inputs: 512 ]
                    │                  │
                 text loss          memory_1
                                       │
Cycle 2: [ memory_1: 512 ][ EOM: 1 ][ text_2: 1023 ]
                                  │
                               EOM/text loss

The cycle-2 text loss backpropagates through memory_1 into cycle 1.
There is no final summary because it has no downstream consumer.
```

For each cycle that writes memory, concatenate `M_t`, one learned `<EOM>` embedding, the embeddings of `X_t`, and `K` copies of the learned summary embedding.
Run the Transformer with causal attention, then take the summary-position output vectors as `M_(t+1)`; the final training cycle omits the summary block and returns no memory.
Text can attend to previous memory and earlier text in its current chunk.
Summary positions can attend to previous memory, all current text, and earlier summary positions.
Current text cannot attend to its own summary suffix.

The complete episode must use separate recurrent calls.
A single attention pass over all chunks would give later chunks direct access to earlier text and bypass the memory bottleneck.
Carry only the latent vectors between cycles; do not carry previous raw text or per-layer KV states across the handoff.

Accumulate the prediction-weighted `<EOM>` and text losses over the full episode and backpropagate once per episode microbatch.
Do not detach memory inside the episode, update weights between cycles, or carry detached memory across optimizer updates in the initial version.
With gradient accumulation, finish backward for each complete episode microbatch and apply the optimizer update after the accumulation group.

The terminal summary suffix receives no training signal under this objective.
The original and initial `<EOM>` measurements retained it to establish comparable baselines.
Current training omits it explicitly through `write_memory=False`, reducing two-cycle positions from 3584 to 3072 without removing any prediction or memory handoff.

## Model integration contract

The existing [model forward path](../nanochat/gpt.py) accepts token IDs, so continuous memory needs an explicit embedding-level path.
Use the following initial contract:

- Represent the shared summary input as one model-owned embedding parameter, repeated across all summary slots and cycles; no tokenizer vocabulary change is needed for this internal query.
- Represent `<EOM>` as a second model-owned embedding parameter shared across cycles, with ordinary first-token loss even in the first cycle.
- Apply the existing input RMS normalization to the concatenated memory, `<EOM>`, text, and summary embeddings.
- Preserve the existing causal smear operation across the combined sequence, including its region boundaries, and construct `x0` afterward as in the current model.
- Keep the existing additional value-embedding lookups for text; use zero additional value embeddings for memory, `<EOM>`, and summary positions, which still have ordinary projected attention values.
- Extract recurrent memory after the existing backout operation and final RMS normalization, before the vocabulary projection.
- Pass those vectors directly into the next cycle without quantization, a learned compression projection, or detachment.
- Reset RoPE positions to zero at the start of each cycle; first and final training cycles have 1536 positions, and intermediate cycles have 2048.
- Apply the LM head jointly to `<EOM>` and text positions during training, avoiding vocabulary logits for the memory and summary blocks.
- Give the summary and `<EOM>` parameters explicit initialization and optimizer groups; verify that the existing custom optimizer includes them and saves their state.

Keep input layout, target selection, memory extraction, and episode unrolling as distinct responsibilities.
Use structured configuration and result objects for dimensions, layout, text loss, and outgoing memory instead of hidden conventions in a generic token tensor.
Handle the first-cycle and later-cycle shapes explicitly in compilation and benchmark their warmup separately.

Identical summary inputs do not require identical learned outputs: positions and causal attention contexts differ.
However, nanochat initially zeroes attention and MLP output projections, so contextual differences need not exist at initialization.
Do not make immediate slot diversity a correctness requirement.
Track slot similarity or effective rank during learning, and prioritize whether the resulting memory improves future text prediction.

## Data, document boundaries, and targets

Use the existing [BOS best-fit data loader](../nanochat/dataloader.py) with an episode length of `U * C`.
For two cycles, request one 2046-token input row and split it into two 1023-token chunks.
For three cycles, request 3069 tokens and split into three chunks; the Transformer still processes at most 2048 positions per cycle.
The loader already reads one extra token to construct shifted targets.
Split both input and target tensors at the same chunk boundaries before inserting memory or summary positions.

```python
# Conceptual flow, not a proposed public API.
x, y = next_episode_batch(length=cycles * text_length)
memory = None
episode_loss = 0
for cycle in range(cycles):
    start = cycle * text_length
    stop = start + text_length
    text_loss, memory = model_cycle(
        memory=memory,
        text_ids=x[:, start:stop],
        targets=y[:, start:stop],
        write_memory=cycle < cycles - 1,
    )
    episode_loss = episode_loss + text_loss / cycles
(episode_loss / accumulation_steps).backward()
# Fetch the next episode microbatch only after this backward has finished.
```

The equal division above assumes all cycles contain the same number of valid targets.
Use valid-target counts if masking or variable text lengths are introduced.

Preserve the existing document convention: each document receives BOS, and pretraining does not append an EOS token.
Do not insert another BOS merely because text enters a new recurrent cycle.
The existing packed rows may contain several documents and may discard cropped document tails; this prototype does not guarantee that an episode is one uninterrupted document.
Reset latent memory between packed rows, and preserve existing attention across internal document boundaries initially.

Measure how often a cycle boundary falls within the same document and how much later-chunk text has relevant earlier context.
Use a held-out set of sufficiently long, contiguous documents to establish memory use.
If packing leaves too few useful continuations for learning, introduce a document-contiguous episode sampler before interpreting a failed training run as an architectural failure.
Evaluate controls on the same documents and target positions.

Preserve shifted next-token supervision at the chunk boundary.
The last text input of cycle 1 predicts the first token of cycle 2 from the old exact context; that target token must not be inserted into cycle 1's summary input.
The first text input of cycle 2 then predicts the following token using memory and its new local context.
The new cycle's `<EOM>` also predicts that first token using only the incoming memory; the first cycle's `<EOM>` predicts its first input, normally BOS.
Build joint targets with `cat([text_ids[:, :1], shifted_text_targets], dim=1)` and apply one cross-entropy across the `<EOM>` plus text readout.
Do not shift targets over the concatenated control layout, which would incorrectly train text to predict summary markers.

Generation summarizes the completed chunk first and uses a prefix-only `[memory][<EOM>]` call to predict the next chunk's first token.
The old chunk's final-position prediction remains a training target but does not select the boundary token during generation.

The current loader returns views into a reusable GPU buffer.
Keep those indices stable until backward completes; fetching another episode early can overwrite tensors saved for embedding backward.
If prefetching is introduced, each in-flight episode must own its storage.

## Batch accounting and memory probe

Count supervised text tokens separately from internal Transformer positions.
Memory slots and summary slots must not inflate the data budget, learning-rate schedule, or reported text throughput.

| Configuration | Text targets per episode | Positions per episode | Text targets at `B=16` | Positions at `B=16` |
| --- | ---: | ---: | ---: | ---: |
| Ordinary 2048-token training | 2048 | 2048 | 32,768 | 32,768 |
| RLM, two cycles | 2046 | 1536 + 1536 = 3072 | 32,736 | 49,152 |
| RLM, three cycles | 3069 | 1536 + 2048 + 1536 = 5120 | 49,104 | 81,920 |
| RLM, four cycles | 4092 | 1536 + 2048 + 2048 + 1536 = 7168 | 65,472 | 114,688 |

For two cycles, batch 16 means 16 parallel episodes processed through two sequential forwards, giving 32 text subsequences.
At the current global batch of 1,047,552 text tokens, one optimizer update contains 32 episode microbatches and 64 cycle forwards on this single GPU, with 1,048,576 supervised predictions including `<EOM>`.
Three cycles require revisiting accumulation or the exact global token budget because that batch no longer divides evenly.
At batch 32, two cycles use 16 accumulation microbatches and four cycles use eight, preserving the same real-text and prediction budgets per update.
Four cycles process 1,835,008 internal positions per update versus 1,572,864 for two cycles, an increase of 16.67%.

The two-cycle version processes 1.5 times as many positions as ordinary 2048-token training after omitting the final summary, versus 1.75 times in the measured complete-suffix baseline.
This is not a FLOP ratio or a peak-memory estimate: attention shapes, joint `<EOM>`/text vocabulary projection, retained activation graphs, and compiler behavior all matter.
The standalone BF16 memory tensor is only 1.5 MiB per lane, or 24 MiB at batch 16; differentiable activations across the unroll are the larger unknown.
Model and optimizer parameters are shared across cycles.

Probe procedure after implementation:

1. Use a fresh artifact root such as `~/.cache/nanochat/spark-rlm`, fresh d24 weights, the existing tokenizer, and real training batches.
2. Start with `B=16`, `U=2`, `C=1023`, `K=512`, full SDPA, BF16, and activation checkpointing disabled to establish the footprint.
3. Keep the 80 GB CUDA allocator limit and 16 GB minimum system-available-memory threshold from the Spark workflow; report decimal GB and account for CPU and external CUDA allocations separately.
4. Complete at least two optimizer updates, using the full 1,047,552-text-token global batch; compare real-text throughput separately from the 1,048,576 supervised predictions.
5. Synchronize measurement boundaries and record per-update and overall peak CUDA allocated/reserved memory, minimum observed system available memory, update time, text tokens/second, internal positions/second, compilation activity, and finite losses/gradients.
6. Include `optimizer.step()` in measurement: the first update creates optimizer state, while the second starts with that state already resident during forward and backward.
7. If batch 16 fails, preserve the failure and measure smaller batches or activation checkpointing as explicitly separate configurations; maintain the text-token global batch where possible.
8. Report batch-16 feasibility and available headroom before selecting the training configuration; retain the existing 72 GB peak-reserved acceptance target for a later full workflow unless it is deliberately revised with measured overhead.

Do not include evaluation or checkpoint saving in the initial training-memory probe.
Measure their additional overhead before choosing a full training configuration.
Keep optimizer updates distinct from recurrent cycles in all logs.
The earlier production horizon of 5568 updates used 1,048,576 real text tokens per update, totaling 5,838,471,168 text tokens.
Preserving that text budget with the current `<EOM>` layout requires `ceil(5,838,471,168 / 1,047,552) = 5574` updates, totaling 5,839,054,848 real text tokens after rounding to a full update.
At the measured optimized two-cycle batch-32 mean of 165.9877 seconds per update, this projects 257.0 hours, or 10.71 days, for training updates alone.
Startup, compilation, checkpoint saving, evaluation, and interruptions add to that estimate.

## Implementation and validation stages

### 1. Implement the recurrent path

- [x] Add continuous memory inputs, shared summary and `<EOM>` parameters, joint `<EOM>`/text logits, and explicit outgoing memory to the model.
- [x] Add the episode wrapper and stable-buffer loader integration with `C=1023`, `K=512`, and `U=2`.
- [x] Keep full attention and the existing tokenizer; add no auxiliary classification loss.
- [x] Separate real-text accounting from internal positions in accumulation, throughput, schedule progress, and evaluation.
- [x] Keep RLM artifacts and checkpoint metadata distinct from the existing completed d24 run.

### 2. Verify the mechanism on a small model

- [x] Verify that changing future text cannot change earlier logits and that a cycle's summary cannot influence its own earlier text predictions.
- [x] Use cycle-2 text loss alone to verify gradients through the first memory tensor into the first summary computation and shared summary parameter; use nonzero attention/MLP output weights or a short warmup for this check.
- [x] Verify dual boundary supervision from old final text and new `<EOM>`, including ordinary first-cycle loss, with no same-position reconstruction or accidental marker prediction.
- [x] Verify memory resets between episodes and that the loader cannot overwrite an in-flight episode's indices.
- [x] Verify all new trainable parameters are optimized and survive checkpoint reload.

These tests should exercise causal, gradient, data-lifetime, and checkpoint behavior rather than merely assert configured shapes or labels.

### 3. Measure d24 resource use

- [x] Run the two-update probe above and publish the measurements, including a batch-16 failure if it occurs.
- [x] Measure activation checkpointing separately after the batch-16 failure; batch 16 then passed, so a smaller batch was unnecessary for feasibility.
- [x] Select a configuration with enough headroom for the later evaluation and checkpoint workflow: batch 16 with activation checkpointing reserved 41.75 GB including checkpoint saving; separate batch-4 evaluation reserved 11.99 GB.

### 4. Run a bounded learning experiment from scratch

- [x] Choose an explicit short training budget after measuring throughput; six d24 updates with batch 16 and activation checkpointing are selected, approximately 21 minutes at the measured speed, within the suggested 30-minute experiment budget.
- [x] Train the recurrent model for the bounded six-update pilot and report held-out loss by cycle.
- [ ] Compare against ordinary 2048-token training and independent 1024-token chunks on matched data and targets before claiming a natural-language quality improvement.
- [x] Evaluate second-cycle loss with intact, zeroed, and batch-shuffled memory on both packed episodes and uninterrupted validation documents; exclude cycle 1 when attributing effects to memory.
- [x] Run a controlled task whose later prediction depends on information in the earlier chunk, and verify that learning depends on the memory handoff; retain its historical results after removing the standalone synthetic diagnostic during cleanup.
- [ ] Report equal-data results together with wall time or measured compute; make a separate equal-compute comparison when claiming efficiency.
- [x] Inspect slot similarity and memory gradients as diagnostics, without treating either as evidence of useful retention on its own: the d24 summary parameter received gradients, while the sampled slots remained highly similar after six updates.

Proceed when training is numerically stable, the recurrence receives gradients, and later predictions demonstrably benefit from relevant memory.
Predefine the held-out comparison and uncertainty reporting; a single noisy loss difference is insufficient.

### 5. Extend the horizon

- [x] Probe a longer unroll with forward episode length equal to gradient unroll length: four cycles at batch 32 passed two updates under the 100 GB CUDA cap.
- [ ] Prefer longer meaningful document context when resources permit; measure memory and throughput again for each unroll length.
- [ ] Evaluate repeated rewriting and inference rollouts beyond the trained horizon before assuming long-term retention.
- [ ] Revisit a document-contiguous sampler if longer packed episodes mostly introduce unrelated documents.
- [ ] Consider detached streaming state only as a later, separately validated training regime.

### 6. Make the training and inference workflow complete

- [x] Store memory size, text chunk length, unroll length, summary-input policy, attention/position convention, loss configuration, and real-text counters in checkpoint metadata.
- [x] Save the model, summary parameter, optimizer, RNG, and data-progress state only after a completed optimizer update, when all episode microbatches in the accumulation group have finished; no live recurrent tensor or partial accumulated gradients need checkpointing.
- [x] Document the loader's existing approximate row-group resume semantics rather than claiming exact replay.
- [x] Implement generation rollover with cleared old KV/smear caches and the same memory extraction, positional reset, and boundary readout used in training; the initial generator recomputes bounded chunks and holds no KV cache.
- [x] Checkpoint/reload and evaluate through dedicated RLM-aware entry points.
- [x] Update the affected model, training, evaluation, and operational documentation together when these behaviors are implemented.

## Deferred token and loss experiments

### Additional boundary tokens: `<bom>`, `<bosum>`, `<eosum>`

`<EOM>` is now implemented as the memory-only first-token readout; the remaining three delimiters are deferred.
The model wrapper already knows the memory, text, and summary regions from their tensor lengths and positions.
Explicit delimiter embeddings might help identify regions when layouts or memory budgets vary, but their value needs measurement.

Changing `K` from 512 to 128 does not inherently require delimiter tokens.
The wrapper can provide 128 incoming memory vectors and 128 summary queries, giving a steady-cycle length of `128 + 1 + 1023 + 128 = 1280`.
This changes the number of latent vectors, not the number of bits in each vector.
A model trained only with `K=512` is not automatically reliable at `K=128`; train or evaluate the variable-budget regime explicitly.

- [ ] Compare delimiters against the same tensor layout without delimiters, holding text and memory budgets fixed.
- [ ] Define whether memory size is fixed per episode or can change between cycles; the latter requires separate incoming-memory and outgoing-summary lengths.
- [ ] Evaluate `K` values such as 128, 256, and 512, including mixed-budget training if runtime budget selection is a goal.
- [ ] Decide how delimiter positions are masked and whether any receive loss; do not accidentally include them in the real-text budget.
- [ ] Resolve context capacity before adding the remaining three markers: the current steady layout already uses 2048 positions, and three extra delimiters would make it 2051.

### Two type targets: `<M>` and `<S>`

Interpretation: a small auxiliary classification loss on memory/read positions and summary/write positions, rather than a binary or bit-level reconstruction objective.
Recommendation: leave it disabled for the first learning experiment, then test it as an ablation.

One concrete formulation labels final hidden states at incoming memory positions as `<M>` and outgoing summary positions as `<S>`.
Its weighted loss would be `L_text + lambda * L_type`, with the type loss normalized separately from the text loss.
The first cycle has no incoming memory positions, and the same latent vector changes role from a summary output to the next cycle's memory input.
Define targets by the current position's role; do not turn latent vectors into discrete token IDs or replace their carried content with type embeddings.

This objective teaches role separability, which could help organize representations, but it does not itself require memory to preserve facts needed later.
Position and layout can make role classification easy without useful compression.
Therefore, choose its weight using future-text loss and memory-dependent evaluation, rather than type accuracy alone.

- [ ] Compare `lambda=0` with small nonzero weights; use 0.01 as a literature reference point rather than an assumed optimum.
- [ ] Specify read/write target positions, normalization, class balancing, and whether the unused terminal summary participates.
- [ ] Compare two vocabulary targets against a small dedicated classification head; the latter tests the objective without adding tokenizer tokens or projecting all memory positions to the full vocabulary.
- [ ] If vocabulary tokens are introduced, update tokenizer/model metadata and examine any coupling between output-class weights and input embeddings.
- [ ] Keep this experiment separate from adding role embeddings to model inputs, delimiter tokens, or a content-reconstruction loss.

## How original Recurrent Memory Transformer differs

The comparison below uses the 2022 RMT paper and a pinned revision of the authors' implementation.
Our design is an RMT-inspired variant.

| Aspect | Original RMT | This initial plan |
| --- | --- | --- |
| Cycle input | Previous memory at both prefix and suffix | Previous memory at prefix; fresh shared summary queries at suffix |
| Initial state | Learned memory table | Empty prefix |
| Learned inputs | Independently trainable memory rows, initialized identically in the code | One permanently shared summary input embedding |
| Attention | Bidirectional within each memory block; causal text | Fully causal across the entire cycle |
| Memory count and boundaries | Configured count and positional slicing in the code | Fixed 512 initially; `<EOM>` readout implemented, three other delimiters deferred |
| Training horizon terminology | BPTT depth counts previous segments | Two total cycles with one differentiable handoff |

The [paper, Section 3](https://arxiv.org/html/2207.06881v2#S3) describes memory recurrence and attention.
The authors' code shows [memory initialization](https://github.com/booydar/LM-RMT/blob/e2895080868afc390386220f2b68a56c4a218126/pytorch/mem_transformer.py#L578-L585), [initial training state](https://github.com/booydar/LM-RMT/blob/e2895080868afc390386220f2b68a56c4a218126/pytorch/train.py#L434-L441), and [input assembly](https://github.com/booydar/LM-RMT/blob/e2895080868afc390386220f2b68a56c4a218126/pytorch/mem_transformer.py#L606-L638).

RMT also provides a precedent for auxiliary supervision: Appendix A.5 describes the Transformer-XL + RMT WikiText-103 experiment using a memory-position loss of weight 0.01 targeting one fixed new special token, rather than separate `<M>` and `<S>` classes.
See the [paper appendix, PDF page 15](https://arxiv.org/pdf/2207.06881v2#page=15).
The pinned implementation's [core loss path](https://github.com/booydar/LM-RMT/blob/e2895080868afc390386220f2b68a56c4a218126/pytorch/mem_transformer.py#L706-L740) selects text positions and returns memory separately, so that path should not be confused with the paper's auxiliary-loss experiment.

- [ ] After establishing the baseline, independently compare fresh summary queries with reused incoming memory at the suffix.
- [ ] Independently compare shared queries with separate learned slots and fully causal attention with RMT's memory-block mask.
- [ ] Keep these changes separate from the token/loss ablations so any improvement can be attributed to a specific choice.

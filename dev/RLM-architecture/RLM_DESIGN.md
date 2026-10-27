# Recurrent latent memory design

This document covers the recurrent model, its data and loss contract, and architectural experiments.
Hardware settings, resource probes, and implementation history belong to the [Spark training record](../spark-training/SPARK_RLM_PLAN.md) and [H100 implementation plan](../h100-training/H100_MULTI_GPU_PLAN.md).
See the [Spark workflow](../spark-training/SPARK_RLM.md) or [H100 workflow](../h100-training/H100_RLM.md) for training commands.

## Objective and design decisions

Train a fresh nanochat model that carries information between text chunks through continuous latent memory.
The forward episode length and gradient unroll length must match initially: all memory transitions inside an episode remain differentiable, and memory resets between episodes.
The shared GPT backbone uses full causal attention with `window_pattern="L"`.

| Item | Design choice |
| --- | --- |
| Text per cycle, `C` | 1023 tokens |
| Memory slots, `K` | 512 continuous vectors |
| Summary inputs | 512 copies of one shared, trainable `<summarize>` embedding |
| Initial memory | Empty, with no learned prefix in the first cycle |
| Cycles per episode, `U` | 2 initially; evaluate 3 and longer after measuring memory and learning |
| Objective | Shifted text cross-entropy plus `<EOM>` predicting each chunk's first token |
| Memory boundary tokens | Learned `<EOM>` every cycle; defer `<bom>`, `<bosum>`, and `<eosum>` |
| Memory type objective | Defer `<M>` / `<S>` classification; initial auxiliary weight is zero |

These choices build on the [Recurrent Latent Memory proposal](https://github.com/ifsheldon/recurrent-latent-memory).
The [interactive memory-cycle illustration](https://recurrent-latent-memory.reify.ing) shows how working context is consolidated into latent memory for the next cycle.
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

The existing [model forward path](../../nanochat/gpt.py) accepts token IDs, so continuous memory needs an explicit embedding-level path.
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

Use the existing [BOS best-fit data loader](../../nanochat/dataloader.py) with an episode length of `U * C`.
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

## Token accounting

Count supervised text tokens separately from internal Transformer positions.
Memory slots and summary slots must not inflate the data budget, learning-rate schedule, or reported text throughput.
See the [Spark batch accounting and memory probe](../spark-training/SPARK_RLM_PLAN.md#batch-accounting-and-memory-probe) for concrete episode, accumulation, and resource calculations.

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

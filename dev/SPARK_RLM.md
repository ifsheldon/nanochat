# Recurrent latent memory on DGX Spark

The RLM model carries differentiable memory between bounded text chunks using the shared GPT backbone.

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
All model parameters remain trainable, including the main input embedding, additional value embeddings, and untied LM head.
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

## Python inference

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

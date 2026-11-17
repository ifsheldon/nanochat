# Long documents and conversations in RLM training

Recorded on November 14, 2026.
This document records how the pretraining and SFT data loaders treat documents and conversations longer than one episode, the measurements behind that, and the decisions for the next runs.
The pretraining decision is implemented.
For SFT the mixed-episode data loader is implemented, measured, and integrated into the recurrent trainer.
The trainer requires `--episode-mix` and an explicit `--loss-weighting` choice between `microbatch` and `group-token`.
The [SFT loss-weighting design discussion](RLM_SFT_LOSS_WEIGHTING.md) records the November 16 follow-up on answer lengths, packed supervision, and the two implemented objectives; their model-quality comparison remains unresolved.
The findings do not change the model already trained, [`rlm-exps/nanochat-rlm`](https://huggingface.co/rlm-exps/nanochat-rlm); they matter for the next pretraining and SFT runs.

Terms follow the [RLM terminology](RLM_DESIGN.md#terminology): an episode is one loader row, a chunk is a fixed `text_length`-token slice of it, and a cycle processes one chunk.
All numbers use 1023-token chunks, so a four-cycle episode holds 4,092 text tokens.

## Summary

| Stage | Problem | Decision |
| --- | --- | --- |
| Pretraining | Documents longer than an episode can never be best-fit. They fill the packer's buffer, about half of those used become a cropped row-end filler, and best-fit then searches a small fraction of the buffer. | Crop each over-long document to one episode when it enters the buffer. |
| SFT | 4,019 conversations are longer than a four-cycle episode. 3,562 of them are skipped entirely, and the other 457 fill the packer's buffer, which raises padding from 0.3% to 11%. | Train with episodes of 4, 12, and 24 cycles, mixed together instead of run one after another. |

## Pretraining

### How the packer works

RLM pretraining uses the original nanochat BOS best-fit packer in [`nanochat/dataloader.py`](../../nanochat/dataloader.py), with a row length of `cycles * text_length`.
The base GPT trainer and the packed evaluation loaders use the same function.
It keeps a buffer of 1,000 tokenized documents and builds each row as follows:

1. Take the largest buffered document that fits the remaining space, and repeat until none fits.
2. When none fits, take the shortest buffered document, crop it to the remaining space, and discard its tail.

Every episode therefore starts at a document start.
Chunk 1 starts there too, while chunks 2 to 4 start wherever the packed text happens to be.

| Chunk | Starts at a document start | Text continuing a document from an earlier chunk | Contains no document start |
| --- | ---: | ---: | ---: |
| 1 | 100% | 0% | 0% |
| 2 | 0.1% | 47% | 19% |
| 3 | 0.2% | 42% | 14% |
| 4 | 0.1% | 44% | 6% |

The logged Spark run agrees with this replay: 99.85% of handoffs fall inside a document, and 43.8% of the text in chunks 2 to 4 precedes the next document start.

### What goes wrong with long documents

About 1.0% of documents exceed 4,096 tokens, and they hold about 13% of the corpus's tokens; see the [document-length census](SPARK_RLM_DOCUMENT_BOUNDARIES.md#full-corpus-document-lengths).
A document longer than a row never fits, so step 1 never selects it.
It leaves the buffer only through step 2, which requires it to be the shortest document in the buffer.

- Over-long documents accumulate until they fill the buffer, after roughly 15,000 four-cycle episodes.
- From then on best-fit searches only the freshly loaded documents: a median of 83 candidates when a row starts, instead of 1,000.
- The longest documents seen so far stay in the buffer for the rest of the run and are never used.
- An over-long document that is used becomes either a whole single-document episode or the cropped filler at the end of a multi-document row.

### Options considered

| Option | What changes | What it costs |
| --- | --- | --- |
| 1. Crop an over-long document to one episode when it arrives | It is the largest candidate for the next empty row, so none waits, none becomes a filler, and best-fit searches the whole buffer again | Text beyond the first episode of a long document is still discarded |
| 2. Cut a long document into consecutive episodes | All tokens are used | Later pieces start mid-document with empty memory, which never happens at inference |
| 3. Let an episode run as many cycles as the document needs | All tokens are used, and memory is trained over many handoffs | Variable-length episodes, and gradients cut every few cycles while memory carries on, which the [design](RLM_DESIGN.md) currently rules out |

### Decision: option 1

Pretraining cannot afford more cycles per episode, so the next pretraining run crops over-long documents on arrival.
The packer does this by default through `crop_long_documents=True`; passing `False` restores the original behaviour.
The base GPT trainer and the packed evaluation loaders share the function and get the same default, so their rows change too.
A replay of both variants gives:

| Measurement | Original packer | Crop on arrival |
| --- | ---: | ---: |
| Over-long documents waiting in the buffer | about 1,000 | 0 |
| Candidates that fit when a row starts, median | 83 | 1,087 |
| Single-document episodes | 3.4% | 7.2% |
| Over-long documents used as a whole episode | 46% | 100% |
| Over-long documents used as a row-end filler | 54%, median 2,015 tokens kept | 0% |
| Episodes ending on a cropped document | 87% | 81% |
| Consumed tokens discarded from over-long documents | 5.7% | 7.0% |
| Consumed tokens discarded by cropping short documents | 1.5% | 1.4% |

Option 1 does not reduce discarded text.
Its benefit is that every long document is used as a full single-document episode, where all three handoffs stay inside one document, and that best-fit works over the whole buffer again.
The real loader on the training shards gives 6.8% single-document episodes over its first 1,920 episodes with the crop, and none without it.

## SFT

### How the packer works

The original SFT packer in [`nanochat/sft_dataloader.py`](../../nanochat/sft_dataloader.py) keeps a buffer of 100 rendered conversations.
The measurements below describe that original packer and the earlier H100 run; the current recurrent trainer uses the mixed-episode loader described later.

- It fills each row with the largest whole conversation that fits, repeating until none fits, then pads the rest of the row.
- It never splits a conversation across rows.
- A conversation that fits no empty row is truncated to the row.
- A conversation with no assistant token inside the row length is skipped.

The H100 SFT run (`pretrain-2`) kept this packer and set the row length to four chunks.
Its logs match a replay of the packer at that row length, both before and after the buffer problem described below:

| Measurement | SFT log | Replay |
| --- | ---: | ---: |
| Conversations per episode, updates 1 to 50 | 6.75 | 6.74 |
| Supervised share of positions, updates 1 to 50 | 61% | 61% |
| Conversations per episode, updates 101 to 504 | 6.0 | 6.0 |
| Supervised share of positions, updates 101 to 504 | 55% | 55% |

### The training mixture

One sample is one whole conversation, rendered with a leading BOS and alternating user and assistant turns; loss is taken on assistant tokens only.

| Source | Conversations | Messages per conversation | Mean tokens | Longer than one chunk |
| --- | ---: | ---: | ---: | ---: |
| SmolTalk train | 460,341 | 4.6 | 865 | 39% |
| MMLU auxiliary train, three passes | 299,526 | 2 | 384 | 0.2% |
| GSM8K train, four passes | 29,892 | 2 | 174 | 0% |

The mixture holds about 522M tokens, of which about 297M are assistant tokens.

Chunks 2 to 4 start in the middle of a conversation in more than 91% of episodes.
After the buffer fills, about 40% of all supervised tokens follow a handoff inside the same conversation, so SFT does exercise memory.

### Conversations longer than a four-cycle episode

4,019 SmolTalk conversations, 0.87% of that source, are longer than four cycles.
None of the sampled MMLU or GSM8K conversations is.
They fall into two groups:

| Group | Conversations | Shape | Share of SFT tokens | Share of assistant tokens | What the original packer does |
| --- | ---: | --- | ---: | ---: | --- |
| Read, then answer | 3,562 | Two messages: one long user message, then an answer with a median of 185 tokens | 8.0% | 0.23% | Skipped: no assistant token falls inside the row |
| Long conversation with early supervision | 457 | 449 multi-turn chats and 8 single-answer conversations | 0.4% | 0.67% | Waits in the buffer and is eventually truncated to one row |

The November 16 full scan separates the eight single-answer conversations whose answer already begins within four cycles from the 449 actual long multi-turn chats.
The earlier analysis grouped all 457 as multi-turn chats.

In the read-then-answer group the answer ends at a median of 10,806 tokens; the 5th and 95th percentiles are 8,184 and 18,398.
These conversations place their answers after about ten chunks of input, so access to earlier chunks passes through recurrent memory.
Whether the answers actually depend on that earlier context was not measured by the length analysis.

### The buffer fills here too

The long conversations with early supervision can never be best-fit, so they accumulate in the 100-conversation buffer.
The buffer is full of them at about episode 24,000, around update 95 of 504.
From then on the packer effectively sees one fresh conversation at a time and pads whatever it cannot fill.

| Measurement | Before | After |
| --- | ---: | ---: |
| Padding, share of all positions | 0.3% | 11% |
| Padding in chunk 4 | 1% | 42% |
| Conversations per episode | 6.75 | 6.0 |
| Supervised share of positions | 61% | 55% |
| Episodes holding one truncated conversation | 0% | 0.3% |

### How many cycles the long conversations need

The table counts read-then-answer conversations whose answer ends inside an episode of the given length.
Each has exactly two messages, so this equals the whole conversation fitting.

| Cycles per episode | Answer ends inside the episode | Share of the group | Their share of SFT tokens |
| ---: | ---: | ---: | ---: |
| 4 | 0 | 0% | 0% |
| 6 | 7 | 0.2% | 0.01% |
| 8 | 181 | 5.1% | 0.27% |
| 10 | 1,476 | 41.4% | 2.55% |
| 12 | 2,395 | 67.2% | 4.52% |
| 14 | 2,937 | 82.5% | 5.90% |
| 16 | 3,254 | 91.4% | 6.82% |
| 20 | 3,464 | 97.2% | 7.54% |
| 24 | 3,540 | 99.4% | 7.87% |

Doubling to eight cycles covers 5% of the group, because answer ending positions cluster between about 8,000 and 13,000 tokens into the conversation.
All 457 long conversations with early supervision fit whole within ten cycles.

### Options considered

- **Crop conversations to four cycles on arrival.** This clears the buffer and recovers the long conversations with early supervision, keeping a median of 89% of their assistant tokens in the earlier sample, but every read-then-answer conversation stays skipped.
- **Crop from the front, keeping the answer.** Rejected: the model would be trained to answer from text it never saw.
- **Raise the cycle count for every episode.** A long episode packed from the whole mixture is mostly unrelated short conversations, so it pays for long backpropagation without training long retention.
- **Use several episode lengths and send each conversation to the shortest one it fits.** Chosen.

### Decision: episodes of 4, 12, and 24 cycles, mixed

SFT can afford more cycles than pretraining, so the next SFT run uses three episode lengths.
The lengths are mixed throughout training instead of run as consecutive phases, so that the run does not end on long read-then-answer episodes at the lowest learning rates.

With each conversation assigned to the shortest length it fits:

| Episode length | Conversations | Share of SFT tokens | Share of assistant tokens |
| --- | ---: | ---: | ---: |
| 4 cycles | 785,740 | 91.6% | 99.1% |
| 12 cycles | 2,852: 2,395 read-then-answer, 449 multi-turn, and 8 single-answer with early supervision | 4.9% | 0.82% |
| 24 cycles | 1,145 read-then-answer | 3.3% | 0.08% |
| Longer than 24 cycles | 22 read-then-answer | 0.1% | under 0.01% |

### Implementation: the mixed-episode loader

`sft_mixed_episode_loader` in [`nanochat/sft_dataloader.py`](../../nanochat/sft_dataloader.py) implements the decision.
[`scripts/rlm_sft.py`](../../scripts/rlm_sft.py) uses this loader for both training and validation in one process.
The mixed loader's `episode_mix` parameter and the trainer's `--episode-mix` option are required and have no default.
`parse_episode_mix` reads it from text as comma-separated `CYCLESxBATCH:MICROBATCHES` entries: cycles per episode, episodes per microbatch, and microbatches of that shape in one optimizer update.

```text
4x8:29,12x2:2,24x1:2    29 microbatches of 8 four-cycle episodes, 2 of 2 twelve-cycle episodes,
                        and 2 of 1 twenty-four-cycle episode per update: 1,024 chunks
4x8:32                  four-cycle episodes only: 1,024 chunks
```

- Each conversation is routed, when it is read, to the shortest episode length that holds it whole, so a long conversation never waits in the four-cycle buffer.
- A longer episode opens with a waiting conversation that needs its length, and the rest of it is filled best-fit from that length and all shorter ones.
- When no conversation of its length is waiting, a longer episode is filled from shorter ones.
- A conversation longer than the longest episode is truncated to it, or skipped when no assistant token would remain.
- Conversations of the longer lengths wait in unbounded queues until an episode of that length takes them; only the shortest length uses the 100-conversation buffer.
- One optimizer update is one pass over a fixed pattern in which every shape contributes its microbatches, evenly interleaved.

The trainer reads the cycle count of a microbatch from its width, and `episode_mix_chunks(mix) * text_length` must equal its inherited or explicit `--total-batch-size` text-position budget.
The old `--cycles` and `--device-batch-size` options are replaced by the mix; `--episode-mix 4x8:32` selects fixed-length episodes.
With a single entry the loader gives fixed-length episodes whose over-long conversations are truncated or skipped on arrival, so its buffer does not fill.
`sft_data_loader_bos_bestfit` retains `crop_long_conversations=False` by default so replays reproduce the earlier run, while `crop_long_conversations=True` truncates over-long conversations on arrival; `scripts/chat_sft.py` requires `--crop-long-conversations` or `--no-crop-long-conversations` explicitly.
The trainer also requires `--loss-weighting {microbatch,group-token}`, with explicit `--group-loss-weights` required only for `group-token`.
It stages a complete mix on CPU, counts assistant text and eligible EOM targets, then transfers and backpropagates one microbatch at a time.
See [recurrent SFT settings](../RLM_TRAINING_SETTINGS.md#recurrent-sft) for the selectors, exact budget, evaluation counts, metrics, and examples.

The earlier packing replay compared one epoch of the real mixture with `4x8:29,12x2:2,24x1:2` against the original packer at four cycles:
The original packer's replay reproduces the published SFT run exactly: 504 updates and 790,367 conversations consumed.

| Measurement | Original packer | Mixed-episode loader |
| --- | ---: | ---: |
| Updates in one epoch | 504 | 498 |
| Padding, share of all positions | 9.05% | 0.25% |
| Supervised text targets per update, excluding EOM readouts | 586,514 | 596,555 |
| Conversations of 5 to 12 cycles trained on whole, of 2,852 | 0 | 2,842 |
| Conversations of 13 to 24 cycles trained on whole, of 1,145 | 0 | 996 |
| Loader time per update on CPU | 406 ms | 488 ms |

| Episode length | Share of positions | Padding | Supervised share | Conversations per episode |
| --- | ---: | ---: | ---: | ---: |
| 4 cycles | 90.6% | 0.27% | 61.3% | 6.7 |
| 12 cycles | 4.7% | 0.10% | 16.1% | 5.3 |
| 24 cycles | 4.7% | 0.05% | 13.3% | 6.7 |

The 24-cycle episodes also absorbed 826 of the 5-to-12-cycle conversations.
149 conversations of 13 to 24 cycles were still waiting when the epoch ended, because two 24-cycle episodes per update is slightly fewer than they arrive.
These are the earlier replay's coverage and queue figures.
The November 16 [loss-weighting investigation](RLM_SFT_LOSS_WEIGHTING.md#data-and-procedure) separately counts unique examples with at least one trained target and distinguishes them from processed occurrences when the loader crosses the epoch boundary.

### Speed of the mixed episodes

Forward and backward of one microbatch were timed on the DGX Spark with the published SFT checkpoint, real SFT conversations, BF16, activation checkpointing, and the compiled cycle forward.

| Microbatch | Text tokens | Positions | Seconds | Text tokens per second | Cost per token against 8 × 4 | Peak CUDA allocated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 episodes × 4 cycles | 32,736 | 57,344 | 6.72 | 4,868 | 1.00 | 24.5 GB |
| 2 episodes × 12 cycles | 24,552 | 47,104 | 6.68 | 3,674 | 1.33 | 18.1 GB |
| 1 episode × 24 cycles | 24,552 | 48,128 | 8.51 | 2,885 | 1.73 | 19.5 GB |

The longer episodes cost more per token because more of their cycles carry both memory and summary positions, and because one or two episodes per microbatch use the GPU less efficiently than eight.
They need less memory than the four-cycle microbatch.

For one optimizer update of 1,024 chunks:

| Measurement | 32 microbatches of 8 × 4 | `4x8:29,12x2:2,24x1:2` | Change |
| --- | ---: | ---: | ---: |
| Forward and backward time, in 8 × 4 microbatches | 32.0 | 33.6 | +4.9% |
| Supervised text targets, against the original packer | 586,514 | 596,555 | +1.7% |
| Updates in one epoch, against the original packer | 504 | 498 | -1.2% |
| Forward and backward time for one epoch | 504 × 32.0 | 498 × 33.6 | +3.7% |
| Loader time on CPU | 0.41 s | 0.49 s | +0.08 s |

The mixed episodes therefore slow an update by about 5% and an epoch by about 4%, and the extra loader time is negligible against an update of more than 90 seconds.
The 24-cycle microbatch accounts for most of the cost.
In this measurement the three training shapes needed nine compiled graphs, more than PyTorch's default recompile limit of eight.
With the default limit the 24-cycle shape silently ran uncompiled and took 9.70 seconds instead of 8.51.
The trainer now raises `torch._dynamo.config.recompile_limit` to at least `8 * number_of_shapes`, preserving larger existing values, to cover initial, first recurrent, later recurrent, and final cycles in both training and validation.
The first memory handoff can have a different tensor stride from later handoffs, requiring a separate compiled variant.
These timings are from one GB10 GPU, one batch per shape, and five timed repetitions after warm-up, without the optimizer step; the ratio on H100s has not been measured.

## Plan

Done:

- Pretraining: crop over-long documents on arrival in the best-fit packer, as the packer's default.
- SFT loader: episode lengths chosen by the required `episode_mix`, each conversation assigned to the shortest length it fits, long episodes filled with shorter conversations, and lengths mixed inside every optimizer update.
- SFT trainer: consume one complete mix per update through required `--episode-mix`, derive cycle counts from batch widths, and require the mix to match the text-position budget.
- SFT objectives: require `--loss-weighting {microbatch,group-token}`, implement both reductions, and record the optimized loss, global token mean, and cycle-group metrics separately.
- Compilation: raise the recompile limit for the mixed shapes in training and validation.

Remaining:

- Hold out some long conversations and score them with intact and shuffled memory, to tell whether long-range retention was learned.

## Open questions

- **Loss normalisation in SFT.** Both equal microbatch means and explicit group weights with token normalization are implemented, but neither has demonstrated better model quality; equal-microbatch averaging remains the recommended first comparison baseline. The November 16 replay measures about 3,954 supervised predictions per two-episode 12-cycle microbatch and 3,256 per one-episode 24-cycle microbatch, including EOM readouts. The earlier figure of about 185 describes one delayed answer, not a packed episode, and target counts do not establish gradient shares. See the [design discussion and measured weighting tradeoff](RLM_SFT_LOSS_WEIGHTING.md).
- **Capacity and speed on H100s.** On the Spark the 12- and 24-cycle microbatches need less memory than the four-cycle one, without optimizer state resident. Neither memory nor the speed ratio has been measured on the H100s.
- **The 22 conversations beyond 24 cycles.** The November 16 scan finds that cropping leaves no answer in 21 and retains 190 targets from a partial answer in one. None fits whole within the selected episode lengths.
- **Training signal.** Pretraining never ran more than four cycles, and about 4,000 conversations would teach longer retention; whether one SFT epoch is enough is unknown.
- **Unused long conversations.** With two 24-cycle episodes per update, 13% of the conversations of 13 to 24 cycles are not reached in one epoch; a different mix changes this.

## How the numbers were measured

- **Pretraining packer:** a length-only replay of the packer algorithm over the 170 training shards, tokenized with the run's tokenizer, for 100,000 four-cycle episodes with the first 25,000 excluded. The chunk table comes from an earlier 60,000-episode replay. The pretraining run was about 1.4 million episodes, so end-of-run figures may differ somewhat.
- **Logged handoff figures:** the last 500 logged updates of the Spark run `pretrain-u4-b32-frozen-20261010`.
- **SFT packer:** the actual loader function replayed on CPU with a 4,092-token row for 88,000 episodes, compared with `sft/updates.jsonl` from the published SFT run.
- **Conversation lengths:** the three length columns in the mixture table use 6,000 sampled conversations per source. Counts of over-long conversations and the cycle table use all 460,341 SmolTalk conversations. Mixture totals combine exact SmolTalk counts with MMLU and GSM8K totals scaled from the samples.
- **Median assistant tokens kept by cropping, and median messages:** an 80,000-conversation SmolTalk sample.
- **Mixed-episode loader:** the loader function replayed on CPU over one epoch of the real mixture, 498 updates, next to the original packer over its 504 updates.
- **Speed:** the published SFT checkpoint at step 504 on the Spark's GB10 with PyTorch 2.14.1, three warm-up and five timed forward-and-backward repetitions per shape. The 24-cycle figure comes from a second process with the recompile limit raised, in which the four-cycle shape measured 6.57 seconds; the update comparison uses each shape's ratio to the four-cycle time from its own process.
- The replay and timing scripts were run from a scratch directory and are not part of the repository.

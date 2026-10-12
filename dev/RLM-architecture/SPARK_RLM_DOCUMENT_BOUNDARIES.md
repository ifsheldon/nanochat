# Pretraining document lengths and RLM memory boundaries

Measured on October 8, 2026.
The packed-boundary measurements below use the original 1024-token chunks without `<EOM>`; they have not been rerun for the current 1023-token layout.
The separate full-corpus document-length census is independent of that recurrent layout.

An exact census of all **14,386,176 documents in the 170 cached training shards** found that **8.894% exceed 1,024 content tokens**.
Those documents contain **35.05% of the corpus's content tokens**.
About one in four documents contributing text to the sampled pretraining batches crosses the memory boundary.
The average spill is approximately 356 next-token targets per crossing document.
These spill targets account for approximately 17.3% of all supervised positions, or 34.7% of second-cycle positions.

## Full-corpus document lengths

This census covers every document in the 170 cached training shards under `~/.cache/nanochat/spark-d24/base_data_climbmix`, excluding the separate validation shard.
It describes the locally cached training corpus, not any additional upstream shards that were not downloaded.
Document length is the number of content tokens produced by the training tokenizer **before adding BOS, packing documents, or cropping them**.
Each source document is counted once, regardless of how much of it the loader would retain.

### Counts above the requested thresholds

| Document length | Documents | Share of documents | Content tokens in these documents | Share of corpus content tokens |
| --- | ---: | ---: | ---: | ---: |
| **> 1,024 tokens** | **1,279,535** | **8.894%** | 3,205,435,633 | **35.05%** |
| **> 2,048 tokens** | **436,534** | **3.034%** | 2,033,584,630 | **22.24%** |
| **> 3,072 tokens** | **228,679** | **1.590%** | 1,518,525,209 | **16.61%** |
| **> 4,096 tokens** | **143,234** | **0.996%** | 1,217,099,944 | **13.31%** |

Thresholds are strictly greater than the stated length, and the rows overlap.
For example, a document of exactly 2,048 tokens appears in the > 1,024 row but not the > 2,048 row.
The token columns count all content tokens in qualifying documents, not only tokens beyond the threshold.
Counts are exact; displayed percentages are rounded.

### Non-overlapping length ranges

| Document length | Documents |
| --- | ---: |
| At most 1,024 tokens | 13,106,641 |
| 1,025–2,048 tokens | 843,001 |
| 2,049–3,072 tokens | 207,855 |
| 3,073–4,096 tokens | 85,445 |
| More than 4,096 tokens | 143,234 |
| **Total** | **14,386,176** |

### Length distribution

| Measurement | Content tokens |
| --- | ---: |
| Total across all documents | 9,144,802,375 |
| Mean per document | 635.67 |
| Minimum | 1 |
| 25th percentile | 242 |
| Median | 527 |
| 75th percentile | 702 |
| 90th percentile | 946 |
| 95th percentile | 1,483 |
| 99th percentile | 4,086 |
| Maximum | 517,602 |

There were no zero-token documents.
The distribution is strongly skewed: approximately 91.1% of documents fit within 1,024 content tokens, while the remaining 8.9% contain approximately 35.1% of the corpus's content tokens.

### Census method and verification

The scan read the text column from all 14,049 Parquet row groups using 20 CPU workers, with one tokenizer thread per worker and no GPU work.
Lengths were accumulated into an exact histogram, along with counts and token totals for each threshold.
The tokenizer's direct array-encoding path was checked against the training tokenizer wrapper on 1,024 corpus documents and Unicode and special-token-literal cases.
Inputs raising a Unicode encoding error used the training wrapper's handling.

Every training shard was counted exactly once, and scanned document counts matched the Parquet metadata.
Shard paths, sizes, and modification times were unchanged during the scan.
The non-overlapping bins sum to the full document count.
The tokenizer artifact's SHA-256 was `387cfc082b0bee45467774fd6f1310a922ad170886a58ccddcb468f275e06a6c`.

### Relationship to memory-boundary crossings

The 8.894% raw-document length rate and the approximately 24% packed-document crossing rate below measure different properties and use different denominators.
The census counts original source documents before packing or cropping; the loader analysis counts documents or fragments that contribute retained content to sampled packed episodes.
A document shorter than 1,024 tokens can cross the boundary when it begins partway through an episode.
Conversely, the full length of a long source document does not imply that all of its tokens become training targets, because the loader can crop its tail.

These token shares describe the source corpus, not the proportion of training loss that depends on memory.
The packed-loader analysis below measures the retained targets occurring after a same-document memory handoff.

## Packed-loader configuration and scope

The boundary analysis ran the actual [BOS best-fit loader](../../nanochat/dataloader.py) on CPU, using the existing tokenizer and cached training corpus under `~/.cache/nanochat/spark-d24`.
It did not train a model or modify the training data.

| Setting | Value |
| --- | --- |
| Recurrent cycles per episode | 2 |
| Text inputs per cycle | 1,024 |
| Text inputs per packed episode | 2,048 |
| Tokens assembled by the loader per episode | 2,049, including the extra shifted target |
| Memory slots in the model being analyzed | 512 |
| Cached training shards | 170; the separate validation shard was excluded |
| Analysis loader batch size | 64 episodes |
| Tokenizer threads | 1 |
| Tokenizer batch size / document buffer size | Loader defaults: 128 / 1,000 |
| BOS token ID | 32,759 |

Two sets of episode rows were measured:

1. **Actual pilot data:** the first 3,072 packed episodes from the beginning of the training loader, reproducing the six-update d24 pilot's 6,291,456 supervised positions.
2. **Broader sample:** 2,048 measured episodes from each of training shards 0, 24, 48, 72, 96, 120, 144, and 169, after discarding 1,024 initial episodes at each starting point to reduce document-buffer startup effects.
   This gives 16,384 measured episodes and 33,554,432 supervised positions.

The broader sample is an estimate from eight starts spread across the cached training shards, not a census of the full corpus.
Its shard-0 portion overlaps the pilot data, so the two columns below are not independent samples.
Packing is independent of the analysis batch size: the loader fills rows sequentially, so changing the number of rows returned together does not change their contents.

## Counting definitions

A **document contributing text** has at least one content token retained in a packed row.
Fragments containing only BOS are excluded from the document denominator.

A **crossing document** has at least one content input token on both sides of the 1,024-position memory boundary.
Cases with only that document's BOS before the boundary are excluded because no earlier content from that document entered memory.

The **spill length** is the number of same-document next-token targets predicted after the memory handoff.
The average spill is conditional on crossing documents, rather than averaged over every document in the batch.
Some crossing documents have zero spill targets: their only content token in the second chunk is the boundary token, which was already predicted by the first cycle.

For a zero-indexed packed row `r` of length 2,049:

- Cycle 1 consumes `r[0:1024]` and predicts `r[1:1025]`.
- Cycle 2 consumes `r[1024:2048]` and predicts `r[1025:2049]`.
- For a crossing document whose exclusive end position is `end`, its spill length is `max(0, end - 1025)`.

Thus, a document containing **1,500 content tokens**, starting at the beginning of the episode, has BOS at position 0 and content at positions 1 through 1,500.
Cycle 1 predicts through content token 1,024, and cycle 2 predicts content tokens 1,025 through 1,500.
Its spill is therefore **476 targets**, matching the motivating example.

## Aggregate measurements

| Measurement | Actual six-update pilot | Broader sample |
| --- | ---: | ---: |
| Packed episodes | 3,072 | 16,384 |
| Inserted documents/fragments, including BOS-only fragments | 12,442 | 69,140 |
| BOS-only fragments | 271 | 1,185 |
| Documents contributing content | 12,171 | 67,955 |
| Crossing documents | 3,062 | 16,332 |
| Crossing documents / documents contributing content | **25.16%** | **24.03%** |
| Episodes containing a crossing document | **99.67%** | **99.68%** |
| Crossing documents with at least one spill target | 3,059 | 16,302 |
| All supervised positions | 6,291,456 | 33,554,432 |
| Spill targets | 1,089,660 | 5,819,269 |
| Average spill per crossing document | **355.87** | **356.31** |
| Average spill among documents with positive spill | 356.21 | 356.97 |
| Median spill per crossing document | 349 | 345 |
| Spill targets / all supervised positions | **17.32%** | **17.34%** |
| Spill targets / second-cycle supervised positions | **34.64%** | **34.69%** |

The supervised-position denominators retain BOS targets, matching the trainer's token accounting.
The broader sample contains approximately 4.15 documents contributing content per episode.
This explains why almost every episode has a crossing document while only about one quarter of documents cross.

## Spill distribution

The broader sample's crossing-document spill lengths have these percentiles:

| Percentile | Spill targets |
| --- | ---: |
| 10th | 86 |
| 25th | 199 |
| 50th | 345 |
| 75th | 460 |
| 90th | Approximately 647 |
| 95th | 827 |
| 99th | 979 |

| Spill range | Crossing documents in broader sample |
| --- | ---: |
| 0 | 30 |
| 1–128 | 2,471 |
| 129–256 | 3,070 |
| 257–512 | 7,874 |
| 513–768 | 1,816 |
| 769–1,024 | 1,071 |

## Results by sampled shard

Each row below contains 2,048 measured episodes after the 1,024-episode startup exclusion.

| Starting training shard | Documents contributing content | Crossing documents | Average spill per crossing document |
| --- | ---: | ---: | ---: |
| 0 | 8,518 | 2,039 | 349.32 |
| 24 | 8,524 | 2,041 | 361.85 |
| 48 | 8,469 | 2,041 | 352.14 |
| 72 | 8,547 | 2,043 | 359.14 |
| 96 | 8,519 | 2,044 | 359.32 |
| 120 | 8,522 | 2,039 | 357.80 |
| 144 | 8,454 | 2,043 | 348.74 |
| 169 | 8,402 | 2,042 | 362.17 |

## Verification against the pilot

The replay was checked against the recorded counters in `~/.cache/nanochat/spark-rlm/learning-6-20261008/result.json`.
It exactly reproduced the trainer's 3,064 handoffs with a non-BOS token at the start of cycle 2 and 1,094,236 input-continuation positions before the next BOS.

Those existing counters differ slightly from this report's definitions.
The trainer counts input positions and includes cases with only BOS from the crossing document before the boundary.
This report instead counts shifted prediction targets and requires earlier same-document content.
An independent calculation using document IDs assigned by cumulative BOS counts reproduced the pilot's **1,089,660 spill targets**.
The 1,500-content-token example was also checked and yielded exactly 476 spill targets.

## Interpretation

Packing determines crossings, so document length alone is insufficient to estimate them.
In the pilot, **2,293 of 3,062 crossing documents, or 74.89%, contained at most 1,024 retained content tokens**.
Such documents cross because they start partway through the packed episode.

These statistics measure opportunities to use memory for earlier same-document context.
They do not establish that every spill prediction requires memory: the model also sees the local second-chunk prefix, which may be sufficient for many predictions.
Conversely, a high episode-crossing rate does not mean that most loss positions train same-document retention: only about 17.3% of all supervised positions are in the measured spill.

The packed-loader measurements cover only text retained by the current loader.
When a document is cropped to finish a row, its remaining tail is discarded rather than continued into another episode.
The boundary analysis does not count hypothetical predictions on those discarded tails or infer useful retention from boundary placement alone.

For the model and training workflow, see [Recurrent latent memory on DGX Spark](../spark-training/SPARK_RLM.md).

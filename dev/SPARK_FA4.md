# FA4 pretraining comparison on DGX Spark

This experiment compares full-context and sliding-window attention across SDPA/FA4 and BF16/FP8 linear layers on the single GB10 GPU.
It starts fresh D24 models using the existing tokenizer and ClimbMix data; the paused production run and its final evaluation remain separate.

## Install and select attention

```bash
uv sync --extra gpu --extra fa4 --group dev
```

The `fa4` extra pins `flash-attn-4==4.0.0b34` with its dependencies in `uv.lock`.
On this ARM64 environment, `uv pip check` reports a pre-existing metadata defect in `nvidia-cusparselt-cu13==0.8.1`: the aarch64 wheel's internal tag says `manylinux2014_sbsa`.
Its library is an aarch64 ELF binary, and the official PyTorch CUDA 13.2 index advertises the same wheel hash as PyPI; changing indexes does not correct the metadata.
Set `NANOCHAT_ATTENTION_IMPL=fa4` to require FA4; unavailable packages, unsupported hardware, and unsupported compute dtypes fail instead of silently switching to SDPA.
The default `auto` retains the existing FA3/SDPA selection, and `sdpa` explicitly bypasses Flash Attention discovery.
FA4 training requires CUDA compute capability major 9, 10, 11, or 12 and BF16 or FP16 compute with this release; GB10 capability 12.1 is the locally validated target.
The adapter normalizes unlimited window sizes and unwraps FA4's output/LSE pair.
Cached inference appends to nanochat's existing KV cache and invokes dense FA4 with bottom-right causal alignment; it does not use a specialized FA4 decode kernel.
The pinned FA4 release works with the training script's `torch.compile(dynamic=False)` through graph breaks; `fullgraph=True` was not supported in the local validation.

Pretraining `--fp8` uses nanochat's existing tensorwise FP8 linear layers.
All eight cases below use `NANOCHAT_DTYPE=bfloat16`, so attention stays BF16 even in the FP8 cases.
This experiment does not test FP8 attention or establish training quality for either precision.

## Run the comparison

```bash
NANOCHAT_RUN_NAME=spark-d24 \
NANOCHAT_TRAIN_SHARDS=170 \
NANOCHAT_CUDA_MEMORY_LIMIT_GB=80 \
NANOCHAT_MIN_SYSTEM_AVAILABLE_GB=16 \
uv run --no-sync python -m scripts.benchmark_pretrain
```

Prepare the selected run tokenizer and the first 170 shared ClimbMix shards before starting.
New benchmark outputs default to `training_run/spark-d24/benchmarks/<UTC timestamp>/`, including all logs and results.
Pass `--output-dir` to choose a different new output directory.
Datasets default to the project `datasets/` directory; `NANOCHAT_DATASETS_DIR` can select an explicit location.
Existing benchmark outputs and production checkpoints remain at their historical paths.
See [project storage](PROJECT_STORAGE.md) for the shared data and run directory contract.
The runner uses a fresh process for each of the eight combinations and writes per-case logs/JSON, aggregate `results.json`, the Git revision/diff, and copies of the benchmark and attention source.
It continues to the next combination after a failed case and exits nonzero if any case failed.
It never reduces an individual case's batch size after OOM.
To repeat the entire matrix with a different common batch, pass `--device-batch-size=4` or another positive divisor of 512 and choose a new output directory.

Every case uses 24 layers, width 1536, 12 heads with head dimension 128, vocabulary 32,768, sequence length 2,048, seed 42, the same initial data order, and global batch 1,048,576 tokens.
The default device batch 8 gives 64 accumulation microbatches per optimizer update.
`L` means full-context attention in every layer.
`SSSL` means 18 short-window layers and 6 full-context layers; the short window includes up to 512 preceding tokens plus the current token.

The original target tokens-to-scaling-parameters ratio of 8 gives 5,568 pretraining updates, or 5,838,471,168 tokens.
Benchmark mode preserves that horizon and all learning-rate, momentum, and weight-decay schedules while stopping after two optimizer updates.
Update one warms up compilation and initializes optimizer state.
Only update two supplies the speed result; it includes all forward/backward microbatches, data packing/prefetch, optimizer work, and gradient reset, with CUDA synchronization at both timing boundaries.
Evaluation, sampling, checkpoint writes, and W&B are disabled.
Warmup duration is recorded separately to account for total experiment runtime.
The reported loss is the last microbatch's unsmoothed loss, not a validation score or batch average.

`compile_stats_delta` records PyTorch Dynamo graph-capture counters during update two.
A nonzero `unique_graphs` value means new graphs were captured during the measured update, so the result may include compilation overhead.
These counters do not prove that all lower-level kernel compilation is absent.
One measured update per case is the requested quick comparison, not a distribution of steady-state timings.

## Historical timing reference

The prior full-context SDPA/BF16 D24 run completed 5,568 updates.
Across its 5,557 counted updates after the initial eleven, the mean was 129.33 seconds, median 92.65 seconds, and range 91.40–324.84 seconds.
Changing GPU load makes the historical mean an uncertain estimate for an idle comparison.
Sixteen updates at that historical mean would take 34.49 minutes before process setup and compilation, and explicit-mask SDPA sliding windows add work.
The initial estimate for the complete eight-case comparison was 45–90 minutes including warmup and setup.

## Batch-16 attempt on October 8, 2026

At the production device batch of 16, full-context SDPA/BF16 completed its measured update in 91.77 seconds with 63.12 GB peak reserved CUDA memory.
SDPA/FP8 took 80.70 seconds with 55.57 GB peak reserved memory.
FA4/BF16 completed warmup but exhausted the 80 GB allocator budget during backward in update two, while requesting another 4 GiB allocation.
The matrix was stopped during the following FA4/FP8 warmup, and every case was restarted at the common device batch of 8 with the same global token batch.
The batch-16 artifacts, including failure and interruption records, are preserved under `~/.cache/nanochat/experiments/d24-attention-20261008`.
These batch-16 measurements are separate from the complete batch-8 comparison.

## Complete batch-8 results on October 8, 2026

All eight cases completed one warmup and one measured update at device batch 8, 64 accumulation microbatches, and 1,048,576 tokens per update.
The environment was NVIDIA GB10 capability 12.1, PyTorch 2.14.0+cu132, Triton 3.8.0, FA4 4.0.0b34, CUTLASS DSL 4.8.0, and quack-kernels 0.6.5.

| Window | Attention | Linear precision | Measured seconds/update | Tokens/second | Peak reserved CUDA GB |
| --- | --- | --- | ---: | ---: | ---: |
| Full (`L`) | SDPA | BF16 | 96.64 | 10,850 | 39.64 |
| Full (`L`) | SDPA | FP8 | 88.17 | 11,893 | 35.05 |
| Full (`L`) | FA4 | BF16 | 114.97 | 9,120 | 49.46 |
| Full (`L`) | FA4 | FP8 | 107.51 | 9,753 | 42.75 |
| Sliding (`SSSL`) | SDPA | BF16 | 144.99 | 7,232 | 39.64 |
| Sliding (`SSSL`) | SDPA | FP8 | 135.83 | 7,720 | 35.13 |
| Sliding (`SSSL`) | FA4 | BF16 | 114.22 | 9,180 | 49.44 |
| Sliding (`SSSL`) | FA4 | FP8 | 105.83 | 9,908 | 42.71 |

FA4 reduced sliding-window update time by 21.2% in BF16 and 22.1% with FP8 linear layers compared with matching SDPA cases.
For full context, FA4 took 19.0% longer in BF16 and 21.9% longer with FP8 linear layers.
The fastest measured configuration was full-context SDPA with FP8 linear layers at 88.17 seconds per update.
Changing the attention window also changes the model's computation, so speed alone does not establish an equivalent quality outcome.

The successful matrix took 2,124.61 seconds, or 35.41 minutes: 19.69 minutes in warmup updates, 15.14 minutes in measured updates, and approximately 35 seconds in process setup and teardown.
Elapsed time from the initial batch-16 attempt through completion of the batch-8 matrix was 2,893.47 seconds, or 48.22 minutes, excluding code integration and subsequent regression tests.
All measured updates had finite logged losses and zero additional Dynamo `calls_captured` and `unique_graphs` counters.
The single-update results do not measure timing variance or establish long-run numerical stability or model quality.

Raw results, logs, and source records are under `~/.cache/nanochat/experiments/d24-attention-20261008-b8`, with all eight records in `results.json`.
The production base checkpoint at update 5,568, SFT checkpoint at update 574, and paused HumanEval evaluation were not resumed or modified by this experiment.

## Verification

```bash
NANOCHAT_ATTENTION_IMPL=sdpa OMP_NUM_THREADS=1 uv run --no-sync python -m pytest -q
uv lock --check
bash -n runs/runspark.sh runs/runspark_d24.sh
git diff --check
```

The final suite passed with 109 tests passed and 10 FA3-dependent tests skipped.
The five real FA4 GPU cases checked full/local attention output and Q/K/V gradients against FP32 SDPA, plus cached prefill, chunked input, and decoding.
The benchmark regression exercised the actual training loop and optimizer on a tiny CPU model, verifying exactly two accumulated updates, the original learning-rate schedule, and no evaluation or checkpoint side effects.
Lockfile, shell syntax, and whitespace checks passed.
FA4/CUTLASS emitted upstream deprecation and JIT argument warnings during successful GPU tests; the separate `uv pip check` metadata exception is described above.

# Phase 3 — Ablation Study & Detailed Benchmark Results

> All experiments run at 32 concurrent users unless otherwise stated.
> Benchmark tool: `vllm.benchmarks.benchmark_serving` · Dataset: ShareGPT
> 500 prompts per run · 50-request warmup · Single NVIDIA A40 (48GB VRAM)

---

## Fixed Baseline Configuration (Phase 2 defaults)

```
--gpu-memory-utilization 0.90
--max-model-len 32768
(all other flags: vLLM defaults)
```

Baseline result at 32 concurrent users:

```
TTFT p50:   420 ms
TTFT p95:  1150 ms
TTFT p99:  1850 ms
TPOT p50:  24.2 ms/tok
TPOT p95:  38.5 ms/tok
Throughput: 420.5 tok/sec
KV Cache:   94.2% peak
Swapped:    14 requests
```

---

## Experiment 1 — `--enable-chunked-prefill`

**Hypothesis:** Prefill stalls are the primary driver of high p95/p99 TTFT at this
concurrency level. Chunked prefill will collapse tail latency without significantly
affecting p50 or TPOT.

*All other flags held at baseline.*

| Config | TTFT p50 | TTFT p95 | TPOT p50 | Throughput |
|---|---|---|---|---|
| `False` (baseline) | 420 ms | 1,150 ms | 24.2 ms/tok | 420.5 tok/sec |
| **`True` (tuned)** | **210 ms** | **440 ms** | **20.1 ms/tok** | **612.3 tok/sec** |
| **Delta** | **-50.0%** | **-61.7%** | **-16.9%** | **+45.6%** |

**Finding:** Hypothesis confirmed. p95 TTFT dropped by 61.7% with no TPOT degradation.
Long RAG prompts no longer block active decode streams. Throughput also improved because
the scheduler can now interleave work more efficiently than the stop-the-world prefill
allowed. This is the single highest-impact change in the entire ablation study.

---

## Experiment 2 — `--kv-cache-dtype`

**Hypothesis:** FP16 KV cache is exhausting available VRAM headroom under 32-user load,
causing the 14 swap events observed in baseline. FP8 will halve KV memory consumption
and eliminate swapping.

*Baseline + chunked prefill enabled.*

| Config | KV Cache VRAM | Peak KV Usage | Swap Events | Throughput |
|---|---|---|---|---|
| `auto` (FP16, baseline+CP) | ~14.8 GB | 88.3% | 4 requests | 612.3 tok/sec |
| **`fp8` (tuned)** | **~7.4 GB** | **54.9%** | **0 requests** | **745.8 tok/sec** |
| **Delta** | **-50.0% VRAM** | **-33.4pp** | **Eliminated** | **+21.8%** |

**Finding:** Hypothesis confirmed. FP8 KV cache freed ~7.4GB of VRAM, reducing peak
KV cache usage from 88.3% (still with some residual swaps even after chunked prefill)
to 54.9%, eliminating all swap events. Throughput increased further because the
scheduler no longer needs to pause and handle eviction/reload cycles for swapped sequences.

> **Note:** 4 swap events remained even after enabling chunked prefill (vs 14 at baseline)
> because chunked prefill addresses compute scheduling but not memory capacity. FP8 KV cache
> addresses memory capacity. Both changes are necessary — neither alone eliminates swapping
> at 32-user load with this model and context length.

---

## Experiment 3 — `--max-num-seqs`

**Hypothesis:** 256 (default) allows too many sequences into the batch simultaneously,
causing scheduler overhead and KV cache pressure during burst periods. 128 will provide
better sustained throughput without queuing at 32-user load.

*Baseline + chunked prefill + FP8 KV cache.*

| Max Seqs | TTFT p95 | TPOT p95 | GPU Compute Util | Queued Requests |
|---|---|---|---|---|
| `256` | 510 ms | 31.2 ms/tok | 98.4% | 0 |
| **`128` (optimal)** | **380 ms** | **24.4 ms/tok** | **92.1%** | **0** |
| `64` | 360 ms | 22.8 ms/tok | 74.5% | 18 |

**Finding:** 128 is the sweet spot. At 256, GPU utilization at 98.4% indicates
the scheduler is thrashing — too many sequences competing for KV cache blocks
simultaneously, causing fragmentation even with FP8 KV cache. At 64, GPU compute
drops to 74.5% (wasted capacity) and 18 requests queue unnecessarily. At 128,
GPU runs at 92.1% with zero queuing — near-optimal saturation without thrashing.

---

## Experiment 4 — `--block-size`

**Hypothesis:** Larger block size better aligns PagedAttention memory access with
A40's 128-byte L2 cache line width, improving HBM bandwidth utilization during decode.

*Baseline + chunked prefill + FP8 KV cache + max-num-seqs=128.*

| Block Size | TTFT p50 | TPOT p50 | Memory BW Util (DCGM) |
|---|---|---|---|
| `16` (default) | 195 ms | 19.8 ms/tok | 78.2% |
| **`32` (optimal)** | **180 ms** | **18.1 ms/tok** | **84.6%** |
| **Delta** | **-7.7%** | **-8.6%** | **+6.4pp** |

**Finding:** Confirmed improvement, smallest absolute impact of all four changes.
Higher memory bandwidth utilization (78.2% → 84.6%) confirms better alignment
with HBM access patterns — the GPU is doing the same decode work but fetching
KV cache values more efficiently from VRAM. Meaningful for a low-risk one-flag change.

---

## Final Best-Combination Run

All four optimal settings applied simultaneously:

```
--gpu-memory-utilization 0.92
--max-num-seqs 128
--block-size 32
--enable-chunked-prefill True
--max-num-batched-tokens 2048
--kv-cache-dtype fp8
```

### Result at 32 concurrent users

```
================================================================================
vLLM BENCHMARK — FINAL TUNED CONFIGURATION
================================================================================
Model:      sandipsingh2007/mistral-nemo-12b-instruct-int8-w8a8
Load:       32 concurrent users · ShareGPT · 500 prompts
--------------------------------------------------------------------------------
TTFT p50:   180.24 ms
TTFT p95:   380.12 ms
TTFT p99:   520.45 ms

TPOT p50:   18.12 ms/tok  →  55.18 tok/sec per user
TPOT p95:   24.41 ms/tok  →  40.96 tok/sec per user
--------------------------------------------------------------------------------
Total benchmark time:      63.67 seconds
Total input tokens:        128,450
Total output tokens:       50,000
Output throughput:         785.23 tok/sec (aggregate)
Request throughput:        4.01 req/sec
--------------------------------------------------------------------------------
Peak VRAM allocated:       44.16 GB / 48.00 GB  (92.0%)
Peak KV cache usage:       68.4%
CPU swap events:           0
================================================================================
```

### Concurrency sweep profile (tuned config)

| Concurrency | TTFT p50 | TTFT p95 | TPOT p95 | Throughput |
|---|---|---|---|---|
| 1 | 95 ms | 130 ms | 19.2 ms/tok | 48.3 tok/sec |
| 8 | 120 ms | 195 ms | 20.8 ms/tok | 210.4 tok/sec |
| 16 | 145 ms | 260 ms | 22.1 ms/tok | 445.7 tok/sec |
| **32** | **180 ms** | **380 ms** | **24.4 ms/tok** | **785.2 tok/sec** |
| 64 | 310 ms | 620 ms | 29.1 ms/tok | 890.4 tok/sec |

**Reading this table:** Throughput increases with concurrency (continuous batching amortizing
memory bandwidth cost across more simultaneous decode work), but TTFT and TPOT grow too.
The 32-concurrency point is the practical operating point for this hardware — 785 tok/sec
aggregate throughput while keeping TTFT p95 under 400ms and TPOT p95 under 25ms. At 64
concurrent users, throughput increases further but TTFT p95 at 620ms may exceed enterprise
SLA targets for interactive use cases.

---

## Comparison: Baseline vs Tuned — Full Percentile Table

| Metric | Baseline p50 | Baseline p95 | Baseline p99 | Tuned p50 | Tuned p95 | Tuned p99 |
|---|---|---|---|---|---|---|
| TTFT (ms) | 420 | 1,150 | 1,850 | 180 | 380 | 520 |
| TPOT (ms/tok) | 24.2 | 38.5 | 52.1 | 18.1 | 24.4 | 30.2 |
| E2E latency (ms) | 5,140 | 8,620 | 12,900 | 3,720 | 5,210 | 6,840 |

**The most important number in this table is TTFT p99.**
1,850ms → 520ms is a 71.8% improvement in worst-case first-token latency.
For a user submitting a RAG query that includes a large document chunk,
this is the difference between a system that feels responsive and one that
feels broken. The improvement comes almost entirely from chunked prefill —
not from the memory or batch-size changes.

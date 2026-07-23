# Phase 3 — vLLM Parameter Tuning & Performance Benchmarking

## What this phase does

Executes a systematic, one-parameter-at-a-time ablation study on the vLLM inference
engine running on a single **NVIDIA A40 (48GB VRAM)**, tuning key serving parameters
to establish a high-throughput, low-latency production profile. Results are captured
via Prometheus/Grafana/DCGM and compared against the Phase 2 baseline.

---

## Results Summary

| Metric | Baseline (Phase 2 defaults) | Tuned (Phase 3 optimal) | Delta |
|---|---|---|---|
| **TTFT p50** | 420 ms | 180 ms | **-57.1%** |
| **TTFT p95** | 1,150 ms | 380 ms | **-66.9%** |
| **TTFT p99** | 1,850 ms | 520 ms | **-71.8%** |
| **TPOT p50** | 24.2 ms/tok | 18.1 ms/tok | **-25.2%** |
| **TPOT p95** | 38.5 ms/tok | 24.4 ms/tok | **-36.6%** |
| **Output throughput** | 420.5 tok/sec | 785.2 tok/sec | **+86.7%** |
| **Request throughput** | 2.15 req/sec | 4.01 req/sec | **+86.5%** |
| **Peak KV cache usage** | 94.2% | 68.4% | **-27.3% headroom gain** |
| **CPU memory swapping** | 14 requests | 0 requests | **Eliminated** |

> All numbers measured at **32 concurrent users**, ShareGPT dataset,
> 500 requests per run, 50-request warmup pass.

---

## Tuned Parameter Set

| Parameter | Default | Optimal | Effect |
|---|---|---|---|
| `--gpu-memory-utilization` | `0.90` | `0.92` | Slightly larger KV cache pool |
| `--max-num-seqs` | `256` | `128` | Prevents batch thrashing on A40 |
| `--block-size` | `16` | `32` | Better HBM memory alignment |
| `--enable-chunked-prefill` | `False` | `True` | Eliminates prefill stalls |
| `--max-num-batched-tokens` | `2048` | `2048` | Unchanged (already optimal) |
| `--kv-cache-dtype` | `auto` | `fp8` | Halves KV cache VRAM, eliminates swapping |

---

## Files in this folder

| File | Purpose |
|---|---|
| `benchmark_suite.sh` | Automated benchmark harness — concurrency sweep across 1, 8, 16, 32, 64 |
| `docs/PHASE3_COMPLETE_REFERENCE.md` | Full parameter explanations, tuned launch command, methodology |
| `docs/PHASE3_BENCHMARK_RESULTS.md` | Per-parameter ablation tables, Grafana observations, final profile |

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/codefordba/AI-Infrastructure-Engineering
cd AI-Infrastructure-Engineering/phase3-vllm-tuning

# 2. Launch vLLM with tuned parameters
# NOTE: Use --runtime nvidia pattern (not --gpus all) for snap-packaged Docker
docker run -d \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --name vllm_gpu_worker_tuned \
  --restart always \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model sandipsingh2007/mistral-nemo-12b-instruct-int8-w8a8 \
  --quantization compressed-tensors \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --block-size 32 \
  --enable-chunked-prefill True \
  --max-num-batched-tokens 2048 \
  --kv-cache-dtype fp8 \
  --port 8000

# 3. Run the benchmark suite
bash benchmark_suite.sh
```

---

## Hardware Context

```
GPU:     NVIDIA A40 · 48GB VRAM · Ampere (sm_86)
VRAM allocation (tuned):
  ~13GB  — INT8 W8A8 model weights (compressed-tensors)
  ~25GB  — PagedAttention KV cache pool (FP8)
  ~10GB  — CUDA context + activation headroom
```

---

## Key Decisions

**Why `--enable-chunked-prefill=True` is the highest-impact change:**
Without it, vLLM treats each prefill as atomic — a 4K-token RAG context blocks
all active decode streams until it completes. This is why baseline TTFT p99 was
1,850ms even though p50 was only 420ms — the tail latency is entirely caused by
prefill-blocking events, not inherent model slowness. Chunked prefill co-schedules
prefill chunks alongside ongoing decode iterations, collapsing the tail.

**Why `--kv-cache-dtype=fp8`:**
At FP16, 14 requests were being evicted to CPU RAM (swapped) under 32-user load
because the KV cache pool was exhausted. FP8 halves each token's KV footprint,
doubling active token capacity from ~32K to ~64K tokens simultaneously in VRAM.
Swap count dropped from 14 to 0. This is the change that makes the system
actually production-safe at this concurrency level.

**Why reduce `--max-num-seqs` from 256 to 128:**
256 allows too many sequences into the batch simultaneously during spikes, which
saturates the KV cache pool and forces evictions before they're necessary. 128
enforces a predictable concurrency boundary that pairs correctly with A40 VRAM
at this model size — GPU compute utilization holds at 92% vs 98%+ (thrashing)
with 256.

**Why `--block-size=32` over 16:**
Larger block size aligns KV cache memory access patterns more closely with the
A40's 128-byte memory bus width, reducing the number of memory transactions
needed per attention fetch. 8.3% improvement in TPOT, low risk change.

---

## Lessons Learned

**1. Never tune multiple parameters simultaneously.**
Changing even two flags at once makes it impossible to isolate which change
caused a latency improvement or regression. The OAT (one-at-a-time) ablation
methodology used here is the only way to build defensible, reproducible results.

**2. Unchunked prefills destroy tail latency for RAG workloads.**
This is not a problem at all for short-prompt chatbot workloads. It's specific
to systems where prompts regularly include large retrieved context (2K-8K tokens).
`--enable-chunked-prefill` is mandatory in RAG serving stacks.

**3. FP8 KV cache is a safe win on Ampere hardware.**
Zero observable degradation in RAG retrieval scores or generation quality was
measured after enabling FP8 KV cache. The accuracy cost of E5M2 format for
attention key-value states is negligible relative to the memory benefit — this
is different from FP8 weight quantization, which requires more care.

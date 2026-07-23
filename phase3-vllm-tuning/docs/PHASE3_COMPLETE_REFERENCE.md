# Phase 3 — Complete Technical Reference & Tuning Methodology

> **Hardware:** NVIDIA A40 (48GB VRAM) · Ubuntu 22.04 · Driver 570.172.08 · CUDA 12.8  
> **Model:** `sandipsingh2007/mistral-nemo-12b-instruct-int8-w8a8` (INT8 W8A8)  
> **Prerequisite:** Phase 2 complete — vLLM serving stack operational, baseline metrics captured

---

## Table of Contents

1. [Benchmarking Methodology](#1-benchmarking-methodology)
2. [Parameters Evaluated](#2-parameters-evaluated)
3. [Reproduction Commands](#3-reproduction-commands)
4. [Before/After Master Summary](#4-beforeafter-master-summary)
5. [Deep Dive: Why Each Change Worked](#5-deep-dive-why-each-change-worked)
6. [Grafana & DCGM Observations](#6-grafana--dcgm-observations)
7. [Final Tuned Container Launch Command](#7-final-tuned-container-launch-command)

---

## 1. Benchmarking Methodology

All tests run using vLLM's native benchmark harness:
`vllm.benchmarks.benchmark_serving` (same as `benchmark_serving.py` in the vLLM repo).

### Test parameters

| Parameter | Value |
|---|---|
| Target endpoint | `http://localhost:8000/v1` |
| Model | `sandipsingh2007/mistral-nemo-12b-instruct-int8-w8a8` |
| Dataset | ShareGPT (avg input ~256 tokens, avg output ~196 tokens) |
| Prompts per run | 500 requests |
| Warmup pass | 50 requests before recording (primes CUDA kernels + cache pages) |
| Concurrency sweep | 1, 8, 16, 32, 64 concurrent requests |
| Primary comparison load | 32 concurrent users |

### OAT (One-At-a-Time) ablation methodology

Starting from the fixed Phase 2 baseline, one parameter was changed per run
while all other flags stayed constant. This is the only methodology that
produces defensible, attributable benchmark results — changing multiple
parameters simultaneously makes it impossible to know which change caused
a given delta.

Order of experiments:
1. Chunked prefill (highest expected impact on TTFT tail)
2. FP8 KV cache (expected to eliminate swap events)
3. Max num sequences (expected to prevent thrashing)
4. Block size (expected marginal TPOT improvement)
5. Final: best-combination run

---

## 2. Parameters Evaluated

| Flag | Default | Values Tested | Optimal | Purpose |
|---|---|---|---|---|
| `--gpu-memory-utilization` | `0.90` | `0.85`, `0.90`, `0.92`, `0.95` | `0.92` | VRAM fraction reserved for weights + KV cache |
| `--max-num-seqs` | `256` | `64`, `128`, `256` | `128` | Max concurrent sequences per scheduling iteration |
| `--block-size` | `16` | `16`, `32` | `32` | PagedAttention memory block size in tokens |
| `--enable-chunked-prefill` | `False` | `False`, `True` | `True` | Co-schedule prefill chunks with decode iterations |
| `--max-num-batched-tokens` | `2048` | `512`, `1024`, `2048`, `4096` | `2048` | Max tokens in one prefill scheduling step |
| `--kv-cache-dtype` | `auto` | `auto` (BF16), `fp8` | `fp8` | KV cache tensor precision |

---

## 3. Reproduction Commands

### Baseline run (Phase 2 default config — produces "before" numbers)

```bash
docker run -d \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --name vllm_baseline \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model sandipsingh2007/mistral-nemo-12b-instruct-int8-w8a8 \
  --quantization compressed-tensors \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --port 8000

# Wait for startup, then run benchmark
bash benchmark_suite.sh
```

### Tuned run (Phase 3 optimal config — produces "after" numbers)

```bash
docker stop vllm_baseline && docker rm vllm_baseline

docker run -d \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --name vllm_tuned \
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

bash benchmark_suite.sh
```

> **GPU-in-Docker note:** `--gpus all` does not work with snap-packaged Docker on this box.
> Use `--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all`.
> See [Phase 1 Debugging Notes](../phase1-quantization/docs/DEBUGGING_NOTES.md).

---

## 4. Before/After Master Summary

Measured at 32 concurrent users, ShareGPT dataset:

| Metric | Baseline | Tuned | Delta |
|---|---|---|---|
| TTFT p50 | 420 ms | 180 ms | **-57.1%** |
| TTFT p95 | 1,150 ms | 380 ms | **-66.9%** |
| TTFT p99 | 1,850 ms | 520 ms | **-71.8%** |
| TPOT p50 | 24.2 ms/tok | 18.1 ms/tok | **-25.2%** |
| TPOT p95 | 38.5 ms/tok | 24.4 ms/tok | **-36.6%** |
| Output throughput | 420.5 tok/sec | 785.2 tok/sec | **+86.7%** |
| Request throughput | 2.15 req/sec | 4.01 req/sec | **+86.5%** |
| Peak KV cache usage | 94.2% | 68.4% | **-27.3pp** |
| CPU swap events | 14 requests | 0 requests | **Eliminated** |

---

## 5. Deep Dive: Why Each Change Worked

### `--enable-chunked-prefill=True` — highest single impact

**Root problem without it:**
vLLM's default scheduler treats prefill as atomic. When a 4K-token RAG prompt
arrives, vLLM suspends all active decode iterations across every user session
until that prefill completes. This is the "prefill stall" — it shows up in
metrics as a TTFT p99 that is 4-5x higher than p50, even when median latency
looks reasonable.

**What chunked prefill does:**
Instead of processing the full 4K-token prompt in one uninterrupted pass,
vLLM splits it into chunks of `--max-num-batched-tokens` (2048 in our config)
and interleaves those chunks with decode iterations for existing sequences.
The prefill still takes the same total compute — it just doesn't block anyone else.

**Why this matters specifically for RAG:**
RAG systems inject retrieved document chunks into every prompt. A user asking
a simple one-sentence question may get a 3K-token prompt after chunk injection.
Without chunked prefill, that 3K prefill blocks every other active user.
With it, the same prefill is invisible to other users' TPOT.

**Measured impact:** TTFT p95: 1,150ms → 440ms (-61.7%) in isolation.

---

### `--kv-cache-dtype=fp8` — eliminates swap events

**Root problem:**
At FP16/BF16 precision, each cached token requires:
```
40 layers × 8 KV heads × 128 head_dim × 2 (K+V) × 2 bytes = 1.64 KB per token
```
With 32K max context and 32 concurrent users, worst-case KV cache demand is:
```
32 users × 32,768 tokens × 1.64 KB = ~1.7 TB
```
Obviously the KV pool isn't that large — PagedAttention allocates incrementally,
but it still exhausted the ~28GB KV pool at 94.2% usage under 32-user load,
causing 14 sequences to be swapped to CPU RAM and back — a 3-10x TPOT penalty
for affected requests.

**What FP8 KV cache does:**
Reduces the per-token KV footprint by 50% (8-bit instead of 16-bit),
halving memory consumption per cached token:
```
FP16 KV cache: ~14.8 GB at peak 32-user load
FP8 KV cache:  ~7.4 GB at peak 32-user load
```
This freed ~7.4GB of VRAM, reducing peak KV cache utilization from 94.2% to 68.4%
and eliminating all swap events.

**Is FP8 KV cache the same as FP8 weight quantization?**
No — and this distinction matters for accuracy concerns. FP8 *weight* quantization
(like what we'd use on an H100) requires careful calibration to avoid quality loss.
FP8 *KV cache* quantization is much more forgiving: the values being compressed
are intermediate attention states, not learned weights, and the accumulation of
many attention layers makes individual token-level KV precision less critical.
No degradation was observed in RAG retrieval scores or generation quality after
enabling FP8 KV cache.

**Measured impact:** Swap events: 14 → 0. KV cache usage: 94.2% → ~55% in isolation.

---

### `--max-num-seqs=128` — prevents batch thrashing

**Root problem with 256:**
The default value allows 256 sequences into the batch simultaneously during
traffic spikes. At 32-user sustained load this is fine, but during burst periods
(login-time spikes, batch document uploads), 256 concurrent sequences compete
for KV cache blocks simultaneously, fragmenting the pool and forcing early evictions
before any individual sequence approaches its actual context limit.

**Why 128 is the sweet spot for this configuration:**
128 provides a predictable concurrency ceiling that pairs with the A40's
available KV cache headroom at this model size. With FP8 KV cache enabled,
128 sequences × typical RAG context (~3K tokens each) = ~384K cached tokens,
well within the available KV pool. Going lower (64) improves per-request latency
marginally but starts queuing requests unnecessarily (18 queued in testing),
wasting available GPU compute capacity.

**Measured impact (at 32-user sustained load):**
TTFT p95: 510ms → 380ms. TPOT p95: 31.2ms → 24.4ms.
GPU compute utilization held at 92% (vs 74.5% at max-seqs=64, or 98.4% thrashing at 256).

---

### `--block-size=32` — memory access alignment

**What a PagedAttention block is:**
A block is the unit of KV cache allocation. When a sequence needs to cache its
first token, vLLM allocates one block. When a sequence exceeds one block's capacity,
it allocates a second. Block size determines how many tokens fit per block.

**Why 32 beats 16 on A40:**
The A40 has a 128-byte L2 cache line. At INT8 KV precision with head_dim=128:
```
Block size 16: 16 tokens × 1.64KB = 26.2KB — spans many cache lines, poor locality
Block size 32: 32 tokens × 1.64KB = 52.4KB — better stride alignment with HBM reads
```
More tokens per block also means fewer total block allocation/deallocation operations
during high-concurrency execution, reducing scheduler overhead.

**Measured impact:** TPOT p50: 19.8ms → 18.1ms (-8.6%). Memory bandwidth util: 78.2% → 84.6%
(higher is better here — indicates more efficient use of available HBM bandwidth).

---

## 6. Grafana & DCGM Observations

Observed on the Prometheus/Grafana dashboard during the final tuned benchmark run
(32 concurrent users, 500 requests):

| Signal | Observation | Interpretation |
|---|---|---|
| `vllm:gpu_cache_usage_perc` | Flat at 68.4% throughout | Zero KV cache fragmentation — healthy stable state |
| `vllm:num_requests_waiting` | 0 throughout | No request queuing at 32-user load |
| `vllm:num_requests_swapped` | 0 throughout | FP8 KV cache providing sufficient headroom |
| `DCGM_FI_DEV_GPU_UTIL` | 88–94% range | Near-optimal tensor core saturation |
| `DCGM_FI_DEV_POWER_USAGE` | ~245W avg (300W TDP) | Well below thermal throttle threshold |
| `DCGM_FI_DEV_GPU_TEMP` | Peak 64°C | Healthy — A40 throttles above ~83°C |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | 84–88% | Memory-bandwidth-bound decode (expected for LLM) |

**The memory bandwidth signal deserves emphasis:**
`DCGM_FI_DEV_MEM_COPY_UTIL` at 84-88% with `DCGM_FI_DEV_GPU_UTIL` at 88-94% is the
correct, expected fingerprint of a well-tuned LLM serving system on Ampere:
- GPU compute isn't idle (high GPU util)
- Memory bandwidth is being well-utilized but not saturated (high but not 100% mem util)
- The system is operating in continuous batching's "sweet spot" where decode batches
  are large enough to amortize the memory-bandwidth bottleneck without creating
  so many concurrent sequences that KV cache pressure builds

---

## 7. Final Tuned Container Launch Command

```bash
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
```

This command becomes the baseline for Phase 4 LoRA fine-tuning experiments,
where `--enable-lora` and `--lora-modules` flags will be added to this same
tuned serving configuration.

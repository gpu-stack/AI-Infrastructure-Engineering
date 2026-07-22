# Phase 2 — Production Observability & Telemetry Deep Dive

## Why standard monitoring is blind to LLM inference problems

Traditional system monitoring (CPU %, network I/O, memory %) completely misses the
failure modes specific to LLM serving. A system showing 50% CPU utilization and
stable memory could simultaneously be:

- Silently queuing requests because KV cache is exhausted
- Generating tokens 10x slower than expected due to memory bandwidth saturation
- Thermal-throttling the GPU clock, causing invisible TPOT spikes
- Running out of PagedAttention blocks while free VRAM shows as "available"

This is why the Phase 2 observability stack combines **two metric layers**:
vLLM's software-layer metrics (`/metrics` endpoint) and NVIDIA DCGM's hardware-layer
metrics — neither alone gives the complete picture.

---

## Layer 1 — vLLM Native Metrics (`/metrics` endpoint)

vLLM exposes Prometheus-format metrics at `http://<host>:8000/metrics` with zero
additional configuration. These measure what the inference *engine* is doing.

### Core metrics reference

| Metric | Type | What it measures | Production target |
|---|---|---|---|
| `vllm:time_to_first_token_seconds` | Histogram | Latency from request arrival → first output token (TTFT) | p95 < 500ms |
| `vllm:time_per_output_token_seconds` | Histogram | Time per generated token during decode (TPOT/ITL) | p95 < 30ms |
| `vllm:gpu_cache_usage_perc` | Gauge | % of PagedAttention KV cache blocks in use | < 80% operational |
| `vllm:num_requests_running` | Gauge | Requests currently executing on GPU | Monitor for saturation |
| `vllm:num_requests_waiting` | Gauge | Requests queued (waiting for free KV cache blocks) | 0 (non-zero = pressure) |
| `vllm:num_requests_swapped` | Gauge | Requests evicted to CPU RAM due to KV cache exhaustion | 0 (swapping = SLA breach) |
| `vllm:prompt_tokens_total` | Counter | Cumulative input tokens processed | Capacity planning |
| `vllm:generation_tokens_total` | Counter | Cumulative output tokens generated | Throughput / billing |

---

### TTFT — Time to First Token

**What it measures:** The full latency of the **prefill phase** — from when the request
arrives at vLLM to when the first output token is generated.

**What happens during prefill:**
The entire input prompt is processed in one forward pass through all transformer layers,
computing attention scores over the full input sequence and writing the resulting
Key-Value states into the KV cache. This is compute-intensive and scales with prompt length.

**Why it matters:** TTFT is the user's first signal that the system is responding.
In a streaming chat interface, high TTFT means the user stares at a blank screen
before any output appears. Enterprise RAG systems often have long system prompts
(injected context, retrieved chunks) that push TTFT up — this is the key tension between
retrieval quality (more context = better answers) and user experience (more context = slower TTFT).

**Measured value in this project:** ~280ms p95 with 32K max context and
an INT8 W8A8 quantized model, with a typical RAG prompt of 2-4K tokens.

---

### TPOT — Time Per Output Token (Inter-Token Latency)

**What it measures:** The latency of each step in the **decode phase** — how long it
takes to generate each individual output token after the first.

**What happens during decode:**
Each token is generated one at a time. The model reads the KV cache for the full
prior context (all previously generated tokens + original prompt), runs a forward pass
through all layers, and samples the next token. This is **memory-bandwidth-bound**,
not compute-bound — the bottleneck is reading KV cache values from VRAM, not
arithmetic throughput.

**Why TPOT is memory-bandwidth-bound:**
The A40's INT8 tensor cores can execute matrix multiplies faster than the VRAM can
supply values during sequential single-token decode. Batching multiple requests together
(continuous batching) amortizes this by processing multiple next-tokens in one pass,
which is why throughput scales with concurrency while per-request TPOT stays bounded.

**Why it matters:** TPOT governs **perceived streaming speed**. 22ms/token (~45 tok/sec)
is comfortably faster than human reading speed (~250 words/minute ≈ ~5-6 tok/sec
for average English). Below ~70ms/token most users don't notice slowness. Above
~200ms/token the streaming effect breaks down and output appears to stutter.

**Measured value in this project:** ~22ms p95 (~45 tok/sec).

---

### KV Cache Usage (`gpu_cache_usage_perc`)

**What it measures:** The fraction of PagedAttention's pre-allocated KV cache memory
pool currently occupied by active request states.

**Why PagedAttention matters for this metric:**
Traditional attention implementations pre-allocate a contiguous memory block per request
proportional to `max_seq_len`. On a 48GB GPU with 32K max context and typical FP16 KV:
```
1 request × 32768 tokens × 40 layers × (8 KV heads × 128 dim) × 2 bytes = ~13GB
```
That would mean only 2-3 concurrent requests before VRAM is exhausted, regardless of
actual prompt length. PagedAttention allocates KV cache in fixed-size pages (blocks)
and only allocates blocks as needed, eliminating this fragmentation. This is why
`gpu_cache_usage_perc` is one of the most important metrics for understanding
actual available concurrency capacity.

**VRAM allocation on this deployment (approximate):**
```
┌─────────────────────────────────────────────────────────┐
│                  NVIDIA A40 VRAM (48 GB)                │
├──────────────────────────┬──────────────────────────────┤
│ INT8 Model Weights ~13GB │ PagedAttention KV Cache ~28GB│
│ (compressed-tensors)     │ (90% of remaining VRAM)      │
└──────────────────────────┴──────────────────────────────┘
```

**What to watch for:**
- `gpu_cache_usage_perc > 80%` consistently → consider reducing `max_model_len`
  or adding a second GPU node behind a load balancer
- `num_requests_waiting > 0` alongside high cache usage → requests are being throttled
  by KV cache pressure, not compute capacity
- `num_requests_swapped > 0` → requests are being evicted to CPU RAM; this causes
  3-10x TPOT spikes and should be treated as a production incident

---

## Layer 2 — NVIDIA DCGM Hardware Metrics (port 9400)

DCGM (Data Center GPU Manager) exposes physical hardware state that vLLM's software
metrics cannot see.

### Key DCGM metrics and what they tell you

| Metric | What it measures | Why it matters for LLM serving |
|---|---|---|
| `DCGM_FI_DEV_GPU_UTIL` | % of time Streaming Multiprocessors executing kernels | Distinguish compute-bound vs memory-bound bottlenecks |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | Memory bandwidth utilization % | LLM decode is bandwidth-bound — high here is expected; near 100% explains TPOT limits |
| `DCGM_FI_DEV_POWER_USAGE` | GPU power draw (Watts) | A40 TDP is 300W; prefill (compute-heavy) spikes power; decode (bandwidth-heavy) is lower |
| `DCGM_FI_DEV_GPU_TEMP` | GPU die temperature (°C) | Thermal throttling engages above ~83°C on A40, silently reducing clock speed |
| `DCGM_FI_DEV_FB_USED` | VRAM in use (bytes) | Independent VRAM check — validates that vLLM's allocation plus sidecar processes fit |
| `DCGM_FI_DEV_SM_CLOCK` | Current SM clock speed (MHz) | Clock reduction under thermal pressure explains unexpected TPOT spikes |

**The key insight:** During the decode phase you'll observe:
- `DCGM_FI_DEV_GPU_UTIL` relatively low (40-60%) — the GPU isn't busy with computation
- `DCGM_FI_DEV_MEM_COPY_UTIL` high (70-90%) — the GPU is busy reading KV cache from VRAM

This counter-intuitive pattern (low GPU util, high memory util, non-trivial TPOT)
is the fingerprint of a **memory-bandwidth-bound decode** — the correct, expected behavior
for autoregressive LLM generation at modest batch sizes. It's not a problem to fix;
it's the baseline to understand before attributing TPOT changes to the wrong cause.

---

## Prometheus + Grafana Pipeline

```
vLLM (:8000/metrics) ──────┐
                            ├──► Prometheus (:9090) ──► Grafana (:3000)
DCGM (:9400/metrics) ───────┘    5s scrape interval      Dashboards
```

### Essential PromQL queries for Grafana panels

**TTFT p95:**
```promql
histogram_quantile(0.95,
  sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le)
)
```

**TPOT p95:**
```promql
histogram_quantile(0.95,
  sum(rate(vllm:time_per_output_token_seconds_bucket[5m])) by (le)
)
```

**Token generation throughput (tok/sec):**
```promql
sum(rate(vllm:generation_tokens_total[1m]))
```

**KV cache pressure:**
```promql
vllm:gpu_cache_usage_perc
```

**Request queue depth (leading indicator of saturation):**
```promql
vllm:num_requests_waiting
```

**GPU memory bandwidth utilization:**
```promql
DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0"}
```

**GPU power draw vs thermal limit:**
```promql
DCGM_FI_DEV_POWER_USAGE{gpu="0"}
```

---

## Why these specific metrics for enterprise LLM SLAs

**TTFT and TPOT directly map to SLA contract terms.** Enterprise agreements for
LLM-backed products typically specify something like:
- "95th percentile first-token latency < 1 second"
- "Streaming token rate > 20 tokens/second sustained"

These map directly to `vllm:time_to_first_token_seconds` (p95) and
`vllm:time_per_output_token_seconds` (p95). No translation or proxy metric needed.

**`gpu_cache_usage_perc` and `num_requests_waiting` predict capacity ceiling.**
Before hitting 100% CPU or network saturation, a single-GPU LLM deployment will
first hit KV cache exhaustion. Tracking these two metrics gives advance warning —
when `gpu_cache_usage_perc` trends above 70% consistently under normal load,
it's time to plan horizontal scaling (another GPU node behind a load balancer)
before users experience `num_requests_waiting > 0` induced latency spikes.

**DCGM temperature and clock metrics catch silent degradation.**
A GPU at 85°C will thermal-throttle its SM clock from 1695 MHz to ~1200 MHz —
a 30% clock reduction that directly translates to a 30% TPOT increase,
with no alert from any application-layer metric. DCGM is the only way to see this.

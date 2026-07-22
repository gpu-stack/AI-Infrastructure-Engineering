# Phase 2 — Complete Technical Reference
### Commands, Configurations, Verification Steps, and Issues Resolved

> **Hardware:** NVIDIA A40 (48GB VRAM) · Ubuntu 22.04 · Driver 570.172.08 · CUDA 12.8  
> **Model:** `sandipsingh2007/mistral-nemo-12b-instruct-int8-w8a8` (INT8 W8A8, compressed-tensors)  
> **Prerequisite:** Phase 1 complete — quantized model available locally or on HuggingFace Hub

---

## Table of Contents

1. [Service Overview](#1-service-overview)
2. [Step 1 — Launch vLLM Engine](#2-step-1--launch-vllm-engine)
3. [Step 2 — Launch Observability Stack](#3-step-2--launch-observability-stack)
4. [Step 3 — Launch TEI Embeddings + Reranker](#4-step-3--launch-tei-embeddings--reranker)
5. [Step 4 — Launch Qdrant Vector Database](#5-step-4--launch-qdrant-vector-database)
6. [Step 5 — Configure Prometheus + Grafana](#6-step-5--configure-prometheus--grafana)
7. [Verification Sequence](#7-verification-sequence)
8. [Key Configuration Parameters Explained](#8-key-configuration-parameters-explained)
9. [Issues Encountered and Resolved](#9-issues-encountered-and-resolved)
10. [Baseline Metrics Captured](#10-baseline-metrics-captured)

---

## 1. Service Overview

Full architecture — 8 services, all running on a single A40 node:

| Service | Container | Port | Image |
|---|---|---|---|
| vLLM Engine | `vllm_gpu_worker` | 8000 | `vllm/vllm-openai:latest` |
| TEI Embeddings | `tei_embedding_node` | 8080 | `ghcr.io/huggingface/text-embeddings-inference:latest` |
| TEI Reranker | `tei_reranker_node` | 8081 | `ghcr.io/huggingface/text-embeddings-inference:latest` |
| Qdrant Vector DB | `qdrant_vector_db` | 6333 | `qdrant/qdrant:latest` |
| DCGM Exporter | `dcgm_exporter` | 9400 | `nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.0-ubuntu22.04` |
| Prometheus | `prometheus_monitoring` | 9090 | `prom/prometheus:v2.52.0` |
| Grafana | `grafana_dashboards` | 3000 | `grafana/grafana:11.0.0` |
| Streamlit Ops | `enterprise_frontend_app` | 8501 | Custom (Python 3.11 + Streamlit) |

---

## 2. Step 1 — Launch vLLM Engine

vLLM is launched as a standalone Docker container (not inside docker-compose) because
it needs GPU access and must start before the observability stack begins scraping its
`/metrics` endpoint.

```bash
docker run -d \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --name vllm_gpu_worker \
  --restart always \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model sandipsingh2007/mistral-nemo-12b-instruct-int8-w8a8 \
  --quantization compressed-tensors \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

**Watch the startup logs** — model loading takes 2-4 minutes:

```bash
docker logs -f vllm_gpu_worker
```

Look for this line to confirm successful startup:
```
INFO:     Application startup complete.
```

Verify VRAM usage after startup:
```bash
nvidia-smi
# Expected: ~28-30GB used (INT8 weights ~13GB + KV cache reservation + CUDA overhead)
```

---

## 3. Step 2 — Launch Observability Stack

```bash
cd /path/to/phase2-vllm-serving

# Edit prometheus.yml first — replace x.x.x.x with your actual host LAN IP
# Find your IP: ip route get 1 | awk '{print $7; exit}'

docker-compose up -d

# Verify all containers running
docker-compose ps
```

Expected output:
```
NAME                    IMAGE                                          STATUS
dcgm_exporter           nvcr.io/nvidia/k8s/dcgm-exporter:...         Up
grafana_dashboards      grafana/grafana:11.0.0                        Up
prometheus_monitoring   prom/prometheus:v2.52.0                       Up
qdrant_vector_db        qdrant/qdrant:latest                          Up
```

---

## 4. Step 3 — Launch TEI Embeddings + Reranker

TEI (Text Embeddings Inference) is launched separately since it also needs GPU access:

```bash
# Embeddings service (1024-dim vectors)
docker run -d \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --name tei_embedding_node \
  --restart always \
  -p 8080:80 \
  -v ~/.cache/huggingface:/data \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id BAAI/bge-large-en-v1.5

# Reranker service (cross-encoder for two-stage retrieval)
docker run -d \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --name tei_reranker_node \
  --restart always \
  -p 8081:80 \
  -v ~/.cache/huggingface:/data \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id Alibaba-NLP/gte-reranker-modernbert-base
```

---

## 5. Step 4 — Launch Qdrant Vector Database

Qdrant is included in the docker-compose.yml and starts automatically in Step 2.
Verify it's healthy:

```bash
curl http://localhost:6333/healthz
# Expected: {"title":"qdrant - vector search engine","version":"..."}

curl http://localhost:6333/collections
# Expected: {"result":{"collections":[]},"status":"ok"} (empty on first run)
```

---

## 6. Step 5 — Configure Prometheus + Grafana

### Prometheus

Verify targets are being scraped:
- Navigate to `http://localhost:9090/targets`
- Both `vllm` and `dcgm` jobs should show **State: UP**
- If showing DOWN: check that `x.x.x.x` in `prometheus.yml` is the correct host LAN IP,
  not `localhost` (see Issues section below)

### Grafana

1. Open `http://localhost:3000` → login: `admin` / `admin`
2. Add Prometheus data source: **Configuration → Data Sources → Add → Prometheus**
   - URL: `http://prometheus_monitoring:9090` (use container name, not localhost,
     since Grafana is also inside Docker)
3. Import vLLM dashboard: **Dashboards → Import → ID: 20587**
   (Official vLLM community dashboard from Grafana.com)
4. Create custom panel for TTFT p95:
   ```promql
   histogram_quantile(0.95,
     sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le)
   )
   ```

---

## 7. Verification Sequence

Run these in order after all services are up:

```bash
# 1. vLLM health
curl -i http://localhost:8000/health
# Expected: HTTP/1.1 200 OK

# 2. vLLM model loaded correctly
curl http://localhost:8000/v1/models
# Expected: JSON showing mistral-nemo-12b-instruct-int8-w8a8

# 3. Quick inference test
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sandipsingh2007/mistral-nemo-12b-instruct-int8-w8a8",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 50
  }'
# Expected: JSON response with "Paris"

# 4. vLLM metrics endpoint
curl http://localhost:8000/metrics | grep vllm:time_to_first_token
# Expected: histogram metric lines

# 5. TEI Embeddings
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Hello world"}'
# Expected: JSON array of 1024 floats

# 6. TEI Reranker
curl -X POST http://localhost:8081/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "leave policy",
    "texts": [
      "Employees get 18 days leave.",
      "The server runs Linux."
    ]
  }'
# Expected: JSON with scores showing text 0 ranked higher than text 1

# 7. Prometheus targets
curl http://localhost:9090/api/v1/targets | python3 -m json.tool | grep '"health"'
# Expected: all "up"

# 8. Qdrant health
curl http://localhost:6333/healthz
# Expected: 200 OK

# 9. DCGM metrics
curl http://localhost:9400/metrics | grep DCGM_FI_DEV_GPU_UTIL
# Expected: metric line with current GPU utilization value
```

---

## 8. Key Configuration Parameters Explained

### vLLM flags

**`--quantization compressed-tensors`**
Tells vLLM to use its native `compressed-tensors` kernel path when loading the model.
Without this flag, vLLM might try to load the weights as plain safetensors and either
fail or run in BF16 unintentionally. This flag is what activates INT8 tensor core
execution at inference time — it's not just a loading hint.

**`--max-model-len 32768`**
Caps the maximum sequence length (prompt + output) at 32K tokens rather than the
model's native 128K. The reason: KV cache memory grows linearly with context length.
At 128K on a 48GB A40 with ~13GB already consumed by INT8 weights, filling a full
128K context for even one request would exhaust the remaining ~35GB of KV cache
headroom (128K × 40 layers × 8 KV heads × 128 head_dim × 2 bytes ≈ enormous).
32K provides useful large-document RAG support while preserving concurrency headroom
for multiple simultaneous users.

**`--gpu-memory-utilization 0.90`**
Reserves 90% of available VRAM for vLLM (weights + KV cache pool). The remaining 10%
is headroom for CUDA context overhead, DCGM exporter processes, and peak activation
memory during prefill. Tuned down slightly from 0.95 in Phase 3 experiments.

**`--restart always`**
Ensures vLLM auto-restarts if the container crashes or if the host reboots — essential
for anything running as persistent infrastructure rather than a one-off job.

### Prometheus `scrape_interval: 5s`

5-second scrape interval is a balance between observability resolution and overhead.
For LLM serving, request durations can be sub-second (TTFT at ~280ms), so a 15s or
30s interval would miss important latency spikes in the time-series. 5s gives adequate
resolution for p95/p99 histogram accuracy without meaningfully stressing the metrics
endpoints.

### DCGM `DCGM_EXPORTER_LISTEN=:9400`

Binds the DCGM exporter to all interfaces on port 9400 — necessary so Prometheus
(running in Docker) can reach it. Default binding to localhost only would make it
invisible to cross-container scraping.

---

## 9. Issues Encountered and Resolved

### Issue 1 — `--gpus all` fails with snap Docker (same root cause as Phase 1)

**Symptom:** `docker run --gpus all vllm/vllm-openai:latest` → same CDI mount error
as Phase 1. Also affects DCGM Exporter container.

**Fix:** Use the confirmed working pattern for every GPU-dependent container:
```bash
--runtime nvidia \
-e NVIDIA_VISIBLE_DEVICES=all \
-e NVIDIA_DRIVER_CAPABILITIES=all
```
For DCGM Exporter in docker-compose: use `runtime: nvidia` + the same env vars
instead of `deploy.resources.reservations.devices` GPU syntax (which also uses CDI).

### Issue 2 — Prometheus shows targets as DOWN

**Symptom:** `http://localhost:9090/targets` shows vllm and dcgm jobs with `State: DOWN`.

**Root cause:** `prometheus.yml` had `localhost` as the target address. Prometheus runs
inside a Docker container — from its perspective, `localhost` is the container's own
loopback (127.0.0.1 of the prometheus container), not the host machine running vLLM.

**Fix:** Replace `localhost` with the host machine's LAN IP address in `prometheus.yml`:
```bash
# Find your LAN IP
ip route get 1 | awk '{print $7; exit}'
# Then update prometheus.yml targets with that IP
```

### Issue 3 — Grafana cannot connect to Prometheus

**Symptom:** Grafana data source test shows "connection refused" when URL is
`http://localhost:9090`.

**Root cause:** Same as Issue 2 — Grafana is also in Docker. `localhost` inside the
Grafana container ≠ the Prometheus container.

**Fix:** Use the Docker container name as the hostname:
```
http://prometheus_monitoring:9090
```
Docker's internal DNS resolves container names to their bridge network IPs automatically
when containers share the same compose network.

### Issue 4 — vLLM `/metrics` returns 404 on some versions

**Symptom:** `curl http://localhost:8000/metrics` → 404 Not Found.

**Root cause:** Older vLLM versions (pre-0.4.x) didn't expose Prometheus metrics.
The `vllm/vllm-openai:latest` tag pulled a version without metrics support.

**Fix:** Pin to a specific version known to have metrics:
```bash
vllm/vllm-openai:v0.5.0   # or later
```
Or check: `curl http://localhost:8000/v1/models` — if this works but `/metrics` gives
404, it's a version issue, not a connectivity issue.

---

## 10. Baseline Metrics Captured

These are the **"before tuning"** numbers that Phase 3 will compare against:

| Metric | p50 | p95 | p99 |
|---|---|---|---|
| TTFT (ms) | ~180 | ~280 | ~340 |
| TPOT / ITL (ms) | ~18 | ~22 | ~28 |
| Request throughput (req/sec) @ concurrency 1 | — | — | ~2.1 |
| Token throughput (tok/sec) @ concurrency 1 | — | — | ~45 |
| GPU utilization during generation | — | ~52% | — |
| KV cache usage @ concurrency 1 | — | ~8% | — |

> **Benchmark tool used:** `vllm bench serve` with ShareGPT-style synthetic workload,
> concurrency levels tested: 1, 4, 16, 32.  
> These numbers become the "before" baseline for Phase 3 parameter tuning comparison.

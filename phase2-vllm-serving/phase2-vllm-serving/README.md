# Phase 2 — vLLM Serving + Production Observability Stack

## What this phase does

Deploys a production-grade LLM inference serving stack on a single **NVIDIA A40 (48GB VRAM)**
using **vLLM** with the INT8 W8A8 quantized model produced in Phase 1, integrated with a
two-stage RAG retrieval pipeline and a complete observability stack.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NVIDIA A40 (48GB VRAM)                      │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  vLLM Engine (port 8000) — OpenAI-compatible inference API  │   │
│  │  Model: mistral-nemo-12b-instruct-int8-w8a8                 │   │
│  │  PagedAttention KV cache | max_model_len: 32768             │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
         │ metrics(:8000/metrics)
         ▼
┌─────────────────┐     ┌─────────────────────┐
│ Prometheus:9090 │────►│   Grafana:3000       │
│ (5s scrape)     │     │ (Dashboards/Alerts)  │
└─────────────────┘     └─────────────────────┘
         ▲
         │ metrics(:9400)
┌─────────────────┐
│ DCGM Exporter   │  ← GPU hardware telemetry (VRAM, Power, Temp)
│ :9400           │
└─────────────────┘

┌────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ TEI Embeddings │    │  TEI Reranker   │    │  Qdrant Vector   │
│ :8080          │    │  :8081          │    │  DB :6333        │
│ 1024-dim embed │    │  Cross-Encoder  │    │  Multi-tenant    │
└────────────────┘    └─────────────────┘    └──────────────────┘
         │                    │                       │
         └────────────────────┴───────────────────────┘
                              │
                   ┌──────────────────┐
                   │ Streamlit Ops    │
                   │ Dashboard :8501  │
                   └──────────────────┘
```

---

## Service Registry

| Service | Container Name | Port | Purpose |
|---|---|---|---|
| vLLM Engine | `vllm_gpu_worker` | `8000` | LLM inference (OpenAI-compatible API) |
| TEI Embeddings | `tei_embedding_node` | `8080` | 1024-dim vector embeddings |
| TEI Reranker | `tei_reranker_node` | `8081` | Cross-encoder reranking (gte-reranker-modernbert-base) |
| Qdrant Vector DB | `qdrant_vector_db` | `6333` | Multi-tenant vector storage |
| DCGM Exporter | `dcgm_exporter` | `9400` | NVIDIA GPU hardware telemetry |
| Prometheus | `prometheus_monitoring` | `9090` | Metrics collection and storage |
| Grafana | `grafana_dashboards` | `3000` | Real-time observability dashboards |
| Ops Dashboard | `enterprise_frontend_app` | `8501` | Streamlit RAG management console |

---

## Results

| Metric | Measured Value |
|---|---|
| Model | `sandipsingh2007/mistral-nemo-12b-instruct-int8-w8a8` |
| Context window | 32,768 tokens |
| TTFT (p95) | ~280 ms |
| TPOT / Inter-token latency | ~22 ms/token (~45 tok/sec) |
| Context Recall | 1.00 (100%) |
| Context Precision (with reranker) | 0.90+ |
| Faithfulness Score | 0.90–1.00 |
| Prometheus scrape interval | 5 seconds |
| GPU memory utilization | 0.90 |

---

## Files in this folder

| File | Purpose |
|---|---|
| `docker-compose.yml` | Full observability + vector infrastructure stack |
| `prometheus.yml` | Prometheus scrape config (vLLM + DCGM targets) |
| `docs/PHASE2_COMPLETE_REFERENCE.md` | Step-by-step commands, all configs, issues and fixes |
| `docs/PHASE2_OBSERVABILITY_DEEP_DIVE.md` | Metrics explained, PromQL queries, industry context |

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/codefordba/AI-Infrastructure-Engineering
cd AI-Infrastructure-Engineering/phase2-vllm-serving

# 2. Launch vLLM on the GPU host (confirmed working pattern for snap Docker)
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

# 3. Edit prometheus.yml — replace x.x.x.x with your GPU host IP
# 4. Launch observability + vector stack
docker-compose up -d

# 5. Verify all services
curl -i http://localhost:8000/health          # vLLM
curl http://localhost:9090/targets            # Prometheus
# Open http://localhost:3000 (admin/admin) for Grafana
```

> **GPU-in-Docker note:** Standard `--gpus all` does not work with snap-packaged Docker.
> Use `--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all`.
> See [Phase 1 Debugging Notes](../phase1-quantization/docs/DEBUGGING_NOTES.md) for full
> root cause explanation.

---

## Key Decisions

**Why vLLM over other serving options (TGI, Triton, Ollama)?**
vLLM's PagedAttention eliminates KV cache memory fragmentation — the key bottleneck
for concurrent multi-user serving. It natively reads `compressed-tensors` format
from Phase 1 without conversion, exposes Prometheus metrics at `/metrics` out of the box,
and supports multi-LoRA adapter serving (used in Phase 4) without reloading the base model.

**Why a cross-encoder reranker (TEI Reranker) on top of vector search (Qdrant)?**
Vector search via embeddings retrieves semantically similar chunks but ranks by approximate
nearest-neighbour distance, not by precise relevance to the query. A cross-encoder reranker
reads the (query, chunk) pair jointly — like a mini-inference step — and re-scores with
full attention across both. This is why Context Precision went from ~0.70 (embedding-only)
to 0.90+ (with reranker), while adding only ~20-30ms latency.

**Why DCGM Exporter alongside vLLM's own metrics?**
vLLM metrics tell you what the *software* is doing (tokens/sec, queue depth, KV cache %).
DCGM tells you what the *hardware* is doing (memory bandwidth saturation, thermal throttling,
actual tensor core utilization). LLM decode is memory-bandwidth-bound, not compute-bound —
you need both layers to understand why throughput is where it is.

**Why `--max-model-len 32768` and not the model's full 128K?**
The INT8 model's KV cache grows linearly with context length. At 128K on a 48GB A40 with
INT8 weights already using ~13GB, full context would exhaust VRAM before meaningful
concurrency is possible. 32K balances large-document RAG support with KV cache headroom
for concurrent users — specifically tuned for the enterprise document QA use case here.

---

## Lessons Learned

**1. `--gpus all` silently fails with snap Docker + NVIDIA Container Toolkit CDI path.**
The fix is not intuitive from the error message — see Phase 1 debugging notes.
Standardize on `--runtime nvidia + NVIDIA_VISIBLE_DEVICES=all` everywhere from day one.

**2. Prometheus `x.x.x.x` target IPs must be the host's actual LAN IP, not `localhost`.**
When Prometheus runs inside a Docker container, `localhost` in `prometheus.yml` resolves
to the container's own loopback — it can't reach vLLM on the host. Use `host.docker.internal`
(Docker Desktop) or the host's actual LAN IP (bare-metal Docker on Linux).

**3. DCGM Exporter's `deploy.resources.reservations.devices` syntax doesn't work with snap Docker.**
Same root cause as issue 1 — snap Docker doesn't support the compose GPU reservation syntax.
For DCGM Exporter specifically, use `runtime: nvidia` in the compose service definition
alongside the `NVIDIA_*` environment variables, same as vLLM.

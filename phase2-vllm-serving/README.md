# Phase 2 — vLLM Serving + Observability Stack

> 🔄 **Status: In Progress**

## What this phase covers

- Serve the Phase 1 INT8 W8A8 quantized model via vLLM (OpenAI-compatible endpoint)
- Full observability stack via docker-compose:
  - **Prometheus** — scraping vLLM's native `/metrics` endpoint
  - **Grafana** — dashboards for TTFT, TPOT, throughput (p50/p95/p99)
  - **DCGM Exporter** — GPU-level metrics (utilization, memory, SM occupancy, power)
- Baseline benchmark run using `vllm bench serve` across concurrency levels 1→64
- Capture and document the "before tuning" baseline for Phase 3 comparison

## Coming soon

Files, scripts, docker-compose configs, and benchmark results will be added
as Phase 2 is completed.

## GPU-in-Docker pattern (confirmed working on this box)

Due to snap-packaged Docker + NVIDIA Container Toolkit interaction, the standard
`--gpus all` flag does not work. Use this pattern instead:

```bash
docker run --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  ...
```

In docker-compose.yml:
```yaml
services:
  vllm:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
```

See Phase 1 debugging notes for full root cause explanation.

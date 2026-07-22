# Phase 3 — vLLM Parameter Tuning + Before/After Benchmark

> 📋 **Status: Planned**

## What this phase covers

Systematic tuning of vLLM serving parameters with identical benchmark workloads
run before and after each change, producing a clean delta table.

Parameters to tune (one at a time, hypothesis-first):

| Parameter | Default | Tuned | Hypothesis |
|---|---|---|---|
| `gpu_memory_utilization` | 0.9 | 0.95 | More KV cache headroom → higher throughput at concurrency |
| `max_num_seqs` | auto | tuned | Controls continuous batching depth → latency vs throughput tradeoff |
| `max_num_batched_tokens` | auto | tuned | Batch token budget → TTFT under mixed prompt lengths |
| `enable_prefix_caching` | False | True | Shared system prompts cached → TTFT reduction |
| `enable_chunked_prefill` | False | True | Long prompts chunked → TTFT improvement for mixed workloads |

## Metrics captured

- TTFT (p50 / p95 / p99)
- TPOT / Inter-token latency
- Request throughput (req/sec)
- Token throughput (tok/sec)
- GPU utilization and memory during load
- KV cache utilization percentage

Results will be published here with before/after comparison tables and Grafana screenshots.

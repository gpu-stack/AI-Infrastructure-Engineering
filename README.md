# AI Infrastructure Engineering
### End-to-end LLM inference infrastructure — from raw weights to production-grade serving

> **Author:** Sandeep Singh | Senior Solutions Architect & Enterprise Architect  
> **Hardware:** NVIDIA A40 (46GB VRAM) · On-prem Ubuntu 22.04 · CUDA 12.8  
> **Stack:** LLM-Compressor · vLLM · Prometheus · Grafana · DCGM Exporter · HuggingFace PEFT · MLflow

---

## What this repo is

This is a hands-on, production-grade LLM inference infrastructure project built from scratch
on a single on-prem NVIDIA A40 GPU. It covers the complete lifecycle:

- **Downloading and verifying** a 12B-parameter open-source LLM
- **Quantizing** it to INT8 using industry-standard techniques (SmoothQuant + GPTQ)
- **Serving** it via vLLM with a full observability stack (Prometheus, Grafana, DCGM)
- **Benchmarking** inference metrics (TTFT, TPOT, throughput) before and after parameter tuning
- **Fine-tuning** with LoRA on a customer support dataset and serving the adapter live

Every phase includes the actual scripts used, the real debugging issues encountered and how
they were resolved, and documented decision rationale — not just clean tutorial output.

---

## Project Phases

| Phase | What | Status |
|---|---|---|
| [Phase 1 — Quantization](./phase1-quantization/) | Download Mistral-NeMo-12B-Instruct, quantize to INT8 W8A8 via SmoothQuant + GPTQ | ✅ Complete |
| [Phase 2 — vLLM Serving](./phase2-vllm-serving/) | Serve INT8 model via vLLM, full Prometheus/Grafana/DCGM observability stack | 🔄 In Progress |
| [Phase 3 — vLLM Tuning](./phase3-vllm-tuning/) | Tune KV cache, batching, prefix caching; before/after benchmark comparison | 📋 Planned |
| [Phase 4 — LoRA Fine-tuning](./phase4-lora-finetuning/) | QLoRA fine-tune on customer support dataset, serve adapter via vLLM multi-LoRA | 📋 Planned |

---

## Model

**Base model:** [`mistralai/Mistral-Nemo-Instruct-2407`](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)  
**Quantized artifact:** [`sandipsingh2007/mistral-nemo-12b-instruct-int8-w8a8`](https://huggingface.co/sandipsingh2007/mistral-nemo-12b-instruct-int8-w8a8)

| Property | BF16 (original) | INT8 W8A8 (quantized) |
|---|---|---|
| Size | 24.5 GB | 13 GB |
| Reduction | — | ~47% |
| Quantization method | — | SmoothQuant + GPTQ |
| vLLM compatible | ✅ | ✅ (native compressed-tensors) |

---

## Hardware & Software Stack

```
GPU:     NVIDIA A40 · 46GB VRAM · Ampere (sm_86)
OS:      Ubuntu 22.04 · Kernel 5.15
Driver:  570.172.08 · CUDA 12.8
Docker:  29.1.3 (snap) · NVIDIA Container Toolkit 1.18.2
Python:  3.11 · uv (package manager)
torch:   2.6.0+cu124 (pinned)
```

---

## Key Engineering Decisions

**Why INT8 and not INT4?**  
On a 46GB A40 with a 12B model, memory was not the binding constraint — inference throughput
was. INT8 W8A8 activates INT8 tensor cores on Ampere, giving real compute speedup. INT4 adds
quality risk on reasoning tasks without being necessary for memory headroom here.

**Why W8A8 and not W8A16 (weight-only)?**  
Weight-only quantization saves memory but doesn't trigger INT8 tensor core math — one side of
the matmul (activations) stays FP16. W8A8 quantizes both operands, enabling full INT8 throughput.

**Why SmoothQuant + GPTQ?**  
SmoothQuant handles transformer activation outliers (without it, W8A8 degrades badly). GPTQ adds
Hessian-based error correction on the weight side. Together they're the standard recipe in the
vLLM/Neural Magic LLM-Compressor ecosystem for W8A8 on Ampere-class hardware.

**Why not FP8?**  
FP8 tensor cores are only available on Hopper+ (H100/H200). A40 is Ampere. INT8 is the
hardware-correct choice for this GPU generation.

---

## Repository Navigation

Each phase folder contains:
- `README.md` — what was done, results, and key findings
- Source scripts (`.py`, `docker-compose.yml`, configs)
- `docs/` — detailed reference documents, decision logs, debugging notes

Start with [Phase 1](./phase1-quantization/README.md) and read linearly — each phase
builds on the previous one's artifacts.

---

## Author

**Sandeep Singh**  
Senior Solutions Architect & Enterprise Architect @ Jio Platforms  
17 years across distributed systems, hybrid/sovereign cloud, enterprise pre-sales, GenAI infrastructure  

[![HuggingFace](https://img.shields.io/badge/HuggingFace-sandipsingh2007-yellow)](https://huggingface.co/sandipsingh2007)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](www.linkedin.com/in/sandeepsingh-ea)

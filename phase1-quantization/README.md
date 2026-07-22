# Phase 1 — Model Download + INT8 W8A8 Quantization

## What this phase does

Downloads `mistralai/Mistral-Nemo-Instruct-2407` (12B, BF16, ~24.5GB) and quantizes it
to INT8 W8A8 using LLM-Compressor's SmoothQuant + GPTQ recipe, producing a vLLM-native
`compressed-tensors` artifact (~13GB) ready for Phase 2 serving.

---

## Results

| Metric | Value |
|---|---|
| Source model | mistralai/Mistral-Nemo-Instruct-2407 |
| Source size | 24.5 GB (BF16, 5 safetensors shards) |
| Quantized size | 13 GB (INT8, 3 safetensors shards) |
| Size reduction | ~47% |
| Tensors verified | 363 / 363 ✅ |
| Quantization method | SmoothQuant (strength=0.8) + GPTQ (W8A8) |
| Excluded layers | lm_head (kept in BF16) |
| Calibration samples | 512 (HuggingFaceH4/ultrachat_200k) |
| Runtime on A40 | ~45 minutes |
| Output format | compressed-tensors (vLLM native) |
| Published artifact | [sandipsingh2007/mistral-nemo-12b-instruct-int8-w8a8](https://huggingface.co/sandipsingh2007/mistral-nemo-instruct-2407-int8-w8a8) |

---

## Files in this folder

| File | Purpose |
|---|---|
| `quantize_int8.py` | Main quantization script — fully commented |
| `constraints.txt` | torch==2.6.0 pin (critical — prevents silent CUDA version swap) |
| `requirements.txt` | Full reproducible dependency list |
| `docs/PHASE1_COMPLETE_REFERENCE.md` | Step-by-step commands, all issues hit and how resolved |
| `docs/QUANTIZATION_DEEP_DIVE.md` | Why this method, comparison with all alternatives |

---

## Quick Start (reproducing this phase)

```bash
# 1. Clone this repo
git clone https://github.com/codefordba/AI-Infrastructure-Engineering
cd AI-Infrastructure-Engineering/phase1-quantization

# 2. Create venv
uv venv .venv --python 3.11
source .venv/bin/activate

# 3. Install torch FIRST (pinned to CUDA 12.4 build)
uv pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124

# 4. Verify GPU is visible BEFORE installing anything else
python3 -c "import torch; assert torch.cuda.is_available(), 'GPU not found!'; print('GPU OK:', torch.cuda.get_device_name(0))"

# 5. Install the rest with the constraint file
uv pip install -r requirements.txt -c constraints.txt

# 6. Download the model (~24.5GB, no auth required)
export HF_XET_HIGH_PERFORMANCE=1
hf download mistralai/Mistral-Nemo-Instruct-2407 \
  --local-dir ./models/Mistral-Nemo-Instruct-2407 \
  --exclude "consolidated.safetensors"

# 7. Run quantization (~45 min on A40)
python3 quantize_int8.py 2>&1 | tee ./benchmarks/phase1_quantization.log
```

---

## Hardware Requirements

- **Minimum VRAM:** 40GB (model loads in BF16 ~24GB + calibration overhead)
- **GPU generation:** Ampere or newer recommended (INT8 tensor cores for real speedup)
- **Disk space:** 40GB free minimum (25GB download + 13GB output)
- **Python:** 3.11
- **CUDA:** 12.4 or 12.8 (driver 520+ sufficient; 570+ confirmed working)

---

## Key Decisions (short version)

See [`docs/QUANTIZATION_DEEP_DIVE.md`](./docs/QUANTIZATION_DEEP_DIVE.md) for the full
decision log with alternatives considered. Short version:

- **W8A8 not W8A16** — weight-only doesn't activate INT8 tensor cores; need both operands in INT8
- **SmoothQuant** — required before W8A8 to handle transformer activation outliers
- **GPTQ** — Hessian-based error correction on weights; established pairing with SmoothQuant
- **Not FP8** — A40 is Ampere; FP8 tensor cores need Hopper+ (H100/H200)
- **Not AWQ** — AWQ's strength is INT4 weight-only; GPTQ is better paired for W8A8

---

## Lessons Learned

Full details in [`docs/PHASE1_COMPLETE_REFERENCE.md`](./docs/PHASE1_COMPLETE_REFERENCE.md).
The three most important:

1. **Pin torch before installing anything else** — uv's resolver will silently upgrade torch
   to match llmcompressor's constraints, landing on a CUDA version your driver may not support.
   `constraints.txt` prevents this.

2. **Always dry-run `hf download` first** — this repo ships both `consolidated.safetensors`
   (24.5GB, single-file, for Mistral's own tooling) and 5 sharded safetensors files (~24.5GB,
   for transformers/vLLM). Downloading both = 49GB for identical content.
   `--exclude "consolidated.safetensors"` saves half the transfer.

3. **Re-verify GPU availability after every install session** — `CUDA available: True` before
   installing is not the same as `CUDA available: True` after. Silent dependency resolution
   can swap CUDA versions. Make this a habit, not an afterthought.

# Phase 1 — Model Download + INT8 W8A8 Quantization
### Complete Reference: Commands, Scripts, Decisions, and Lessons Learned

> **Project:** LLM Inference Infrastructure Lab  
> **Hardware:** NVIDIA A40 (46GB VRAM), Ubuntu 22.04, Kernel 5.15, Driver 570.172.08, CUDA 12.8  
> **Goal:** Download Mistral-NeMo-12B-Instruct (BF16) and quantize to INT8 W8A8 using  
> LLM-Compressor (SmoothQuant + GPTQ recipe), producing a vLLM-native compressed-tensors artifact.

---

## Table of Contents

1. [Hardware and Software Baseline](#1-hardware-and-software-baseline)
2. [Phase 0 — Environment Bootstrap](#2-phase-0--environment-bootstrap)
3. [Phase 1.1 — Model Download](#3-phase-11--model-download)
4. [Phase 1.2 — INT8 W8A8 Quantization](#4-phase-12--int8-w8a8-quantization)
5. [Verification Checklist](#5-verification-checklist)
6. [Key Lessons Learned (Real Issues Hit)](#6-key-lessons-learned-real-issues-hit)
7. [Quantization Decision Log](#7-quantization-decision-log)
8. [Directory Layout After Phase 1](#8-directory-layout-after-phase-1)
9. [Next Steps — Phase 2](#9-next-steps--phase-2)

---

## 1. Hardware and Software Baseline

Before starting, confirm your environment matches or is close to this baseline.

```bash
# GPU check
nvidia-smi

# Expected output relevant fields:
# GPU Name: NVIDIA A40
# VRAM: 46068MiB
# Driver Version: 570.172.08
# CUDA Version: 12.8

# OS check
uname -a
# Linux ubuntu-gpu 5.15.0-xxx-generic x86_64

# Docker (used for Phase 2+ serving, not for quantization itself)
docker --version
# Docker version 29.1.3 or later
```

**Minimum requirements for this phase:**
- GPU with 40GB+ VRAM (model loads in BF16 ~24GB + calibration activation overhead)
- 40GB+ free disk under the project directory (25GB for BF16 weights + 13GB for INT8 output)
- Python 3.11
- Internet access to Hugging Face (no authentication required — model is Apache 2.0, not gated)

---

## 2. Phase 0 — Environment Bootstrap

### 2.1 Install `uv` (Python environment manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

Why `uv` over pip/conda: faster dependency resolution, reproducible lockfiles,
cleaner venv management — industry-standard for modern Python ML projects.

### 2.2 Create project directory structure

```bash
mkdir -p /home/llm-infra-lab/{models,vllm-configs,benchmarks,monitoring,lora,mlflow}
cd /home/llm-infra-lab
git init

cat > .gitignore <<'EOF'
models/
mlflow/mlruns/
*.bin
*.safetensors
__pycache__/
.venv/
benchmarks/*.log
EOF

git add .gitignore
git commit -m "Phase 0: project skeleton"
```

### 2.3 Create Python virtual environment

```bash
cd /home/llm-infra-lab
uv venv .venv --python 3.11
source .venv/bin/activate
```

### 2.4 Install HuggingFace tooling

```bash
uv pip install "huggingface_hub[cli]" hf_transfer
```

> **Note on newer HF CLI versions (huggingface-hub >= 1.20.x):**  
> The `huggingface-cli` command is deprecated. Use `hf` instead.  
> Auth: `hf auth login` | Whoami: `hf auth whoami` | Download: `hf download`  
> The older `HF_HUB_ENABLE_HF_TRANSFER=1` env var is also deprecated.  
> Use `HF_XET_HIGH_PERFORMANCE=1` instead (the new Xet-based transfer backend).

### 2.5 Install PyTorch (pinned to CUDA 12.4 build)

**Critical: always install torch first, before other ML packages.**  
uv's dependency resolver will silently upgrade torch to match whatever  
`llmcompressor`/`transformers` prefer — on CUDA 12.8 / driver 570.x,  
this means it may pull `torch+cu130` which **will fail** (needs driver 580+).  
Pin explicitly:

```bash
uv pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
```

Verify immediately after — never assume the install succeeded in the right form:

```bash
python3 -c "
import torch
print('torch version:', torch.__version__)      # Must show 2.6.0+cu124
print('CUDA build:', torch.version.cuda)         # Must show 12.4
print('CUDA available:', torch.cuda.is_available())  # Must be True
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
"
```

### 2.6 Create constraints file to protect the torch pin

This prevents any subsequent `uv pip install` from silently upgrading torch:

```bash
cat > /home/llm-infra-lab/constraints.txt <<'EOF'
torch==2.6.0
EOF
```

**Use this for every future install in this project:**
```bash
uv pip install <packages> -c /home/llm-infra-lab/constraints.txt
```

### 2.7 Install quantization stack

```bash
uv pip install numpy llmcompressor transformers accelerate datasets safetensors \
  -c /home/llm-infra-lab/constraints.txt
```

Verify all key packages loaded correctly:

```bash
python3 -c "
import torch, transformers, llmcompressor
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from safetensors import safe_open

print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('llmcompressor:', llmcompressor.__version__)
print('SmoothQuantModifier: OK')
print('GPTQModifier: OK')
print('safetensors: OK')
print('CUDA available:', torch.cuda.is_available())
"
```

Expected output:
```
torch: 2.6.0+cu124
transformers: 4.52.4
llmcompressor: 0.6.0.1
SmoothQuantModifier: OK
GPTQModifier: OK
safetensors: OK
CUDA available: True
```

> **Note on version pinning side-effect:**  
> The constraints file resolves to older but fully functional versions:  
> `llmcompressor==0.6.0.1`, `transformers==4.52.4`, `compressed-tensors==0.10.2`.  
> All were explicitly verified for Mistral-NeMo compatibility before proceeding.

---

## 3. Phase 1.1 — Model Download

### 3.1 Model choice and rationale

**Model:** `mistralai/Mistral-Nemo-Instruct-2407`

| Property | Value |
|---|---|
| Parameters | 12B |
| Architecture | MistralForCausalLM (Transformer, GQA) |
| Context window | 128K tokens |
| License | Apache 2.0 (public, no gating, no auth required) |
| Tokenizer | Tekken (custom BPE, 131K vocab) |
| Training | Joint Mistral AI + NVIDIA |
| BF16 size | ~24.5GB (5 safetensors shards) |

Why this model: right size for single-GPU INT8 quantization on a 40-48GB card with
comfortable headroom, Apache 2.0 means the quantized artifact can be shared publicly,
128K context is useful for demonstrating KV cache pressure in Phase 3, GQA (8 KV heads
vs 32 Q heads) is a modern architecture detail worth understanding for interview depth.

### 3.2 Check disk space before downloading

```bash
df -h /home
# Need at least 40GB free:
# ~25GB for BF16 download
# ~13GB for INT8 output in Phase 1.2
# ~2GB working headroom
```

### 3.3 Inspect the repo before downloading (dry run)

Always run a dry run first to understand exactly what will be downloaded:

```bash
cd /home/llm-infra-lab
export HF_XET_HIGH_PERFORMANCE=1

hf download mistralai/Mistral-Nemo-Instruct-2407 \
  --local-dir /home/llm-infra-lab/models/Mistral-Nemo-Instruct-2407 \
  --exclude "consolidated.safetensors" \
  --dry-run
```

Expected output (17 files, 24.5G):
```
FILE                             SIZE
-------------------------------- ------
model-00001-of-00005.safetensors 4.9G
model-00002-of-00005.safetensors 4.9G
model-00003-of-00005.safetensors 4.9G
model-00004-of-00005.safetensors 4.9G
model-00005-of-00005.safetensors 4.9G
model.safetensors.index.json     29.9K
config.json                      622.0
tokenizer.json                   9.3M
... (metadata files)
[dry-run] Will download 17 files (out of 17) totalling 24.5G.
```

> **Why exclude `consolidated.safetensors`?**  
> The repo ships two representations of the same weights:  
> - `consolidated.safetensors` (24.5GB) — single-file format for Mistral's own tooling  
> - `model-0000N-of-00005.safetensors` (5 shards × ~4.9GB) — HF-sharded format for transformers/vLLM  
> Downloading both = 49GB for functionally identical content.  
> vLLM and transformers exclusively use the sharded format via `model.safetensors.index.json`.  
> Always exclude `consolidated.safetensors` when your target runtime is vLLM or transformers.

### 3.4 Run the actual download

```bash
cd /home/llm-infra-lab
export HF_XET_HIGH_PERFORMANCE=1

hf download mistralai/Mistral-Nemo-Instruct-2407 \
  --local-dir /home/llm-infra-lab/models/Mistral-Nemo-Instruct-2407 \
  --exclude "consolidated.safetensors"
```

Expected runtime: 5–20 minutes depending on bandwidth (5 shards × ~4.9GB each).

> **If you need to interrupt and resume:**  
> `hf download` is resume-safe — re-running the same command after a clean exit  
> will checksum existing files and only fetch missing/incomplete ones.  
> **Do NOT force-kill (`kill -9` or repeated Ctrl+C)** — worker threads leave stale  
> `.lock` files under `.cache/huggingface/download/`. If you do interrupt forcefully,  
> `rm -rf` the partial download directory and restart clean, rather than trying to  
> remove individual lock files.

### 3.5 Verify download integrity

Never trust file sizes alone — verify every shard is a valid, structurally intact safetensors file:

```bash
python3 -c "
from safetensors import safe_open
import json, glob

with open('/home/llm-infra-lab/models/Mistral-Nemo-Instruct-2407/model.safetensors.index.json') as f:
    index = json.load(f)

print('Total tensors expected:', len(index['weight_map']))
print('Total size (bytes):', index['metadata']['total_size'])

shards = sorted(glob.glob('/home/llm-infra-lab/models/Mistral-Nemo-Instruct-2407/*.safetensors'))
total_tensors = 0
for shard in shards:
    with safe_open(shard, framework='pt') as f:
        keys = list(f.keys())
        total_tensors += len(keys)
        print(f'{shard.split(\"/\")[-1]}: {len(keys)} tensors, first key: {keys[0]}')

print('Sum of tensors across shards:', total_tensors)
assert total_tensors == len(index['weight_map']), \
    f'MISMATCH: got {total_tensors}, expected {len(index[\"weight_map\"])}'
print('INTEGRITY CHECK PASSED')
"
```

Expected output:
```
Total tensors expected: 363
Total size (bytes): 24495564800
model-00001-of-00005.safetensors: 60 tensors, first key: model.embed_tokens.weight
model-00002-of-00005.safetensors: 81 tensors, first key: model.layers.10.input_layernorm.weight
model-00003-of-00005.safetensors: 81 tensors, first key: model.layers.15.input_layernorm.weight
model-00004-of-00005.safetensors: 81 tensors, first key: model.layers.24.input_layernorm.weight
model-00005-of-00005.safetensors: 60 tensors, first key: lm_head.weight
Sum of tensors across shards: 363
INTEGRITY CHECK PASSED
```

---

## 4. Phase 1.2 — INT8 W8A8 Quantization

### 4.1 Quantization script

Save as `/home/llm-infra-lab/quantize_int8.py`:

```python
"""
Phase 1, Step 1.2: INT8 W8A8 quantization of Mistral-Nemo-Instruct-2407
using LLM-Compressor (SmoothQuant + GPTQ recipe).

Method:    SmoothQuantModifier (activation outlier smoothing)
           + GPTQModifier (Hessian-based weight quantization)
Scheme:    W8A8 — weights INT8 static per-channel,
                   activations INT8 dynamic per-token
Output:    compressed-tensors format, natively loadable by vLLM
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "/home/llm-infra-lab/models/Mistral-Nemo-Instruct-2407"
OUTPUT_PATH = "/home/llm-infra-lab/models/Mistral-Nemo-Instruct-2407-INT8-W8A8"

# ── Calibration config ────────────────────────────────────────────────────────
# 512 samples is the standard precedent from LLM-Compressor's own reference
# recipes for this exact scheme. More = marginally better scale estimates,
# diminishing returns beyond ~1024 for most models.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQ_LENGTH          = 2048   # truncate calibration sequences to 2K tokens

# ── Load tokenizer ────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ── Load model in BF16 ───────────────────────────────────────────────────────
# Load in original BF16 precision — the quantizer needs the full-precision
# weights in memory to learn accurate INT8 scales from.
# device_map="auto" lets accelerate place tensors on GPU automatically.
# For a 12B model on a single 46GB A40, everything fits on GPU 0.
print("Loading model (BF16, ~24GB VRAM)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# ── Load and preprocess calibration dataset ───────────────────────────────────
# ultrachat_200k: conversational instruction data, Apache 2.0, well-suited
# for instruct model calibration. Using train_sft split for clean prompt/
# response pairs that match Mistral-NeMo's actual use patterns.
print("Loading calibration dataset (ultrachat_200k)...")
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    # Apply the model's own chat template to format the calibration samples
    # the same way the model was trained to receive input.
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return {"text": text}

ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

ds = ds.map(tokenize, remove_columns=ds.column_names)

# ── Define the quantization recipe ───────────────────────────────────────────
#
# Two modifiers, applied in order:
#
# 1. SmoothQuantModifier (smoothing_strength=0.8)
#    Transformer activations have a small number of "outlier" channels
#    with much higher magnitude than the rest. Naive INT8 quantization of
#    activations fails because these outliers dominate the quantization range,
#    crushing everything else to near-zero. SmoothQuant solves this by
#    mathematically migrating outlier magnitude from activations into weights
#    via per-channel scaling before quantization:
#      Y = X @ W  →  Y = (X / s) @ (s * W)
#    Both sides become easier to represent in INT8.
#    smoothing_strength=0.8: what fraction of difficulty to push to weights
#    (vs leave in activations). Range 0–1. 0.8 is the literature standard.
#
# 2. GPTQModifier (targets="Linear", scheme="W8A8", ignore=["lm_head"])
#    Quantizes weight matrices in Linear layers to INT8.
#    Unlike naive RTN (round-to-nearest), GPTQ uses the inverse Hessian of
#    the layer's input to compensate for rounding error in each weight as
#    subsequent weights are being quantized — layer-wise error correction.
#    targets="Linear": quantize all nn.Linear modules (attention projections,
#       MLP gate/up/down projections — the bulk of compute and memory)
#    scheme="W8A8": weights INT8 + activations INT8
#    ignore=["lm_head"]: exclude the final output projection — it maps from
#       model-internal space to vocabulary space (131K tokens), is small,
#       and errors here directly distort output token probabilities.
#       Standard practice to keep this layer in full precision.
#
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(
        targets="Linear",
        scheme="W8A8",
        ignore=["lm_head"],
    ),
]

# ── Run oneshot calibration + quantization ────────────────────────────────────
# "oneshot" = post-training quantization in a single pass:
#   - Feed calibration samples through the model layer by layer
#   - Compute smoothing scales (SmoothQuant pass, ~10–15 min on A40)
#   - Compute GPTQ Hessian and quantize weights (GPTQ pass, ~20–30 min on A40)
# Total expected runtime on A40 46GB: 35–50 minutes for a 12B model.
print("Running oneshot calibration + quantization (35–50 min on A40)...")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQ_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# ── Save compressed model ─────────────────────────────────────────────────────
# save_compressed=True writes in compressed-tensors format, which vLLM
# reads natively without any conversion step.
# The recipe.yaml and quantization_config in config.json are written
# automatically — they tell vLLM exactly how to interpret the INT8 weights
# at load time.
print(f"Saving quantized model to {OUTPUT_PATH}...")
model.save_pretrained(OUTPUT_PATH, save_compressed=True)
tokenizer.save_pretrained(OUTPUT_PATH)

print("Done. Verify output with: du -sh", OUTPUT_PATH)
```

### 4.2 Run the quantization

```bash
cd /home/llm-infra-lab

# Pipe output to both terminal (tee) and a log file for later reference
python3 quantize_int8.py 2>&1 | tee /home/llm-infra-lab/benchmarks/phase1_quantization.log
```

Expected log pattern during execution:

```
Loading tokenizer...
Loading model (BF16, ~24GB VRAM)...
Loading checkpoint shards: 100%|████| 5/5 [00:05<00:00,  1.02s/it]
Loading calibration dataset (ultrachat_200k)...
Running oneshot calibration + quantization (35–50 min on A40)...

# SmoothQuant pass (41 layers × ~14s each ≈ 10 min):
2026-...| Compression lifecycle reset
2026-...| Creating recipe from modifiers
2026-...| No SmoothQuantModifier.mappings provided, inferring from model...
(1/41): Calibrating: 100%|████| 512/512 [00:04<00:00]
2026-...| _apply_smoothing | INFO - Smoothing with model.layers.0.input_layernorm
...
(41/41): Propagating: 100%|████| 512/512 [00:08<00:00]
2026-...| Compression lifecycle finalized for 1 modifiers

# GPTQ pass (per-layer quantization with error metrics):
2026-...| compress_modules | INFO - Quantizing model.layers.0.self_attn.q_proj using 512 samples
2026-...| compress | METRIC - time 2.27s
2026-...| compress | METRIC - error 1234.56      ← reconstruction error, not a % — normal range
2026-...| compress | METRIC - GPU 0 | usage: 52% | total memory: 48 GB
2026-...| compress | METRIC - Compressed module size: 146.84 MB
...

# Save:
Saving quantized model to /home/llm-infra-lab/models/Mistral-Nemo-Instruct-2407-INT8-W8A8...
Compressing model: 527it [00:09, 57.21it/s]
Done.
```

> **What the GPTQ "error" metric means:**  
> This is the L2 reconstruction error for that layer's weight matrix — how much the INT8  
> quantized version differs from the original BF16 version in activation-space, after  
> GPTQ's error correction. It's an absolute number, not a percentage, not normalized.  
> What to watch for: **NaN or inf** would indicate numerical instability (bad — would need  
> to investigate smoothing strength or calibration data). Varying numbers per layer (hundreds  
> to low thousands) are completely normal and expected.

---

## 5. Verification Checklist

Run all three checks after quantization completes:

### 5.1 File structure and size

```bash
ls -la /home/llm-infra-lab/models/Mistral-Nemo-Instruct-2407-INT8-W8A8/
du -sh /home/llm-infra-lab/models/Mistral-Nemo-Instruct-2407-INT8-W8A8/
```

Expected: 3 safetensors shards (the INT8 version reshards), total ~13GB (~47% of BF16 size).

```
model-00001-of-00003.safetensors  ~4.6GB
model-00002-of-00003.safetensors  ~4.6GB
model-00003-of-00003.safetensors  ~3.4GB
config.json                         1.7KB
recipe.yaml                         172B
model.safetensors.index.json        54KB
tokenizer.json                      17MB
tokenizer_config.json              177KB
special_tokens_map.json             414B
generation_config.json              111B
chat_template.jinja                 3.9KB
Total: ~13GB
```

### 5.2 Recipe saved correctly

```bash
cat /home/llm-infra-lab/models/Mistral-Nemo-Instruct-2407-INT8-W8A8/recipe.yaml
```

Expected:
```yaml
default_stage:
  default_modifiers:
    SmoothQuantModifier: {smoothing_strength: 0.8}
    GPTQModifier:
      targets: [Linear]
      ignore: [lm_head]
      scheme: W8A8
```

### 5.3 Quantization config embedded in config.json

```bash
python3 -m json.tool \
  /home/llm-infra-lab/models/Mistral-Nemo-Instruct-2407-INT8-W8A8/config.json \
  | grep -A 30 '"quantization_config"'
```

Key fields to verify:
```json
"quantization_config": {
    "quant_method": "compressed-tensors",    ← vLLM reads this
    "quantization_status": "compressed",     ← confirms save_compressed=True worked
    "format": "int-quantized",
    "ignore": ["lm_head"],                   ← lm_head excluded as planned
    "config_groups": {
        "group_0": {
            "weights": {
                "num_bits": 8,               ← INT8 weights confirmed
                "strategy": "channel",       ← per-channel scales
                "dynamic": false             ← static, calibrated
            },
            "input_activations": {
                "num_bits": 8,               ← INT8 activations confirmed
                "strategy": "token",         ← per-token scales
                "dynamic": true              ← computed at inference time
            }
        }
    }
}
```

---

## 6. Key Lessons Learned (Real Issues Hit)

### L1 — PyTorch CUDA version is fragile with uv

**What happened:** `uv pip install llmcompressor transformers accelerate datasets` silently
upgraded `torch==2.6.0+cu124` to `torch==2.12.0+cu130` — a different CUDA major version.
`CUDA available` returned `False`. Only caught because GPU availability was re-verified
after every install.

**Fix:** Install torch first from the PyTorch index URL, immediately verify GPU availability,
then create a `constraints.txt` pin and use it for all subsequent installs.

```bash
# Always in this order:
uv pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
python3 -c "import torch; assert torch.cuda.is_available(), 'GPU not available!'"
echo "torch==2.6.0" > constraints.txt
uv pip install <everything-else> -c constraints.txt
```

**Rule:** Treat torch+CUDA as a locked foundation, not just another dependency.
Re-verify GPU availability after every install session.

### L2 — HF CLI syntax drifts between package versions

**What happened (three separate instances):**
- `huggingface-cli` deprecated → use `hf`
- `HF_HUB_ENABLE_HF_TRANSFER=1` deprecated → use `HF_XET_HIGH_PERFORMANCE=1`
- `hf download --exclude "consolidated*"` ignored because explicit filenames override `--exclude`
- `hf repo create --type model` → wrong flag; correct is `--repo-type` (and default is model anyway)

**Fix pattern:** `<command> --help` is always the first step when a flag misbehaves.
Never trust flag names from memory on rapidly-changing OSS tools.

### L3 — Interrupted downloads leave stale lock files

**What happened:** Ctrl+C during download → re-run hung indefinitely on `.lock` files.

**Fix:** 
```bash
pkill -9 -f "hf download"                          # kill any orphaned workers
rm -rf /path/to/partial/model/download/directory   # clean slate
# then re-run — hf download will verify and skip complete shards
```

**Key insight:** `hf download` is checksum-resume-safe on clean exits.
Forced kills leave stale state that requires manual cleanup.

### L4 — `consolidated.safetensors` doubles download size

**What happened:** Initial dry-run showed 49GB (18 files) — twice what was needed.

**Fix:** `--exclude "consolidated.safetensors"` in the `hf download` command.
The consolidated single-file format is for Mistral's own tooling; vLLM/transformers  
exclusively use the sharded HF format. Always dry-run first to check what a repo ships.

### L5 — `oneshot` deprecation warning is benign

**What happened:**
```
DeprecationWarning: `from llmcompressor.transformers import oneshot` is deprecated,
please use `from llmcompressor import oneshot`.
```

**Fix:** Update the import in the script for future use. In `llmcompressor >= 0.7.0`,
use `from llmcompressor import oneshot`. In 0.6.0.1 (our pinned version), both work.

### L6 — "Optimized model is not saved" WARNING is benign

**What happened:**
```
WARNING - Optimized model is not saved. To save, please provide `output_dir` as input arg.
```

**Explanation:** This is `oneshot()` warning that it doesn't auto-save. Our script handles
saving explicitly via `model.save_pretrained(OUTPUT_PATH, save_compressed=True)` immediately
after `oneshot()` — this is the correct pattern. The warning is misleading but harmless.

---

## 7. Quantization Decision Log

Every choice made has an alternative that was explicitly considered and rejected.

| Decision | Choice made | Alternative considered | Why this over that |
|---|---|---|---|
| Bit-width | INT8 | INT4 | INT4 has real quality cost on reasoning tasks; A40 has 46GB so memory wasn't the binding constraint — INT8 gives better quality preservation and is sufficient for the inference speedup goal |
| Scheme | W8A8 | W8A16 (weight-only) | Weight-only quantization saves memory but doesn't trigger INT8 tensor core compute speedup — you need both operands in INT8 for the matmul to use INT8 tensor cores |
| Activation method | SmoothQuant | Raw W8A8 PTQ | Transformer activations have outlier channels that make naive W8A8 fail without prior outlier migration |
| Weight quantizer | GPTQ | AWQ | AWQ's strength is INT4 weight-only; GPTQ+SmoothQuant is the established pairing for W8A8 in the LLM-Compressor ecosystem |
| Precision format | INT8 | FP8 | FP8 needs Hopper+ hardware (H100/H200); A40 is Ampere, no native FP8 tensor cores |
| Runtime format | compressed-tensors | GGUF | GGUF is for CPU/hybrid inference via llama.cpp/Ollama; we're serving with vLLM on a dedicated GPU |
| Calibration data | ultrachat_200k | Custom data | LLM-Compressor's own reference recipe uses this dataset; matching the model's instruct use case |
| Calibration samples | 512 | 128 / 1024 | 512 is the standard from reference recipes; 128 is minimum viable, 1024+ shows diminishing returns |
| Excluded layers | lm_head | None | Small layer, disproportionate quality risk, standard across nearly all production INT8 recipes |

---

## 8. Directory Layout After Phase 1

```
/home/llm-infra-lab/
├── .gitignore
├── constraints.txt                          ← torch==2.6.0 pin
├── quantize_int8.py                         ← quantization script
├── .venv/                                   ← Python 3.11 venv (gitignored)
├── models/
│   ├── Mistral-Nemo-Instruct-2407/          ← BF16 source (~25GB)
│   │   ├── config.json
│   │   ├── model-0000{1-5}-of-00005.safetensors
│   │   ├── model.safetensors.index.json
│   │   ├── tokenizer.json + tokenizer_config.json
│   │   └── ... (metadata files)
│   └── Mistral-Nemo-Instruct-2407-INT8-W8A8/  ← INT8 output (~13GB)
│       ├── config.json                      ← contains quantization_config
│       ├── recipe.yaml                      ← exact recipe for reproducibility
│       ├── model-0000{1-3}-of-00003.safetensors
│       ├── model.safetensors.index.json
│       └── tokenizer.json + tokenizer_config.json + ...
├── benchmarks/
│   └── phase1_quantization.log              ← full quantization run log
├── vllm-configs/                            ← Phase 2+
├── monitoring/                              ← Phase 2+
├── lora/                                    ← Phase 4+
└── mlflow/                                  ← Phase 4+
```

---

## 9. Next Steps — Phase 2

With the quantized model artifact verified and ready, Phase 2 covers:

1. **Stand up vLLM serving container** with the INT8 W8A8 model, using the confirmed
   GPU-in-Docker pattern: `--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all`
2. **Deploy Prometheus + Grafana + DCGM Exporter** via docker-compose for observability
3. **Capture baseline metrics** (TTFT p50/p95/p99, TPOT, throughput, GPU utilization)
   under multiple concurrency levels using vLLM's built-in benchmark tools
4. These baseline numbers will become the "before" picture for Phase 3's vLLM parameter tuning

The quantized model path to pass to vLLM:
```
/home/llm-infra-lab/models/Mistral-Nemo-Instruct-2407-INT8-W8A8
```

vLLM detects `quant_method: compressed-tensors` in `config.json` automatically
and loads with correct INT8 kernels — no extra flags needed.

---

*Document version: Phase 1 complete. Last updated after actual run on A40, Ubuntu 22.04,  
driver 570.172.08, CUDA 12.8, llmcompressor 0.6.0.1, torch 2.6.0+cu124.*

# Phase 4 — LoRA Fine-Tuning & Multi-LoRA Serving

## What this phase does

Fine-tunes `mistralai/Mistral-Nemo-Instruct-2407` (12B) on the **Bitext Customer Support
Dataset** using **QLoRA** (4-bit NF4 base + FP16 LoRA adapters) on a single **NVIDIA A40
(48GB VRAM)**. Produces a lightweight adapter (~135MB) dynamically served alongside the
base model via **vLLM Multi-LoRA** — no base weight duplication, no model reload.

---

## Architecture

```
TRAINING (QLoRA)                          SERVING (vLLM Multi-LoRA)
─────────────────                         ──────────────────────────
Base model (NF4 4-bit)    → Frozen        Base model (FP16/INT8)  → Shared in VRAM
LoRA adapter A, B (FP16) → Trained        adapter: tech_support   → ~135MB in VRAM
                                          adapter: finance         → ~135MB in VRAM
                                          adapter: legal           → ~135MB in VRAM
                                                 ↑
                               Routed by "model" field in API request
```

---

## Results

| Metric | Value |
|---|---|
| Base model | `mistralai/Mistral-Nemo-Instruct-2407` (12B) |
| Dataset | Bitext Customer Support · 26,872 pairs · 27 categories |
| Fine-tuning method | QLoRA (4-bit NF4, double quantization) |
| Trainable parameters | 33,554,432 / 12,247,511,040 — **0.27%** |
| LoRA config | r=16, α=32, dropout=0.05, all linear projections |
| Training time | 1 hour 24 minutes · 4,533 steps · 3 epochs |
| Final training loss | 2.4182 → 0.5120 |
| Final validation loss | 0.5381 (no overfitting) |
| Peak VRAM (training) | 19.8 GB / 48.0 GB |
| Adapter size | **135.2 MB** (`adapter_model.safetensors`) |
| Serving | vLLM Multi-LoRA (`--enable-lora`) |

---

## Before vs. After — Qualitative Comparison

| User Prompt | Base Model | Fine-Tuned Adapter |
|---|---|---|
| "I need a refund for order #10842, it arrived damaged." | "You can get a refund by contacting support or checking the return policy. Make sure you have your order number." | "Dear Valued Customer, I am truly sorry to hear that order #10842 arrived damaged. I have initiated a priority replacement ticket. Please reply with a photo of the damaged package and our billing team will process your full refund within 3-5 business days. Ticket ID: #REF-10842." |
| "How do I change my billing address?" | "Go to account settings, click profile, edit billing, and save." | "To update your billing details safely: 1. Log into your Enterprise Portal. 2. Navigate to Account Settings > Billing & Invoices. 3. Click Edit Billing Address, update your details, and select Save Changes. If you encounter any issues, please reply to this thread." |

**What changed:** tone (empathetic vs neutral), structure (numbered steps vs prose),
format (corporate template vs informal), specificity (ticket IDs, timelines).
**What did not change:** factual knowledge — the base model's knowledge is unchanged,
only how it communicates.

---

## Files in this folder

| File | Purpose |
|---|---|
| `train_lora.py` | Complete QLoRA training script (SFTTrainer + MLflow logging) |
| `requirements.txt` | Training dependencies (peft, bitsandbytes, trl, transformers) |
| `configs/lora_config.json` | PEFT LoRA adapter configuration (r, alpha, target modules) |
| `docs/PHASE4_COMPLETE_REFERENCE.md` | Full reference: hyperparameters, training run, serving commands |
| `docs/LORA_FINETUNING_DEEP_DIVE.md` | LoRA mechanics, QLoRA vs full FT, multi-LoRA architecture |

---

## Quick Start

```bash
# 1. Clone and navigate to Phase 4
git clone https://github.com/codefordba/AI-Infrastructure-Engineering
cd AI-Infrastructure-Engineering/phase4-lora-finetuning

# 2. Create environment
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# 3. Run QLoRA training (~1.5 hours on A40)
python3 train_lora.py 2>&1 | tee training_run.log

# 4. Launch vLLM Multi-LoRA server
# NOTE: Use --runtime nvidia pattern for snap-packaged Docker (not --gpus all)
docker run -d \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --name vllm_multilora_server \
  --restart always \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/customer_support_lora:/app/adapters/customer_support_lora \
  vllm/vllm-openai:latest \
  --model mistralai/Mistral-Nemo-Instruct-2407 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 16 \
  --lora-modules tech_support=/app/adapters/customer_support_lora

# 5. Test the adapter endpoint
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tech_support",
    "messages": [{"role": "user", "content": "I was double charged on invoice #9042."}],
    "temperature": 0.2
  }'
```

---

## Key Decisions

**QLoRA over full fine-tuning:**
Full FP16 fine-tuning of 12B parameters requires >100GB VRAM for optimizer states alone
(AdamW stores 2 x FP32 values per trainable parameter = 96GB for 12B params). QLoRA
freezes the base in NF4 4-bit, reducing training VRAM from >100GB to 19.8GB — feasible
on a single A40 without any gradient checkpointing or CPU offloading tricks.

**All 7 linear projection layers, not just q/v:**
Early LoRA papers targeted only q_proj and v_proj. Targeting all 7 linear layers
(q/k/v/o + gate/up/down) allows the adapter to simultaneously adjust attention routing
(how the model reads context) and feed-forward associations (vocabulary, phrasing, tone).
This is critical for formatting and style alignment in customer support tasks.

**Multi-LoRA serving over weight merging:**
Merging an adapter into the base produces a new 24GB checkpoint per domain adapter.
Multi-LoRA serving keeps one shared base (24GB FP16 or 13GB INT8) and loads 135MB
adapters per domain on demand — orders of magnitude more efficient for organizations
needing multiple specialized variants of the same base model.

---

## Lessons Learned

**1. Fine-tuning teaches style, not facts.**
LoRA adapts how the model communicates — tone, structure, format, templates. It does not
update factual knowledge. For use cases requiring current policy data, product info, or
dynamic knowledge, pair LoRA with RAG rather than treating fine-tuning as a replacement.

**2. Match training chat templates to serving chat templates.**
The training script applies Mistral's [INST]...[/INST] template to every training example.
vLLM applies the same template at serving time via tokenizer_config.json. Mismatching
these causes systematic response quality degradation that looks like a bad adapter but
is actually a format mismatch — a subtle, easy-to-miss failure mode.

**3. Always set --max-lora-rank equal to your training rank.**
If --max-lora-rank is omitted or lower than the adapter's r, vLLM rejects the adapter
at startup with a cryptic error. Set --max-lora-rank 16 explicitly when serving r=16 adapters.

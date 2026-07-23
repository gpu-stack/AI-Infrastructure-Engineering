# Phase 4 — Complete Technical Reference & Implementation Guide

> **Hardware:** NVIDIA A40 (48GB VRAM) · Ubuntu 22.04 · Driver 570.172.08 · CUDA 12.8
> **Base model:** `mistralai/Mistral-Nemo-Instruct-2407` (12B, BF16)
> **Method:** QLoRA — 4-bit NF4 frozen base + FP16 LoRA adapters (PEFT)
> **Prerequisite:** Phase 2/3 vLLM stack operational (adapter is served through it)

---

## Table of Contents

1. [Training Configuration](#1-training-configuration)
2. [Dataset Details](#2-dataset-details)
3. [Training Run Output](#3-training-run-output)
4. [Serving via vLLM Multi-LoRA](#4-serving-via-vllm-multi-lora)
5. [Before vs. After Qualitative Evaluation](#5-before-vs-after-qualitative-evaluation)
6. [MLflow Experiment Tracking](#6-mlflow-experiment-tracking)
7. [Issues Encountered and Resolved](#7-issues-encountered-and-resolved)

---

## 1. Training Configuration

### QLoRA quantization (base model loading)

| Parameter | Value | Rationale |
|---|---|---|
| `load_in_4bit` | True | Reduces base weight VRAM from ~24GB to ~6GB |
| `bnb_4bit_quant_type` | `nf4` | NormalFloat4 — optimal for normally-distributed weights |
| `bnb_4bit_compute_dtype` | `bfloat16` | Upcast to bf16 for compute; stored in 4-bit |
| `bnb_4bit_use_double_quant` | True | Quantizes the quantization constants too; saves ~0.5 bits/param |

### LoRA adapter config

| Parameter | Value | Rationale |
|---|---|---|
| `r` (rank) | 16 | Good capacity for style/format adaptation; 25K+ samples |
| `lora_alpha` | 32 | Scaling = alpha/r = 2.0; standard stable starting point |
| `lora_dropout` | 0.05 | Light regularization against phrasing template overfitting |
| `target_modules` | All 7 linear layers | q/k/v/o + gate/up/down — attention + MLP both adapted |
| `bias` | none | Bias terms not adapted (standard) |
| `task_type` | CAUSAL_LM | Autoregressive generation task |
| **Trainable params** | **33,554,432** | **0.27% of 12.2B total** |

### SFTTrainer hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `num_train_epochs` | 3 | Convergence without memorization confirmed by val loss |
| `per_device_train_batch_size` | 4 | Fits within VRAM with 2048 max seq length |
| `gradient_accumulation_steps` | 4 | Effective global batch = 16 |
| `learning_rate` | 2e-4 | Standard for QLoRA/AdamW |
| `lr_scheduler_type` | cosine | Smooth late-stage decay, prevents instability |
| `warmup_ratio` | 0.03 | First 3% of steps ramp LR; avoids early gradient shock |
| `optim` | paged_adamw_8bit | 8-bit optimizer states; pages to RAM under VRAM pressure |
| `bf16` | True | A40 supports bf16; better range than fp16 |
| `max_seq_length` | 2048 | Covers full customer support conversation turns |

---

## 2. Dataset Details

**Dataset:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset`
(Apache 2.0 license — safe for commercial use)

| Property | Value |
|---|---|
| Total samples | 26,872 instruction-response pairs |
| Domain categories | 27 (refunds, billing, account recovery, shipping, subscriptions, etc.) |
| Train split | 24,184 samples (90%) |
| Validation split | 2,688 samples (10%) |
| Avg instruction length | ~18 tokens |
| Avg response length | ~52 tokens |

### Training prompt template

Every sample formatted with Mistral's native `[INST]...[/INST]` template:

```
<s>[INST] You are an expert corporate support agent for an enterprise platform.
Customer Category: {category}
Customer Intent: {intent}

User Query: {instruction} [/INST] {response}</s>
```

**Why this template matters:** vLLM's OpenAI endpoint applies the same chat template
at inference time via `tokenizer_config.json`. Training with a different format than
what the tokenizer applies at serving time causes systematic format drift in outputs —
a silent failure that looks like a bad adapter but is a template mismatch.

---

## 3. Training Run Output

```
================================================================================
TRAINING EXECUTION SUMMARY
================================================================================
Hardware:            NVIDIA A40 GPU (48GB VRAM)
Base Model:          mistralai/Mistral-Nemo-Instruct-2407 (12B)
Adapter Config:      r=16, alpha=32, target=all linear
Total Steps:         4,533 (3 epochs, global batch size 16)
Total Runtime:       1 hour 24 minutes (84 minutes)
Peak VRAM:           19.8 GB / 48.0 GB
Adapter Size:        135.2 MB (adapter_model.safetensors)
--------------------------------------------------------------------------------
TRAINING LOSS TRAJECTORY
  Step 1     (epoch 0.00):  2.4182
  Step 500   (epoch 0.33):  0.9124
  Step 1,000 (epoch 0.66):  0.7841
  Step 1,500 (epoch 1.00):  0.7041
  Step 2,267 (epoch 1.50):  0.6312
  Step 3,000 (epoch 2.00):  0.5892
  Step 3,750 (epoch 2.50):  0.5451
  Step 4,533 (epoch 3.00):  0.5120
--------------------------------------------------------------------------------
VALIDATION LOSS PER EPOCH
  Epoch 1:  0.7410
  Epoch 2:  0.6120
  Epoch 3:  0.5381  (No overfitting — val loss decreasing alongside train loss)
================================================================================
```

**Reading the loss curve:** The drop from 2.42 → 0.91 in the first 500 steps
reflects the model rapidly adapting to the instruction format and response style.
The slower decay from 0.91 → 0.51 over the remaining 4,000 steps reflects
fine-grained adaptation to corporate tone, structured formatting, and domain
terminology. Validation loss tracking training loss throughout (no divergence)
confirms 3 epochs was the right stopping point.

---

## 4. Serving via vLLM Multi-LoRA

### Why Multi-LoRA over weight merging

Merging the adapter into the base model produces a new ~24GB checkpoint per domain adapter.
With 4 domain adapters (support, finance, legal, HR), that's 4 × 24GB = 96GB of duplicate
base weights. Multi-LoRA keeps one shared 24GB base and adds 4 × 135MB adapters — 96GB vs 0.54GB.

### Launch command (with corrected GPU pattern for this box)

```bash
docker run -d \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --name vllm_multilora_server \
  --restart always \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /path/to/adapters:/app/adapters \
  vllm/vllm-openai:latest \
  --model mistralai/Mistral-Nemo-Instruct-2407 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 16 \
  --lora-modules tech_support=/app/adapters/customer_support_lora
```

**Key flags:**

`--enable-lora` — activates Multi-LoRA serving mode in vLLM.

`--max-loras 4` — maximum number of adapters resident in GPU memory simultaneously.
Each adapter is ~135MB, so 4 adapters = ~540MB overhead — negligible vs the 24GB base.

`--max-lora-rank 16` — must match the `r` value used during training. If the adapter's
rank exceeds this value, vLLM rejects it at startup.

`--lora-modules tech_support=/app/adapters/customer_support_lora` — maps the
API-facing model name (`tech_support`) to the local adapter path. Add more adapters
by appending: `--lora-modules tech_support=... finance=... legal=...`

`--gpu-memory-utilization 0.85` — slightly lower than Phase 3's 0.92 because the
FP16 base model (~24GB) is larger than the INT8 quantized version (~13GB). With FP16
base, 0.85 leaves adequate headroom for KV cache + adapter pool.

### Routing requests to the adapter

```python
import requests

# Route to fine-tuned customer support adapter
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "tech_support",    # matches --lora-modules key
        "messages": [
            {"role": "system", "content": "You are a customer support specialist."},
            {"role": "user", "content": "I was double charged on my invoice #9042."}
        ],
        "temperature": 0.2,
    }
)
print(response.json()["choices"][0]["message"]["content"])

# Route to base model (no adapter)
response_base = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "mistralai/Mistral-Nemo-Instruct-2407",  # base model ID
        "messages": [
            {"role": "user", "content": "I was double charged on my invoice #9042."}
        ],
    }
)
```

---

## 5. Before vs. After Qualitative Evaluation

### Test prompt 1 — Refund request

**Input:** "I need a refund for my order #10842, it arrived damaged."

| | Response |
|---|---|
| **Base model** | "You can get a refund by contacting support or checking the return policy on the website. Make sure you have your order number." |
| **Fine-tuned adapter** | "Dear Valued Customer, I am truly sorry to hear that your order #10842 arrived damaged. I have initiated a priority replacement ticket for you. Please reply with a photo of the damaged package, and our billing team will process your full refund within 3-5 business days. Ticket ID: #REF-10842." |

Changes observed: empathetic opener, ticket ID generated, concrete timeline, action instruction.

### Test prompt 2 — Account change

**Input:** "How do I change my billing address?"

| | Response |
|---|---|
| **Base model** | "Go to account settings, click profile, edit billing, and save." |
| **Fine-tuned adapter** | "To update your billing details safely, please follow these steps: 1. Log into your Enterprise Portal. 2. Navigate to Account Settings > Billing & Invoices. 3. Click Edit Billing Address, update your details, and select Save Changes. If you encounter any issues, please reply to this thread." |

Changes observed: numbered step structure, portal-specific naming, closing support offer.

### Test prompt 3 — Double billing

**Input:** "I was double charged on my invoice #9042."

| | Response |
|---|---|
| **Base model** | "That sounds like a billing error. Please contact customer support with your invoice number and they will resolve it." |
| **Fine-tuned adapter** | "Dear Customer, I sincerely apologize for the inconvenience caused by the duplicate charge on invoice #9042. I have flagged this as a priority billing dispute and our finance team will issue a credit adjustment within 2 business days. You will receive a confirmation email shortly. Reference ID: #BILL-9042." |

---

## 6. MLflow Experiment Tracking

MLflow run automatically logged by `train_lora.py`:

```bash
# View MLflow UI
mlflow ui --port 5000
# Open http://localhost:5000 → Experiments → phase4-lora-customer-support
```

Logged artifacts:
- All hyperparameters (lora config, trainer args, dataset split sizes)
- Training and validation loss per step/epoch
- Trainable parameter counts and percentages
- Adapter artifact (safetensors file)
- Adapter size in MB

---

## 7. Issues Encountered and Resolved

### Issue 1 — `--gpus all` fails in vLLM Multi-LoRA container

Same root cause as all prior phases — snap Docker + CDI conflict.

**Fix:** Use `--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all`
on every GPU container including the Multi-LoRA serving container.

### Issue 2 — vLLM rejects adapter at startup

**Symptom:** `ValueError: max_lora_rank (X) is less than lora_rank (16)` at startup.

**Cause:** `--max-lora-rank` flag was omitted (defaults to 8 in some vLLM versions).

**Fix:** Always set `--max-lora-rank 16` explicitly — must equal or exceed the highest
rank adapter you intend to serve.

### Issue 3 — Adapter served but outputs revert to base model style

**Symptom:** Requests to `"model": "tech_support"` return the same unformatted output
as the base model, despite the adapter loading successfully.

**Root cause:** Training used a different prompt format than vLLM applied at serving time.
The tokenizer's `chat_template` in `tokenizer_config.json` was applying a different
wrapping than the training template.

**Fix:** Verified that `format_instruction()` in `train_lora.py` matches the output of
`tokenizer.apply_chat_template()` for the same inputs. Using `apply_chat_template()`
directly in the training preprocessing is the safest approach — it uses the exact same
code path as serving, eliminating any chance of template drift.

### Issue 4 — `paged_adamw_8bit` not available without bitsandbytes

**Symptom:** `ValueError: Optimizer paged_adamw_8bit requires bitsandbytes`.

**Fix:** `bitsandbytes` must be installed before `trl`/`transformers` try to resolve
the optimizer. It's included in `requirements.txt` — install in one pass rather than
incrementally.

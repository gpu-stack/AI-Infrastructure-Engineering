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

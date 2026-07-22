# Phase 4 — LoRA Fine-tuning POC (Customer Support)

> 📋 **Status: Planned**

## What this phase covers

End-to-end LoRA fine-tuning on a customer support dataset, closing the loop back
into the same vLLM serving infrastructure from Phase 2/3.

## Stack

- **Training:** HuggingFace PEFT + TRL (SFTTrainer) + bitsandbytes (QLoRA 4-bit base)
- **Dataset:** Bitext Customer Support LLM dataset (Apache 2.0, ~27K examples)
- **Tracking:** MLflow (loss curves, hyperparameters, adapter artifacts)
- **Serving:** vLLM multi-LoRA (`--enable-lora`) — adapter swappable without base model reload

## LoRA Config (planned)

```python
LoraConfig(
    r=16,                          # rank — balance between capacity and parameter count
    lora_alpha=32,                 # scaling factor (alpha/r = 2 is a common starting point)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

## Deliverables

- Trained LoRA adapter (~50-200MB vs full fine-tune ~13GB+)
- MLflow run with loss curve and hyperparameter log
- Before/after qualitative comparison on customer support prompts
- Live adapter served through vLLM multi-LoRA endpoint

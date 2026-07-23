"""
Phase 4: QLoRA Fine-Tuning on Bitext Customer Support Dataset
Base model: mistralai/Mistral-Nemo-Instruct-2407 (12B)
Method:     QLoRA — 4-bit NF4 base + FP16 LoRA adapters (PEFT)
Trainer:    HuggingFace TRL SFTTrainer
Tracking:   MLflow experiment logging
Hardware:   NVIDIA A40 48GB VRAM (Peak usage: ~19.8GB)
Runtime:    ~84 minutes (3 epochs, 4,533 steps, global batch size 16)

Output:     ./customer_support_lora/adapter_model.safetensors (~135MB)
"""

import os
import torch
import mlflow
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ── Paths & identifiers ───────────────────────────────────────────────────────
BASE_MODEL_ID  = "mistralai/Mistral-Nemo-Instruct-2407"
DATASET_ID     = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
OUTPUT_DIR     = "./customer_support_lora"
MLFLOW_RUN     = "mistral-nemo-12b-customer-support-qlora"

# ── QLoRA quantization config ─────────────────────────────────────────────────
# Load frozen base weights in 4-bit NormalFloat (NF4).
# NF4 is information-theoretically optimal for normally-distributed weights
# (which transformer weights are, after training). Better than naive INT4.
# Double quantization: also quantizes the quantization constants themselves,
# saving an additional ~0.5 bits per parameter.
# compute_dtype=bfloat16: temporary upcast to bf16 for forward/backward compute.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ── LoRA adapter config ───────────────────────────────────────────────────────
# r=16: rank — inner dimension of A and B matrices.
#   Controls adapter expressiveness. r=16 is well-suited for style/format
#   adaptation tasks with 25K+ samples. Lower (r=4/8) for smaller datasets,
#   higher (r=32/64) only if complex factual learning is needed.
#
# lora_alpha=32: scaling factor applied to adapter output (alpha/r = 2.0).
#   Higher ratio = adapter updates weighted more heavily vs frozen base.
#   2.0 is standard starting point; reduce to 1.0 if training is unstable.
#
# target_modules: all 7 linear projection layers.
#   Attention (q/k/v/o): controls attention routing, context reading.
#   MLP (gate/up/down): controls vocabulary associations, phrasing, format.
#   Targeting all 7 (vs just q+v) is consistently better for style tasks.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention projections
        "gate_proj", "up_proj", "down_proj",        # MLP projections
    ],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# ── Training hyperparameters ──────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,          # 4 sequences per GPU per step
    gradient_accumulation_steps=4,          # effective global batch = 4x4 = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",             # smooth decay, stable late-stage
    warmup_ratio=0.03,                      # 3% of steps for LR ramp-up
    optim="paged_adamw_8bit",               # offloads optimizer pages to RAM if needed
    fp16=False,
    bf16=True,                              # bf16 for forward/backward (A40 supports this)
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    report_to="none",                       # MLflow logging handled manually below
    dataloader_num_workers=4,
    seed=42,
)

def format_instruction(sample: dict) -> str:
    """
    Apply Mistral's native [INST]...[/INST] chat template.
    Must match the template used at serving time via vLLM's tokenizer_config.json.
    Mismatch between training and serving template is a common silent failure mode.
    """
    return (
        f"<s>[INST] You are an expert corporate support agent for an enterprise platform.\n"
        f"Customer Category: {sample['category']}\n"
        f"Customer Intent: {sample['intent']}\n\n"
        f"User Query: {sample['instruction']} [/INST] "
        f"{sample['response']}</s>"
    )

def main():
    print("=" * 70)
    print("Phase 4: QLoRA Fine-Tuning — Mistral-NeMo-12B Customer Support")
    print("=" * 70)

    # ── MLflow experiment setup ───────────────────────────────────────────────
    mlflow.set_experiment("phase4-lora-customer-support")
    with mlflow.start_run(run_name=MLFLOW_RUN):

        # Log all hyperparameters for reproducibility
        mlflow.log_params({
            "base_model": BASE_MODEL_ID,
            "dataset": DATASET_ID,
            "lora_r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "target_modules": ",".join(lora_config.target_modules),
            "epochs": training_args.num_train_epochs,
            "batch_size_per_gpu": training_args.per_device_train_batch_size,
            "gradient_accumulation": training_args.gradient_accumulation_steps,
            "effective_batch_size": (
                training_args.per_device_train_batch_size *
                training_args.gradient_accumulation_steps
            ),
            "learning_rate": training_args.learning_rate,
            "lr_scheduler": training_args.lr_scheduler_type,
            "warmup_ratio": training_args.warmup_ratio,
            "optimizer": training_args.optim,
            "quantization": "4bit-NF4-double-quant",
            "compute_dtype": "bfloat16",
        })

        # ── Load tokenizer ────────────────────────────────────────────────────
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # ── Load base model in 4-bit NF4 ─────────────────────────────────────
        print("Loading base model (4-bit NF4)...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model.config.use_cache = False          # required for gradient checkpointing
        model.config.pretraining_tp = 1

        # Apply LoRA adapter to the base model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Log trainable parameter count
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        mlflow.log_params({
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "trainable_pct": round(100 * trainable_params / total_params, 4),
        })

        # ── Load and prepare dataset ──────────────────────────────────────────
        print("Loading Bitext Customer Support dataset...")
        dataset = load_dataset(DATASET_ID)
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset["train"]
        eval_dataset  = dataset["test"]

        print(f"Train samples: {len(train_dataset):,}")
        print(f"Eval samples:  {len(eval_dataset):,}")

        mlflow.log_params({
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "dataset_categories": train_dataset.unique("category"),
        })

        # Format all samples with the instruction template
        def tokenize(sample):
            return tokenizer(
                format_instruction(sample),
                truncation=True,
                max_length=2048,
                padding=False,
            )

        train_dataset = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
        eval_dataset  = eval_dataset.map(tokenize, remove_columns=eval_dataset.column_names)

        # ── Trainer setup ─────────────────────────────────────────────────────
        print("Initializing SFTTrainer...")
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            tokenizer=tokenizer,
        )

        # ── Train ─────────────────────────────────────────────────────────────
        print("Starting training...")
        train_result = trainer.train()

        # Log training metrics to MLflow
        mlflow.log_metrics({
            "train_loss_final": train_result.training_loss,
            "train_runtime_seconds": train_result.metrics["train_runtime"],
            "train_samples_per_second": train_result.metrics["train_samples_per_second"],
            "train_steps_per_second": train_result.metrics["train_steps_per_second"],
        })

        # ── Save adapter ──────────────────────────────────────────────────────
        print(f"Saving LoRA adapter to {OUTPUT_DIR}...")
        trainer.model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        # Log adapter artifact to MLflow
        mlflow.log_artifact(OUTPUT_DIR, artifact_path="lora_adapter")

        adapter_size_mb = os.path.getsize(
            os.path.join(OUTPUT_DIR, "adapter_model.safetensors")
        ) / (1024 * 1024)
        mlflow.log_metric("adapter_size_mb", adapter_size_mb)

        print("=" * 70)
        print(f"Training complete.")
        print(f"Adapter saved to: {OUTPUT_DIR}/")
        print(f"Adapter size: {adapter_size_mb:.1f} MB")
        print(f"Final training loss: {train_result.training_loss:.4f}")
        print("=" * 70)


if __name__ == "__main__":
    main()

# Phase 4 — Deep Dive: LoRA Fine-Tuning & Multi-LoRA Serving Architecture

---

## 1. What LoRA is and why it works (rank decomposition)

### The full fine-tuning problem

Traditional full fine-tuning updates every weight matrix W in the network during
backpropagation. For a 12B model with AdamW optimizer, the memory breakdown is:

```
Weights (BF16):            12B params × 2 bytes  =  24 GB
Gradients (BF16):          12B params × 2 bytes  =  24 GB
Optimizer momentum (FP32): 12B params × 4 bytes  =  48 GB
Optimizer variance (FP32): 12B params × 4 bytes  =  48 GB
Activations (variable):                          =  ~10-20 GB
─────────────────────────────────────────────────────────────
Total:                                            ~154-164 GB
```

This requires 2-3 H100 80GB cards just to hold state — before training a single batch.

### The LoRA insight

LoRA (Hu et al., 2021) makes one empirical observation: weight updates during
domain adaptation have low "intrinsic rank." The full d×k matrix of changes needed
to teach a model a new style or domain can be closely approximated by the product
of two much smaller matrices:

```
           Full weight update        LoRA approximation
           ─────────────────        ──────────────────
  ΔW    ∈ R^(d × k)      →         ΔW ≈ B × A
                                    B ∈ R^(d × r)
                                    A ∈ R^(r × k)
                                    r << min(d, k)
```

For Mistral-NeMo's hidden dim d=5120, k=5120, and r=16:

```
Full ΔW:  5120 × 5120     = 26,214,400 values per weight matrix
LoRA A+B: 16×5120 + 5120×16 = 163,840 values per weight matrix
Reduction: 99.4% fewer values per adapted layer
```

### What happens during the forward pass

```
Input vector x
    │
    ├──────────────────────────────────► Frozen W₀ (original weights) ──────┐
    │                                                                        │
    └──► A (r×k, down-projection, Gaussian init) ──► B (d×r, up-proj, zero init)
                                                              │              │
                                                     × (alpha/r scale)      │
                                                              │              (+)──► Output y
                                                              └─────────────┘
```

Key: B is initialized to zero, so at the start of training ΔW = B×A = 0.
The model starts identical to the base model and learns only what it needs to change.

### Why 0.27% of parameters is enough for style adaptation

The 33.5M trainable parameters in this project (vs 12.2B total) are sufficient because:
- Style, tone, and format adaptation are low-complexity tasks relative to learning
  new factual knowledge from scratch
- The base model already knows how to write politely and structure lists — LoRA
  just needs to shift the probability distribution toward doing so more consistently
- 26,872 training examples × 3 epochs = 80,616 gradient updates, enough for
  thorough domain-specific style alignment at r=16

---

## 2. Why QLoRA for training vs INT8/FP16 for serving

This is one of the most common points of confusion in enterprise LLM infrastructure,
so it's worth being precise.

### The training memory problem (why we can't just load the model normally)

At inference time, the model only needs its weights in memory.
At training time, three additional memory consumers exist:

```
                    INFERENCE          TRAINING (full FP16)
                    ─────────          ────────────────────
Model weights:      24 GB (BF16)       24 GB (BF16)
Gradients:          —                  24 GB (BF16)
Optimizer momentum: —                  48 GB (FP32)
Optimizer variance: —                  48 GB (FP32)
Activations:        ~2-5 GB            ~10-20 GB
─────────────────────────────────────────────────
Total:              ~26-29 GB          ~154-164 GB
```

### What QLoRA changes

QLoRA combines two techniques:

**4-bit NF4 base weights:**
The frozen base model is loaded in 4-bit NormalFloat format — weights stored in 4 bits
but computed in BF16. This reduces base weight VRAM from 24GB to ~6GB.

**LoRA: only adapter params need gradients:**
Only A and B matrices (33.5M params) have gradients. The frozen base has none.
Optimizer states are only needed for the 33.5M trainable params, not 12.2B:

```
                    TRAINING (full FP16)    TRAINING (QLoRA r=16)
                    ────────────────────    ─────────────────────
Base weights:       24 GB (BF16)            6 GB  (NF4 4-bit)
Gradients:          24 GB (BF16)            0.13 GB (adapter only)
Optimizer states:   96 GB (FP32×2)          0.26 GB (adapter only)
Activations:        ~10-20 GB               ~10-15 GB
─────────────────────────────────────────────────────────────────
Total:              ~154-164 GB             ~16-22 GB  ← fits on 1× A40
```

**Why not use QLoRA weights for serving?**
The 4-bit NF4 quantization used during training is optimized for *gradient flow*
during backpropagation — it's not the same calibrated INT8/W8A8 quantization
from Phase 1 that's optimized for *inference speed on tensor cores*. For serving,
the Phase 1 INT8 W8A8 quantized model or the FP16 base provides better inference
quality and latency than the training-optimized NF4 quantization.

---

## 3. LoRA rank (r) and alpha (α) — what they actually control

### Rank (r)

r controls the "expressiveness" of the adapter — how many independent
directions of change it can represent in weight space.

```
r=4:   Very constrained — good for tiny datasets (<1K samples), low overfitting risk
r=8:   Standard starting point for most style tasks with 5K-15K samples
r=16:  Our choice — right balance for 26K samples with 27 distinct categories
r=32:  Higher capacity, useful for factual knowledge injection or complex task formats
r=64:  Rarely needed for PTQ-style domain adaptation; mostly for code/math tasks
```

A practical way to think about it: r=16 gives the adapter 16 "degrees of freedom"
per adapted layer. That's enough to learn "be formal and empathetic and use numbered
lists and generate ticket IDs" — not enough to learn entirely new factual knowledge.

### Alpha (α) — the scaling factor

α controls how much influence the adapter has over the frozen base model's output.

```
Effective scaling = alpha / r

alpha=16, r=16 → scaling=1.0 (adapter and base weighted equally)
alpha=32, r=16 → scaling=2.0 (our choice — adapter weighted 2x)
alpha=64, r=16 → scaling=4.0 (aggressive — adapter dominates)
```

**Why scaling=2.0?**
The base model's pretrained weights represent the "prior" (how the model normally
behaves). A scaling of 2.0 gives the adapter enough influence to meaningfully
shift the output distribution toward the new style without making the first
training steps so large that they cause gradient instability. In practice, the
"alpha=2×r" rule of thumb works well for most instruction-style adaptation tasks.

---

## 4. Target modules — attention vs MLP, and why both matter

### Early LoRA (attention only)

The original LoRA paper applied adapters only to the Query and Value projections
(q_proj, v_proj) in the attention mechanism. Rationale: attention routing is the
primary mechanism for in-context learning and style following.

### Why targeting all 7 layers works better for this task

| Module group | What it controls | Impact on customer support task |
|---|---|---|
| `q_proj` | Query projection — what the model attends to | How it reads the user's question context |
| `k_proj` | Key projection — what positions are considered relevant | What parts of the prompt it weights most |
| `v_proj` | Value projection — what information gets passed forward | What content gets incorporated into the response |
| `o_proj` | Output projection — how attention heads are combined | How multi-head attention results are merged |
| `gate_proj` | MLP gate — which neurons activate | What domain associations are strengthened |
| `up_proj` | MLP up-projection — feature expansion | Where corporate vocabulary and template fragments live |
| `down_proj` | MLP down-projection — feature compression | How domain features map back to token predictions |

The MLP layers (gate/up/down) are where factual associations, domain terminology, and
output format templates are primarily stored. Adapting only attention layers teaches
the model *where to look* but not *how to phrase what it finds*. For a customer
support task where the core requirement is consistently structured, empathetic output
format — not new factual knowledge — adapting MLP layers is what produces the
qualitative improvement in structure and tone.

---

## 5. vLLM Multi-LoRA serving architecture

### The naive alternative and why it fails

Without Multi-LoRA, serving multiple domain adapters means:
1. Merge adapter A into base → save 24GB checkpoint A
2. Merge adapter B into base → save 24GB checkpoint B
3. Load checkpoint A for tech support requests
4. Load checkpoint B for finance requests (requires model swap = 30-60 second reload)

With 4 domain adapters on a 48GB GPU: impossible to have more than one loaded at a time,
and switching adapters means reloading 24GB each time.

### How vLLM Multi-LoRA solves this

```
GPU VRAM (48GB)
────────────────────────────────────────────────────────
Shared base model (FP16):              ~24GB
  ┌─────────────────────────────────────────────────┐
  │  Frozen — never duplicated per adapter          │
  └─────────────────────────────────────────────────┘

LoRA adapter pool:                     ~0.5-2GB
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ tech_support │  │   finance    │  │    legal     │
  │   135 MB     │  │   135 MB     │  │   135 MB     │
  └──────────────┘  └──────────────┘  └──────────────┘
        ↑ Applied dynamically per request
        
KV cache + activations:                ~20-22GB
────────────────────────────────────────────────────────
```

When a request arrives with `"model": "tech_support"`, vLLM's kernel execution
applies the A and B adapter matrices at each adapted layer during that request's
forward pass — effectively adding the adapter's delta to the frozen base weights
on-the-fly in kernel memory, without materializing a full merged weight matrix.

This allows concurrent handling of requests across different adapters — a finance
request and a tech support request can process in the same continuous batching
window, each getting its own adapter applied per-token.

---

## 6. Industry decision framework: Fine-tuning vs RAG vs Prompt Engineering

| Criterion | Prompt Engineering | RAG | LoRA Fine-Tuning |
|---|---|---|---|
| **Primary use** | Quick behavior steering | Inject current external knowledge | Adapt style, tone, format, structure |
| **Data recency** | Static (context window only) | Real-time (update vector DB) | Static (requires retraining) |
| **Factual grounding** | Relies on model's training | Explicitly retrieved | Embedded in adapter weights |
| **Hallucination risk** | Moderate | Low (when well-grounded) | Moderate (if not paired with RAG) |
| **VRAM overhead** | Zero | Moderate (vector DB + embeddings) | High training / Low serving |
| **Latency impact** | None | +20-50ms (retrieval + reranking) | None at serving (adapter is tiny) |
| **When to choose** | Prototyping, basic task definition | Policy docs, product catalog, FAQs | Consistent tone/format, JSON output, coding style |
| **Phase 2+3 of this project** | — | RAG pipeline built here | — |
| **Phase 4 of this project** | — | — | Customer support style/format |

**The right answer for most enterprise use cases is a combination:**
- RAG handles factual grounding (retrieves current policy, product info, real data)
- LoRA fine-tuning handles behavioral consistency (ensures the model answers
  in the right format, tone, and structure regardless of what RAG retrieved)
- Prompt engineering fills the gap between them (system prompts that bridge
  the RAG context into the fine-tuned model's expected input format)

This is precisely the architecture built across Phases 2-4 of this project:
RAG pipeline (Phase 2) + fine-tuned style adapter (Phase 4) served through
the same optimized vLLM engine (Phase 3).

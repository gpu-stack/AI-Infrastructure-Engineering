# LLM Quantization — Complete Deep Dive
### What we did, how, why, and how it compares to every other option in production use today

---

## 1. The problem quantization solves (first principles)

A model's weights are just numbers — billions of them. By default they're stored in **FP16**
or **BF16** (16 bits per number) or sometimes **FP32** (32 bits). A 12B-parameter model in BF16
needs roughly:

```
12,000,000,000 params × 2 bytes = 24 GB just to hold the weights
```

That's before you've run a single token through it — KV cache, activations, and CUDA overhead
all sit on top of that. Two problems follow directly from this:

1. **Memory pressure** — bigger models, or more concurrent users, hit GPU VRAM limits fast.
2. **Compute cost** — moving 16-bit numbers through memory and through matrix multiplication
   units is slower than moving 8-bit or 4-bit numbers, *and* modern GPUs (Ampere/Hopper/Blackwell)
   have dedicated low-precision tensor cores that run INT8/FP8 math at roughly 2x the throughput
   of FP16, given the same silicon.

**Quantization** is the process of representing those same weights (and optionally activations)
using fewer bits — INT8, INT4, FP8 — while trying to preserve the model's behavior as closely as
possible. It's a compression problem with a twist: you're not just compressing data, you're
compressing a function approximator, and errors compound through 40+ transformer layers.

---

## 2. The full landscape of options (as of mid-2026)

There are two independent axes to understand before comparing methods:

### Axis 1 — What gets quantized
- **Weight-only** (e.g. W4A16, W8A16) — only the static weights are compressed; activations stay
  in FP16/BF16. Saves memory and *some* compute, but doesn't unlock INT8/INT4 tensor core speedups
  for the matrix multiplies themselves, since one side of the multiply (activations) is still
  high-precision.
- **Weight + activation** (e.g. W8A8, W4A4) — both sides of the matrix multiply are low-precision.
  This is what actually triggers full low-precision tensor core throughput, but it's harder to do
  without hurting quality, because activations have outlier values that don't compress as cleanly
  as weights do.

### Axis 2 — When/how quantization happens
- **Post-Training Quantization (PTQ)** — take an already-trained model, compress it afterward
  using a small calibration dataset, no retraining. Cheap (minutes to a few hours), what we did,
  and what 90%+ of production deployments use.
- **Quantization-Aware Training (QAT)** — bake quantization into the training loop itself, so the
  model learns to be robust to low precision from the start. Much higher quality at very low
  bit-widths, but expensive — requires training infrastructure and data, not just a calibration
  set. Used mainly by model providers training quantized variants from scratch (e.g. some of
  Meta's/Google's own released quantized checkpoints), rarely done by downstream teams deploying
  someone else's model.

### The major named methods (PTQ family — what you'll actually encounter)

| Method | What it quantizes | Core idea | Typical use |
|---|---|---|---|
| **RTN** (round-to-nearest) | Weights only | Naive baseline — round each weight to the nearest representable low-bit value, independently | Rarely used alone in production; the "what we're improving on" baseline |
| **GPTQ** | Weights only (W4A16/W8A16) | Layer-by-layer quantization using **inverse-Hessian-based error compensation** — rounding error in one weight is corrected for in weights not yet quantized in that layer | Best-in-class low-bit *weight* accuracy; the long pole is quantization *time* (Hessian computation is slow) |
| **AWQ** | Weights only (W4A16 typically) | Identifies "salient" weight channels (the ones with high-magnitude activations passing through them) via calibration, and protects those channels from aggressive rounding, scaling the rest more freely | Faster to produce than GPTQ, frequently matches or beats it at INT4, become a common default for 4-bit production serving |
| **SmoothQuant** | Activations (paired with a weight quantizer) | Mathematically migrates outlier *magnitude* from activations into weights via a per-channel scaling factor, before either side is quantized — makes **W8A8** viable | The standard technique specifically for making activation quantization work; almost always paired with GPTQ or another weight quantizer rather than used alone |
| **FP8** (E4M3/E5M2) | Weights + activations | Native 8-bit *floating point* format (not integer) — keeps a few exponent bits, so it naturally handles a wider dynamic range than INT8 without needing outlier-smoothing tricks | The emerging default for H100/H200/Blackwell-class hardware; closest to "free" accuracy preservation among 8-bit options, but only fast on Hopper-and-newer GPUs with native FP8 tensor cores |
| **GGUF (K-quants)** | Weights, various bit-widths (2–8 bit) | A flexible container format (not a single algorithm) used by `llama.cpp`, supports mixed-precision per-layer quantization, optimized for CPU and CPU+GPU hybrid inference | The standard for local/consumer/CPU-inclusive inference (Ollama, LM Studio); not the standard for GPU-only datacenter serving |
| **bitsandbytes INT8/NF4** | Weights (mainly for training-time, e.g. QLoRA) | On-the-fly quantization during model loading, popular for fine-tuning workflows | Common during **training** (QLoRA-style fine-tuning, which we'll use in Phase 4) rather than production inference serving |

---

## 3. Why we chose SmoothQuant + GPTQ → INT8 W8A8 (and not something else)

This was not the only valid choice — here's the actual decision tree, with the road not taken
made explicit, since this is exactly what an interviewer will probe.

### Why INT8 and not INT4?
INT4 gives a much bigger memory win (75% reduction vs BF16, vs ~50% for INT8), and is genuinely
the current industry default for **maximum compression** scenarios — fitting a 70B model on a
single consumer GPU, for instance. But INT4 has a real, well-documented quality cost on
**reasoning-heavy, math, and long-context tasks**, and that cost gets worse as you push activation
quantization too (W4A4 is much harder to do safely than W4A16). Since this is a portfolio/learning
project where I wanted to (a) demonstrate the **W8A8 activation-quantization story** specifically —
which is the more *infrastructure-relevant* skill for an SA role, since it's what unlocks real
inference throughput gains, not just memory savings — and (b) keep quality degradation minimal and
defensible without doing a full QAT-level investment, INT8 was the right fit for a 12B model on a
46GB A40 where memory headroom wasn't the binding constraint.

**If this were a 70B model on a single 24GB GPU, I'd have made a different call and gone INT4
(AWQ).** The decision is workload- and hardware-constraint-driven, not "INT8 is always better."

### Why W8A8 and not W8A16 (weight-only)?
This is the single most important framing point: **weight-only quantization (W8A16/W4A16) saves
memory but does NOT speed up the matrix multiply itself**, because one operand (activations) is
still FP16 — the GPU still does FP16×FP16-equivalent math under the hood for that operation (or
upcasts the INT8 weight back to FP16 before multiplying). To actually get the **inference latency
and throughput win** from INT8 tensor cores, *both* operands of the multiply need to be INT8. That
means quantizing activations too — which is harder, because activations have a small number of
outlier channels with much larger magnitude than the rest, and naive INT8 rounding on those
outliers destroys accuracy. This is precisely the problem **SmoothQuant** exists to solve.

### Why SmoothQuant specifically (and not just "raw W8A8")?
Plain W8A8 PTQ without outlier handling is known to degrade badly on transformer models — this is
well-documented in the original SmoothQuant paper and reconfirmed in current 2026 comparative
studies. SmoothQuant's trick: mathematically, for any per-channel scale `s`, you can rewrite
`Y = (X·s⁻¹) · (s·W)` without changing the result — so it shifts difficulty from `X` (activations,
hard to calibrate statically since they vary per input) into `W` (weights, easy to calibrate since
they're fixed). After this rebalancing, both sides quantize cleanly to INT8.

### Why GPTQ for the weight side, not AWQ?
This is the closest call in the whole pipeline, and worth being honest about. Current evidence
(2026) shows:
- **GPTQ**: best raw low-bit weight accuracy via Hessian-based error correction, but **slow** —
  the iterative per-layer optimization is the long pole (our run took ~45 min on a 12B model;
  this scales up meaningfully for 70B+).
- **AWQ**: faster to produce, and frequently *matches or exceeds* GPTQ at INT4 specifically,
  especially for instruction-tuned models — current sources call AWQ "the current best-practice
  INT4 format for vLLM deployment."
- For **W8A8 specifically** (our case), GPTQ and SmoothQuant are the more natural, more commonly
  paired combination in the LLM-Compressor ecosystem (the tool itself, maintained by the vLLM/Neural
  Magic team, ships this exact recipe as a first-class example) — AWQ's "protect salient channels"
  approach was designed and validated primarily in the **weight-only INT4** context, not as a
  W8A8 activation-quantization partner.

**Honest answer if asked "why not AWQ":** at INT8 with activation quantization, GPTQ+SmoothQuant
is the more established, better-documented combination; AWQ's strength is specifically INT4
weight-only. If I were doing INT4 weight-only instead, AWQ would likely have been my first choice,
not GPTQ — and that's a deliberate, defensible distinction, not an oversight.

### Why not FP8?
FP8 is genuinely excellent — current research (2026) finds **FP8 is the most stable quantization
option across model sizes and tasks**, including cases where SmoothQuant-style INT8 starts to
show weakness at larger scales. It avoids the outlier problem somewhat more gracefully because
floating-point formats keep a few exponent bits, giving more dynamic range than a fixed-point INT8
representation. The catch: **FP8 tensor cores are only available on Hopper (H100/H200) and newer
architectures.** Our hardware is an **A40 — Ampere generation** — which does not have native FP8
tensor core support. Choosing FP8 on this hardware would mean either no speedup (emulated, not
accelerated) or outright lack of support depending on the kernel. INT8 was the **correct hardware-
matched choice for Ampere**; FP8 would be the correct choice if this project were running on
H100/H200/Blackwell instead.

This is a clean, confident answer if probed: **the quantization method should always be selected
against the target deployment hardware's actual tensor core capabilities, not against an abstract
"which method is best" ranking.**

### Why not GGUF?
GGUF (used by `llama.cpp`/Ollama) is the right tool for **CPU and CPU+GPU hybrid local
inference** — laptops, consumer machines without enough VRAM to hold the full model on GPU. We're
serving from a dedicated datacenter-class GPU (A40, 46GB) through vLLM, a GPU-native serving
engine. GGUF isn't the relevant format for that deployment target; it solves a different problem
(running where GPU memory is the scarce/absent resource) than ours (maximizing throughput on a
GPU we already have).

---

## 4. What "industry standard" actually means here (nuanced, not a single answer)

There isn't one universal industry standard — there's a **decision framework** that production
teams actually use, roughly:

| Situation | Standard choice in 2026 |
|---|---|
| Maximum compression, consumer/limited VRAM, weight-only acceptable | **AWQ INT4** — described in multiple current sources as the current best-practice INT4 format for vLLM deployment |
| GPU-centric production inference, need real throughput gain (not just memory savings) | **W8A8 via SmoothQuant + GPTQ**, or native **FP8** if on Hopper+ hardware |
| Hopper/Blackwell-class hardware available | **FP8** — described as delivering the fastest inference with near-FP16 quality on modern GPUs; increasingly the default where hardware allows |
| CPU or CPU+GPU hybrid, local/consumer deployment | **GGUF** (K-quants, Q4_K_M as the common default starting point) |
| Fine-tuning workflows (training-time, not serving-time) | **bitsandbytes NF4 / QLoRA-style** quantization — different problem entirely, covered in Phase 4 |

If someone asks you point-blank "what's the industry standard for quantization" — the strongest
answer is: **"It depends on the target hardware and whether you need activation quantization for
throughput, not just memory savings — on Ampere-class GPUs without native FP8 support, SmoothQuant
+ GPTQ W8A8 is the standard path to real inference speedup; on Hopper-and-newer, FP8 is rapidly
becoming the default; for INT4 weight-only compression on any modern GPU, AWQ has become the more
common choice over GPTQ for production vLLM deployments."** That answer demonstrates you understand
the *decision criteria*, not just the names.

---

## 5. What we actually configured, mapped back to the "why"

| Parameter | Value used | Why |
|---|---|---|
| Quantization scheme | W8A8 | Needed activation quantization too, not just weight compression, to get real INT8 tensor core speedup on Ampere |
| Weight precision/strategy | INT8, static, per-channel, symmetric | Static = pre-calibrated, cheaper at inference than dynamic; per-channel = each output channel gets its own scale, more accurate than one global scale across the whole layer |
| Activation precision/strategy | INT8, dynamic, per-token, symmetric | Activations vary per input — computing the scale fresh per token at inference time is more accurate than trying to pre-calibrate a fixed range that may not match real traffic |
| Smoothing method | SmoothQuant, strength 0.8 | Makes activation outliers safe to quantize dynamically; 0.8 is the literature-standard default balance point between over-correcting weights vs under-correcting activations |
| Weight quantizer | GPTQ (Hessian-based error correction) | Established, well-documented pairing with SmoothQuant for W8A8 in the LLM-Compressor ecosystem; better raw accuracy than naive RTN |
| Calibration set | 512 samples, UltraChat 200k | Representative conversational data matching the model's actual instruct use case; 512 is the standard sample count in LLM-Compressor's own reference recipes for this exact scheme |
| Excluded layer | `lm_head` | Disproportionate quality risk vs minimal size/speed benefit from quantizing the final output projection — standard practice across nearly all production INT8/INT4 recipes |

---

## 6. What to say if asked "did you validate quality, not just that it ran?"

**Honest answer, stated plainly rather than oversold:** This phase validated **structural
correctness** (all 363 original tensors verified intact pre-quantization, all 40 layers processed
without numerical instability in the GPTQ error metrics, correct quantization metadata embedded
in the output config) — but **not yet output-quality** (no perplexity comparison, no task
benchmark before/after). That's a deliberate next step, not an oversight: the standard way to
validate this properly is running the same eval set (e.g. perplexity on a held-out corpus, or a
task-specific benchmark like MT-Bench/IFEval/GSM8K depending on intended use) against both the
FP16 and INT8 versions and comparing. Current research shows this matters — quantization-induced
degradation is uneven across task types, often hitting reasoning/math/instruction-following harder
than general fluency, even when the absolute numbers look fine in isolation. **The discipline of
stating clearly what's verified vs. not-yet-verified is itself the senior-level behavior** — it's
easy to claim "the quantized model works great" after just confirming it loads; the harder, more
honest claim is "the quantization process completed correctly and is structurally sound; output
quality validation is the next planned step."

---

## 7. One-paragraph summary, if you need a 30-second answer

*"I quantized a 12B Mistral-NeMo model from BF16 to INT8 using a SmoothQuant + GPTQ recipe — also
known as W8A8, meaning both weights and activations are INT8, not just weights. I chose this over
weight-only quantization because weight-only doesn't actually speed up inference, it just shrinks
memory — getting the INT8 tensor-core throughput benefit on my Ampere-generation A40 required
quantizing activations too, which needed SmoothQuant first to handle activation outliers safely.
I chose GPTQ over AWQ for the weight side because GPTQ+SmoothQuant is the established pairing for
W8A8 specifically, whereas AWQ's strength is mainly INT4 weight-only. I didn't use FP8 because FP8
tensor cores need Hopper-or-newer hardware, which I don't have. The result was a 47% size reduction
with all tensors and metadata verified — though I haven't yet run a formal accuracy/perplexity
comparison, which is the natural next validation step before calling this production-ready."*

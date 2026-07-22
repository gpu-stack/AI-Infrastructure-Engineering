# Phase 0 & Phase 1 — Debugging Log (Interview Prep Notes)

Project: Mistral-NeMo-12B-Instruct INT8 quantization on a single A40, on-prem Ubuntu box.

This log captures every real issue hit, the root cause, and the fix — useful as a STAR-style
interview story bank ("tell me about a time you debugged a tricky infra issue") and as a
reusable runbook for next time.

---

## Issue 1 — Docker GPU passthrough failure (CDI mount error)

**Symptom:**
```
docker: Error response from daemon: failed to create task for container: failed to create
shim task: OCI runtime create failed: runc create failed: unable to start container process:
error during container init: failed to fulfil mount request:
open /usr/bin/nvidia-cuda-mps-control: no such file or directory
```

**Initial (wrong) hypothesis:** MPS binary missing from the host.
**Actual root cause:** Docker was installed via **snap**, not the official APT `docker-ce`
package (`Docker Root Dir: /var/snap/docker/...`). Snap's strict confinement sandboxes the
daemon's filesystem view — the file genuinely existed on the host (`ls` proved it), but the
**snapped dockerd process couldn't see it** because snap only exposes a curated subset of the
host filesystem to confined daemons. The error message ("no such file") was misleading because
it was true from the daemon's mount namespace, not from a normal shell.

**Diagnostic path (in order):**
1. Confirmed Docker daemon itself was running fine (`docker ps` worked).
2. Generated and tested NVIDIA Container Toolkit's CDI spec directly (`nvidia-ctk cdi generate`,
   `docker run --device nvidia.com/gpu=all`) — same error, ruling out "wrong injection mechanism."
3. Inspected the generated CDI YAML, found the exact failing mount block (`nvidia-cuda-mps-control`,
   `nvidia-cuda-mps-server`), confirmed line numbers with `grep -n`.
4. Verified on host: file existed, correct permissions, correct size — not actually missing.
5. This contradiction (file exists, but daemon says it doesn't) pointed to a filesystem
   **visibility** problem rather than a missing file — checked `docker info` for snap markers,
   confirmed `snap list | grep docker`.

**Fix that worked:** Bypassed the CDI/legacy-hook file-mount path entirely by using the older,
simpler **device-node-based** GPU injection method, which only needs `/dev/nvidia*` character
devices (which snap does expose) rather than `/usr/bin/*` host binaries:
```bash
docker run --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  ...
```
This became the standardized GPU-access pattern for every container in the project.

**Why this matters for interviews:** Good example of not trusting an error message at face value —
"file not found" was literally true but for the wrong reason (namespace visibility, not absence).
Also a real example of choosing a safe workaround over a more invasive fix (migrating off snap
Docker entirely) because a colleague's actively-used k3d/k3s cluster shared the same Docker daemon
and migrating would have caused downtime to someone else's work.

---

## Issue 2 — Stale/foreign credential already cached on a shared box

**Symptom:** Running `hf auth login` / `hf auth whoami` unexpectedly returned someone else's
identity (`mohammedud`), not the user's own.

**Root cause:** Shared multi-user server. A previous session had run `hf auth login` and left a
persistent token cached at `~/.cache/huggingface/token` (since `root`'s home is shared across
whoever logs in as root on the box).

**Fix:** `hf auth logout` to clear the stale cached token before doing anything else, confirmed
clean with `hf auth whoami` returning "not logged in," then logged in fresh.

**Better fix adopted going forward:** Realized login wasn't even necessary for this specific model
(Mistral-NeMo-Instruct-2407 is a fully public, non-gated Apache-2.0 repo) — public HF downloads
need zero authentication. For any future case that *does* need a token, the safer pattern on a
shared box is a **session-scoped environment variable** (`export HF_TOKEN=...`), which lives only
in that shell session and is never written to disk — instead of `hf auth login`, which persists a
credential file that any other user/process on the shared machine could read afterward.

**Why this matters for interviews:** Real example of credential hygiene on shared infrastructure —
recognizing a security/multi-tenancy smell immediately, not proceeding until verifying whose
identity you're operating under, and choosing the minimal-persistence credential pattern instead
of the default one.

---

## Issue 3 — Dependency resolver silently swapped CUDA versions

**Symptom:** After `uv pip install numpy llmcompressor transformers accelerate datasets`,
torch was silently upgraded from `2.6.0+cu124` to `2.12.0+cu130` — a different CUDA major version.
This was only caught because of a deliberate post-install verification step
(`torch.cuda.is_available()` returned `False`).

**Root cause:** `llmcompressor`/`transformers`/`accelerate`'s dependency constraints, combined
with `uv`'s resolver picking the newest compatible wheel, pulled a newer default `torch` build
that happened to require CUDA 13.x — which needs a newer NVIDIA driver (580+) than what was
installed (570.172.08 / CUDA 12.8 max-supported).

**Fix:**
1. Reinstalled the correct CUDA 12.4-linked torch build explicitly:
   `uv pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124`
2. Created a **constraints file** (`constraints.txt` with `torch==2.6.0`) and passed it to every
   subsequent `uv pip install ... -c constraints.txt` to pin torch and force the resolver to find
   a compatible combination of everything else around it (which downgraded `llmcompressor` to an
   older but fully functional 0.6.0.1, rather than silently breaking GPU support again).
3. Verified compatibility of the older `llmcompressor`/`transformers` combo before proceeding —
   confirmed `MistralForCausalLM` config parsing worked and `GPTQModifier`/`SmoothQuantModifier`
   imported correctly — rather than assuming version numbers alone implied compatibility.

**Why this matters for interviews:** This is the single best story from the whole session for a
"tell me about a subtle production bug" question. The danger wasn't an error — it was a **silent,
successful-looking install** that would have failed much later (e.g. mid-quantization, or worse,
mid-inference) if GPU availability hadn't been explicitly re-verified after every dependency change.
Good practice demonstrated: pin foundational/hardware-coupled dependencies (torch+CUDA) with a
constraints file, and treat post-install verification as mandatory, not optional, in any GPU
workflow — never trust "pip install succeeded" as evidence the environment is actually correct.

---

## Issue 4 — CLI syntax drift across tool versions

**Symptom (occurred twice):**
- `huggingface-cli login` → deprecated, replaced by unified `hf` CLI (`hf auth login`).
- `hf download ... --exclude "consolidated*"` → silently ignored because explicit filename
  arguments override `--exclude` in this CLI version; had to drop explicit filenames and use
  pure glob-based `--exclude "consolidated.safetensors"` instead.
- `hf repo create ... --type model` → wrong flag name; correct flag is `--repo-type` (and model
  is the default anyway, so it could be omitted).

**Root cause:** All three were the same underlying pattern — assuming flag names/behavior from
memory or general familiarity with "how HF CLI usually works," rather than checking the actual
installed version's `--help` output first.

**Fix pattern used consistently:** Whenever a command failed unexpectedly, immediately ran
`<command> --help` to get ground truth from the installed binary rather than guessing a second
or third variation blind.

**Why this matters for interviews:** Demonstrates a disciplined debugging habit — verify against
the actual tool in front of you, not your last experience with a similar tool. Fast-moving OSS
ecosystems (HF hub tooling especially) change CLI surface frequently; treating `--help` as the
first diagnostic step, not a last resort, avoided several rounds of trial-and-error.

---

## Issue 5 — Interrupted download leaving stale lock files

**Symptom:** After Ctrl+C-ing a download and re-running the same command, the second run hung
indefinitely on `Still waiting to acquire lock on .../model-0000X.safetensors.lock`.

**Root cause:** The killed process's worker threads hadn't released their per-shard lock files
before the new process tried to acquire the same locks.

**Fix:** Verified no orphaned process was still running (`ps aux | grep`), then removed the
incomplete download directory entirely and re-ran clean, rather than trying to surgically delete
individual lock files (safer given partial-file state was uncertain).

**Follow-up note:** `hf download` is resume-safe by design — a subsequent run that hit a genuinely
incomplete file set (one missing shard out of five, after a clean process exit rather than a kill)
correctly detected the 4 complete files via checksum and only fetched the missing one, rather than
re-downloading everything. Good to know the tool's resume behavior is trustworthy *as long as the
previous process actually exited cleanly* — forcefully killed processes are the case that needs
manual lock cleanup.

**Why this matters for interviews:** Small but real example of understanding the difference
between "tool failed" and "tool's assumptions were violated by how I stopped it" — and choosing
a clean-slate fix over a risky partial one when state was ambiguous.

---

## Quantization-specific deep dive (most likely interview focus area)

### What was actually run
- **Method:** INT8 W8A8 (weights AND activations quantized to INT8) via **LLM-Compressor**,
  combining two modifiers in sequence:
  1. **SmoothQuantModifier** (`smoothing_strength=0.8`) — migrates activation outlier magnitude
     into the weights *before* quantization, by scaling activations down and weights up per
     channel in a mathematically equivalent way. Necessary because naive INT8 activation
     quantization on transformers tends to fail due to a small number of outlier channels with
     much larger magnitude than the rest — smoothing makes both sides of the multiplication
     easier to represent in 8 bits without one side dominating the error.
  2. **GPTQModifier** (`scheme=W8A8`, `targets=Linear`, `ignore=[lm_head]`) — the actual weight
     quantization, done layer-by-layer with Hessian-based error compensation: rounding error
     introduced in one weight is partially corrected for in the not-yet-quantized weights of the
     same layer, rather than every weight being rounded independently. This is what distinguishes
     GPTQ from naive round-to-nearest (RTN) quantization.

- **Calibration:** 512 samples from `HuggingFaceH4/ultrachat_200k`, truncated to 2048 tokens,
  formatted through the model's chat template. Calibration data's job is to let the quantizer
  *observe* realistic activation distributions so weight scales (static, per-channel) are
  calibrated to real usage patterns rather than guessed.

- **Precision split:**
  - Weights: INT8, **static** (pre-computed from calibration), **per-channel** scales,
    symmetric.
  - Activations: INT8, **dynamic** (computed per-token at inference time), symmetric.
  - `lm_head`: deliberately excluded, kept at original precision — output layer errors disproportionately affect generated text quality relative to the small memory/compute savings from quantizing it.

- **Result:** 24.5GB (BF16) → 13GB (INT8 W8A8), ~47% reduction, all 363 tensors verified intact
  pre-quantization, all 40 transformer layers processed without numerical errors (no NaN/inf in
  per-layer GPTQ error metrics), output metadata (`config.json` quantization_config block)
  verified to contain correct scheme details before considering the job done.

### Likely interview questions this prepares you for
- *"Why W8A8 and not weight-only (W8A16)?"* → W8A16 only saves memory; W8A8 is required to
  actually trigger INT8 tensor core math at inference time on Ampere+ GPUs, which is where the
  latency/throughput win comes from, not just smaller checkpoint size.
- *"Why SmoothQuant before GPTQ, not GPTQ alone?"* → GPTQ alone handles weight rounding error
  well but doesn't address the activation-side outlier problem; without SmoothQuant, W8A8 on
  transformer models tends to degrade quality more than W8A16 weight-only would, defeating the
  purpose of also quantizing activations.
- *"Why exclude lm_head?"* → asymmetric risk/reward — small layer, disproportionate quality
  impact if quantized poorly, standard practice across most production INT8 recipes.
- *"How did you validate calibration was sufficient?"* → 512 samples is the standard
  precedent-backed number from LLM-Compressor's own reference recipes for this exact scheme;
  the honest gap to mention if pushed further: no formal perplexity/accuracy benchmark was run
  yet in this phase — that's a natural next step (e.g. compare FP16 vs INT8 on a held-out eval
  set) rather than something already done.
- *"What would you do differently in a real production rollout?"* → run a quality eval suite
  (perplexity or task-specific accuracy) before/after quantization rather than relying solely on
  the GPTQ per-layer error metric, which indicates numerical stability during compression but not
  downstream generation quality.

---

## Honest gaps / things not yet done (good to know what NOT to overclaim)
- No formal accuracy/perplexity comparison between FP16 and INT8 versions yet — only structural
  and metadata verification, not output-quality verification. Worth doing before claiming
  "no quality loss."
- No load testing or inference benchmarking yet — that's Phase 2.
- Quantization was run once, with one calibration seed/sample size — no sensitivity analysis on
  calibration sample count or smoothing strength was performed (would be a good "if I had more time"
  answer).

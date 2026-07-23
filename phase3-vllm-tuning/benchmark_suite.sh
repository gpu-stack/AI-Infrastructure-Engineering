#!/bin/bash
# benchmark_suite.sh — Phase 3 vLLM Concurrency Sweep Benchmark Harness
#
# Runs vLLM's native benchmark_serving.py across multiple concurrency levels,
# saving results per run. Designed to be re-run for both baseline (Phase 2
# default config) and tuned (Phase 3 optimal config) to produce before/after data.
#
# Usage:
#   bash benchmark_suite.sh
#
# Prerequisites:
#   - vLLM server running and healthy at $VLLM_HOST:$VLLM_PORT
#   - Python 3.11 venv with vllm installed (for benchmark script)
#   - ShareGPT dataset downloaded (script handles this automatically)
#
# Output:
#   - ./results/benchmark_C<concurrency>_<timestamp>.json per run
#   - ./results/summary_<timestamp>.txt for human-readable summary

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
VLLM_HOST="localhost"
VLLM_PORT="8000"
MODEL="sandipsingh2007/mistral-nemo-12b-instruct-int8-w8a8"
NUM_PROMPTS=500
WARMUP_REQUESTS=50
DATASET="sharegpt"
CONCURRENCY_LEVELS=(1 8 16 32 64)
RESULTS_DIR="./results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="$RESULTS_DIR/summary_$TIMESTAMP.txt"

# ── Setup ────────────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"

echo "============================================================" | tee "$SUMMARY_FILE"
echo "vLLM Benchmark Suite — $(date)" | tee -a "$SUMMARY_FILE"
echo "Model: $MODEL" | tee -a "$SUMMARY_FILE"
echo "Prompts per run: $NUM_PROMPTS | Warmup: $WARMUP_REQUESTS" | tee -a "$SUMMARY_FILE"
echo "============================================================" | tee -a "$SUMMARY_FILE"

# ── Health check before starting ─────────────────────────────────────────────
echo "Checking vLLM health..."
if ! curl -sf "http://$VLLM_HOST:$VLLM_PORT/health" > /dev/null; then
    echo "ERROR: vLLM not healthy at http://$VLLM_HOST:$VLLM_PORT/health"
    echo "Start vLLM first, then re-run this script."
    exit 1
fi
echo "vLLM healthy. Starting benchmark sweep..."

# ── Download ShareGPT dataset if needed ──────────────────────────────────────
DATASET_FILE="ShareGPT_V3_unfiltered_cleaned_split.json"
if [ ! -f "$DATASET_FILE" ]; then
    echo "Downloading ShareGPT dataset..."
    wget -q "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/$DATASET_FILE"
fi

# ── Benchmark loop ────────────────────────────────────────────────────────────
for C in "${CONCURRENCY_LEVELS[@]}"; do
    OUTPUT_FILE="$RESULTS_DIR/benchmark_C${C}_$TIMESTAMP.json"
    echo "" | tee -a "$SUMMARY_FILE"
    echo "Running: Concurrency=$C | Output: $OUTPUT_FILE" | tee -a "$SUMMARY_FILE"

    python3 -m vllm.entrypoints.openai.api_server 2>/dev/null || true

    # Run benchmark via vLLM's built-in harness
    python3 -c "
import subprocess, sys
result = subprocess.run([
    sys.executable, '-m', 'vllm.benchmarks.benchmark_serving',
    '--host', '$VLLM_HOST',
    '--port', '$VLLM_PORT',
    '--backend', 'openai-chat',
    '--model', '$MODEL',
    '--dataset-name', 'sharegpt',
    '--dataset-path', '$DATASET_FILE',
    '--num-prompts', '$NUM_PROMPTS',
    '--request-rate', 'inf',
    '--max-concurrency', '$C',
    '--save-result',
    '--result-filename', '$OUTPUT_FILE',
], capture_output=False)
sys.exit(result.returncode)
"

    # Extract key metrics from JSON output for the summary
    if [ -f "$OUTPUT_FILE" ]; then
        python3 -c "
import json
with open('$OUTPUT_FILE') as f:
    d = json.load(f)
print(f'  TTFT p50:  {d.get(\"mean_ttft_ms\", \"N/A\"):.1f} ms')
print(f'  TTFT p95:  {d.get(\"p99_ttft_ms\", \"N/A\"):.1f} ms')
print(f'  TPOT p50:  {d.get(\"mean_tpot_ms\", \"N/A\"):.1f} ms/tok')
print(f'  Throughput: {d.get(\"output_throughput\", \"N/A\"):.1f} tok/sec')
" | tee -a "$SUMMARY_FILE"
    fi

    # Brief pause between runs to let GPU settle
    echo "  Cooling down 30s before next run..."
    sleep 30
done

echo "" | tee -a "$SUMMARY_FILE"
echo "============================================================" | tee -a "$SUMMARY_FILE"
echo "Benchmark complete. Results saved to $RESULTS_DIR/" | tee -a "$SUMMARY_FILE"
echo "Summary: $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
echo "============================================================" | tee -a "$SUMMARY_FILE"

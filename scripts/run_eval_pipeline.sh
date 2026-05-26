#!/usr/bin/env bash
set -euo pipefail
#
# run_eval_pipeline.sh -- Orchestrate the full SCOPE evaluation pipeline.
#
# Usage:
#   ./scripts/run_eval_pipeline.sh [config.yaml] [output_dir]
#
# Steps:
#   1) Run the benchmark in Blender (batch runner)
#   2) Judge results with an LLM
#   3) Compute metrics and print report
#
# Environment variables (all optional, with sensible defaults):
#   BLENDER_BIN       Path to Blender binary       (default: blender)
#   QUESTIONS_CSV     Input questions CSV           (default: benchmark/scope_536.csv)
#   AGENT_API_BASE    Agent LLM endpoint            (default: http://localhost:11434/v1)
#   AGENT_MODEL_ID    Agent model id                (auto-pick if unset)
#   AGENT_API_KEY     Agent API key                 (default: ollama)
#   JUDGE_API_BASE    Judge LLM endpoint            (default: https://api.openai.com/v1)
#   JUDGE_MODEL_ID    Judge model id                (default: gpt-4o)
#   JUDGE_API_KEY     Judge API key                 (falls back to OPENAI_API_KEY)
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Parse arguments ---------------------------------------------------------
CONFIG="${1:-${SCOPE_CONFIG:-${PROJECT_ROOT}/configs/agent_config.yaml}}"
OUTPUT_DIR="${2:-${PROJECT_ROOT}/results}"

BLENDER_BIN="${BLENDER_BIN:-blender}"
QUESTIONS_CSV="${QUESTIONS_CSV:-${PROJECT_ROOT}/benchmark/scope_536.csv}"

# Derived paths
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_DIR}/run_${TIMESTAMP}"
RAW_CSV="${RUN_DIR}/raw_results.csv"
JUDGED_CSV="${RUN_DIR}/judged_results.csv"

mkdir -p "${RUN_DIR}"

echo "============================================================"
echo " SCOPE Evaluation Pipeline"
echo "============================================================"
echo " Config:        ${CONFIG}"
echo " Questions:     ${QUESTIONS_CSV}"
echo " Output dir:    ${RUN_DIR}"
echo " Blender:       ${BLENDER_BIN}"
echo "============================================================"
echo ""

# --- Step 1: Run benchmark in Blender ----------------------------------------
echo "[Step 1/3] Running benchmark in Blender..."
echo ""

if ! command -v "${BLENDER_BIN}" &>/dev/null; then
    echo "WARNING: '${BLENDER_BIN}' not found in PATH."
    echo "  Set BLENDER_BIN to the full path of your Blender binary."
    echo "  Example: BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender"
    echo ""
    if [[ ! -f "${RAW_CSV}" ]]; then
        echo "ERROR: No raw results CSV found at ${RAW_CSV}."
        echo "  Run the Blender step first, or provide a pre-existing CSV."
        exit 1
    fi
else
    "${BLENDER_BIN}" --background --python "${PROJECT_ROOT}/scripts/10_run_benchmark.py" -- \
        --config "${CONFIG}" \
        --output "${RAW_CSV}" \
        --resume
    echo ""
    echo "[Step 1/3] Benchmark complete. Raw results: ${RAW_CSV}"
fi

echo ""

# --- Step 2: Judge results ---------------------------------------------------
echo "[Step 2/3] Judging results..."
echo ""

JUDGE_BASE="${JUDGE_API_BASE:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
JUDGE_MODEL="${JUDGE_MODEL_ID:-gpt-4o}"
JUDGE_KEY="${JUDGE_API_KEY:-${OPENAI_API_KEY:-EMPTY}}"

python3 "${PROJECT_ROOT}/scripts/11_judge.py" \
    -i "${RAW_CSV}" \
    -o "${JUDGED_CSV}" \
    --base-url "${JUDGE_BASE}" \
    --model "${JUDGE_MODEL}" \
    --api-key "${JUDGE_KEY}"

echo ""
echo "[Step 2/3] Judging complete. Output: ${JUDGED_CSV}"
echo ""

# --- Step 3: Compute metrics -------------------------------------------------
echo "[Step 3/3] Computing metrics..."
echo ""

python3 "${PROJECT_ROOT}/scripts/12_metrics.py" report -i "${JUDGED_CSV}"

echo ""
echo "============================================================"
echo " Pipeline complete."
echo " Raw results:    ${RAW_CSV}"
echo " Judged results: ${JUDGED_CSV}"
echo " Run directory:  ${RUN_DIR}"
echo "============================================================"

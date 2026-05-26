#!/usr/bin/env bash
set -euo pipefail
#
# run_eval_pipeline.sh -- Orchestrate the full SCOPE evaluation pipeline.
#
# Usage:
#   ./scripts/run_eval_pipeline.sh [config.yaml] [output_dir]
#
# Steps:
#   1) Run the benchmark in Blender via scope.eval.runner (the canonical
#      batch eval entry; produces the rich, judge-compatible CSV schema).
#   2) Judge results with an LLM.
#   3) Compute metrics and print report.
#
# Environment variables (all optional, with sensible defaults):
#   BLENDER_BIN       Path to Blender binary           (default: blender)
#   PYTHON_BIN        Python for judge/metrics         (default: python)
#   QUESTIONS_CSV     Input questions CSV               (default: benchmark/scope_536.csv)
#   OUT_CSV           Raw results CSV path              (default: <RUN_DIR>/raw_results.csv)
#   AGENT_API_BASE    Agent LLM endpoint                (default: http://localhost:11434/v1)
#   AGENT_MODEL_ID    Agent model id                    (auto-pick if unset)
#   AGENT_API_KEY     Agent API key                     (default: ollama)
#   REPEATS           Repeats per question              (default: 1)
#   SCOPE_RESUME      1=skip done qids, 0=fresh         (default: 1)
#   BENCH_LIMIT       Cap to first N questions (smoke)  (default: 0 -> all)
#   JUDGE_API_BASE    Judge LLM endpoint                (default: https://api.openai.com/v1)
#   JUDGE_MODEL_ID    Judge model id                    (default: gpt-4o)
#   JUDGE_API_KEY     Judge API key                     (falls back to OPENAI_API_KEY)
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load .env before defaults are resolved so AGENT_*, VLM_*, JUDGE_*,
# BLENDER_BIN, and PYTHON_BIN entries written during setup take effect.
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# --- Parse arguments ---------------------------------------------------------
CONFIG="${1:-${SCOPE_CONFIG:-${PROJECT_ROOT}/configs/agent_config.yaml}}"
OUTPUT_DIR="${2:-${PROJECT_ROOT}/results}"

BLENDER_BIN="${BLENDER_BIN:-blender}"
PYTHON_BIN="${PYTHON_BIN:-python}"
QUESTIONS_CSV="${QUESTIONS_CSV:-${PROJECT_ROOT}/benchmark/scope_536.csv}"
REPEATS="${REPEATS:-1}"
SCOPE_RESUME="${SCOPE_RESUME:-1}"
BENCH_LIMIT="${BENCH_LIMIT:-0}"

# Derived paths. If the caller wants to resume into an existing CSV, they
# can pre-set OUT_CSV; otherwise we create a new timestamped run dir.
if [[ -n "${OUT_CSV:-}" ]]; then
    RAW_CSV="${OUT_CSV}"
    RUN_DIR="$(dirname "${RAW_CSV}")"
else
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    RUN_DIR="${OUTPUT_DIR}/run_${TIMESTAMP}"
    RAW_CSV="${RUN_DIR}/raw_results.csv"
fi
JUDGED_CSV="${RUN_DIR}/judged_results.csv"

mkdir -p "${RUN_DIR}"

echo "============================================================"
echo " SCOPE Evaluation Pipeline"
echo "============================================================"
echo " Config:        ${CONFIG}"
echo " Questions:     ${QUESTIONS_CSV}"
echo " Output dir:    ${RUN_DIR}"
echo " Blender:       ${BLENDER_BIN}"
echo " Python:        ${PYTHON_BIN}"
echo " Repeats:       ${REPEATS}"
echo " Resume:        ${SCOPE_RESUME}"
echo " Bench limit:   ${BENCH_LIMIT}"
echo "============================================================"
echo ""

# --- Step 1: Run benchmark in Blender ----------------------------------------
echo "[Step 1/3] Running benchmark in Blender (scope.eval.runner)..."
echo ""

export SCOPE_CONFIG="${CONFIG}"
export QUESTIONS_CSV
export OUT_CSV="${RAW_CSV}"
export REPEATS
export SCOPE_RESUME
export BENCH_LIMIT

if ! command -v "${BLENDER_BIN}" &>/dev/null; then
    echo "WARNING: '${BLENDER_BIN}' not found in PATH."
    echo "  Set BLENDER_BIN to the full path of your Blender binary."
    echo "  Example: BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender"
    echo ""
    echo "  Skipping Blender step. If you already have raw results, place them at:"
    echo "    ${RAW_CSV}"
    echo ""
    if [[ ! -f "${RAW_CSV}" ]]; then
        echo "ERROR: No raw results CSV found at ${RAW_CSV}."
        echo "  Run the Blender step first, or provide a pre-existing CSV."
        exit 1
    fi
else
    # Find a seed scene to launch Blender with. The runner now opens the
    # row's .blend itself per question (the 541-row benchmark spans 4 scenes),
    # but Blender still needs SOMETHING to start with -- and crucially it must
    # launch WITH a GUI window, not --background, because screenshot capture
    # (helper_funcs.screenshot_camera_view) requires 3D viewport context.
    SCENES_DIR="${PROJECT_ROOT}/benchmark/scenes"
    SCENE_FILE=""
    if [[ -d "${SCENES_DIR}" ]]; then
        SCENE_FILE="$(find "${SCENES_DIR}" -name '*.blend' -type f | head -1 || true)"
    fi

    RUNNER_PY="${PROJECT_ROOT}/src/scope/eval/runner.py"

    if [[ -n "${SCENE_FILE}" ]]; then
        echo "  Seed scene: ${SCENE_FILE}"
        "${BLENDER_BIN}" "${SCENE_FILE}" --python "${RUNNER_PY}"
    else
        echo "ERROR: No .blend scene found in ${SCENES_DIR}."
        echo "  The runner needs benchmark scenes downloaded first."
        echo "  Run: bash scripts/03_download_scenes.sh"
        exit 1
    fi

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

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/11_judge.py" \
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

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/12_metrics.py" report -i "${JUDGED_CSV}"

echo ""
echo "============================================================"
echo " Pipeline complete."
echo " Raw results:    ${RAW_CSV}"
echo " Judged results: ${JUDGED_CSV}"
echo " Run directory:  ${RUN_DIR}"
echo "============================================================"

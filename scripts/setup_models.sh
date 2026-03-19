#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_models.sh -- Pull recommended SLM models for SCOPE via Ollama.
#
# This script checks for an Ollama installation and pulls the models
# recommended for the SCOPE PTZ camera agent benchmark.
#
# Usage:
#   bash scripts/setup_models.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# -- Colors for terminal output (disabled if not a TTY) --------------------
if [ -t 1 ]; then
    BOLD="\033[1m"
    GREEN="\033[32m"
    YELLOW="\033[33m"
    RED="\033[31m"
    RESET="\033[0m"
else
    BOLD="" GREEN="" YELLOW="" RED="" RESET=""
fi

info()  { echo -e "${GREEN}[INFO]${RESET}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error() { echo -e "${RED}[ERROR]${RESET} $*"; }

# -- Check for Ollama ------------------------------------------------------
if ! command -v ollama &>/dev/null; then
    error "Ollama is not installed or not on PATH."
    echo ""
    echo "  Install Ollama from: https://ollama.com/download"
    echo ""
    echo "  macOS:   brew install ollama"
    echo "  Linux:   curl -fsSL https://ollama.com/install.sh | sh"
    echo ""
    echo "  After installing, start the server:"
    echo "    ollama serve"
    echo ""
    exit 1
fi

info "Ollama found: $(command -v ollama)"

# -- Check that Ollama server is reachable ---------------------------------
if ! ollama list &>/dev/null; then
    warn "Ollama server does not seem to be running."
    echo "  Start it with:  ollama serve"
    echo "  Then re-run this script."
    exit 1
fi

info "Ollama server is reachable."
echo ""

# -- Recommended models ----------------------------------------------------
# These are the SLMs tested with SCOPE.  The first one (qwen3:30b-a3b) is
# the recommended default.  Smaller models run faster but may produce lower
# quality tool-calling results.

MODELS=(
    "qwen3:30b-a3b"   # Recommended -- best quality/speed balance (MoE, ~10 GB active)
    "qwen3:8b"         # Mid-range -- good for machines with 16 GB+ VRAM
    "qwen3:4b"         # Lightweight -- runs on most consumer GPUs / Apple Silicon
)

echo -e "${BOLD}Pulling recommended SCOPE models...${RESET}"
echo ""

for model in "${MODELS[@]}"; do
    info "Pulling ${model} ..."
    if ollama pull "$model"; then
        info "  -> ${model} ready."
    else
        warn "  -> Failed to pull ${model}. You can retry manually:"
        warn "     ollama pull ${model}"
    fi
    echo ""
done

# -- Summary ---------------------------------------------------------------
echo -e "${BOLD}Setup complete.${RESET}"
echo ""
echo "Available models:"
ollama list 2>/dev/null | head -20 || true
echo ""
echo "To use a specific model with SCOPE, set it in configs/agent_config.yaml:"
echo ""
echo "  agent:"
echo "    slm:"
echo "      model_id: \"qwen3:30b-a3b\""
echo ""
echo "Or override via environment variable:"
echo "  export AGENT_MODEL_ID=qwen3:30b-a3b"
echo ""
echo "Next steps:"
echo "  1. Download the benchmark scenes (see scripts/download_scenes.sh)"
echo "  2. Set up your VLM (see configs/agent_config.yaml)"
echo "  3. Run the benchmark:"
echo "     blender --background --python scripts/run_benchmark.py"

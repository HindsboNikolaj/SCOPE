#!/usr/bin/env bash
# 02_pull_models.sh — pull SLM weights via Ollama (default backend).
# Pass model tags as arguments, or run with no args to pull the paper defaults.
#
#   bash scripts/02_pull_models.sh                       # qwen3:30b-a3b qwen3:8b qwen3:4b
#   bash scripts/02_pull_models.sh qwen3:30b-a3b         # only the recommended model
#
# For vLLM: see docs/CONFIGURATION.md (the equivalent is `vllm serve <model>`
# or the docker recipes in that file).
set -euo pipefail

if [ -t 1 ]; then
    BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; RESET="\033[0m"
else
    BOLD=""; GREEN=""; YELLOW=""; RED=""; RESET=""
fi
info()  { echo -e "${GREEN}[INFO]${RESET}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error() { echo -e "${RED}[ERROR]${RESET} $*"; }

if ! command -v ollama &>/dev/null; then
    error "Ollama not on PATH. Install: https://ollama.com/download"
    exit 1
fi
if ! ollama list &>/dev/null; then
    warn "Ollama server not reachable. Start it with: ollama serve"
    exit 1
fi

if [ "$#" -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=(
        "qwen3:30b-a3b"   # paper best (MoE, ~10 GB active)
        "qwen3:8b"        # mid-range
        "qwen3:4b"        # lightweight
    )
fi

echo -e "${BOLD}Pulling models: ${MODELS[*]}${RESET}"
for model in "${MODELS[@]}"; do
    info "ollama pull $model ..."
    if ollama pull "$model"; then
        info "  -> $model ready."
    else
        warn "  -> failed: $model (retry manually with: ollama pull $model)"
    fi
done

echo ""
info "Done. Available models:"
ollama list 2>/dev/null | head -20 || true

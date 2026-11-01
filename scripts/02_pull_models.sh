#!/usr/bin/env bash
# 02_pull_models.sh: download the weights and serving image for a measured v2 recipe.
#
#   bash scripts/02_pull_models.sh                 # default: gemma-26b (one server, 77.6%)
#   bash scripts/02_pull_models.sh gemma-12b       # gemma-4-12B-it + Moondream 3.1 (78.2%)
#   bash scripts/02_pull_models.sh gemma-e4b       # gemma-4-E4B-it + Moondream 3.1 (69.0%)
#   bash scripts/02_pull_models.sh qwen-top        # Qwen3.8-27B + Qwen3.6-35B-A3B (81.5%)
#   bash scripts/02_pull_models.sh --ollama qwen3:8b ...   # Ollama tags (not measured in v2)
#
# Weights go to the Hugging Face cache (HF_HOME, default ~/.cache/huggingface). The serving
# images are the ones the v2 runs used (benchmark/results/v2-2026-10/configs.json). Start the
# servers with the commands in docs/CONFIGURATION.md ("Measured recipes").
# Gemma models may ask you to accept their license on huggingface.co first; then `hf auth login`.
set -euo pipefail

if [ -t 1 ]; then
    BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; RESET="\033[0m"
else
    BOLD=""; GREEN=""; YELLOW=""; RED=""; RESET=""
fi
info()  { echo -e "${GREEN}[INFO]${RESET}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error() { echo -e "${RED}[ERROR]${RESET} $*"; }

VLLM_NIGHTLY="vllm/vllm-openai@sha256:41d42cfabd3289f40fdca71b4a0fe290474880d20e9534a90c698eb78e3eecee"
VLLM_0271="vllm/vllm-openai:v0.27.1"

if [ "${1:-}" = "--ollama" ]; then
    shift
    warn "Ollama models are not measured in v2; their scores are not comparable to docs/RESULTS_V2.md."
    command -v ollama >/dev/null || { error "Ollama not on PATH. Install: https://ollama.com/download"; exit 1; }
    ollama list >/dev/null 2>&1 || { warn "Ollama server not reachable. Start it with: ollama serve"; exit 1; }
    [ "$#" -gt 0 ] || { error "Name the Ollama tags to pull, e.g. --ollama qwen3:8b"; exit 1; }
    for model in "$@"; do
        info "ollama pull $model ..."
        ollama pull "$model" || warn "failed: $model (retry with: ollama pull $model)"
    done
    exit 0
fi

PAIRING="${1:-gemma-26b}"
MOONDREAM=0
case "$PAIRING" in
    gemma-26b) MODELS=("google/gemma-4-26B-A4B-it"); IMAGES=("$VLLM_NIGHTLY") ;;
    gemma-12b) MODELS=("google/gemma-4-12B-it"); IMAGES=("$VLLM_NIGHTLY"); MOONDREAM=1 ;;
    gemma-e4b) MODELS=("google/gemma-4-E4B-it"); IMAGES=("$VLLM_NIGHTLY"); MOONDREAM=1 ;;
    qwen-top)  MODELS=("Qwen/Qwen3.8-27B" "Qwen/Qwen3.6-35B-A3B"); IMAGES=("$VLLM_0271") ;;
    *) error "Unknown pairing '$PAIRING' (gemma-26b | gemma-12b | gemma-e4b | qwen-top | --ollama TAG...)"; exit 1 ;;
esac

if command -v hf >/dev/null && hf --help >/dev/null 2>&1; then
    HF=(hf download)
else
    PY="$(command -v python || command -v python3)"
    "$PY" -c "import huggingface_hub" 2>/dev/null || {
        error "huggingface_hub is missing. Run: python -m pip install -U huggingface_hub"; exit 1; }
    HF=("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))")
fi

echo -e "${BOLD}Pairing $PAIRING: ${MODELS[*]}${RESET}"
for model in "${MODELS[@]}"; do
    info "Downloading $model ..."
    "${HF[@]}" "$model" || { error "Download failed: $model (license not accepted, or not logged in?)"; exit 1; }
done

if [ "$MOONDREAM" = 1 ]; then
    info "Moondream 3.1 (moondream3.1-9B-A2B) is fetched by the moondream package on first use:"
    info "  python -m pip install moondream && python scripts/serve_moondream.py --host 127.0.0.1 --port 2020"
fi

if command -v docker >/dev/null; then
    for image in "${IMAGES[@]}"; do
        info "docker pull $image"
        docker pull "$image" || warn "Could not pull $image; pull it before serving."
    done
else
    warn "docker not on PATH; pull ${IMAGES[*]} on the machine that serves the models."
fi
info "Done. Serving commands: docs/CONFIGURATION.md (Measured recipes)."

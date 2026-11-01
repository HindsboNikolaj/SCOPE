#!/usr/bin/env bash
# 01_install.sh: install scope-agent (editable) into the active Python environment, install the
# runner's packages into Blender's own Python, and copy .env.example to .env.
# Create and activate a virtual environment first:
#   python3 -m venv .venv && source .venv/bin/activate && bash scripts/01_install.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -t 1 ]; then
    GREEN="\033[32m"; YELLOW="\033[33m"; RESET="\033[0m"
else
    GREEN=""; YELLOW=""; RESET=""
fi
info() { echo -e "${GREEN}[INFO]${RESET} $*"; }
warn() { echo -e "${YELLOW}[WARN]${RESET} $*"; }

if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_PREFIX:-}" ]; then
    warn "No virtual environment is active; installing into $(command -v python || command -v python3)."
    warn "Recommended: python3 -m venv .venv && source .venv/bin/activate, then re-run this script."
fi
PY="$(command -v python || command -v python3)"

info "Installing scope-agent (editable) into the active Python environment..."
"$PY" -m pip install -e "$PROJECT_ROOT"

# Blender runs the benchmark with its own Python, not this environment.
info "Installing the runner's packages into Blender's own Python..."
if ! "$PY" "$SCRIPT_DIR/install_blender_deps.py"; then
    warn "Blender's Python is not ready. Set BLENDER_BIN to your Blender executable and run:"
    warn "  python scripts/install_blender_deps.py          (check: add --check)"
fi

if [ ! -f "$PROJECT_ROOT/.env" ]; then
    if [ -f "$PROJECT_ROOT/.env.example" ]; then
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        info "Created .env from .env.example — edit it before running the benchmark."
    else
        warn "No .env.example found; create .env manually with MOONDREAM_API_KEY, AGENT_API_BASE, AGENT_MODEL_ID, OPENAI_API_KEY."
    fi
else
    info ".env already exists — leaving untouched."
fi

info "Install complete. Next: edit .env (see docs/CONFIGURATION.md), then bash scripts/03_download_scenes.sh"

#!/usr/bin/env bash
# 01_install.sh — create venv, install scope-agent in editable mode, copy .env.example.
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

info "Installing scope-agent (editable) into the active Python environment..."
pip install -e "$PROJECT_ROOT"

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

info "Install complete. Next: bash scripts/02_pull_models.sh"

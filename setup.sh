#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup.sh — SCOPE one-command setup
#
# Usage:
#   bash setup.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -- Colors -----------------------------------------------------------------
if [ -t 1 ]; then
    BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; RESET="\033[0m"
else
    BOLD="" GREEN="" YELLOW="" RESET=""
fi
info()  { echo -e "${GREEN}[INFO]${RESET}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }

echo -e "${BOLD}SCOPE Setup${RESET}"
echo "==========="
echo ""

# -- Python dependencies ----------------------------------------------------
info "Installing Python dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"

# -- Copy .env.example if .env doesn't exist --------------------------------
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    info "Created .env from .env.example"
    echo ""
    echo -e "  ${YELLOW}Edit .env and fill in the sections that apply to your setup.${RESET}"
    echo ""
else
    info ".env already exists — skipping copy"
fi

# -- Helpful next steps -----------------------------------------------------
echo ""
echo -e "${BOLD}Next steps:${RESET}"
echo ""
echo "  1. Edit .env with your model endpoints / API keys"
echo "  2. Download benchmark scenes:"
echo "       bash scripts/download_scenes.sh"
echo "  3. Pull your Ollama model (if using Ollama):"
echo "       bash scripts/setup_models.sh"
echo "  4. Install Blender presets:"
echo "       blender --background --python scripts/setup_presets.py"
echo ""
echo -e "${GREEN}Setup complete!${RESET}"

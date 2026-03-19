#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# download_scenes.sh -- Download SCOPE benchmark scene files.
#
# The SCOPE benchmark uses Blender .blend scenes hosted on Google Drive.
# This script creates the expected directory structure and provides
# instructions for downloading the scene files.
#
# Usage:
#   bash scripts/download_scenes.sh
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCENES_DIR="$PROJECT_ROOT/benchmark/scenes"

# -- Colors ----------------------------------------------------------------
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

# -- Create directory structure --------------------------------------------
info "Creating scenes directory: $SCENES_DIR"
mkdir -p "$SCENES_DIR"

# -- Download scenes -------------------------------------------------------
GDRIVE_FOLDER_ID="1Wj9NThod8CD4Aa1K8B8MO2vZtSJKt8CN"
GDRIVE_FOLDER_URL="https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}"

echo ""
echo -e "${BOLD}SCOPE Benchmark Scene Download${RESET}"
echo "=============================="
echo ""

# Check if gdown is available (used for Google Drive downloads)
if command -v gdown &>/dev/null; then
    info "gdown is available: $(command -v gdown)"
else
    warn "gdown is not installed. Install it with:"
    echo "  pip install gdown"
    echo ""
fi

# Check if scenes are already present
scene_count=$(find "$SCENES_DIR" -name "*.blend" 2>/dev/null | wc -l | tr -d ' ')
if [ "$scene_count" -gt 0 ]; then
    info "Found $scene_count .blend files already in $SCENES_DIR"
    echo ""
    echo "If you want to re-download, remove the existing scenes first:"
    echo "  rm -rf $SCENES_DIR/*"
    echo ""
    exit 0
fi

echo "Downloading SCOPE benchmark scenes from Google Drive..."
echo ""

if command -v gdown &>/dev/null; then
    info "Using gdown to download scenes folder..."
    echo ""
    echo "  Note: Some scenes include external texture files in subfolders."
    echo "  Keep the folder structure intact after downloading."
    echo ""
    gdown --folder "$GDRIVE_FOLDER_URL" -O "$SCENES_DIR" && \
        info "Download complete. Scenes saved to: $SCENES_DIR" || \
        warn "gdown download failed. Try the manual instructions below."
else
    warn "gdown is not installed. Install it with:  pip install gdown"
    echo ""
    echo "Then re-run this script, or follow the manual steps below."
fi

echo ""
echo -e "${BOLD}── Manual download ──────────────────────────────────────────────${RESET}"
echo ""
echo "  Option A: Download via browser"
echo "    1. Open: $GDRIVE_FOLDER_URL"
echo "    2. Right-click the folder → 'Download' (creates a zip archive)"
echo "    3. Unzip and move to:"
echo "       $SCENES_DIR/"
echo "    4. The expected layout:"
echo "         benchmark/scenes/"
echo "           after-the-rain-vr-sound/Whitechapel.blend    (+ textures/)"
echo "           book-nook/Book_Nook.blend"
echo "           city-street-one-way/city-street-camera-on-street.blend"
echo "           postwar-city-exterior-scene/d77c0cc….blend"
echo ""
echo "  Option B: gdown (once installed: pip install gdown)"
echo "    gdown --folder '$GDRIVE_FOLDER_URL' -O $SCENES_DIR"
echo ""
echo "  Note: Some scenes ship with external texture folders."
echo "        Keep the folder structure intact — Blender resolves textures"
echo "        relative to the .blend file."
echo ""
echo "  After downloading, verify:"
echo "    find $SCENES_DIR -name '*.blend' | wc -l   # should print 4"
echo ""

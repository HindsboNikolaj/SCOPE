#!/usr/bin/env bash
# Download SCOPE benchmark scenes from Hugging Face Hub.
#
# Public dataset: huggingface.co/datasets/HindsboNikolaj/scope-benchmark
#
# Falls back to Google Drive (legacy mirror) if HF is unreachable.
# Usage:
#   bash scripts/03_download_scenes.sh
#   bash scripts/03_download_scenes.sh --revision <git-sha>   # pin a version
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCENES_DIR="$PROJECT_ROOT/benchmark/scenes"
HF_REPO="HindsboNikolaj/scope-benchmark"
REVISION="${1:-main}"

GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; RESET="\033[0m"
info()  { echo -e "${GREEN}[INFO]${RESET}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error() { echo -e "${RED}[ERROR]${RESET} $*"; }

mkdir -p "$SCENES_DIR"

# Skip if already downloaded
existing=$(find "$SCENES_DIR" -name "*.blend" 2>/dev/null | wc -l | tr -d ' ')
if [ "$existing" -ge 4 ]; then
    info "Found $existing .blend files in $SCENES_DIR. Skipping download."
    info "To force re-download:  rm -rf $SCENES_DIR/* && bash $0"
    exit 0
fi

# Prefer Hugging Face — better bandwidth, no auth needed for public datasets
if command -v huggingface-cli >/dev/null; then
    info "Downloading from Hugging Face Hub: $HF_REPO@$REVISION"
    huggingface-cli download "$HF_REPO" \
        --repo-type dataset \
        --revision "$REVISION" \
        --local-dir "$SCENES_DIR" \
        --local-dir-use-symlinks False
else
    warn "huggingface-cli not installed. Install with: pip install huggingface_hub"
    warn "Falling back to Google Drive mirror (slower, throttled)."
    if ! command -v gdown >/dev/null; then
        error "Need either huggingface_hub or gdown. Install one:"
        echo "  pip install huggingface_hub   # preferred"
        echo "  pip install gdown              # fallback"
        exit 1
    fi
    GDRIVE_FOLDER_ID="1Wj9NThod8CD4Aa1K8B8MO2vZtSJKt8CN"
    gdown --folder --remaining-ok \
        "https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}" \
        -O "$SCENES_DIR"
fi

# Verify
final=$(find "$SCENES_DIR" -name "*.blend" 2>/dev/null | wc -l | tr -d ' ')
if [ "$final" -lt 4 ]; then
    error "Only got $final/4 .blend files. Re-run or check network."
    exit 1
fi
info "Downloaded $final scenes. Expected layout:"
echo "  benchmark/scenes/whitechapel/whitechapel.blend         (+ textures/)"
echo "  benchmark/scenes/book-nook/book-nook.blend"
echo "  benchmark/scenes/city-street/city-street.blend"
echo "  benchmark/scenes/postwar-city/postwar-city.blend       (+ textures/)"

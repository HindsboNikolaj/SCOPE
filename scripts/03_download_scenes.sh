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
# The HF dataset has `scenes/` as its top-level directory, so we point the
# download at PROJECT_ROOT/benchmark and let the dataset's own `scenes/`
# prefix land at PROJECT_ROOT/benchmark/scenes/.
HF_LOCAL_DIR="$PROJECT_ROOT/benchmark"
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

# Prefer Hugging Face — better bandwidth, no auth needed for public datasets.
# Note: `huggingface-cli` was deprecated and removed in huggingface_hub 1.17+.
# The new entry point is `hf`. We prefer `hf`, fall back to `huggingface-cli`
# for older installs, then to gdown as a last resort.
if command -v hf >/dev/null; then
    info "Downloading from Hugging Face Hub: $HF_REPO@$REVISION (via hf)"
    hf download "$HF_REPO" \
        --repo-type dataset \
        --revision "$REVISION" \
        --local-dir "$HF_LOCAL_DIR"
elif command -v huggingface-cli >/dev/null; then
    info "Downloading from Hugging Face Hub: $HF_REPO@$REVISION (via huggingface-cli)"
    huggingface-cli download "$HF_REPO" \
        --repo-type dataset \
        --revision "$REVISION" \
        --local-dir "$HF_LOCAL_DIR" \
        --local-dir-use-symlinks False
else
    warn "No huggingface client on PATH. Install with: pip install -U huggingface_hub"
    warn "Falling back to Google Drive mirror (slower, throttled)."
    if ! command -v gdown >/dev/null; then
        error "Need either huggingface_hub or gdown. Install one:"
        echo "  pip install -U huggingface_hub   # preferred"
        echo "  pip install gdown                 # fallback"
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

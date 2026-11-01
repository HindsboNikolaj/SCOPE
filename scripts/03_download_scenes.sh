#!/usr/bin/env bash
# Download SCOPE benchmark scenes from Hugging Face Hub.
#
# Public dataset: huggingface.co/datasets/HindsboNikolaj/scope-benchmark
# Only scenes/ is downloaded. The benchmark CSVs the scripts read are the ones in this
# repository (benchmark/scope_v2.csv, benchmark/archive/); copies on Hugging Face never
# overwrite them.
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
REVISION=main
[ "${1:-}" = "--revision" ] && REVISION="${2:?--revision needs a git sha or branch}"

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

# Prefer Hugging Face: better bandwidth, no auth needed for public datasets.
# huggingface_hub is a dependency of scope-agent, so `pip install -e .` provides both the
# `hf` command and the Python API. Each client is tried in turn and a failure moves on to the
# next one: an `hf` left on PATH by another Python (a "bad interpreter" error) must not stop
# the download. Google Drive is the last resort.
PY="$(command -v python || command -v python3 || true)"

download_hf_cli() {
    command -v hf >/dev/null || return 1
    info "Downloading from Hugging Face Hub: $HF_REPO@$REVISION (via hf)"
    hf download "$HF_REPO" \
        --repo-type dataset \
        --revision "$REVISION" \
        --include "scenes/*" \
        --local-dir "$HF_LOCAL_DIR" && return 0
    warn "hf download failed; install or repair huggingface_hub in the active environment (python -m pip install -U huggingface_hub). Trying the next client."
    return 1
}

download_legacy_cli() {
    command -v huggingface-cli >/dev/null || return 1
    info "Downloading from Hugging Face Hub: $HF_REPO@$REVISION (via huggingface-cli)"
    huggingface-cli download "$HF_REPO" \
        --repo-type dataset \
        --revision "$REVISION" \
        --include "scenes/*" \
        --local-dir "$HF_LOCAL_DIR" \
        --local-dir-use-symlinks False && return 0
    warn "huggingface-cli download failed. Trying the next client."
    return 1
}

download_python_api() {
    [ -n "$PY" ] && "$PY" -c "import huggingface_hub" 2>/dev/null || return 1
    info "Downloading from Hugging Face Hub: $HF_REPO@$REVISION (via $PY and huggingface_hub)"
    "$PY" - "$HF_REPO" "$REVISION" "$HF_LOCAL_DIR" <<'PY' && return 0
import sys
from huggingface_hub import snapshot_download
repo, revision, local_dir = sys.argv[1:4]
snapshot_download(repo_id=repo, repo_type="dataset", revision=revision,
                  allow_patterns=["scenes/*"], local_dir=local_dir)
PY
    warn "huggingface_hub download from Python failed."
    return 1
}

if ! download_hf_cli && ! download_legacy_cli && ! download_python_api; then
    warn "No working Hugging Face client. Install one with: python -m pip install -U huggingface_hub"
    warn "Falling back to Google Drive mirror (slower, throttled)."
    if ! command -v gdown >/dev/null; then
        error "Need either huggingface_hub or gdown. Install one:"
        echo "  python -m pip install -U huggingface_hub   # preferred"
        echo "  python -m pip install gdown                 # fallback"
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

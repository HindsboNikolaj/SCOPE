#!/usr/bin/env bash
# Upload SCOPE benchmark to Hugging Face Hub.
#
# Prerequisites:
#   1. pip install huggingface_hub
#   2. huggingface-cli login   # paste your write token from huggingface.co/settings/tokens
#   3. Create the dataset repo via web UI: huggingface.co/new-dataset
#      Name: scope-benchmark   Type: Dataset   Visibility: Public
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HF_USER="HindsboNikolaj"
HF_DATASET="scope-benchmark"
HF_SPACE="scope"

# Stage the dataset contents
STAGE="${PROJECT_ROOT}/.hf_stage"
rm -rf "$STAGE"
mkdir -p "$STAGE/scenes"

# Pack each scene from local SCOPE/ folder (the texture-rich source)
LOCAL_SCOPE="${LOCAL_SCOPE:?set LOCAL_SCOPE to the folder holding the texture-rich scene sources}"
for s in whitechapel book-nook city-street postwar-city; do
    mkdir -p "$STAGE/scenes/$s"
    # Prefer packed if it exists, else copy raw .blend
    if [ -f "$LOCAL_SCOPE/$s/${s}.packed.blend" ]; then
        cp "$LOCAL_SCOPE/$s/${s}.packed.blend" "$STAGE/scenes/$s/${s}.blend"
    elif [ -f "$LOCAL_SCOPE/$s/${s}.blend" ]; then
        cp "$LOCAL_SCOPE/$s/${s}.blend" "$STAGE/scenes/$s/${s}.blend"
    fi
done

# Include benchmark CSVs: v2 (default) and the archived v1 files, with the change list
cp "$PROJECT_ROOT/benchmark/scope_v2.csv" "$STAGE/scope_v2.csv"
cp "$PROJECT_ROOT/benchmark/CHANGES_v2.md" "$STAGE/CHANGES_v2.md"
mkdir -p "$STAGE/archive"
cp "$PROJECT_ROOT/benchmark/archive/"*.csv "$PROJECT_ROOT/benchmark/archive/README.md" "$STAGE/archive/"

# Include dataset card as README.md
cp "$SCRIPT_DIR/dataset_card.md" "$STAGE/README.md"

# Upload
huggingface-cli upload "${HF_USER}/${HF_DATASET}" "$STAGE" . \
    --repo-type dataset \
    --commit-message "Initial upload from SCOPE refactor"

echo ""
echo "Dataset uploaded to: https://huggingface.co/datasets/${HF_USER}/${HF_DATASET}"
echo ""
echo "To upload the Space (project landing page):"
echo "  1. Create at: https://huggingface.co/new-space  (SDK: gradio)"
echo "  2. cd $SCRIPT_DIR && cp space_app.py /tmp/scope-space/app.py"
echo "  3. cp space_readme.md /tmp/scope-space/README.md"
echo "  4. cd /tmp/scope-space && git push"

rm -rf "$STAGE"

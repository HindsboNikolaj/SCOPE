#!/usr/bin/env bash
# Download CC0 textures from AmbientCG by asset code, unzip, and place
# files into the scene textures/ folder where the .blend expects them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCENES_DIR="${PROJECT_ROOT}/benchmark/scenes"
STAGING="${SCENES_DIR}/.ambientcg_staging"
CODES_FILE="${SCRIPT_DIR}/ambientcg_codes.txt"

if ! command -v curl >/dev/null; then
    echo "ERROR: curl is required." >&2; exit 1
fi
if ! command -v unzip >/dev/null; then
    echo "ERROR: unzip is required." >&2; exit 1
fi

mkdir -p "$STAGING"

while IFS= read -r code; do
    [ -z "$code" ] && continue
    echo "→ $code"
    # Try the standard 2K-PNG variant first; fall back to 2K-JPG, then 1K-PNG
    for variant in "2K-PNG" "2K-JPG" "1K-PNG" "1K-JPG"; do
        url="https://ambientcg.com/get?file=${code}_${variant}.zip"
        out="${STAGING}/${code}_${variant}.zip"
        if [ -f "$out" ]; then echo "  cached: $variant"; break; fi
        if curl -sfL --retry 2 -o "$out" "$url"; then
            echo "  got $variant ($(du -h "$out" | cut -f1))"
            break
        fi
        rm -f "$out"
    done
done < "$CODES_FILE"

echo ""
echo "Unzipping into ${STAGING}/extracted ..."
mkdir -p "${STAGING}/extracted"
for z in "${STAGING}"/*.zip; do
    [ -f "$z" ] || continue
    code="$(basename "$z" .zip | sed 's/_[12]K-.*//')"
    mkdir -p "${STAGING}/extracted/${code}"
    unzip -oq "$z" -d "${STAGING}/extracted/${code}" || true
done

echo ""
echo "Distributing textures into scene folders..."
# For each scene, scan its .blend for referenced filenames and copy matching files
# from the staging area.
python3 - <<'PY'
import os, shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent if "__file__" in dir() else Path("/tmp/scope-refactor")
PROJECT_ROOT = Path("/tmp/scope-refactor")
SCENES_DIR = PROJECT_ROOT / "benchmark" / "scenes"
STAGING = SCENES_DIR / ".ambientcg_staging" / "extracted"

# Build index of staged files by basename
idx = {}
if STAGING.exists():
    for root, _, files in os.walk(STAGING):
        for f in files:
            idx.setdefault(f, os.path.join(root, f))

# For each scene, read its missing list and copy matches
import json
for scene_dir in ("whitechapel", "book-nook", "city-street", "postwar-city"):
    scene_path = SCENES_DIR / scene_dir
    missing_log = SCENES_DIR / f".{scene_dir}_missing.txt"
    if not missing_log.exists():
        continue
    needs = [l.strip() for l in missing_log.read_text().splitlines() if l.strip()]
    copied = 0
    target = scene_path / "textures"
    target.mkdir(exist_ok=True)
    for name in needs:
        if name in idx:
            shutil.copy2(idx[name], target / name)
            copied += 1
    print(f"  {scene_dir}: copied {copied}/{len(needs)}")
PY

echo ""
echo "Done. Now re-pack each scene:"
echo "  for s in whitechapel book-nook city-street postwar-city; do"
echo "    blender --background --python scripts/repack_scene.py -- benchmark/scenes/\$s/\$s.blend"
echo "  done"

#!/usr/bin/env bash
# Run one benchmark shard (one scene) in the GPU Blender container, with its own display,
# scratch directory and output CSV.
#
#   bash scripts/run_shard.sh <scene> <gpu> <planner_port> <model_id> <tag> [limit] [shard_idx]
#   bash scripts/run_shard.sh book-nook 1 8003 qwen3-4b smoke 3
#
# Shard by scene, not by row: runner.py opens the .blend a row names, so a one-scene shard
# opens it once, and a mixed shard would reopen scenes at about a minute each. Run the four
# scenes as four shards to use four cards.
#
# Every host path comes from the environment (see .env.example); nothing here assumes a
# particular machine.
#   SCOPE_RESULTS_DIR   where <tag>/<scene>.csv and .log land      (default ./results)
#   SCOPE_SCENES_DIR    directory holding <scene>/<scene>.blend    (default ./benchmark/scenes)
#   SCOPE_SHARDS_DIR    directory holding <scene>.csv question files; default: split
#                       benchmark/scope_536.csv by scene into $SCOPE_RESULTS_DIR/<tag>/shards
#   SCOPE_PRESETS_HOST  Blender camera presets directory
#   HF_CACHE_HOST       Hugging Face cache to mount at /hf         (default ~/.cache/huggingface)
#   BLENDER_IMAGE       GPU Blender image from docker/build-gpu.sh (default scope-blender:gpu)
set -euo pipefail
SCENE=$1; GPU=$2; PLANNER_PORT=$3; MODEL=$4; TAG=$5; LIMIT=${6:-0}; SHARD_IDX=${7:-0}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="${SCOPE_RESULTS_DIR:-$REPO/results}"
SCENES="${SCOPE_SCENES_DIR:-$REPO/benchmark/scenes}"
PRESETS="${SCOPE_PRESETS_HOST:-$HOME/.config/blender/4.4/scripts/presets/camera}"
HF="${HF_CACHE_HOST:-$HOME/.cache/huggingface}"
IMAGE="${BLENDER_IMAGE:-scope-blender:gpu}"
OUT="$RESULTS/$TAG"
SHARDS="${SCOPE_SHARDS_DIR:-$OUT/shards}"
mkdir -p "$OUT" "$SHARDS"

if [ ! -f "$SHARDS/$SCENE.csv" ]; then
  python3 - "$REPO/benchmark/scope_536.csv" "$SHARDS" <<'EOF'
import csv, sys, collections
src, dst = sys.argv[1], sys.argv[2]
rows = collections.defaultdict(list)
with open(src, encoding="utf-8", newline="") as f:
    r = csv.DictReader(f); header = r.fieldnames
    for row in r:
        rows[row["file_location"].split("/")[1]].append(row)
for scene, rs in rows.items():
    with open(f"{dst}/{scene}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerows(rs)
EOF
fi

# The X display must be unique per shard, not per GPU: shards pinned to one card otherwise
# race for one display and all but one die with "GHOST: unable to initialize".
DISPLAY_NUM=$(( 90 + ${SCOPE_PORT_SLOT:-0} * 10 + SHARD_IDX ))
# Own scratch per shard: shared scratch means Blenders writing captures into one directory and
# Xvfb sockets colliding in one /tmp. Nothing errors; the shards just interfere.
SCRATCH="$OUT/scratch-$SCENE"
mkdir -p "$SCRATCH"
NAME="scope-bench-$TAG-$SCENE"
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --rm --name "$NAME" --gpus "\"device=$GPU\"" --network host \
  --user "$(id -u):$(id -g)" --entrypoint bash \
  -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/repo/src \
  -e SCOPE_PRESET_DIR=/presets -e SCOPE_CAPTURE=opengl -e SCOPE_OUTPUT_DIR=/tmp/scope \
  -e SCOPE_SHADING="${SCOPE_SHADING:-MATERIAL}" \
  -e SCOPE_SAMPLE_EVERY="${SCOPE_SAMPLE_EVERY:-25}" -e SCOPE_SAMPLE_EDGE="${SCOPE_SAMPLE_EDGE:-512}" \
  -e SCOPE_KEEP_CAPTURES="${SCOPE_KEEP_CAPTURES:-0}" -e SCOPE_CAPTURE_RING="${SCOPE_CAPTURE_RING:-8}" \
  -e SCOPE_MAX_TURNS="${SCOPE_MAX_TURNS:-40}" \
  -e SCOPE_VLM_COORD_SPACE="${SCOPE_VLM_COORD_SPACE:-auto}" \
  -e AGENT_API_BASE="http://localhost:$PLANNER_PORT/v1" -e AGENT_MODEL_ID="$MODEL" \
  -e AGENT_API_KEY="${AGENT_API_KEY:-local}" \
  -e VLM_MODEL="${VLM_MODEL:-Moondream2}" -e VLM_MODEL_ID="${VLM_MODEL_ID:-}" \
  -e VLM_MODEL_URL="${VLM_MODEL_URL:-http://localhost:2020}" \
  -e MOONDREAM_API_KEY="${MOONDREAM_API_KEY:-}" -e HF_HOME=/hf \
  -e QUESTIONS_CSV="/shards/$SCENE.csv" -e OUT_CSV="/out/$SCENE.csv" \
  -e BENCH_LIMIT="$LIMIT" -e SCOPE_RESUME="${SCOPE_RESUME:-1}" \
  -v "$REPO":/repo:ro -v "$SCENES":/scenes:ro -v "$PRESETS":/presets:ro \
  -v "$SHARDS":/shards:ro -v "$OUT":/out -v "$SCRATCH":/tmp -v "$HF":/hf \
  "$IMAGE" -c "
    # A relaunched shard reuses its scratch, and the dead container's X lock is still there;
    # Xvfb then refuses to start and Blender reports that there is no display.
    rm -f /tmp/.X$DISPLAY_NUM-lock /tmp/.X11-unix/X$DISPLAY_NUM
    Xvfb :$DISPLAY_NUM -screen 0 1920x1080x24 >/dev/null 2>&1 &
    sleep 3; export DISPLAY=:$DISPLAY_NUM
    cd /repo
    exec vglrun -d egl0 blender /scenes/$SCENE/$SCENE.blend --python /repo/src/scope/eval/runner.py \
         > /out/$SCENE.log 2>&1
  " >/dev/null
echo "launched $NAME on GPU $GPU (display :$DISPLAY_NUM) -> $OUT/$SCENE.csv"

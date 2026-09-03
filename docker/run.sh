#!/usr/bin/env bash
# Run a Blender script inside the SCOPE container, with a working viewport and the GPUs.
#
#   ./run.sh blender -b /scenes/whitechapel/whitechapel.blend --python /work/impl/foo.py
#   GPU=5 ./run.sh blender /scenes/whitechapel/whitechapel.blend --python /work/impl/bar.py
#
# Note the second form has no -b. That is the point of this image: Blender runs with a UI
# against the container's virtual display, so `screenshot_area` and `render.opengl` work.
set -euo pipefail

IMAGE="${IMAGE:-scope-blender:4.4.3}"
GPU="${GPU:-}"                    # e.g. GPU=5. Empty means no GPU is claimed at all.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENES="${SCENES:-$REPO_ROOT/benchmark/scenes}"
WORK="${WORK:-$REPO_ROOT}"
REPO="${REPO:-$REPO_ROOT}"
PRESETS="${PRESETS:-$HOME/.config/blender/4.4/scripts/presets/camera}"
OUT="${OUT:-$REPO_ROOT/results}"
mkdir -p "$OUT"

# Claim GPUs explicitly. The daemon's default runtime is nvidia, so without this a container
# can see every card on a box whose cards mostly belong to other people.
gpu_args=()
if [ -n "$GPU" ]; then
  gpu_args=(--gpus "\"device=$GPU\"")
else
  gpu_args=(--env NVIDIA_VISIBLE_DEVICES=void)
fi

# Run as the calling user so files written to the mounts are owned by that user rather than
# by root, which is the usual way a container quietly makes a shared directory unusable.
# Forward every SCOPE_* variable. Without this the container runs with defaults and the
# caller's settings are silently ignored, which looks like the settings not working.
env_args=()
while IFS='=' read -r k _; do
  case "$k" in SCOPE_*) env_args+=(--env "$k=${!k}") ;; esac
done < <(env)

exec docker run --rm -i \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  "${env_args[@]}" \
  "${gpu_args[@]}" \
  -v "$SCENES":/scenes:ro \
  -v "$REPO":/repo:ro \
  -v "$WORK":/work \
  -v "$PRESETS":/presets:ro \
  -v "$OUT":/out \
  -w /work \
  "$IMAGE" "$@"

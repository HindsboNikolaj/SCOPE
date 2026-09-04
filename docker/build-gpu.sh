#!/usr/bin/env bash
#
# Build a SCOPE Blender image that draws on the GPU, on a headless Linux server.
#
#   ./docker/build-gpu.sh                 # build scope-blender:gpu
#   ./docker/build-gpu.sh --check         # only report whether this host needs it
#   SCOPE_GPU_TAG=myimage ./docker/build-gpu.sh
#
# You probably do not need this. If your machine has a desktop and a working driver, Blender
# already draws on the GPU and a capture already costs about 0.2 seconds. This is for the case
# where it does not: a server with NVIDIA hardware whose driver was installed for compute, so
# CUDA works and OpenGL falls back to the CPU. There a capture costs about 15 seconds, which is
# the same code drawing the same image 74 times slower.
#
# What it does, and why each step is needed:
#
#   1. Reads the host driver version from nvidia-smi. The userspace libraries must match the
#      running kernel module exactly, so the version is read rather than chosen.
#   2. Downloads that exact .run installer from NVIDIA and extracts it WITHOUT installing.
#      Nothing on the host is modified. A driver installed for compute ships CUDA and no
#      graphics libraries, and the container toolkit can only pass through what the host has,
#      so NVIDIA_DRIVER_CAPABILITIES=all injects nothing and reports no error.
#   3. Stages libEGL_nvidia, libGLX_nvidia and their dependencies with the sonames they expect,
#      plus the libglvnd vendor JSON that says where the EGL driver is.
#   4. Fetches VirtualGL from its GitHub release. Blender links libGL and libGLX and has no EGL
#      backend, and under a virtual X server GLX resolves to Mesa with no way to point it at
#      NVIDIA. vglrun interposes the GLX calls onto the EGL device, which does work.
#   5. Builds the image and proves the result by asking OpenGL its renderer name inside it.
#
# The last step is the one that matters. Every failure mode here is quiet: the wrong library
# version loads and falls back, the vendor JSON is missing and EGL finds no device, the
# VirtualGL download returns an HTML error page and dpkg is told to ignore it. So the script
# refuses to declare success on anything except the GPU's name coming back from inside the
# container.
set -euo pipefail

# Resolved once, absolutely, because the script cd's into the build directory later and any
# relative path to itself stops working the moment it does.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TAG="${SCOPE_GPU_TAG:-scope-blender:gpu}"
BASE="${SCOPE_BASE_IMAGE:-scope-blender:4.4.3}"
VGL_VERSION="${VGL_VERSION:-3.1.1}"
WORK="${SCOPE_GPU_WORKDIR:-${HERE}/../.gpu-build}"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 0. is this needed at all?

say "Checking whether this host needs the GPU bridge"

command -v nvidia-smi >/dev/null || die "no nvidia-smi. This script is only for NVIDIA hosts."
command -v docker     >/dev/null || die "no docker."

# -i 0 rather than piping through head: under `set -o pipefail`, head closing the pipe sends
# nvidia-smi a SIGPIPE and the whole script exits 141 before printing anything.
DRIVER="$(nvidia-smi -i 0 --query-gpu=driver_version --format=csv,noheader | tr -d ' \n')"
GPUNAME="$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader | tr -d '\n')"
echo "  GPU:    $GPUNAME"
echo "  driver: $DRIVER"

# Does the host already expose graphics libraries the container toolkit could inject?
if ls /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.* >/dev/null 2>&1; then
  echo "  The host already has libEGL_nvidia. The container toolkit can inject it, so you may"
  echo "  not need this image at all. Try NVIDIA_DRIVER_CAPABILITIES=all on the base image"
  echo "  first and check the renderer name."
else
  echo "  No libEGL_nvidia on the host: this driver was installed compute-only, which is"
  echo "  exactly the case this script exists for."
fi

docker image inspect "$BASE" >/dev/null 2>&1 || die \
  "base image $BASE not found. Build it first:
    docker build -t $BASE -f ${HERE}/Dockerfile ${HERE}"

[ "$CHECK_ONLY" = 1 ] && { echo; echo "  --check only, stopping here."; exit 0; }

# ---------------------------------------------------------------- 1. driver userspace

mkdir -p "$WORK"
cd "$WORK"
RUN="NVIDIA-Linux-x86_64-${DRIVER}.run"

if [ ! -d "NVIDIA-Linux-x86_64-${DRIVER}" ]; then
  say "Fetching the matching driver archive ($DRIVER)"
  if [ ! -f "$RUN" ]; then
    # Datacentre cards are under /tesla/, consumer under /XFree86/. Try both rather than
    # guessing from the GPU name, which does not reliably tell you which.
    ok=0
    for url in \
      "https://us.download.nvidia.com/tesla/${DRIVER}/${RUN}" \
      "https://us.download.nvidia.com/XFree86/Linux-x86_64/${DRIVER}/${RUN}" ; do
      echo "  trying ${url}"
      if curl -fL --retry 2 --max-time 900 --progress-bar -o "$RUN" "$url" && [ -s "$RUN" ] \
         && head -c 2 "$RUN" | grep -q '#!' ; then ok=1; break; fi
      rm -f "$RUN"
    done
    [ "$ok" = 1 ] || die "could not download $RUN. Fetch it manually into $WORK and re-run."
  fi
  sh "$RUN" --extract-only >/dev/null || die "could not extract $RUN"
fi

say "Staging the graphics libraries"
SRC="NVIDIA-Linux-x86_64-${DRIVER}"
rm -rf nvidia-gl && mkdir -p nvidia-gl/lib nvidia-gl/egl_vendor.d

# Everything the EGL and GLX vendor libraries need at runtime.
#
# This list is empirical rather than derived, because `ldd` does not produce it. libEGL_nvidia
# declares exactly one dependency and libGLX_nvidia three; the rest are opened with dlopen at
# runtime and appear in no link table. An image built from what ldd reports loads, finds no
# device, and falls back to llvmpipe with no error at all, which is why the verification step
# at the end of this script exists.
#
# libnvidia-allocator and libnvidia-gpucomp are the two that are easiest to leave out and
# hardest to diagnose the absence of. If you extend this list, extend it and then re-run: the
# only reliable test is the renderer name coming back from inside the container.
for stem in libEGL_nvidia libGLX_nvidia libnvidia-allocator libnvidia-eglcore \
            libnvidia-glcore libnvidia-glsi libnvidia-glvkspirv libnvidia-gpucomp \
            libnvidia-rtcore libnvidia-tls; do
  for f in "$SRC/${stem}.so."*; do
    [ -e "$f" ] || continue
    cp "$f" nvidia-gl/lib/
    b="$(basename "$f")"
    # every consumer looks for the .so.0 soname, not the versioned filename
    ln -sf "$b" "nvidia-gl/lib/${stem}.so.0"
    ln -sf "$b" "nvidia-gl/lib/${stem}.so"
  done
done
[ -e nvidia-gl/lib/libEGL_nvidia.so.0 ] || die \
  "libEGL_nvidia is not in $SRC. This archive may be a compute-only build."

cat > nvidia-gl/egl_vendor.d/10_nvidia.json <<'JSON'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "/opt/nvgl/lib/libEGL_nvidia.so.0"
    }
}
JSON
echo "  staged $(ls nvidia-gl/lib | wc -l) files"

# ---------------------------------------------------------------- 2. VirtualGL

say "Fetching VirtualGL ${VGL_VERSION}"
if [ ! -s vgl.deb ] || ! file vgl.deb | grep -q Debian; then
  # Deliberately the GitHub release. The SourceForge URL returns an HTML page, and a
  # `dpkg -i ... || apt-get -f install` will swallow that and build an image with no vglrun.
  curl -fL --retry 2 --max-time 300 --progress-bar -o vgl.deb \
    "https://github.com/VirtualGL/virtualgl/releases/download/${VGL_VERSION}/virtualgl_${VGL_VERSION}_amd64.deb" \
    || die "could not download VirtualGL ${VGL_VERSION}"
fi
file vgl.deb | grep -q Debian || die "vgl.deb is not a Debian package (got: $(file -b vgl.deb))"
echo "  $(du -h vgl.deb | cut -f1) ok"

# ---------------------------------------------------------------- 3. build

say "Building $TAG"
cp "${HERE}/Dockerfile.hardware-gl" ./Dockerfile.hardware-gl
docker build --build-arg BASE="$BASE" -t "$TAG" -f Dockerfile.hardware-gl . \
  || die "docker build failed"

# ---------------------------------------------------------------- 4. prove it

say "Verifying the GPU is actually drawing"
OUT="$(docker run --rm --gpus '"device=0"' --entrypoint bash "$TAG" -c '
  Xvfb :99 -screen 0 1280x1024x24 >/dev/null 2>&1 &
  sleep 3; export DISPLAY=:99
  timeout 60 vglrun -d egl0 /opt/VirtualGL/bin/glxspheres64 -n 20 2>&1 | grep -i "OpenGL Renderer"
' 2>&1 || true)"

echo "  $OUT"
case "$OUT" in
  *llvmpipe*|*softpipe*|*swrast*) die "still on the software rasteriser. The bridge is not engaged." ;;
  *"OpenGL Renderer"*)            : ;;
  *)                              die "no renderer reported. Output was: $OUT" ;;
esac

cat <<EOF

  Built and verified: $TAG

  Run a capture with it by prefixing Blender with vglrun:

    docker run --rm --gpus '"device=0"' --entrypoint bash $TAG -c '
      Xvfb :99 -screen 0 1920x1080x24 & sleep 3; export DISPLAY=:99
      vglrun -d egl0 blender /scenes/<scene>.blend --python <script>.py'

  Build artefacts are in $WORK and can be deleted.
EOF

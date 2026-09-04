# Capturing on the GPU

A SCOPE capture is a viewport draw, not a render. On a machine with a working graphics driver
it costs about **0.2 seconds**. On a machine falling back to software rasterisation it costs
about **15 seconds**, and on the most expensive scene a full panorama took hours.

That is a 74x difference in the same code, and it is worth knowing which side of it you are on
before concluding that something is slow.

```
                       renderer                          per capture
  software             llvmpipe (LLVM 15.0.7)               14.63 s
  hardware             NVIDIA A100-SXM4-80GB                 0.198 s
```

Measured on postwar-city, three captures, discarding the first (which pays for shader
compilation).

## Are you on the slow path?

```bash
glxinfo -B | grep -i "OpenGL renderer"
```

If it says `llvmpipe`, `softpipe` or `swrast`, every capture is being drawn on the CPU.

Two symptoms that look like something else:

- **A capture takes seconds rather than milliseconds.** Easy to read as "Blender is slow" or
  "this scene is heavy". The cost is also roughly flat in resolution, because what dominates is
  submitting draw calls rather than filling pixels, so lowering the resolution does not help and
  that in turn looks like evidence that the scene is the problem.
- **One CPU core is pinned.** On a 256-core machine a capture at 102% CPU looks like a small
  job. It is one thread doing the work of a GPU.

## Getting onto the fast path

The rest of this page is for a headless Linux machine with NVIDIA hardware. If you have a
desktop with a working driver, you are already on the fast path.

### Why it is not simply a matter of passing the GPU through

Three things have to be true, and on a compute-configured host the third usually is not.

1. **The container can see the device.** `--gpus` handles this.
2. **The graphics libraries are present.** `NVIDIA_DRIVER_CAPABILITIES=all` asks the container
   toolkit to inject them, but it can only inject what the host has. A datacentre driver
   installed for compute ships CUDA and no `libEGL_nvidia`, `libGLX_nvidia` or
   `libnvidia-glcore`. The request succeeds and nothing graphical arrives, which is a confusing
   way to fail.
3. **Blender can reach them.** Blender links `libGL.so.1` and `libGLX.so.0`. Under a virtual X
   server, GLX resolves to Mesa, and there is no way to point it at NVIDIA without an NVIDIA X
   server. EGL does not have this problem, but Blender does not offer an EGL backend.

### The two pieces

**The libraries.** Download the `.run` installer matching the host driver version
(`nvidia-smi --query-gpu=driver_version --format=csv,noheader`) and extract it without
installing, so nothing on the host changes:

```bash
sh NVIDIA-Linux-x86_64-<version>.run --extract-only
mkdir -p nvidia-gl/lib nvidia-gl/egl_vendor.d
cp NVIDIA-Linux-x86_64-<version>/libEGL_nvidia.so.<version>      nvidia-gl/lib/
cp NVIDIA-Linux-x86_64-<version>/libGLX_nvidia.so.<version>      nvidia-gl/lib/
cp NVIDIA-Linux-x86_64-<version>/libnvidia-glcore.so.<version>   nvidia-gl/lib/
cp NVIDIA-Linux-x86_64-<version>/libnvidia-eglcore.so.<version>  nvidia-gl/lib/
cp NVIDIA-Linux-x86_64-<version>/libnvidia-glsi.so.<version>     nvidia-gl/lib/
# then create the sonames each one expects, e.g. libEGL_nvidia.so.0 -> libEGL_nvidia.so.<version>
```

`nvidia-gl/egl_vendor.d/10_nvidia.json` tells libglvnd where the EGL driver is:

```json
{ "file_format_version": "1.0.0", "ICD": { "library_path": "/opt/nvgl/lib/libEGL_nvidia.so.0" } }
```

**The bridge.** [VirtualGL](https://github.com/VirtualGL/virtualgl/releases) interposes
Blender's GLX calls onto the EGL device. Take the `.deb` from the GitHub releases page; the
SourceForge URL returns an HTML page rather than a package, and `dpkg -i ... || apt-get -f
install` will swallow that failure and leave you with an image that has no `vglrun` in it.

Then build `docker/Dockerfile.hardware-gl` with `nvidia-gl/` and `vgl.deb` in the build context.

### Check it worked

```bash
docker run --rm --gpus '"device=0"' --entrypoint bash scope-blender:gpu -c '
  Xvfb :99 -screen 0 1280x1024x24 & sleep 3; export DISPLAY=:99
  vglrun -d egl0 /opt/VirtualGL/bin/glxspheres64 -n 40 | grep -i "OpenGL Renderer"'
```

Expected: the name of your GPU. If it says `llvmpipe`, the bridge is not engaged.

### Running a capture

Prefix the Blender command. Everything else is unchanged.

```bash
Xvfb :99 -screen 0 1920x1080x24 & sleep 3; export DISPLAY=:99
vglrun -d egl0 blender <scene>.blend --python <script>.py
```

## What this changes about the benchmark

Nothing about the images: the same code path draws the same viewport, and the captures are
comparable. What changes is what is practical.

- city-street, previously described in these docs as impractical to capture in Material
  Preview, sweeps a full panorama in about **10 seconds**.
- Capturing the full set of preset artefacts across all four scenes is minutes rather than a
  day.
- A demo recorded at the frame rate the viewport actually runs at becomes possible.

The claim that a capture costs fifteen seconds appears in older revisions of `docs/SETUP.md`.
It was a measurement of this machine's fallback, stated as a property of the method.

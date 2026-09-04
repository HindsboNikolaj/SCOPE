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
- **One CPU core is pinned.** On a many-core machine a capture sitting at 102% CPU looks
  like a small job. It is one thread doing the work of a GPU.

## Getting onto the fast path

If your machine has a desktop and a working driver, you are already on it and there is nothing
to do here. The rest of this page is for a headless Linux server with NVIDIA hardware.

### One command

```bash
./docker/build-gpu.sh --check     # does this host need it?
./docker/build-gpu.sh             # build and verify
```

It reads the driver version off `nvidia-smi`, downloads that exact `.run` installer, extracts
it **without installing anything on the host**, stages the graphics libraries into an image,
adds VirtualGL, builds, and then proves the result by asking OpenGL its renderer name from
inside the container it just built. It exits non-zero unless a GPU name comes back.

That last step is not ceremony. Every failure here is quiet: the wrong library version loads
and falls back, the vendor JSON is missing so EGL finds no device, the download returns an HTML
error page and `dpkg` is told to ignore it. In each case you get a working image that draws on
the CPU and says nothing about it.

### Running a capture with it

```bash
docker run --rm --gpus '"device=0"' --entrypoint bash scope-blender:gpu -c '
  Xvfb :99 -screen 0 1920x1080x24 & sleep 3; export DISPLAY=:99
  vglrun -d egl0 blender /scenes/<scene>.blend --python <script>.py'
```

Only two things differ from the CPU image: a virtual display, and `vglrun -d egl0` in front of
Blender. Everything downstream is unchanged.

### Why it is not simply a matter of passing the GPU through

Three things have to be true, and on a compute-configured host the third usually is not.

1. **The container can see the device.** `--gpus` handles this.
2. **The graphics libraries are present.** `NVIDIA_DRIVER_CAPABILITIES=all` asks the container
   toolkit to inject them, but it can only inject what the host has. A datacentre driver
   installed for compute ships CUDA and no `libEGL_nvidia`, `libGLX_nvidia` or
   `libnvidia-glcore`. The request succeeds and nothing graphical arrives.
3. **Blender can reach them.** Blender links `libGL.so.1` and `libGLX.so.0`. Under a virtual X
   server, GLX resolves to Mesa, and there is no way to point it at NVIDIA without an NVIDIA X
   server. EGL does not have this problem, but Blender offers no EGL backend. VirtualGL
   interposes the GLX calls onto the EGL device, which does work.

### If you would rather do it by hand

The script is readable and each step says why it exists. Two things in it were learned the hard
way and are easy to get wrong:

**The library list cannot be derived from `ldd`.** `libEGL_nvidia` declares one dependency and
`libGLX_nvidia` three; the rest are opened with `dlopen` at runtime and appear in no link
table. An image built from what `ldd` reports loads, finds no device, and silently falls back.
`libnvidia-allocator` and `libnvidia-gpucomp` are the two easiest to omit and hardest to
diagnose.

**Take VirtualGL from its GitHub release.** The SourceForge URL returns an HTML page. Paired
with the common `dpkg -i pkg.deb || apt-get -f install -y`, that produces an image with no
`vglrun` in it and no error anywhere in the build log.

## What this changes about the benchmark

Nothing about the images: the same code path draws the same viewport, and the captures are
comparable. What changes is what is practical.

- city-street, previously described in these docs as impractical to capture in Material
  Preview, sweeps a full panorama in about **10 seconds**.
- Capturing the full set of preset artefacts across all four scenes is minutes rather than a
  day.
- A demo recorded at the frame rate the viewport actually runs at becomes possible.

The claim that a capture costs fifteen seconds appears in older revisions of `docs/SETUP.md`.
It was a measurement of a machine without a graphics driver, stated as a property of the
method.

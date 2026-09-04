# Blender with a working 3D viewport, headless, on a machine that has no graphics stack.
#
# Why this image exists. SCOPE captures by photographing a real Blender viewport, through
# `screenshot_area` or `render.opengl`. Both need an OpenGL context, which needs a display.
# A GPU box provisioned for compute typically has CUDA and nothing else: no GLX, no EGL vendor
# library, no libGL at all. So Blender refuses, with "Cannot use OpenGL render in background
# mode (no opengl context)", and the benchmark cannot run there.
#
# The host could be fixed with `apt install xvfb libgl1-mesa-dri libglx-mesa0`, but on a
# shared machine you may not own the update policy, and on a managed one you may not have
# root. A container gets the same result, changes nothing outside itself, and is thrown away
# by deleting an image.
#
# The earlier attempt used a CentOS 7 era X server from conda, whose GLX cannot create a
# modern context: Blender reported "GLX_ARB_create_context not available" and quit. Ubuntu
# 22.04 ships Xorg 21.1 with current Mesa, which can.
#
# Rendering is by llvmpipe, Mesa's software rasteriser, because a compute-only NVIDIA driver
# exposes no OpenGL. That has a real cost: MATERIAL preview takes about a minute a frame,
# against a few seconds for SOLID with textures. docs/HEADLESS.md has the measurements and
# the trade. It also has one advantage a benchmark values, which is that software rendering
# gives the same image on every machine. Cycles still uses the real GPUs through CUDA.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# xvfb            an X server that draws into memory, so Blender has a display
# openbox         a window manager. Without one, nothing maps and paints Blender's window
#                 into the X framebuffer, so `screen.screenshot_area` photographs a black
#                 rectangle. That operator reads the window as drawn on screen, unlike
#                 `render.opengl` which renders offscreen and needs no window manager.
# libgl1-mesa-dri the Mesa drivers, including llvmpipe
# libglx-mesa0    the GLX that can create the modern context Blender asks for
# libx*           what the Blender binary links against
RUN apt-get update && apt-get install -y --no-install-recommends \
      xvfb libgl1-mesa-dri libglx-mesa0 libgl1 libegl1 \
      openbox \
      libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 libice6 \
      libxcb1 libx11-6 libxext6 libglu1-mesa \
      ca-certificates curl xz-utils python3 \
 && rm -rf /var/lib/apt/lists/*

ARG BLENDER_VERSION=4.4.3
ARG BLENDER_SERIES=4.4
RUN curl -fsSL -o /tmp/blender.tar.xz \
      "https://download.blender.org/release/Blender${BLENDER_SERIES}/blender-${BLENDER_VERSION}-linux-x64.tar.xz" \
 && mkdir -p /opt/blender \
 && tar xf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 \
 && rm /tmp/blender.tar.xz \
 && ln -s /opt/blender/blender /usr/local/bin/blender

# SCOPE's helper_funcs imports PIL, and the stitch needs numpy. These go into Blender's own
# bundled Python, not the system one, because that is the interpreter Blender scripts run in.
RUN /opt/blender/${BLENDER_SERIES}/python/bin/python3.11 -m pip install --no-cache-dir \
      Pillow numpy

# Force software GL. Without this Mesa probes /dev/dri, finds cards it cannot open, and the
# probe failures are the first thing in every log.
ENV LIBGL_ALWAYS_SOFTWARE=1 \
    GALLIUM_DRIVER=llvmpipe \
    MESA_LOADER_DRIVER_OVERRIDE=swrast \
    DISPLAY=:99

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["blender", "--version"]

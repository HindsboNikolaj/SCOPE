# Running SCOPE on a machine with no display

The evaluator opens Blender with a user interface on purpose, because capture photographs a
real 3D viewport. On a laptop or a workstation that is fine. On a GPU server it usually is
not: a datacentre NVIDIA driver is installed for compute and ships no OpenGL, so there is no
display and no GL, and Blender will not start its interface at all.

This document is what was learned making it work anyway, on a box with eight A100s, driver
570.148.08, and no graphics stack of any kind.

## This is not the recommended setup

Read this as a way to make SCOPE run where it otherwise could not, not as the way to run it.
A headless box gives you software OpenGL and nothing else: an Xvfb display has no hardware GL,
so the GPUs on the machine accelerate the vision model and do not touch the viewport at all.

What that costs, measured at a 640 pixel long edge in Material Preview: 5 to 60 seconds a frame
on three of the four scenes, and 790 to 8500 seconds a frame on `city-street`, whose materials
are expensive to shade in software. The same captures on a workstation with a real display and
a GL driver are not in that range.

So the order of preference is a workstation, then a cloud instance with a virtual workstation
driver (the GRID and vWS variants do ship OpenGL), and only then this. If you are here because
a headless box is what you have, the rest of the document works. If you are here because it
seemed simpler, it is not.

## What fails, and the exact message

Neither capture function works in `--background` mode. Blender says so itself:

```
screenshot_camera_view   bpy.ops.screen.screenshot_area.poll() failed, context is incorrect
fast_opengl_screenshot   Cannot use OpenGL render in background mode (no opengl context)
```

The second is a refusal built into Blender rather than a driver limitation. No amount of
driver work changes it. Blender must run with an interface, which means it must have a
display.

## What does not solve it

Recorded so the same afternoon is not spent twice.

| Attempt | Outcome |
|---|---|
| Xvfb from a conda `cos7` package | Starts, but Blender quits: `GLX_ARB_create_context not available`. That X server is from the CentOS 7 era and its GLX cannot create a modern context. |
| Mesa surfaceless EGL, no X at all | Blender's GL loader crashes. |
| A newer X server from conda-forge | There is not one. Only the `cos7` build is packaged. |
| Hardware OpenGL through the NVIDIA container runtime | Nothing to expose. `nvidia-container-cli list` shows no GL libraries, because the driver was installed compute-only. Getting hardware GL means reinstalling the driver with graphics support. |

## What does solve it

A container with a current X server and Mesa. Ubuntu 22.04 ships Xorg 21.1, which provides
the context Blender asks for.

```bash
docker build -t scope-blender:4.4.3 docker/
SCOPE_CAPTURE=opengl docker/run.sh \
  blender /scenes/whitechapel/whitechapel.blend --python /repo/src/scope/eval/runner.py
```

Inside the container Blender reports:

```
windows = 1        areas = [... 'VIEW_3D']
GL renderer = llvmpipe (LLVM 15.0.7, 256 bits)
GL version  = 4.5 (Core Profile) Mesa 23.2.1
```

No root is required on the host if the user is in the `docker` group.

## The part that surprises people: only one of the two captures works

With a display, both work. In a container, only one does.

| Function | Mechanism | In a container |
|---|---|---|
| `screenshot_camera_view` | `screen.screenshot_area`, photographs the window as drawn | **black image, always** |
| `fast_opengl_screenshot` | `render.opengl`, renders offscreen | **works** |

`screenshot_area` reads back what was painted on screen. Under a virtual X server with
software GL, nothing paints it and the read returns nothing. This was tested with a window
manager running, without one, after forcing a full draw and buffer swap with
`wm.redraw_timer(type='DRAW_WIN_SWAP')`, and with the whole-window `screen.screenshot`
operator. Every one of them returned a uniformly black image.

So set `SCOPE_CAPTURE=opengl` when running headless. The default is `viewport`, which is
what produced the published results and what you should keep when you have a display.

## Shading: the flags that decide whether the picture is right

`screenshot_camera_view` sets no shading at all. It photographs the viewport as it stands,
so it inherits whatever the `.blend` was saved with. `fast_opengl_screenshot` has to choose,
and its choice used to be wrong in two ways that are worth understanding before overriding
anything.

`whitechapel.blend` is saved as `type=MATERIAL`, `light=STUDIO`. `STUDIO` lights the scene
from a built-in studio image and **ignores the scene world**. That single fact explains why
the desktop capture of that scene looks correct even though its sky texture is missing.

Two flags change the answer:

- **`shading.type`.** `SOLID` without a `color_type` gives a flat grey model with no
  textures at all. Setting `color_type='TEXTURE'` brings them back.
- **`use_scene_world`.** Setting it to true pulls the scene's world into the picture. For a
  scene whose world texture is missing, that turns the entire frame magenta. It now defaults
  to off.

Measured on whitechapel, same camera, colourfulness meaning mean channel spread:

| Shading | Time per frame | Colourfulness | What you get |
|---|---|---|---|
| `SOLID`, no `color_type` | 9 s | 4.7 | grey massing model, no textures |
| `SOLID` + `TEXTURE`, studio light | **2.7 s** | 27.4 | textured, correct colour |
| `MATERIAL` preview, studio light | 64 s | 15.3 | matches the desktop capture |
| `MATERIAL` preview, scene world on | 56 s | 181.9 | full detail, entirely magenta |

`MATERIAL` preview with the studio light was compared against a capture of the same scene
and camera taken on a Mac through the normal viewport path. Normalised cross correlation of
the two edge maps is **0.799 at zero shift**, which for two different renderers on two
different operating systems means the same view with the same geometry in the same places.

### Choosing

- **Reproducing published numbers:** `MATERIAL` preview. It matches, and it costs about a
  minute a frame under software OpenGL, so a full 541-row run is measured in days.
- **A practical headless run:** `SCOPE_SHADING=SOLID` with `SCOPE_SHADING_COLOR=TEXTURE`,
  about twenty times faster and keeps the textures. The images differ from the published
  ones, so report them as their own configuration rather than comparing directly.

Software rendering is the reason for the gap. A machine with a real GPU-backed display does
`MATERIAL` preview quickly.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `SCOPE_CAPTURE` | `viewport` | `opengl` to use the offscreen path that works headless |
| `SCOPE_SHADING` | as saved in the `.blend` | `SOLID`, `MATERIAL` or `RENDERED` |
| `SCOPE_SHADING_COLOR` | as saved | `TEXTURE`, `MATERIAL`, `OBJECT`, `SINGLE`. `SOLID` only |
| `SCOPE_SHADING_WORLD` | `0` | `1` to light from the scene world instead of the studio light |

## Two things that will waste your time otherwise

**Blender with an interface never exits on its own.** It enters its event loop and waits.
Every script must end with `bpy.ops.wm.quit_blender()`, and anything driving it needs a
timeout as a backstop. Containers were left running for half an hour before this was
understood.

**A container sees every GPU by default** when `nvidia-container-runtime` is the default
runtime, which it commonly is on a shared box. Claim cards explicitly with
`--gpus '"device=N"'`, or set `NVIDIA_VISIBLE_DEVICES=void` when no GPU is wanted. On a
shared machine this is a courtesy that matters.

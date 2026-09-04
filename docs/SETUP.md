# Setting SCOPE up, and checking that it worked

SCOPE photographs a real Blender viewport. That one fact decides everything about setup: it
needs a machine that can draw a 3D window, and the quality of what the model sees depends on
how that window is drawn.

Pick the row that matches your machine.

| Your machine | What to do | Capture speed |
|---|---|---|
| Laptop or desktop with a screen | Install Blender and run it. Nothing special. | Fast. A real GPU draws a frame in well under a second. |
| Remote desktop, VM, or a fresh Linux install with a display | Install Blender plus the OpenGL libraries below. | Fast if the machine has a GPU-backed display. |
| Headless server, no display at all | Use the container in `docker/`. | Slow. Software rendering, seconds to a minute a frame. |

## A laptop or desktop

Install Blender, clone the repository, run the quickstart in the README. There is nothing to
configure, because your machine already has a window system and a GPU driver that can draw
into it.

## A remote desktop or a fresh Linux install

Blender needs an OpenGL context, and a minimal server install usually has no OpenGL at all.
On Debian or Ubuntu:

```bash
sudo apt-get install --no-install-recommends \
    libgl1 libglx-mesa0 libgl1-mesa-dri libegl1 \
    libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 libice6
```

If there is no display attached, add a virtual one and a window manager:

```bash
sudo apt-get install --no-install-recommends xvfb openbox
```

Two notes from doing this the hard way. The X server has to be a current one: an X server
from the CentOS 7 era starts fine and then Blender quits with `GLX_ARB_create_context not
available`, because that server cannot create the modern context Blender asks for. And a
datacentre NVIDIA driver is usually installed for compute only and ships no OpenGL at all,
so Mesa's software renderer does the drawing even on a machine full of GPUs.

If you would rather not touch a shared machine, the container does the same thing in a box.

## A headless server

```bash
docker build -t scope-blender:4.4.3 docker/
docker/run.sh blender /scenes/whitechapel/whitechapel.blend --python /repo/scripts/06_verify_setup.py -- /out
```

The image is Ubuntu 22.04 with Xvfb, Mesa, openbox and Blender. It needs no root on the host
if you are in the `docker` group. `docs/HEADLESS.md` covers what works there and what does
not, in particular that `screenshot_camera_view` cannot work without a real display and
`SCOPE_CAPTURE=opengl` is required.

## Which shading to capture with

Two modes work. They cost very different amounts and they do not show the same things.

| | `SCOPE_SHADING=SOLID` + `SCOPE_SHADING_COLOR=TEXTURE` | `SCOPE_SHADING=MATERIAL` |
|---|---|---|
| Speed, software OpenGL | 7 to 9 s a frame | 40 to 90 s a frame |
| Speed, real GPU | well under a second | around a second |
| Textures, signage, graffiti | yes, all legible | yes |
| Seen through glass, for example a shop interior | **no, glass renders dark** | yes |
| Road and ground surface detail | flatter | full |
| Match to a desktop capture, as edge correlation | 0.24 | **0.80** |

**Use MATERIAL when you can.** It is what the scenes are saved as, it is what produced the
published numbers, and on a machine with a real GPU it is not meaningfully slower.

**Use SOLID with TEXTURE when software rendering makes MATERIAL impractical**, which is any
headless container. It keeps the textures and the lettering, so most questions are still
answerable, but be aware of what it loses. A question about what is inside a shop window
cannot be answered from an image where the window is dark. Report such a run as its own
configuration rather than comparing it against the published table.

## Then check that it worked

```bash
blender <scene.blend> --python scripts/06_verify_setup.py -- verify_out
blender <scene.blend> --python scripts/06_verify_setup.py -- verify_out --vlm   # also ask a VLM
```

The check exists because every setup failure met so far was silent. A blank capture, a
camera inside a wall, a missing sky texture tinting every frame, a scene photographed before
its textures finished loading: in each case the benchmark produced numbers and the numbers
were about an image nobody had looked at.

It runs in two tiers.

**Tier one is arithmetic on the pixels.** Free, no network, a few seconds a view. It reports:

- whether Blender has a viewport at all, and whether OpenGL is hardware or software
- every texture the scene refers to and cannot find
- how long the scene takes to reach a stable image, measured rather than assumed
- for each preset: the image size, whether the frame is blank, whether it is sharp enough to
  have finished drawing, whether it carries the magenta cast of a missing texture, and
  whether it has enough colour to be textured rather than a grey model
- whether the image is large enough that a vision model will see small objects and lettering

**Tier two asks a vision model.** One call per image, using the same model the benchmark
already needs, so it adds no dependency. It answers the question arithmetic cannot: does
this look like a coherent place, seen from a sensible position, with its surfaces intact. It
catches the case where every number is fine and the picture is still wrong.

The check writes a JSON report and the images beside it. Look at them. That is the point.

## Things worth knowing before you trust a run

**The scenes disagree about output size.** This is a property of the `.blend` files, not of
your machine, so it is the same everywhere:

| Scene | Render size | Aspect | Bit depth |
|---|---|---|---|
| book-nook | 3240 x 3240 | 1.00, square | 16 |
| city-street | 2309 x 1867 | 1.24 | 8 |
| postwar-city | 1920 x 1080 | 1.78 | 8 |
| whitechapel | 1920 x 1080 | 1.78 | 8 |

The model therefore sees a differently shaped frame depending on which scene a question came
from, and book-nook writes 84 MB images at 16 bits per channel, which is slow to produce and
slow to send anywhere. Worth normalising before a large run.

**Cold start is real and it is slower than it looks.** Opening a `.blend` returns before the
scene can be drawn. On whitechapel none of its 193 textures are resident at that moment; a
capture taken straight away is a blank grey rectangle, one at four seconds is still about a
third wrong, and the image is not stable until roughly twenty seconds. Fifteen seconds is a
sensible floor, and the verifier measures the real figure for your machine.

**Image quality is partly your choice.** With `SCOPE_CAPTURE=opengl` the output size comes
from the scene's own render settings, not from your window, so it is reproducible across
machines. With the default viewport capture it comes from the window, cropped to the camera
frustum, so a small window gives a small image. If the verifier warns that an image is under
about 1280 pixels on its long edge, make the window bigger or raise the scene's resolution:
a vision model asked to count objects or read a sign needs the pixels.

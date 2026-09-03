# Setting SCOPE up, and checking that it worked

SCOPE photographs a real Blender viewport. That one fact decides everything about setup: it
needs a machine that can draw a 3D window, and what the model sees depends on how that window
is drawn.

Pick the row that matches your machine.

| Your machine | What to do | Capture speed |
|---|---|---|
| Laptop or desktop with a screen | Install Blender and run it. Nothing else. | Fast. A real GPU draws a frame in well under a second. |
| Remote desktop, VM, or a fresh Linux install | Install Blender plus the OpenGL libraries below. | Fast if the display is GPU-backed. |
| Headless server, no display at all | Use the container in `docker/`. | Slow. Software rendering, seconds to a minute a frame. |

## A laptop or desktop

Install Blender, clone the repository, follow the quickstart in the README. There is nothing
to configure, because your machine already has a window system and a driver that can draw
into it.

## A remote desktop or a fresh Linux install

Blender needs an OpenGL context, and a minimal server install usually has no OpenGL at all.
On Debian or Ubuntu:

```bash
sudo apt-get install --no-install-recommends \
    libgl1 libglx-mesa0 libgl1-mesa-dri libegl1 \
    libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 libice6
```

With no display attached, add a virtual one and a window manager:

```bash
sudo apt-get install --no-install-recommends xvfb openbox
```

Two notes from doing this the hard way. The X server must be a current one: a CentOS 7 era
server starts and then Blender quits with `GLX_ARB_create_context not available`, because that
server cannot create the context Blender asks for. And a datacentre NVIDIA driver is usually
installed for compute only and ships no OpenGL at all, so Mesa's software renderer does the
drawing even on a machine full of GPUs.

If you would rather not change a shared machine, the container does the same thing in a box.

## A headless server

```bash
docker build -t scope-blender:4.4.3 docker/
SCOPE_CAPTURE=opengl docker/run.sh \
  blender /scenes/whitechapel/whitechapel.blend --python /repo/scripts/06_verify_setup.py -- /out
```

Ubuntu 22.04 with Xvfb, Mesa, openbox and Blender, giving GL 4.5 Core through llvmpipe. No
root on the host if you are in the `docker` group. `docs/HEADLESS.md` covers what works there
and what does not: in particular `screenshot_camera_view` cannot work without a real display,
so `SCOPE_CAPTURE=opengl` is required.

## Which shading to capture with

Two modes work. They cost very different amounts and they do not show the same things.

| | `SCOPE_SHADING=SOLID` + `SCOPE_SHADING_COLOR=TEXTURE` | `SCOPE_SHADING=MATERIAL` |
|---|---|---|
| Textures, signage, graffiti | yes, legible | yes |
| Anything behind glass, such as a shop interior | **no, glass renders dark** | yes |
| Match to a desktop capture, as edge correlation | 0.24 | **0.80** |

Asked what is inside the bookshop window, a vision model given a Material Preview capture
answers "bookshelves filled with books". Given a Solid capture of the same camera it answers
"a dark interior". Signage and object counts survive both; anything behind glass does not.

### What each costs

Measured per frame at a 640 pixel long edge, under software OpenGL, which is what a headless
container gets: an Xvfb display has no hardware GL, so a GPU on the box does not accelerate the
viewport. The first frame of each scene includes shader compilation and is reported separately
because it is not representative of the rest.

| scene | MATERIAL, first frame | MATERIAL, after | SOLID+TEXTURE |
|---|---|---|---|
| `book-nook` | 91 s | 41 to 62 s | 7 to 9 s |
| `postwar-city` | 32 s | 5 to 9 s | 1.2 to 1.4 s |
| `whitechapel` | 60 s | 10 to 46 s | not measured |
| `city-street` | 8500 s | 790 to 4200 s | 1.7 to 13 s |

`city-street` is the outlier and it is not a fluke: its materials are expensive enough to shade
in software that a nine frame panorama would take most of a day. Halving the capture resolution
brings it back into range, since software shading cost scales with pixel count.

These are software numbers. No figure here should be read as the cost on a machine with a real
display and a GPU driver, which was not measured.

**Use MATERIAL when you can.** It is what the scenes are saved as, and what produced the
published numbers. On three of the four scenes it costs tens of seconds a frame even in the
worst case, which is affordable.

**Drop the capture resolution before you drop the shading mode.** Resolution costs image
detail; shading mode costs whole categories of answer. A shop interior that is dark in SOLID is
dark at every resolution.

**Use SOLID with TEXTURE only when MATERIAL is genuinely impractical.** Report such a run as its
own configuration rather than comparing it against the published table.

## Then check that it worked

```bash
blender <scene.blend> --python scripts/06_verify_setup.py -- verify_out
blender <scene.blend> --python scripts/06_verify_setup.py -- verify_out --vlm
```

This exists because every setup failure met so far was silent. A blank capture, a camera
inside a wall, a missing sky tinting every frame, a scene photographed before its textures
loaded: in each case the benchmark produced numbers about an image nobody had looked at.

The failure is quiet rather than loud. Shown a panorama that was mostly empty viewport
background, a vision model described "a street scene under a dark sky". There was no sky in
that image. Nothing raised.

**Tier one is arithmetic on the pixels.** Free, no network, a few seconds a view. It reports
whether Blender has a viewport and whether OpenGL is hardware or software, every texture the
scene cannot find, how long the scene takes to reach a stable image, and for each preset
whether the frame is blank, sharp enough to have finished drawing, carrying a magenta cast, or
too small for a vision model to read a sign.

**Tier two asks a vision model**, one call an image, using the model the benchmark already
needs. It answers what arithmetic cannot: does this look like a coherent place, from a
sensible position, with its surfaces intact.

It writes a JSON report and the images beside it. Look at them. That is the point.

## Things worth knowing before you trust a run

**The scenes disagree about output size.** This is a property of the `.blend` files, so it is
the same on every machine:

| Scene | Render size | Aspect | Bit depth |
|---|---|---|---|
| book-nook | 3240 x 3240 | 1.00, square | 16 |
| city-street | 2309 x 1867 | 1.24 | 8 |
| postwar-city | 1920 x 1080 | 1.78 | 8 |
| whitechapel | 1920 x 1080 | 1.78 | 8 |

The model sees a differently shaped frame depending on which scene a question came from, and
book-nook writes 84 MB images at 16 bits per channel. Worth normalising before a large run.

**Cold start is real.** Opening a `.blend` returns before the scene can be drawn. On
whitechapel none of its 193 textures are resident at that moment: a capture taken straight
away is a blank grey rectangle, one at four seconds is still about a third wrong, and the
image is not stable until roughly twenty seconds. Fifteen seconds is a sensible floor, and it
costs almost nothing over a whole run because the runner only reopens a file when the scene
changes and the benchmark CSV is grouped by scene, which is four opens in 541 rows.

**Image size depends on how you capture.** With `SCOPE_CAPTURE=opengl` it comes from the
scene's own render settings, so it is reproducible across machines. With the default viewport
capture it comes from your window, cropped to the camera frustum, so a small window gives a
small image, and a workspace with several editors open gives a smaller one still. If the
verifier warns that an image is under about 1280 pixels on its long edge, enlarge the window
or raise the scene's resolution: a model asked to count objects or read a sign needs the
pixels.

**The scenes have no sky in the viewport.** The benchmark scenes are saved with studio
lighting, which means the viewport does not draw the scene's environment. For a single frame
pointed at a building this hardly matters. For a 360 degree panorama it matters a great deal,
because most directions in an outdoor scene are sky, and they come back as flat grey. The
published answers were labelled from captures taken this way, so turning the world on would
produce a prettier image that no longer matches what a labeller saw. It is a property of the
dataset rather than a defect to fix.

---
name: scope-new-world
description: Use when adding a new scene to SCOPE. Covers preparing the .blend, packing textures, creating camera presets, choosing the right shape of full view for each viewpoint, and pre-capturing panoramas so questions load them instead of rebuilding them.
---

# Adding a world

Budget an hour or two per scene, most of it unattended. Doing it properly once is much cheaper
than a benchmark that regenerates a panorama for every question.

## 1. Prepare the .blend

- **Pack the textures** (`File > External Data > Pack Resources`). Unpacked textures use paths
  relative to wherever the author had them, and those paths do not survive being shared. Every
  missing texture in the shipped scenes is this.
- **Check what is still missing** after packing, and record it in
  `benchmark/expected_assets.json`. Absent files are then reported as expected rather than as a
  broken download.
- **Do not rely on the world/sky.** Captures use studio lighting with the scene world off, so
  the sky is never drawn. A scene with no sky is fine.
- Note that scenes are often **built to be seen from one side**. Turning right round may show
  empty space. That is not a fault, but it changes what a full view should be (step 3).

## 2. Create camera presets

Each viewpoint the benchmark starts from is a Blender camera preset: a small Python file that
assigns location, rotation and lens directly.

```bash
blender <scene.blend>            # position the camera, then save a camera preset
blender --background --python scripts/04_install_presets.py
```

Guidance from the existing scenes:

- **Two to four viewpoints per scene** is enough. Each one carries dozens of questions.
- **Point them at something answerable.** Signage, countable objects, things behind glass.
- **Vary the height.** One elevated viewpoint per scene is valuable and behaves differently
  from street level, which is the point of step 3.
- A preset whose camera equals the scene's saved camera is legitimate; applying it is a no-op
  and the capture is identical to the file as opened.

Verify by capturing each one and looking at it, then add it to `docs/VISUAL_SMOKE_TEST.md`.

## 3. Choose the full view per viewpoint, by measuring

Questions whose `answer_view` is `full` need the agent to look around. **A 360&deg; sweep is
not always the right shape**, and which shape is right is not guessable from the camera height.

Two measurements decide it:

- **content** — how much of the output is scene rather than flat viewport background
- **scene coverage** — how many of the scene's mesh objects fall inside the captured frames

Coverage decides first: a full-view question needs to see the thing at all. Among candidates
that see everything, keep the sweep if it is also a good picture; otherwise take the cheapest
candidate that sees comparably much.

What that produced across the shipped scenes: roughly half the viewpoints keep a 360&deg;
sweep and half are better served by one wide frame. From above a bounded scene, a single 90&deg;
frame contained every object a nine-frame sweep did, in a tenth of the time. And **levelling a
pitched camera before sweeping**, which is right at street level, was the *worst* option for an
elevated one, because it aims the camera at empty sky.

`docs/FULL_VIEW.md` has the measurements and the candidate shapes.

## 4. Pre-capture the full views

Stitched panoramas are fussy: the result depends on frame overlap, the step angle, the camera
pitch and the capture resolution, and those interact. Seam quality stops improving somewhere
around forty percent overlap while the frame count keeps rising, and a low-resolution capture
stitches into something blurry however good the geometry is.

**So do not tune this per question.** Capture once, look at the result, keep it:

```bash
SCOPE_PANO_CACHE=benchmark/panoramas SCOPE_PANO_CACHE_MODE=write \
  blender benchmark/scenes/<scene>/<scene>.blend \
    --python scripts/precapture_panoramas.py
python3 scripts/index_panorama_cache.py        # writes the metadata a lookup validates
```

Then look at every result. One or two good panoramas per viewpoint beat a hundred mediocre ones
generated on demand.

`docs/PANORAMA_CACHE.md` explains how an entry is validated, which matters if you are shipping
the cache: entries are keyed by scene and preset but **served only when the live camera pose
matches**, so a stale or mislabelled entry cannot answer the wrong question.

## 5. Wire it up

- Add rows to the benchmark CSV — see the `scope-authoring-rows` skill.
- Add the scene to `benchmark/expected_assets.json`.
- Add its pictures to `docs/VISUAL_SMOKE_TEST.md`.
- Run the smoke test — see the `scope-smoke-test` skill.

## Watch out for

**A scene whose materials are expensive to shade.** One of the shipped scenes costs hundreds of
times more per frame than the others under software rendering, and the cost does not fall with
resolution because it is per-material shader compilation rather than per pixel. If a new scene
is unexpectedly slow, measure at two resolutions before trying to optimise: if the time barely
changes, resolution is not the lever.

# What "the full view" should be, and why it is not always a panorama

93 of the benchmark's 541 rows have `answer_view = full`. The question cannot be answered from
the starting frame, so the agent has to look around before answering. Every one of the ten
scene-and-preset combinations carries some of these rows, so every viewpoint needs a full view
that works.

The obvious implementation is a 360 degree sweep: turn the camera on the spot, capture a frame
every so often, reproject the frames onto a cylinder. That is right for most of the viewpoints
and wrong for at least one of them, and the difference is not a parameter to tune. It is the
shape of the thing.

## The measurement that decides it

Two numbers, because the obvious one is misleading on its own.

**Content** is the fraction of the output that is scene rather than flat viewport background.
It catches a sweep that is mostly empty sky.

**Scene coverage** is the fraction of the scene file's mesh objects that fall inside at least
one captured frame. It catches the opposite error, and it is the one that matters for a
full-view question: a picture can be completely full and still show you a quarter of the scene.

Content alone would have got this wrong. At street level a 360 strip scores 95.0 percent and a
single wide frame 91.8, which reads as interchangeable. They are not: one sees all the way
round and the other sees a quarter of it.

Scene coverage ignores occlusion, so an object behind a wall counts as visible. That is a real
limitation and it is worth stating, but it does not affect the conclusion below, where the
candidates differ by a factor of seven.

## Measured, on postwar-city

Two viewpoints in the same scene, one at street level and one above the rooftops. 61 mesh
objects, 900 pixel captures, 40 percent overlap.

### `street-view-1`, at z = 0.1, pitch already level

| shape | frames | wall clock | scene coverage | content |
|---|---:|---:|---:|---:|
| **360 strip** | 9 | 84 s | **100% (61/61)** | **95.0%** |
| 360 strip, levelled first | 9 | 53 s | 100% (61/61) | 93.0% |
| four 90° frames, stacked | 4 | 24 s | 100% (61/61) | 93.6% |
| one 90° frame | 1 | 9 s | 93.4% (57/61) | 91.8% |
| one 110° frame | 1 | 9 s | 93.4% (57/61) | 88.0% |
| one 130° frame | 1 | 9 s | 93.4% (57/61) | 79.1% |

A single frame, however wide, misses four objects. Widening the lens does not recover them,
because they are behind the camera. The strip is the right shape here, and it is also the
sharpest result.

### `diff-view-1`, at z = 3.4, pitched 48° down at a plaza

| shape | frames | wall clock | scene coverage | content |
|---|---:|---:|---:|---:|
| 360 strip | 9 | 85 s | 100% (61/61) | 60.5% |
| **one 90° frame** | **1** | **8 s** | **100% (61/61)** | 58.2% |
| one 110° frame | 1 | 14 s | 100% (61/61) | 42.0% |
| four 90° frames, stacked | 4 | 25 s | 100% (61/61) | 26.0% |
| one 130° frame | 1 | 9 s | 100% (61/61) | 26.9% |
| 360 strip, levelled first | 9 | 36 s | **13.1% (8/61)** | 10.8% |

One frame contains every object in the scene, exactly as many as nine frames do, in a tenth of
the time and with less empty sky in it. The camera is above a bounded scene looking down, so the
scene is *in front of* it rather than around it, and turning on the spot adds only sky.

## The correction that made it worse

Levelling the camera before sweeping is the obviously right thing to do, and on this viewpoint
it is the worst option by a wide margin: coverage falls from 61 objects to 8.

A real pan-tilt head levels itself before a preset tour, because a sweep about the vertical axis
traces a cone when the camera is pitched, and a cone through a street looks wrong. That
reasoning is sound and it assumes the scene is around the camera. When the camera is above the
rooftops, levelling aims it at the horizon, and there is nothing at the horizon.

So the pipeline sweeps both ways at any viewpoint whose pitch is not already level, measures
both, and keeps the better. It is two extra minutes per viewpoint and it removes a guess.

## The rule

| the camera is | the full view should be | because |
|---|---|---|
| at or near ground level, looking along | a 360° strip | a single frame misses what is behind it |
| well above the scene, looking down | one wide frame | the scene is already in front of it |
| pitched, at ground level | levelled, then a 360° strip | the cone through a street reads wrong |
| pitched, above the scene | left at its own pitch | levelling points it at empty sky |

Which case a viewpoint falls into is decided by measuring both, not by a rule of thumb about
height. `SCOPE_PANO_AB=0` disables the comparison and keeps the preset's own pitch.

## Reproducing it

```bash
# compare every candidate shape at one viewpoint, with both metrics
blender benchmark/scenes/postwar-city/postwar-city.blend \
  --python impl/exp_p_fullview.py -- out diff-view-1 \
    strip,strip_level,wide90,wide110,wide130,sweep4
```

Writes `out/fullview.json` with the numbers and `out/img/*.jpg` with each candidate, so the
ranking can be checked by looking rather than taken on trust. That matters here: an earlier
version of this comparison ran with a camera that was never moved to the preset, because
`preset_helpers.apply_preset` returns `False` rather than raising when it cannot find the
preset directory, and every log line claimed success. The numbers were meaningless and looked
fine.

See also [SCENES_AT_A_GLANCE.md](SCENES_AT_A_GLANCE.md) for every viewpoint and its full view,
and [HEADLESS.md](HEADLESS.md) if there is no display.

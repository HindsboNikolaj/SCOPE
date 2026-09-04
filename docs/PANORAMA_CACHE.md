# Reusing panoramas instead of re-sweeping them

93 of the benchmark's 541 rows have `answer_view = full`. The agent cannot answer from the
starting frame, so it sweeps the camera around and looks at the result. Those 93 rows start
from ten fixed camera positions, and nothing in these scenes moves between rows, so the same
ten sweeps were being recomputed for answers that cannot differ.

A sweep is nine or ten viewport captures and a reprojection. On a machine with a real GL driver
that is quick. Under software OpenGL it is 60 to 190 seconds, and on `city-street` it is hours,
because that scene costs roughly 700 seconds a frame whatever the resolution
(see [SETUP.md](SETUP.md)).

So a sweep is captured once per viewpoint, stored, and loaded thereafter.

## Using it

```bash
SCOPE_PANO_CACHE=benchmark/panoramas     # where entries live; default <output>/pano_cache
SCOPE_PANO_CACHE_MODE=use                # read and write. The default.
SCOPE_PANO_CACHE_MODE=write              # ignore what is stored, sweep, overwrite the entry
SCOPE_PANO_CACHE_MODE=off                # never read or write
```

Nothing else changes. `answer_view = full` behaves exactly as before; a hit simply returns
without yielding any frame steps, so a caller that renders sweep progress sees none.

Entries are ordinary files and can be pre-populated. Capturing the expensive scenes once,
offline and unhurried, then shipping the directory, is the intended way to use this:

```bash
# capture every viewpoint of one scene into the cache, however long it takes
SCOPE_PANO_CACHE=benchmark/panoramas SCOPE_PANO_CACHE_MODE=write \
  blender benchmark/scenes/city-street/city-street.blend \
    --python scripts/precapture_panoramas.py
```

## What makes a hit valid

**The key is a hint. The pose is the guarantee.**

Entries are named after the scene and the preset, because `whitechapel__preset-sor-viewpoint.png`
is a useful thing to see in a directory listing. But a name is never enough to serve on. Every
hit is checked against the process's actual state before it is returned:

| checked | why |
|---|---|
| the camera's location, rotation and focal length | the only check that cannot go stale, because it is read from the camera at lookup time |
| the `.blend` path actually open | two scenes in different directories can share a filename |
| the file's modification time | an edited scene invalidates its panoramas |
| the shading mode | a Solid panorama is not a substitute for a Material Preview one: Solid draws glass opaque, and on `city-street` it also loses the texture colour |

If any disagree, the entry is discarded and the sweep runs. A miss costs a sweep. A wrong hit
costs a wrong answer, so the checks are ordered to fail closed.

## Running several benchmarks at once

This is the case the design is built around: four scenes being benchmarked in parallel, sharing
one cache directory, and a question in one of them must never be answered with another's
picture.

**Preset names are not trusted across scenes or across file opens.** The name of the last
applied preset is a module global, so it survives `open_mainfile`: a second scene opened without
applying a preset would otherwise still report the first scene's viewpoint. Two things prevent
that becoming a wrong answer. A `load_post` handler clears the record when a file is opened, and
`last_applied_preset()` returns a name only when the camera is still exactly where applying it
put it, in the same `.blend`. Code that moves the camera directly is covered by the second even
though it defeats the first.

**Writes are atomic.** Both the image and its metadata are written to a temporary named with the
writing process's pid and renamed into place, so a reader in another process sees either the
previous entry or the new one, never a half-copied PNG. The image is renamed before the metadata,
and a reader requires both, so the transient state is a miss rather than a wrong hit.

**Concurrent writers of the same entry are harmless.** Two processes can only collide on a key
if they are at the same pose in the same scene in the same shading mode, in which case they are
producing the same picture.

## What is not handled

An entry captured at a different resolution than the current run is still served, because
resolution does not change what is in the picture and the benchmark's questions are about
content. If that matters for a particular use, run with `SCOPE_PANO_CACHE_MODE=write`.

The cache does not notice a scene changed by a script at runtime, only one changed on disk.
Nothing in the benchmark does that, but a tool that moved objects would need
`SCOPE_PANO_CACHE_MODE=off`.

## Related

- [FULL_VIEW.md](FULL_VIEW.md) for why the full view is not a 360 sweep at every viewpoint
- [SETUP.md](SETUP.md) for the per-scene capture costs that make this worth doing
- [SCENES_AT_A_GLANCE.md](SCENES_AT_A_GLANCE.md) for the ten viewpoints themselves

---
name: scope-verify-capture
description: Use after installing SCOPE, or whenever captures look wrong (black, grey, magenta, blank, low detail, or the wrong place). Verifies the viewport is producing correct pictures before a run is trusted. For installation itself, follow AGENT_INSTRUCTIONS.md first.
---

# Is this setup producing correct pictures?

**Install first.** `AGENT_INSTRUCTIONS.md` at the repo root is the end-to-end setup: models,
environment, scenes, presets, and a five-task smoke run. Follow it, then come back here.

This covers the part it does not: SCOPE grades models on screenshots of a Blender viewport, and
almost everything that goes wrong with those goes wrong *silently*. The capture succeeds, the
model answers, and the answer is about a picture nobody looked at. A run that passes
`run_eval_pipeline.sh` can still be measuring the wrong image.

So before trusting a long run, look at one picture.

## 1. Decide how you are capturing

| you have | use |
|---|---|
| a desktop with a display | nothing to set, the default works |
| a server with a GPU but no display | `SCOPE_CAPTURE=opengl`, and read `docs/HEADLESS.md` |

A datacentre GPU driver usually ships no OpenGL, so Blender will not start its interface at
all. `docs/HEADLESS.md` explains the virtual-display setup. **It is not the recommended
configuration**: without a GL driver the viewport falls back to software rendering, and one
scene becomes hundreds of times slower. Prefer a machine with a real display.

## 2. Look at a picture before trusting anything

```bash
blender <scene.blend> --python scripts/06_verify_setup.py -- verify_out
```

Then open `docs/VISUAL_SMOKE_TEST.md` and compare. It shows every camera position the benchmark
uses. If yours does not look like the reference, stop and fix it; a run started now produces
numbers about the wrong image.

| what you see | what it is |
|---|---|
| black rectangle | no display. Set `SCOPE_CAPTURE=opengl`. |
| small picture in a big grey field | the camera frame is not filling the render. Update. |
| everything magenta | a missing sky is being drawn. Captures must use studio lighting with the scene world off. |
| grey shapes, no textures | Solid shading without textures. |
| blank or half-drawn | captured too soon after opening the file. See below. |
| the wrong place | the preset did not apply. Re-run `04_install_presets.py`. |

## 3. Give a scene time to load

`open_mainfile` returns before the scene can be drawn. Textures stream in afterwards. A capture
taken immediately is a blank rectangle, and at four seconds it still looks plausible while a
third of the pixels are wrong, which is the dangerous part.

`SCOPE_SETTLE` sets the wait in seconds; the default of 15 is on the safe side. This applies to
**opening a file**, not to each camera move: the per-frame wait during a panorama is a third of
a second and does not need raising. `docs/COLD_START.md` has the measured curve.

A resumed run is a cold start too, because it reopens the scene.

## 4. Missing textures are expected

Some are absent and unrecoverable, and `benchmark/expected_assets.json` records exactly which.
The verifier compares against it and distinguishes "as expected" from "worse than expected".
Absent textures do not affect a run: captures use studio lighting, so a missing sky is never
drawn.

## Not covered here

- **Installing anything** — `AGENT_INSTRUCTIONS.md`.
- **The agent's system prompt and the judge rubrics** — `prompts/`. Those are the benchmark's
  measured apparatus, not configuration: changing one changes what the numbers mean. Treat them
  as versioned experiment inputs and never edit them to make a run pass.
- **Adding a scene, or authoring rows** — `docs/creating_scenes.md`.

## References

- `references/silent-failures.md` — the failure modes that produce confident wrong answers, and
  the check that catches each. Read this if something looks *almost* right.
- `docs/COLD_START.md`, `docs/HEADLESS.md`, `docs/SETUP.md`, `docs/VISUAL_SMOKE_TEST.md`

## The rule worth carrying

**Render it and look at it.** Every real defect found while building this was invisible to the
metric that was supposed to catch it, and obvious in the picture.

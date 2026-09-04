---
name: scope-smoke-test
description: Use when verifying a SCOPE setup is producing correct captures and correct tool behaviour, including how to use an LLM or VLM as a judge and when not to trust one. Covers deterministic checks, comparison against a known-good run, and judged checks.
---

# Verifying a setup

Three layers, cheapest first. Each catches things the others cannot, and the last one is the
least trustworthy, which is the part people get wrong.

## Layer 1: deterministic checks

Assertions with exact expected answers. Fast, free, and they cannot tell you that a picture
came back looking wrong in a way nobody thought to assert.

```bash
blender <scene.blend> --python scripts/06_verify_setup.py -- verify_out
```

Checks the capture is not blank, the scene has settled, the assets match
`benchmark/expected_assets.json`, and the camera lands where its preset says.

For the tools themselves, each call states what should happen before it runs, then the camera
is measured and the frames either side are kept.

**Check the picture, not the parameter.** A rotation check passes for an implementation that is
internally consistent and still aims the camera the wrong way. If the camera turns right, the
scene must slide left; that check cannot be satisfied by a sign error. See `docs/TOOL_DEMO.md`.

## Layer 2: comparison against a known-good run

Structural comparison of this machine's captures against a stored reference. Deliberately
loose: two machines will not produce identical pixels, so the comparison is of edge structure
rather than colour, and a generous threshold decides whether a setup is in the right ballpark.

Catches: a camera in the wrong place, a scene half-loaded, a shading mode that changed.
Does not catch: both machines being wrong the same way.

## Layer 3: a judge

Show a model the image and ask whether it is right. This catches what an assertion cannot
phrase — a missing texture, a magenta cast, a scene that loaded only partly — and it is the
layer most likely to mislead you.

### Using a VLM as a judge

Ask for **structured output with a reason**, and keep the reason. A score with no reason
attached cannot be checked, and a model will happily produce a confident number.

```
Reply as strict JSON, these keys only:
  "seam":  0-10, 10 means no visible join at all
  "blank": 0-10, 10 means the crop is all scene
  "reason": one short sentence naming what you can actually see wrong, or "nothing"
```

### Where a VLM judge fails, from experience

**Aspect ratio.** A wide panorama squashed to fit a model's input is unreadable before the
model sees it. Judge square crops at close to native scale instead of the whole strip.

**Calibrate before trusting.** Run the judge on one output you know is good and one you know is
bad, *first*. A judge that scores them the same is not measuring anything. This has happened:
a 4B vision model scored a panorama that was 90% empty grey the same as a correct one, and
rated one with a large void *higher*.

**Prefer a measurement where one exists.** "How much of this image is flat background" is a
deterministic calculation, and it discriminated cleanly where the judge did not. Use the model
for what cannot be computed — does this look like a plausible street, is anything duplicated —
and compute the rest.

### Using an LLM as a judge

For grading text answers against `expected_answer`, give it the question, the expected answer
and the model's answer, and ask for a verdict plus a one-line justification. Watch for it
rewarding fluency: a wrong answer stated confidently should not outscore a right answer stated
plainly.

## A judge is not a substitute for looking

Every defect worth finding in this project was caught by rendering the thing and looking at it,
after a metric had already reported success. Budget a few minutes for that on any new setup.

See `.claude/skills/scope-setup/references/silent-failures.md` for the catalogue.

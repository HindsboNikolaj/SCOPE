# Failures that answer instead of erroring

Every defect found while building the capture path had the same shape: nothing raised, the
pipeline completed, and the output was confidently wrong. They are collected here because the
pattern generalises, and because recognising it is faster than rediscovering each one.

Shown a panorama that was mostly empty viewport, a vision model described "a street scene under
a dark sky". There was no sky in the image. That is the whole problem in one sentence: a broken
capture does not fail, it answers.

## The catalogue

| symptom | cause | what actually catches it |
|---|---|---|
| capture is solid black | `screenshot_area` returns black on a virtual display and raises nothing | look at the file; a warning now fires on a flat capture |
| a tool reports success after moving nothing | detection returned nothing and the code fell back to a whole-frame box, which is a 2% zoom *out* | check the `found` field, not the camera |
| panorama is mirrored | the output azimuth was composed with the sweep index the wrong way round | read text in the picture; no metric sees a mirror |
| capture is 77% grey border | in camera view the camera frame is inset in the region, and the render includes the background | measure how much of the frame is scene |
| preset silently not applied | `apply_preset` returns `False` rather than raising | check the return; the runner now warns |
| a "full" panorama is 90% empty sky | the camera is above the scene, so sweeping adds nothing | measure scene coverage, not image fullness |
| scene graded before it loaded | `open_mainfile` returns early | wait, and see `docs/COLD_START.md` |

## Why the metrics missed them

Three of these survived a check designed to catch them.

**A reference built by the code under test cannot detect a mirror in that code.** The panorama
round trip scored 0.67 mean absolute error out of 255 and passed twice, because it generated its
reference frames with the same azimuth convention the stitch used. Ground truth has to come from
outside: a real capture with readable text in it.

**"Not blank" is not "correct".** A content metric counting local gradient scored a panorama at
88.8% scene when it was visibly half flat grey, because the grey ran 58 to 61 and a box average
of that clears a small threshold. Raising the threshold then failed on a genuinely detailed but
low-contrast scene. The fix was to identify the background colour rather than infer it.

**"Full" is not "complete".** At street level a 360 sweep and a single wide frame score 95% and
92% on how much of the picture is scene, which reads as interchangeable. One sees all the way
round and the other sees a quarter of it. Measuring against the scene's own geometry separated
them.

## The habits that follow

1. **Render it and look at it.** Before believing a number about an image.
2. **Ground truth from outside the code.** A test that shares a convention with the code under
   test validates the convention, not the code.
3. **Prefer a loud failure to a plausible default.** A fallback that returns something usable
   turns a setup error into a wrong answer.
4. **Check the return value.** Especially from anything that returns `False` instead of raising.
5. **Give a long job a deadline and a completion marker.** Otherwise "still running" and "stuck"
   look identical, and a finished job that never exits looks like a hang.

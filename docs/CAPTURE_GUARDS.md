# Capture guards

What the Blender side now checks on every frame and every run, and the knobs. All of it came
out of a 29-run campaign in which the recurring failure was a run that completed, produced 536
rows and a plausible score, and measured nothing.

## Running headless

`bpy.app.timers` never fires without a window manager, and `bpy.context.window` / `.screen`
are `None` outside the event loop. The runner now probes for a timer and drives the loop
directly when there is none, and every tool resolves its view through the window manager, so
the benchmark runs in a container with no display (see [`HEADLESS.md`](HEADLESS.md)).
Setting `show_region_ui` / `toolbar` / `header` outside the event loop segfaulted Blender; those
calls are gone and never affected the image. `go_to_preset` now honours `SCOPE_PRESET_DIR`
like everything else.

`scripts/run_shard.sh` runs one scene per container on the GPU image from
`docker/build-gpu.sh`, with its own X display and scratch. Host paths come from the environment
(`.env.example`).

## Per frame: blank and under-drawn

[`src/scope/tools/frame_probe.py`](../src/scope/tools/frame_probe.py) classifies each capture
as `blank`, `under-drawn` or `fine` from the image the capture path already holds, in about
0.06 ms. It replaced a guard that decoded a 3240 x 3240 PNG twice per frame at 510 ms each.

- **Under-drawn** frames (textures still streaming) wait in 120 ms steps, up to 400 ms, and
  re-capture (`SCOPE_DETAIL_WAIT_MS`, `SCOPE_DETAIL_WAIT_STEP_MS`).
- **Blank** frames are counted and kept as evidence, and **not redrawn.** The redraw was
  measured and removed: 11 blanks, 22 retries, 0 recoveries, and the second draw never changed
  one pixel. The blanks were real: whitechapel has geometry on one side of the street only, so
  a 180-degree pan faces open sky.

Counts are written to `_capture_stats.json` beside the screenshots.

## Per run: colour

A single grey frame cannot tell "the world rendered without materials" from "the camera is
looking at something grey". A whole run can. `run_colour_health()` fails a shard whose median
colourfulness is below `SCOPE_RUN_COLOUR_FLOOR` (default 6.0), or where more than
`SCOPE_RUN_GREY_FRACTION` (default 25%) of frames fall below `SCOPE_COLOUR_MIN` (3.0), and prints
`SCENE RENDERED WITHOUT COLOUR` on the shard log. It exists because
`benchmark/preset_views/city-street__mailbox.png` measures 3.1 (flat grey walls, a colourless
sign) while the same view captured live measures 21.7.

## Disk and the vision model

- **Capture ring.** Full-resolution captures are about 15 MB each; keeping all of them left
  tens of GB per run. The default keeps the most recent 8 (`SCOPE_CAPTURE_RING`);
  `SCOPE_KEEP_CAPTURES=1` keeps everything.
- **Thumbnails for audit.** Every Nth frame is kept as a small JPEG (`SCOPE_SAMPLE_EVERY`,
  default 25; `SCOPE_SAMPLE_EDGE`, default 512 px). A run stays auditable in under 2 MB where
  full frames had reached 59 GB.
- **Pixel budget.** Frames reach the VLM capped at 1920 x 1080 pixels (`SCOPE_VLM_MAX_PIXELS`),
  scaled with BOX (43 ms against 152 ms for LANCZOS). book-nook renders at 3240 x 3240; on it
  the median vision call fell from 7.41 s to 2.57 s.
- **Box coordinates.** A served VLM answers boxes in pixels by default while the tools read
  0..1 fractions, which sent every non-Moondream zoom to the top-left corner. Coordinates above
  1 are now normalised by the frame size (`SCOPE_VLM_COORD_SPACE`).

## Agent loop

- `SCOPE_MAX_TURNS` (default 40) ends a looping planner's row, not the run.
- A tool call with malformed arguments returns the error to the model instead of killing the
  shard.
- Reasoning-channel markers from several model families are stripped before an answer is
  recorded, so they are not graded as part of it.
- `runner.py` writes a `turn_timings_json` column with per-turn planner, vision and render
  times, and resume upgrades a CSV whose header is a prefix of the new one instead of aborting.

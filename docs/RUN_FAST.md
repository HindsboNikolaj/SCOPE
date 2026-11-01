# Run a pairing in parallel

After the prerequisites below, run the smoke set, then the complete question set, with the planner and
vision settings from `.env` (the default is the one-server Gemma recipe in
[`CONFIGURATION.md`](CONFIGURATION.md#measured-recipes-dataset-v2)):

```bash
set -a; source .env; set +a
python3 scripts/run_fast.py --questions benchmark/scope_v2_smoke.csv --out results/smoke --judge
python3 scripts/run_fast.py --out results/my-pairing --judge    # all 536 tasks
```

These run six workers on `--gpu`. If the model server is on the same card, add `--workers 2`
(prerequisite 4).

Run the 24-task smoke set first ([`benchmark/scope_v2_smoke.csv`](../benchmark/scope_v2_smoke.csv);
the README gives the default setup's v2 score on it). Two environment variables set where the
run lands; the defaults are shown:

| Variable | Default | Use |
| --- | --- | --- |
| `SCOPE_GPU_DEVICE` | `0` | Capture card (index or UUID) for `run_fast.py` and `docker/build-gpu.sh`; `--gpu` overrides it |
| `SCOPE_CONTAINER_PREFIX` | `scope` | Start of every container name both scripts (and the commands below) create, for hosts with naming rules |

`--judge` runs the existing judge and metrics after answer coverage passes.
Configure the judge endpoint, model and key as in the setup guide. Answer and
judge wall times are recorded separately. A judge failure returns nonzero and
leaves answer results intact for resume. Without `--judge`, the command stops
after answer generation; to grade those results separately:

```bash
python3 -m scope.eval.judge -i results/my-pairing/raw_results.csv \
  -o results/my-pairing/judged_results.csv --concurrency 8
python3 -m scope.eval.metrics report -i results/my-pairing/judged_results.csv
```

Use the existing judge endpoint/model/key environment variables. To compare
with serial judging, set `--concurrency 1`. Concurrency allows the server to
schedule several requests; it does not guarantee GPU batching or equal latency.

`--judge` uses one judge. To grade like v2, run the three-judge panel on `raw_results.csv`
([`JUDGING.md`](JUDGING.md#3-the-v2-judge-panel)).

## Prerequisites

1. Linux, Docker with NVIDIA GPU access, and Python dependencies from the
   repository install. Set API credentials in the environment, not this command.
2. Download the benchmark scenes and install the camera presets using
   [the setup guide](../AGENT_INSTRUCTIONS.md). Keep original scene/texture
   licences with the assets. Scenes are not in this repository; they come from Hugging Face.
   Host Blender is not needed: after step 3, the Blender check and the presets can run in the
   image, with no GPU, writing the presets to `.scope-presets/` in the repository:

   ```bash
   docker run --rm --name "${SCOPE_CONTAINER_PREFIX:-scope}-blender-check" \
     --user "$(id -u):$(id -g)" -e HOME=/tmp --entrypoint bash -v "$PWD:/repo:ro" scope-blender:gpu \
     -c 'cd /repo && python3 scripts/install_blender_deps.py --check'
   mkdir -p .scope-presets
   docker run --rm --name "${SCOPE_CONTAINER_PREFIX:-scope}-presets" \
     --user "$(id -u):$(id -g)" -e HOME=/tmp -e BLENDER_USER_SCRIPTS=/scripts \
     -v "$PWD:/repo:ro" -v "$PWD/.scope-presets:/scripts" --entrypoint blender scope-blender:gpu \
     --background --python /repo/scripts/04_install_presets.py
   ```

   `run_fast.py` uses `.scope-presets/presets/camera` when it exists, before host Blender's
   folder; it prints the folder it used and stops if a preset the tasks need is missing.
3. Build the base image with `docker build -t scope-blender:4.4.3 docker/`, then build and
   verify `scope-blender:gpu` with [docker/build-gpu.sh](../docker/build-gpu.sh). This adds
   hardware OpenGL to the base image; Xvfb alone can leave rendering on the CPU.
4. Start the planner and vision services first. `--gpu` selects the capture card; every worker
   calls the same planner and vision endpoints, so one loaded model serves all shards. The
   default Gemma server can share the capture card: on an 80 GB A100 it holds 67.4 GiB at
   `--gpu-memory-utilization 0.85` (48.5 GiB weights, 17.8 GiB cache), and a Blender worker
   takes about 2.5 GiB (3.5 GiB at most), so two workers fit beside it (`--workers 2`). The six
   default workers need about 21 GiB; give them a card of their own, as the timed runs below did.
5. Set `--scenes` and `--presets` if the downloaded scenes or Blender camera
   presets are elsewhere. The defaults are `benchmark/scenes` and
   `.scope-presets/presets/camera` if present, else the Blender 4.4 camera-preset
   directory. Host-network Docker allows workers to reach model servers on localhost.

The launcher does not select free GPUs or start/reconfigure model services.
An agent can check endpoints, use `--dry-run`, invoke this command and inspect
its coverage and sample images. It should not alter questions or rubrics to
get a completed run.

## What makes this execution shape faster

The default input is dataset v2, `benchmark/scope_v2.csv` (536 rows); `--dataset v1` selects
the paper's 536 tasks. Both span four worlds. Workers for different worlds overlap;
book-nook and city-street each get two workers, producing six homogeneous
scene shards. Every worker retains its loaded scene, uses its own display and
capture directory, and shares externally served model instances. Workers use host networking,
which shares X display sockets across containers, so each worker's Xvfb takes a free display
number itself (`-displayfd`); two launches on one host do not collide either. Upstream
already retains a loaded scene when the next row names it. The addition here
is parallel homogeneous shards and complete-output validation.

`SCOPE_CAPTURE=opengl` and `SCOPE_SHADING=MATERIAL` retain material detail on
a virtual display. Each worker requires an NVIDIA OpenGL renderer before
starting Blender. This is **not** Blender `--background` mode, which cannot
run the viewport capture path. Renderer output is recorded in the shard log.

Stored panoramas are read with `SCOPE_PANO_CACHE_MODE=use`, using the existing
cache validation. The repository is mounted read-only, so cache misses cannot
persist replacements there; prebuild panoramas before the run. Cache availability
does not imply that an arbitrary camera pose will hit the cache.

The fast launcher sets the vision input the v2 runs used: a 2,073,600-pixel (1080p-area)
budget and `SCOPE_VLM_COORD_SPACE=auto` (0..1, or pixels when a value exceeds 1). Outside the
launcher, served vision models (Gemma, Qwen, Molmo) get the same budget by default, while the
Moondream clients default to no reduction (`SCOPE_VLM_MAX_PIXELS=0`); `.env.example` sets the
budget for both. A pixel budget changes model input and must be recorded when comparing
accuracy. `unit`, `pixel` and `percent` force one coordinate reading.

Captures retain a short rolling set and sampled thumbnails. Set
`SCOPE_KEEP_CAPTURES=1` in direct runner use if every full frame is required;
the fast launcher uses the bounded capture policy. Inspect sampled frames
against [VISUAL_SMOKE_TEST.md](VISUAL_SMOKE_TEST.md) before accepting scores.

## Expected time and hardware

On a shared A100 80GB server, Qwen3.8-27B with Moondream 3.1 answered all 536 tasks in 50.8 min
plus 6.9 min loading models, on three cards (planner, vision, and one card for the six Blender
workers). Judging, downloads, image builds and scene preparation are excluded. These are
observations from retained runs with the same execution shape, not predictions. That run
used an earlier six-shard launcher; `run_fast.py` itself is covered by CPU
tests and a full 536-row dry run, and has no recorded GPU timing of its own yet. Its timer
starts once the model servers are ready. With gemma-4-26B-A4B-it as the vision model instead,
the same shape took 46.8 min plus 12.3 min loading. The four worlds run at once, so their
times overlap and do not add up to the total.

## Resume, failures and re-runs

Re-use the exact command and `--out` to resume. A manifest binds the question
bytes, source code, prompts/schema, model settings, image ID, scenes, presets and panorama bytes. An
input or image change requires a fresh output directory. Concurrent writers to one output
directory are rejected.

An interrupted shard resumes successful `(question_id, repeat_idx)` pairs.
The launcher retries incomplete shards at most twice by default. Completed
shards, including explicit model non-answers, are retained on the next
invocation. Use `--rerun-failed` to retry non-answers; success is never replaced
by a later empty answer when merging. `REPEATS` is included in the run manifest.

The final CSV has one selected row per expected question/repeat. The report
counts missing keys, failed answers and duplicate attempts separately. Missing
rows or a failed worker produce a nonzero exit. An attempted non-answer is
reported as a failed answer, not hidden as missing work or treated as success.
The new 40-turn planner cap also changes the evaluation contract and is
recorded in the manifest.

Use a new `--out` for a full fresh run. For an intentional subset, supply a
CSV using `--questions`, keep its hash and label it as a subset. Never present
reused answers plus a selective re-run as newly measured full-pairing time.
Raw per-shard files and invocation reports remain available beside the merged
CSV; keep both with the manifest when auditing a resumed run.

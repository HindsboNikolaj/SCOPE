#!/usr/bin/env python3
"""Run one pairing in parallel scene shards, using already-served models.

Requires Linux Docker, an image built by docker/build-gpu.sh, downloaded scenes,
installed camera presets, and explicit AGENT_MODEL_ID / VLM_MODEL settings.
See docs/RUN_FAST.md. --dry-run plans without Docker or filesystem writes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ENV = (
    "AGENT_API_BASE", "AGENT_MODEL_ID", "VLM_MODEL", "VLM_MODEL_ID", "VLM_MODEL_URL",
    "SCOPE_MAX_TURNS", "SCOPE_VLM_MAX_PIXELS", "SCOPE_VLM_COORD_SPACE",
    "SCOPE_PANO_CACHE_MODE", "SCOPE_SHADING", "SCOPE_CAPTURE", "REPEATS",
)
SECRET_ENV = ("AGENT_API_KEY", "VLM_API_KEY", "MOONDREAM_API_KEY")
# Workers share the host network, and with it the abstract X socket namespace, so a fixed
# display number lets only one Xvfb per host start. -displayfd makes Xvfb take a free display
# and report it, which keeps displays unique across shards and across launches on one host;
# the October 2026 v2 runs started their workers this way. glxspheres64 draws until it is
# killed unless -f bounds its frames.
INNER = r'''
set -eu
command -v vglrun >/dev/null
command -v Xvfb >/dev/null
Xvfb -displayfd 3 -screen 0 1920x1080x24 -nolisten tcp 3>/tmp/xdisplay > /run/xvfb-$2.log 2>&1 &
for n in $(seq 1 100); do
  [ ! -s /tmp/xdisplay ] || break
  sleep 0.1
done
test -s /tmp/xdisplay
export DISPLAY=":$(cat /tmp/xdisplay)"
echo "X display $DISPLAY"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
renderer=$(timeout 60 vglrun -d egl0 /opt/VirtualGL/bin/glxspheres64 -n 2 -f 10 2>&1) || {
  printf '%s\nrenderer probe failed\n' "$renderer"; exit 1; }
printf '%s\n' "$renderer"
printf '%s\n' "$renderer" | grep -qi 'OpenGL Renderer.*NVIDIA'
cd /repo
exec vglrun -d egl0 blender "$1" --python-exit-code 1 --python /repo/src/scope/eval/runner.py
'''


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def write_csv(path, header, rows):
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def plan(rows):
    """Keep each world loaded; split the two longest worlds into two workers."""
    by_scene = {}
    seen = set()
    for row in rows:
        qid = row.get("question_id", "")
        if not qid or qid in seen:
            raise ValueError("questions require nonempty, unique question_id values")
        seen.add(qid)
        scene = row.get("file_location", "")
        path = Path(scene)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".blend":
            raise ValueError(f"invalid relative scene path: {scene}")
        by_scene.setdefault(scene, []).append(row)
    if not rows:
        raise ValueError("question set is empty")
    shards = []
    for scene, items in sorted(by_scene.items()):
        parts = 2 if Path(scene).stem in {"city-street", "book-nook"} else 1
        parts = min(parts, len(items))
        name = re.sub(r"[^A-Za-z0-9_-]", "-", Path(scene).stem)
        name += "-" + hashlib.sha256(scene.encode()).hexdigest()[:8]
        for i in range(parts):
            # Contiguous halves preserve the dataset's preset order.
            chunk = items[len(items) * i // parts:len(items) * (i + 1) // parts]
            shards.append({"id": f"{name}-{i+1}", "scene": scene, "rows": chunk})
    return shards


def merge(shards, out, source_rows, repeats):
    expected = {(r["question_id"], str(i)) for r in source_rows for i in range(1, repeats + 1)}
    selected, header, duplicates = {}, None, 0
    for shard in shards:
        path = out / (shard["id"] + ".csv")
        if not path.exists():
            continue
        fields, rows = read_csv(path)
        if header is not None and fields != header:
            raise ValueError("shards have incompatible output schemas")
        header = fields
        for row in rows:
            key = (row.get("question_id"), row.get("repeat_idx") or "1")
            if key not in expected:
                raise ValueError(f"unexpected result key: {key}")
            if key in selected:
                duplicates += 1
            # A failed retry cannot replace an already successful answer.
            if key not in selected or (row.get("final_answer") or "").strip():
                selected[key] = row
    missing = expected - selected.keys()
    failures = sum(not (r.get("final_answer") or "").strip() for r in selected.values())
    if header:
        ordered = [selected[(r["question_id"], str(i))] for r in source_rows
                   for i in range(1, repeats + 1) if (r["question_id"], str(i)) in selected]
        write_csv(out / "raw_results.csv", header, ordered)
    return {"expected": len(expected), "rows": len(selected), "missing": len(missing),
            "failed_answers": failures, "duplicate_attempts": duplicates}


def judge_and_report(out):
    """Use the existing judge and metrics modules; keep their wall time separate."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    judged = out / "judged_results.csv"
    start = time.monotonic()
    subprocess.run([sys.executable, "-m", "scope.eval.judge", "-i", str(out / "raw_results.csv"),
                    "-o", str(judged), "--concurrency", os.getenv("JUDGE_CONCURRENCY", "8")],
                   env=env, check=True)
    seconds = round(time.monotonic() - start, 3)
    subprocess.run([sys.executable, "-m", "scope.eval.metrics", "report", "-i", str(judged)],
                   env=env, check=True)
    return seconds


HOST_PRESETS = Path.home() / ".config/blender/4.4/scripts/presets/camera"


def default_presets():
    """The in-image install's .scope-presets folder (docs/RUN_FAST.md) wins over host Blender's."""
    repo = ROOT / ".scope-presets/presets/camera"
    return repo if repo.is_dir() else HOST_PRESETS


def missing_presets(presets, rows):
    """Presets the questions start from or offer (preset_start, presets_available) that the folder lacks."""
    wanted = set()
    for row in rows:
        wanted.update(json.loads(row.get("presets_available") or "[]"))
        wanted.add(row.get("preset_start") or "")
    return sorted(n for n in wanted - {""} if not (presets / f"{n}.py").is_file())


DATASETS = {"v2": ROOT / "benchmark/scope_v2.csv",
            "v1": ROOT / "benchmark/archive/scope_536_v1.csv"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="v2",
                    help="benchmark version: v2 (default, corrected) or v1 (the paper's 536 tasks)")
    ap.add_argument("--questions", type=Path, help="explicit questions CSV; overrides --dataset")
    ap.add_argument("--out", type=Path, required=True, help="stable directory; reuse it to resume")
    ap.add_argument("--gpu", default=os.getenv("SCOPE_GPU_DEVICE", "0"),
                    help="one capture GPU index or UUID (default: $SCOPE_GPU_DEVICE, else 0)")
    ap.add_argument("--image", default="scope-blender:gpu")
    ap.add_argument("--scenes", type=Path, default=ROOT / "benchmark/scenes")
    ap.add_argument("--presets", type=Path,
                    help="camera preset folder (default: .scope-presets/presets/camera if present, "
                         "else Blender 4.4's user presets folder)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--attempts", type=int, default=2, help="bounded retry of incomplete shards")
    ap.add_argument("--rerun-failed", action="store_true", help="retry terminal non-answers from a previous run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--judge", action="store_true", help="also judge the merged CSV and print metrics")
    args = ap.parse_args(argv)
    if args.workers < 1 or args.attempts < 1:
        ap.error("workers and attempts must be positive")
    prefix = os.getenv("SCOPE_CONTAINER_PREFIX", "scope")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", prefix):
        ap.error("SCOPE_CONTAINER_PREFIX must be a valid Docker container name")
    args.presets = args.presets or default_presets()
    args.questions = args.questions or DATASETS[args.dataset]
    header, rows = read_csv(args.questions)
    shards = plan(rows)
    env = {k: v for k in PUBLIC_ENV if (v := os.getenv(k)) is not None}
    defaults = {"SCOPE_MAX_TURNS": "40", "SCOPE_VLM_MAX_PIXELS": "2073600",
                "SCOPE_VLM_COORD_SPACE": "auto", "SCOPE_PANO_CACHE_MODE": "use",
                "SCOPE_SHADING": "MATERIAL", "SCOPE_CAPTURE": "opengl", "REPEATS": "1"}
    env = {**defaults, **env}
    repeats = int(env["REPEATS"])
    if repeats < 1:
        ap.error("REPEATS must be positive")
    fingerprint = {"questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
                   "environment": env, "image": args.image, "scenes": str(args.scenes.resolve()),
                   "presets": str(args.presets.resolve()), "workers": args.workers,
                   "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "source_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in
                       sorted((ROOT / "src").rglob("*.py")))).hexdigest()}
    overview = [{"id": s["id"], "scene": s["scene"], "questions": len(s["rows"])} for s in shards]
    if args.dry_run:
        print(json.dumps({"shards": overview, "expected_rows": len(rows) * repeats,
                          "workers": args.workers, "capture_gpu": args.gpu}, indent=2))
        return 0
    for key in ("AGENT_MODEL_ID", "VLM_MODEL"):
        if not env.get(key):
            ap.error(f"set {key} explicitly; launch model servers first")
    endpoint = urlparse(env.get("VLM_MODEL_URL", ""))
    if (not env.get("VLM_MODEL_URL", "").startswith(("http://", "https://"))
            or endpoint.scheme not in {"http", "https"} or not endpoint.netloc):
        ap.error("set an HTTP(S) VLM_MODEL_URL for the shared vision server")
    if not args.scenes.is_dir():
        ap.error(f"scene directory not found: {args.scenes}")
    if not args.presets.is_dir():
        ap.error(f"camera preset folder not found: {args.presets}; install the presets "
                 "(docs/RUN_FAST.md, prerequisite 2) or pass --presets")
    if missing := missing_presets(args.presets, rows):
        ap.error(f"camera presets missing from {args.presets}: {', '.join(missing)}")
    print(f"run_fast: camera presets from {args.presets.resolve()}", file=sys.stderr)
    for s in shards:
        if not (args.scenes / Path(s["scene"]).relative_to("scenes")).is_file():
            ap.error(f"missing scene: {s['scene']}")
    # Pin the bytes used by capture and model prompting, not just their directory names.
    inputs = {f"scene/{s['scene']}": args.scenes / Path(s["scene"]).relative_to("scenes") for s in shards}
    inputs.update({f"preset/{p.name}": p for p in sorted(args.presets.glob("*.py"))})
    for directory in (ROOT / "prompts", ROOT / "benchmark/panoramas"):
        inputs.update({str(p.relative_to(ROOT)): p for p in sorted(directory.rglob("*")) if p.is_file()})
    inputs["src/scope/tools/schema.json"] = ROOT / "src/scope/tools/schema.json"
    fingerprint["input_sha256"] = {label: digest(p) for label, p in inputs.items()}
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / ".run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        ap.error("another launcher is writing this output directory")
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != fingerprint:
            ap.error("run inputs changed; use a new --out directory")
    elif any(p.name != ".run.lock" for p in args.out.iterdir()):
        ap.error("output has files without a manifest; use a new directory")
    manifest_path.write_text(json.dumps(fingerprint, indent=2) + "\n")
    image_id = subprocess.check_output(["docker", "image", "inspect", args.image,
                                       "--format", "{{.Id}}"], text=True).strip()
    image_path = args.out / "image-id.txt"
    if image_path.exists() and image_path.read_text().strip() != image_id:
        ap.error("Docker image changed; use a new --out directory")
    image_path.write_text(image_id + "\n")
    run_id = uuid.uuid4().hex[:10]

    def worker(shard):
        name = shard["id"]
        csv_path = args.out / f"{name}.csv"
        # Completed attempts, including explicit model failures, are terminal until requested.
        if csv_path.exists() and not args.rerun_failed:
            _, previous = read_csv(csv_path)
            present = {(r.get("question_id"), r.get("repeat_idx") or "1") for r in previous}
            required = {(r["question_id"], str(i)) for r in shard["rows"] for i in range(1, repeats + 1)}
            if required <= present:
                return {"id": name, "resumed": True, "seconds": 0, "exit_code": 0}
        write_csv(args.out / f"{name}-questions.csv", header, shard["rows"])
        start = time.monotonic()
        code = 1
        for attempt in range(args.attempts):
            command = ["docker", "run", "--rm", "--init", "--name", f"{prefix}-{run_id}-{name}",
                       "--gpus", f"device={args.gpu}", "--network", "host", "--entrypoint", "bash",
                       "-v", f"{ROOT}:/repo:ro", "-v", f"{args.out}:/run",
                       "-v", f"{args.scenes.resolve()}:/repo/benchmark/scenes:ro",
                       "-v", f"{args.presets.resolve()}:/presets:ro"]
            settings = {**env, "SCOPE_PRESET_DIR": "/presets", "SCOPE_RESUME": "1",
                        "QUESTIONS_CSV": f"/run/{name}-questions.csv", "OUT_CSV": f"/run/{name}.csv",
                        "SCOPE_OUTPUT_DIR": f"/run/captures/{name}", "SCOPE_SAMPLE_EVERY": "25",
                        "SCOPE_SAMPLE_DIR": f"/run/samples/{name}", "SCOPE_CAPTURE_RING": "8",
                        "SCOPE_KEEP_CAPTURES": "0", "PYTHONUNBUFFERED": "1"}
            for key, value in settings.items():
                command += ["-e", f"{key}={value}"]
            for key in SECRET_ENV:
                if key in os.environ:
                    command += ["-e", key]
            command += [args.image, "-c", INNER, "scope-worker", f"/repo/benchmark/{shard['scene']}", name]
            with (args.out / f"{name}.log").open("a") as log:
                log.write(f"\nAttempt {attempt + 1}\n")
                log.flush()
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
            _, got = read_csv(csv_path) if csv_path.exists() else ([], [])
            # Count all terminal outcomes, not physical CSV lines (transcripts contain newlines).
            present = {(r.get("question_id"), r.get("repeat_idx") or "1") for r in got}
            required = {(r["question_id"], str(i)) for r in shard["rows"] for i in range(1, repeats + 1)}
            if code == 0 and required <= present:
                break
        return {"id": name, "seconds": round(time.monotonic() - start, 3), "exit_code": code}

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes = list(pool.map(worker, shards))
    coverage = merge(shards, args.out, rows, repeats)
    report = {**coverage, "wall_seconds": round(time.monotonic() - start, 3),
              "shards": outcomes, "judging_included": False}
    failed = bool(coverage["missing"] or any(s["exit_code"] for s in outcomes))
    # Persist coverage before any API failure in the optional grading stage.
    (args.out / f"run-{run_id}.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.judge and not failed:
        report["judge_wall_seconds"] = judge_and_report(args.out)
        report["judging_included"] = True
        (args.out / f"run-{run_id}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(failed)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"run_fast: {exc}", file=sys.stderr)
        raise SystemExit(1)

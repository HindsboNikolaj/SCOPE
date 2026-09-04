"""Capture a panorama for every benchmark preset in a scene, into the panorama cache.

Run this once per scene, unhurried, and the benchmark never pays for a sweep again. That is
worth doing on any machine without a GL driver and close to mandatory for `city-street`, which
costs roughly 700 seconds a frame in Material Preview whatever the resolution.

    SCOPE_PANO_CACHE=benchmark/panoramas SCOPE_PANO_CACHE_MODE=write \
      blender benchmark/scenes/<scene>/<scene>.blend \
        --python scripts/precapture_panoramas.py

Only the presets that the benchmark CSV actually starts from are captured, because a preset no
row uses is not worth an hour. Pass names on the command line to override:

    ... --python scripts/precapture_panoramas.py -- road1 store-front

Set SCOPE_PANO_CACHE_MODE=write so that an existing entry is replaced rather than served.
"""
import csv
import os
import sys
import time
from pathlib import Path

import bpy

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from scope.blender import panorama_cache                      # noqa: E402
from scope.blender.preset_helpers import apply_preset         # noqa: E402
from scope.eval.runner import prepare_view_for_capture        # noqa: E402
from scope.tools import blender_tools as BT                   # noqa: E402

SETTLE = float(os.environ.get("SCOPE_SETTLE", "15"))
CSV_PATH = os.environ.get("SCOPE_CSV", str(_HERE.parent / "benchmark" / "scope_536.csv"))


def presets_for_this_scene():
    """The presets the CSV starts from for the scene currently open."""
    stem = os.path.splitext(os.path.basename(bpy.data.filepath))[0].lower().replace("_", "-")
    want = []
    try:
        with open(CSV_PATH, newline="") as fh:
            for row in csv.DictReader(fh):
                parts = (row.get("file_location") or "").split("/")
                if len(parts) < 2:
                    continue
                if parts[1].lower().replace("_", "-") != stem:
                    continue
                p = (row.get("preset_start") or "").strip()
                if p and p not in want:
                    want.append(p)
    except OSError as e:
        print(f"[pre] could not read {CSV_PATH}: {e}")
    return want


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    names = argv or presets_for_this_scene()
    if not names:
        print("[pre] no presets to capture for this scene")
        return

    if panorama_cache.MODE == "use":
        print("[pre] note: SCOPE_PANO_CACHE_MODE is 'use', so existing entries will be served "
              "rather than recaptured. Set it to 'write' to replace them.")

    print(f"[pre] scene {os.path.basename(bpy.data.filepath)}, {len(names)} presets: {names}")
    print(f"[pre] cache {panorama_cache.cache_dir()}")
    print(f"[pre] settling {SETTLE}s after open")
    time.sleep(SETTLE)

    done, failed = 0, []
    for name in names:
        if not apply_preset(name):
            print(f"[pre] {name}: preset not found, skipping. Has 04_install_presets.py run?")
            failed.append(name)
            continue
        prepare_view_for_capture()
        time.sleep(1.0)
        t0 = time.time()
        try:
            it = BT._iter_full_panorama()
            frames = 0
            while True:
                try:
                    step = next(it)
                except StopIteration as stop:
                    path = stop.value
                    break
                if (step.get("panorama") or {}).get("frame"):
                    frames += 1
        except Exception as e:
            print(f"[pre] {name}: sweep failed, {type(e).__name__}: {e}")
            failed.append(name)
            continue
        secs = time.time() - t0
        cached = panorama_cache.lookup()
        print(f"[pre] {name:18} {frames:2} frames  {secs:8.1f}s  -> "
              f"{cached or 'NOT CACHED'}")
        done += 1

    print(f"[pre] {done} captured, {len(failed)} failed{': ' + ', '.join(failed) if failed else ''}")


main()

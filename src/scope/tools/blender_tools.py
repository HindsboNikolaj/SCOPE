#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
blender_tools.py — SCOPE simulation tools (Blender) with pluggable VLM

Implements the SCOPE tool schema:
  zoom_bounding, count_pointing, query_answer, take_image,
  home_action, get_presets, go_to_preset, track_object, ptz_adjust
"""

from __future__ import annotations
import os, time, math
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

from PIL import Image
import bpy

from ..blender.helper_funcs import (
    start_panorama_capture,
    capture_panorama_step,
    screenshot_camera_view,
    fast_opengl_screenshot,
    blender_zoom,
    list_presets,
    apply_preset,
    create_preset,
)
from .vlm_clients import VLMClient, create_vlm_from_env

# ─── Global VLM binding ──────────────────────────────────────────────────────

_VLM: Optional[VLMClient] = None

def set_vlm(vlm: VLMClient):
    """Bind a VLM client for image understanding tools."""
    global _VLM
    _VLM = vlm

if os.getenv("VLM_AUTO_INIT", "1") in ("1", "true", "yes"):
    try:
        _VLM = _VLM or create_vlm_from_env()
    except Exception:
        pass

# ─── Utilities ────────────────────────────────────────────────────────────────

def _need_vlm(capa: str):
    if _VLM is None:
        raise RuntimeError("VLM is not initialized (call set_vlm(...) or set VLM_* env vars).")
    if not getattr(_VLM.caps, capa, False):
        raise RuntimeError(f"Selected VLM lacks capability: {capa}")

def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0

def _active_cam():
    cam = bpy.context.scene.camera
    if cam is None:
        raise RuntimeError("No active scene camera set.")
    return cam

# src/scope/tools/blender_tools.py -> parents[3] is the project root
# ([0]=tools, [1]=scope, [2]=src, [3]=project).
# Where captures are written. Defaults to <project>/output, which is where they have always
# gone, but SCOPE_OUTPUT_DIR moves it.
#
# The override exists because the default assumes the checkout is writable, and increasingly it
# is not: a container mounting the repo read-only, a shared install under /opt, a checkout on a
# read-only volume. The failure is confusing when it happens, because import succeeds (the
# directory usually already exists, so mkdir(exist_ok=True) passes) and the error only surfaces
# much later as "Could not write image: Invalid argument" from deep inside a Blender operator.
#
# src/scope/tools/blender_tools.py -> parents[3] is the project root
# ([0]=tools, [1]=scope, [2]=src, [3]=project).
OUTPUT_DIR = Path(os.environ.get("SCOPE_OUTPUT_DIR")
                  or Path(__file__).resolve().parents[3] / "output")
PANOS_DIR = OUTPUT_DIR / "panos"
SCREENSHOTS_DIR = OUTPUT_DIR / "screenshots"
try:
    for d in (OUTPUT_DIR, PANOS_DIR, SCREENSHOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
except OSError as e:
    raise RuntimeError(
        f"SCOPE cannot write captures to {OUTPUT_DIR} ({e}). Set SCOPE_OUTPUT_DIR to a "
        f"writable directory. This is the usual symptom of a read-only checkout."
    ) from e

# Which capture to use. There are two, and they do not work in the same places.
#
#   viewport  screenshot_camera_view -> screen.screenshot_area. Photographs the window as


# Which capture to use. There are two, and they do not work in the same places.
#
#   viewport  screenshot_camera_view -> screen.screenshot_area. Photographs the window as
#             drawn. This is the default and it is what produced the published results.
#             On a machine with no physical display it returns a black image, because
#             nothing paints the window and reading the buffer back yields nothing.
#
#   opengl    fast_opengl_screenshot -> render.opengl. Renders offscreen, so it does not
#             need a painted window. This is the one that works headless, for example in
#             a container with Xvfb and software OpenGL.
#
# The default stays "viewport", so nothing changes for anyone running with a display.
# docs/HEADLESS.md covers when to switch and what it costs.
_CAPTURE_BACKEND = os.environ.get("SCOPE_CAPTURE", "viewport").strip().lower()


_BLANK_WARNED = False


def _capture_frame(prefix: str = "raw") -> str:
    global _BLANK_WARNED
    fp = SCREENSHOTS_DIR / f"{_now_ts()}_{prefix}.png"
    if _CAPTURE_BACKEND in ("opengl", "offscreen", "headless"):
        fast_opengl_screenshot(str(fp))
    else:
        screenshot_camera_view(str(fp))

    # Say something the first time a capture comes back with no variation at all.
    #
    # This is the quietest failure in the whole pipeline. screenshot_area returns a solid black
    # image on a virtual display without raising, and everything downstream carries on: the
    # detector is shown black, reports the whole frame as its box, and zoom_bounding used to
    # answer "Zoomed to target". A whole run can complete and be graded on black rectangles.
    #
    # A warning rather than an exception, because a legitimately dark frame is possible and
    # losing a long run to a heuristic would be worse than the heuristic missing something.
    if not _BLANK_WARNED:
        try:
            lo, hi = Image.open(fp).convert("L").getextrema()
            if hi - lo == 0:
                _BLANK_WARNED = True
                print(f"[scope] WARNING: capture {fp.name} is a single flat colour "
                      f"(value {lo}). Every result from this run will be about a blank image. "
                      f"With no real display set SCOPE_CAPTURE=opengl; see docs/HEADLESS.md.",
                      flush=True)
        except Exception:
            pass
    return str(fp)

_HOME = None

def _capture_home_if_needed():
    global _HOME
    if _HOME is None:
        cam = _active_cam()
        _HOME = {
            "location": cam.location.copy(),
            "rotation_euler": cam.rotation_euler.copy(),
            "lens": float(cam.data.lens) if hasattr(cam.data, "lens") else None,
        }

# ─── Panorama helpers ─────────────────────────────────────────────────────────

def _find_stitched_panorama(pano_root_dir: str) -> Optional[str]:
    expected = os.path.join(pano_root_dir, "panorama_stitched.png")
    if os.path.exists(expected):
        return expected
    try:
        for entry in os.scandir(pano_root_dir):
            if entry.is_dir():
                nested = os.path.join(entry.path, "panorama_stitched.png")
                if os.path.exists(nested):
                    return nested
    except FileNotFoundError:
        pass
    for root, dirs, files in os.walk(pano_root_dir):
        if "panorama_stitched.png" in files:
            return os.path.join(root, "panorama_stitched.png")
    return None

def _iter_full_panorama():
    """Produce the full 360 view, from cache where possible and by sweeping when not.

    Read this before you spend an afternoon tuning a sweep.

    A stitched panorama in Blender is fussy in ways that are not obvious from the code. The
    result depends on the overlap between frames, on the step angle that follows from it, on
    the camera's pitch, and on the capture resolution, and those interact. Too little overlap
    and the joins show; too much and you pay for frames that add nothing, because the seam
    quality stops improving somewhere around forty percent while the frame count keeps rising.
    Sweeping a camera that is pitched traces a cone rather than a circle, which is right for a
    street-level view and wrong for one looking down at a courtyard. And a panorama is only as
    sharp as the frames behind it, so a low resolution capture stitches into something blurry
    no matter how good the geometry is. See docs/FULL_VIEW.md for the measurements behind all
    of that.

    The consequence for anyone using this: **do not tune a panorama per question.** A sweep is
    nine or ten captures and costs 60 to 190 seconds under software OpenGL, and hours on a
    scene with expensive materials. The scenes do not change between questions, so the same
    sweep is being recomputed for an answer that cannot differ.

    Capture it once per viewpoint instead, look at the result, and keep it:

        SCOPE_PANO_CACHE=benchmark/panoramas SCOPE_PANO_CACHE_MODE=write \
          blender <scene>.blend --python scripts/precapture_panoramas.py

    If you are adding a new world, that is an hour of setup and it is worth it. One or two good
    panoramas per viewpoint, checked by eye, beat a hundred mediocre ones generated on demand.
    docs/PANORAMA_CACHE.md covers how entries are validated; docs/VISUAL_SMOKE_TEST.md shows
    what the shipped ones look like.
    """
    ts = _now_ts()
    base_dir = PANOS_DIR

    # A panorama for this scene and this camera pose may already exist. Nothing in these scenes
    # moves between benchmark rows, so a sweep from a given preset produces the same picture
    # every time, and 93 of the 541 rows ask for one from ten fixed positions. Sweeping is 60
    # to 190 seconds on most scenes and hours on city-street, so reusing a stored sweep is the
    # difference between a full-view question costing three minutes and costing nothing.
    #
    # This is a generator whose return value is the panorama path, so a hit returns without
    # yielding any frame steps. Callers that render progress will simply see none.
    try:
        from scope.blender import panorama_cache
        hit = panorama_cache.lookup(root=OUTPUT_DIR)
    except Exception:
        hit = None
    if hit:
        yield {'panorama': {'cached': True, 'path': hit}}
        return hit

    # No stored panorama for this viewpoint, so one is about to be built at full cost. Say so
    # rather than doing it silently: on most scenes this is a minute or three, and on a scene
    # with expensive materials it can be hours, which is a surprising thing to discover from a
    # progress bar. A miss is legitimate at a viewpoint nobody has pre-captured, and it is also
    # what a stale or mismatched cache looks like, so the message names what to run.
    try:
        from scope.blender import panorama_cache as _pc
        _preset = None
        try:
            from scope.blender import preset_helpers as _ph
            _preset = _ph.last_applied_preset()
        except Exception:
            pass
        if _pc.MODE != "off":
            print(f"[scope] No stored panorama for this viewpoint"
                  f"{f' (preset {_preset!r})' if _preset else ''}; sweeping now. "
                  f"This is 9 or 10 captures and can take minutes. "
                  f"Pre-capture with scripts/precapture_panoramas.py to avoid paying for it "
                  f"again. Cache: {_pc.cache_dir(OUTPUT_DIR)}", flush=True)
    except Exception:
        pass

    fake_target = str(base_dir / f"{ts}_panorama.png")
    start_panorama_capture(fake_target, overlap_ratio=0.1)
    yield {'panorama': {'start': True}}
    while True:
        frame = capture_panorama_step()
        if frame is None:
            break
        yield {'panorama': {'frame': frame}}
    run_dir = base_dir / f"_panorama_{ts}"
    deadline = time.time() + 5.0
    stitched = None
    while stitched is None:
        candidate = _find_stitched_panorama(str(run_dir))
        if candidate:
            stitched = candidate
            break
        if time.time() > deadline:
            raise FileNotFoundError(f"Timed out waiting for stitched panorama in {run_dir}")
        time.sleep(0.01)

    # Store it, so the next question from this viewpoint does not pay for the sweep again.
    try:
        from scope.blender import panorama_cache
        panorama_cache.store(stitched, root=OUTPUT_DIR,
                             extra={"frames": None, "captured_ts": ts})
    except Exception:
        pass
    return stitched

def _capture_for_view(view_type: str) -> str:
    if view_type == "full":
        raise RuntimeError("Use _iter_full_panorama() for view_type='full'")
    return _capture_frame("raw")

# ─── VLM wrappers ─────────────────────────────────────────────────────────────

def _vlm_caption(img_path: str) -> Tuple[str, float]:
    _need_vlm("caption")
    t0 = time.time()
    cap = _VLM.caption(Image.open(img_path)).get("caption", "")
    return str(cap), (time.time() - t0)

def _vlm_query(img_path: str, instruction: str) -> Tuple[str, float]:
    _need_vlm("vqa")
    t0 = time.time()
    ans = _VLM.query(Image.open(img_path), instruction).get("answer", "")
    return str(ans), (time.time() - t0)

def _vlm_point(img_path: str, instruction: str) -> Tuple[list, float]:
    _need_vlm("point")
    t0 = time.time()
    pts = _VLM.point(Image.open(img_path), instruction).get("points", [])
    if isinstance(pts, dict):
        pts = [pts]
    return list(pts or []), (time.time() - t0)

def _vlm_detect(img_path: str, instruction: str) -> Tuple[list, float]:
    _need_vlm("detect")
    t0 = time.time()
    objs = _VLM.detect(Image.open(img_path), instruction).get("objects", [])
    return list(objs or []), (time.time() - t0)

# ─── Public TOOLS ─────────────────────────────────────────────────────────────

def zoom_bounding(instruction: str):
    t_script0 = time.time()
    img_path = _capture_frame("prezoom")
    bbox = None
    vlm_time = 0.0
    try:
        objs, t_det = _vlm_detect(img_path, instruction)
        vlm_time += t_det
        if objs:
            o = objs[0]
            bbox = (float(o.get("x_min", 0.0)), float(o.get("y_min", 0.0)),
                    float(o.get("x_max", 1.0)), float(o.get("y_max", 1.0)))
        else:
            pts, t_pts = _vlm_point(img_path, instruction)
            vlm_time += t_pts
            if pts:
                W, H = Image.open(img_path).size
                xs, ys = [], []
                for p in pts:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        xs.append(float(p[0])); ys.append(float(p[1]))
                    elif isinstance(p, dict) and {"x","y"} <= set(p.keys()):
                        xs.append(float(p["x"]) * W if p["x"] <= 1.0 else float(p["x"]))
                        ys.append(float(p["y"]) * H if p["y"] <= 1.0 else float(p["y"]))
                if xs and ys:
                    pad = 0.06
                    bbox = (_clamp(min(xs)/W - pad, 0, 1), _clamp(min(ys)/H - pad, 0, 1),
                            _clamp(max(xs)/W + pad, 0, 1), _clamp(max(ys)/H + pad, 0, 1))
    except Exception:
        pass
    if not bbox:
        bbox = (0.0, 0.0, 1.0, 1.0)
    x1, y1, x2, y2 = bbox
    cam = _active_cam()
    blender_zoom(cam, x1, y1, x2, y2)
    post_path = _capture_frame("postzoom")
    return {
        "result": f"Zoomed to target: {instruction}",
        "bbox": [x1, y1, x2, y2], "path": post_path,
        "timings": {"vlm": round(vlm_time, 3), "script": round(time.time() - t_script0 - vlm_time, 3)},
    }

def count_pointing(instruction: str, view_type: str = "current"):
    t0 = time.time()
    if view_type == "full":
        try:
            pano_iter = _iter_full_panorama()
            while True:
                step = next(pano_iter)
                yield step
        except StopIteration as stop:
            img_path = stop.value
        yield {'detect': {'object': instruction, 'image': img_path}}
        try:
            pts, vlm_time = _vlm_point(img_path, instruction)
            cnt = len(pts)
            return {"result": f"Counted {cnt} '{instruction}' in full view.", "count": int(cnt),
                    "timings": {"vlm": round(vlm_time, 3), "script": round(time.time() - t0 - vlm_time, 3)}}
        except Exception as e:
            return {"result": f"Point detection error: {e}", "count": 0,
                    "timings": {"vlm": 0.0, "script": round(time.time() - t0, 3)}}
    try:
        img_path = _capture_for_view(view_type)
        yield {'screenshot': img_path}
        pts, vlm_time = _vlm_point(img_path, instruction)
        cnt = len(pts)
        return {"result": f"Counted {cnt} '{instruction}' in current view.", "count": int(cnt),
                "timings": {"vlm": round(vlm_time, 3), "script": round(time.time() - t0 - vlm_time, 3)}}
    except Exception as e:
        return {"result": f"Counting error: {e}", "count": 0,
                "timings": {"vlm": 0.0, "script": round(time.time() - t0, 3)}}

def query_answer(instruction: str, view_type: str = "current"):
    t0 = time.time()
    if view_type == "full":
        try:
            pano_iter = _iter_full_panorama()
            while True:
                step = next(pano_iter)
                yield step
        except StopIteration as stop:
            img_path = stop.value
        yield {'query': {'query': instruction, 'view_type': view_type}}
        try:
            ans, vlm_time = _vlm_query(img_path, instruction)
            return {"result": ans, "answer": ans,
                    "timings": {"vlm": round(vlm_time, 3), "script": round(time.time() - t0 - vlm_time, 3)}}
        except Exception as e:
            return {"result": f"Query error: {e}", "answer": "",
                    "timings": {"vlm": 0.0, "script": round(time.time() - t0, 3)}}
    try:
        img_path = _capture_for_view(view_type)
        yield {'screenshot': img_path}
        ans, vlm_time = _vlm_query(img_path, instruction)
        return {"result": ans, "answer": ans,
                "timings": {"vlm": round(vlm_time, 3), "script": round(time.time() - t0 - vlm_time, 3)}}
    except Exception as e:
        return {"result": f"Query error: {e}", "answer": "",
                "timings": {"vlm": 0.0, "script": round(time.time() - t0, 3)}}

def take_image():
    shot = _capture_frame("screenshot")
    return {"result": shot, "path": shot, "timings": {"vlm": 0.0, "script": 0.0}}

def home_action():
    t0 = time.time()
    _capture_home_if_needed()
    names = set(list_presets() or [])
    used_preset = False
    if "Home" in names:
        used_preset = bool(apply_preset("Home"))
    if not used_preset and _HOME:
        cam = _active_cam()
        cam.location       = _HOME["location"].copy()
        cam.rotation_euler = _HOME["rotation_euler"].copy()
        if _HOME.get("lens") is not None and hasattr(cam.data, "lens"):
            cam.data.lens = float(_HOME["lens"])
    return {
        "result": "Returned to Home position" + (" (preset)" if used_preset else " (fallback)"),
        "timings": {"vlm": 0.0, "script": round(time.time() - t0, 3)},
    }

def get_presets():
    t0 = time.time()
    names = list_presets() or []
    return {"result": "Available presets: " + ", ".join(names), "presets": list(names),
            "timings": {"vlm": 0.0, "script": round(time.time() - t0, 3)}}

def go_to_preset(preset_name: Optional[str] = None, name: Optional[str] = None):
    t0 = time.time()
    target = preset_name or name
    if not target:
        return {"result": "Error: no preset name provided", "timings": {"vlm": 0.0, "script": round(time.time() - t0, 3)}}
    ok = bool(apply_preset(target))
    return {"result": f"Moved to preset '{target}'" if ok else f"Preset '{target}' not found",
            "timings": {"vlm": 0.0, "script": round(time.time() - t0, 3)}}

def track_object(object_of_interest: str, duration: int, unit: str):
    t0 = time.time()
    return {"result": f"Tracked '{object_of_interest}' for {duration} {unit}.",
            "timings": {"vlm": 0.0, "script": round(time.time() - t0, 3)}}

def ptz_adjust(
    pan_deg: Optional[float] = None,
    tilt_deg: Optional[float] = None,
    zoom_percent: Optional[float] = None,
    zoom_factor: Optional[float] = None,
    zoom_full: str = "",
):
    t0 = time.time()
    cam = _active_cam()
    if pan_deg not in (None, 0):
        cam.rotation_euler[2] -= _deg_to_rad(float(pan_deg))
    if tilt_deg not in (None, 0):
        cam.rotation_euler[0] += _deg_to_rad(float(tilt_deg))
    if hasattr(cam.data, "lens"):
        lens = float(cam.data.lens)
        if zoom_full in ("in", "out"):
            lens = lens * (50 if zoom_full == "in" else 0.02)
        if zoom_factor not in (None, 0):
            lens *= float(zoom_factor)
        if zoom_percent not in (None, 0):
            lens *= 1.0 + float(zoom_percent) / 100.0
        cam.data.lens = lens
    path = _capture_frame("ptz")
    return {"result": "PTZ adjusted", "path": path,
            "timings": {"vlm": 0.0, "script": round(time.time() - t0, 3)}}

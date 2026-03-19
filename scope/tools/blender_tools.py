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

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"
PANOS_DIR = OUTPUT_DIR / "panos"
SCREENSHOTS_DIR = OUTPUT_DIR / "screenshots"
for d in (OUTPUT_DIR, PANOS_DIR, SCREENSHOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

def _capture_frame(prefix: str = "raw") -> str:
    fp = SCREENSHOTS_DIR / f"{_now_ts()}_{prefix}.png"
    screenshot_camera_view(str(fp))
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
    ts = _now_ts()
    base_dir = PANOS_DIR
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

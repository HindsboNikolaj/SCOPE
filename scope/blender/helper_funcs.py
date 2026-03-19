#!/usr/bin/env python3
"""
helper_funcs.py — Blender camera helpers for SCOPE simulation.

Provides fast camera screenshots, FOV zoom, and panorama capture.
"""

import bpy
from PIL import Image
import math
import time
from bpy_extras import view3d_utils


def corrected_persp_area_zoom_fov(cam_obj, u0, v0, u1, v1, margin=1.02, max_zoom=6.0):
    """
    Perspective-aware area zoom with cosine correction.
    Returns (applied_zoom, (delta_pan, delta_tilt)).
    """
    cam = cam_obj.data
    orig_hFOV = cam.angle_x
    orig_vFOV = cam.angle_y
    span_x = u1 - u0
    span_y = v1 - v0
    uc = 0.5 * (u0 + u1)
    vc = 0.5 * (v0 + v1)
    dx = uc - 0.5
    dy = 0.5 - vc

    raw_factor = max(
        span_x / math.cos(abs(dx * orig_hFOV)),
        span_y / math.cos(abs(dy * orig_vFOV))
    ) * margin
    factor = max(raw_factor, 1.0 / max_zoom)
    cam.lens /= factor

    x_ndc = dx * 2.0
    y_ndc = dy * 2.0
    delta_pan = math.atan(math.tan(orig_hFOV * 0.5) * x_ndc)
    delta_tilt = math.atan(math.tan(orig_vFOV * 0.5) * y_ndc)

    cam_obj.rotation_euler[0] += delta_tilt
    cam_obj.rotation_euler[2] += -delta_pan

    return 1.0 / factor, (delta_pan, delta_tilt)


# Alias used by blender_tools.py
blender_zoom = corrected_persp_area_zoom_fov


def screenshot_camera_view(out_path: str, wait: float = 0.05):
    """
    Capture a camera-view screenshot, cropped to the camera frustum.
    """
    C     = bpy.context
    scene = C.scene

    area   = next(a for a in C.window.screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    space  = area.spaces.active
    rv3d   = space.region_3d

    # Save originals
    orig_persp = rv3d.view_perspective
    orig_cam   = space.camera
    orig_overlay = {attr: getattr(space.overlay, attr) for attr in (
        'show_overlays','show_floor','show_axis_x','show_axis_y','show_axis_z',
        'show_wireframes','show_outline_selected','show_extras',
        'show_relationship_lines','show_cursor')}
    orig_gizmo          = space.show_gizmo
    orig_region_ui      = space.show_region_ui
    orig_region_toolbar = space.show_region_toolbar
    orig_region_header  = space.show_region_header

    try:
        rv3d.view_perspective = 'CAMERA'
        space.camera          = scene.camera

        ov = space.overlay
        ov.show_overlays            = False
        ov.show_floor               = False
        ov.show_axis_x              = False
        ov.show_axis_y              = False
        ov.show_axis_z              = False
        ov.show_wireframes          = False
        ov.show_outline_selected    = False
        ov.show_extras              = False
        ov.show_relationship_lines  = False
        ov.show_cursor              = False

        space.show_gizmo            = False
        space.show_region_ui        = False
        space.show_region_toolbar   = False
        space.show_region_header    = False

        for a in C.window.screen.areas:
            if a.type == 'VIEW_3D':
                a.tag_redraw()
        region.tag_redraw()
        time.sleep(wait)

        raw_path = out_path.replace(".png", "_raw.png")
        override = {'window':C.window, 'screen':C.screen, 'area':area, 'region':region}
        with C.temp_override(**override):
            bpy.ops.screen.screenshot_area(filepath=raw_path, hide_props_region=False)

        # Crop to camera frustum
        cam = scene.camera
        local_corners = cam.data.view_frame(scene=scene)
        world_corners = [cam.matrix_world @ v for v in local_corners]
        pts2d = [
            view3d_utils.location_3d_to_region_2d(region, rv3d, wc)
            for wc in world_corners
        ]
        pts2d = [p for p in pts2d if p]
        if pts2d:
            xs, ys = [p.x for p in pts2d], [p.y for p in pts2d]
            x0, x1 = int(min(xs)), int(max(xs))
            y0, y1 = int(min(ys)), int(max(ys))
            img = Image.open(raw_path)
            W, H = img.size
            crop = img.crop((x0, H-y1, x1, H-y0))
            crop.save(out_path)

    finally:
        rv3d.view_perspective = orig_persp
        space.camera          = orig_cam
        for k, v in orig_overlay.items():
            setattr(space.overlay, k, v)
        space.show_gizmo            = orig_gizmo
        space.show_region_ui        = orig_region_ui
        space.show_region_toolbar   = orig_region_toolbar
        space.show_region_header    = orig_region_header
        for a in C.window.screen.areas:
            if a.type == 'VIEW_3D':
                a.tag_redraw()


def fast_opengl_screenshot(out_path: str, scale_crop: bool = True):
    """
    Capture a screenshot via OpenGL render, then crop to camera aspect.
    """
    C = bpy.context
    win = C.window
    scene = C.scene

    area = next(a for a in C.screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    space = area.spaces.active
    space.region_3d.view_perspective = 'CAMERA'
    space.camera = scene.camera
    space.shading.type = 'SOLID'
    space.shading.use_scene_lights = True
    space.shading.use_scene_world = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = out_path

    override = {'window': win, 'screen': win.screen, 'area': area, 'region': region, 'space': space}
    with C.temp_override(**override):
        bpy.ops.render.opengl(write_still=True, view_context=True)

    if scale_crop:
        try:
            scale = scene.render.resolution_percentage / 100.0
            cam_w = int(scene.render.resolution_x * scale)
            cam_h = int(scene.render.resolution_y * scale)
            cam_asp = cam_w / cam_h
            img = Image.open(out_path)
            rw, rh = img.size
            reg_asp = rw / rh
            if abs(reg_asp - cam_asp) > 1e-3:
                if reg_asp > cam_asp:
                    new_w = int(rh * cam_asp)
                    left = (rw - new_w) // 2
                    img = img.crop((left, 0, left + new_w, rh))
                else:
                    new_h = int(rw / cam_asp)
                    top = (rh - new_h) // 2
                    img = img.crop((0, top, rw, top + new_h))
                img.save(out_path)
        except Exception:
            pass


# ─── Panorama (stub - implement based on your scene setup) ────────────────────

_PANO_STATE = {}

def start_panorama_capture(output_path: str, overlap_ratio: float = 0.1):
    """Initialize a 360-degree panorama sweep."""
    _PANO_STATE.clear()
    _PANO_STATE["output_path"] = output_path
    _PANO_STATE["overlap"] = overlap_ratio
    _PANO_STATE["started"] = True

def capture_panorama_step():
    """Capture the next frame in the panorama sweep. Returns frame path or None when done."""
    if not _PANO_STATE.get("started"):
        return None
    _PANO_STATE["started"] = False
    return None


# Re-export for convenience
list_presets = None
apply_preset = None
create_preset = None

try:
    from ..blender.preset_helpers import list_presets, apply_preset, create_preset
except ImportError:
    # When running inside Blender, preset_helpers may be imported differently
    pass

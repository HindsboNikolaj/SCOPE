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
from mathutils import Vector


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

    # Aim the camera at the box by pointing it, not by adding to its Euler components.
    #
    # The previous version did `rotation_euler[2] -= delta_pan`, which rotates about the *world*
    # vertical. That is the camera's own horizontal axis only while the camera is level, so the
    # error grew with pitch: measured against a known world point, the boxed target landed this
    # far from the frame centre, as a fraction of frame width.
    #
    #     camera pitch   90 deg (level)   0.005
    #                    70 deg           0.073
    #                    50 deg           0.213
    #                    30 deg           0.419
    #
    # A purely vertical offset had zero error at every pitch, because tilt was already applied
    # about the camera's own X axis. Only the pan was wrong. On a viewpoint pitched 48 degrees
    # down, a 6x zoom onto a correctly detected object put it outside the frame.
    #
    # Pointing the camera instead makes it exact at any pitch: take the direction the box centre
    # sits in, in world space, and build the rotation whose -Z looks along it with Y up. That is
    # a camera look-at, so it also keeps the horizon level rather than accumulating roll.
    d_cam = Vector((x_ndc * math.tan(orig_hFOV * 0.5),
                    y_ndc * math.tan(orig_vFOV * 0.5),
                    -1.0))
    R = cam_obj.rotation_euler.to_matrix()
    look = (R @ d_cam).normalized()
    aimed = look.to_track_quat('-Z', 'Y').to_euler(cam_obj.rotation_mode)

    # Wrap each component into (-pi, pi]. Rebuilding the Euler from a quaternion can land on an
    # equivalent branch many turns away from where the camera started: a preset sitting at a
    # yaw of 144 radians came back reading -8236 degrees. The orientation is identical, since
    # each component is 2*pi periodic, but a log or a delta computed from it is not readable,
    # and anything comparing angles numerically would be misled.
    for _i in range(3):
        aimed[_i] = (aimed[_i] + math.pi) % (2.0 * math.pi) - math.pi
    cam_obj.rotation_euler = aimed

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

    This is the capture that works without a physical display. `screenshot_camera_view`
    uses `screen.screenshot_area`, which photographs the window as drawn, and reading
    that back returns a black image under a virtual X server with software OpenGL.
    `render.opengl` renders offscreen, so it does not depend on anything having been
    painted to a screen. See docs/HEADLESS.md.

    Shading follows what the .blend was saved with, unless an environment variable says
    otherwise. That default matters. This function previously forced
    `shading.type = 'SOLID'` with no `color_type`, which discards every texture and
    returns a flat grey massing model, and forced `use_scene_world = True`, which pulls
    the scene's world into the picture. For a scene whose world texture is missing, that
    second flag tints the entire frame magenta. whitechapel is saved as MATERIAL preview
    with STUDIO lighting, and STUDIO deliberately ignores the scene world, which is why
    the desktop capture of that scene looks correct despite the missing sky.

    Overrides, for when the saved shading is not what a run wants:
      SCOPE_SHADING       SOLID | MATERIAL | RENDERED
      SCOPE_SHADING_COLOR MATERIAL | TEXTURE | OBJECT | SINGLE   (SOLID only)
      SCOPE_SHADING_WORLD  1 or 0 to force the scene world on or off
      SCOPE_SHADING_LIGHTS 1 or 0 to force the scene's own lamps on or off

    Both lighting flags are left as the .blend saved them when unset. Forcing either one
    produced a wrong picture on whitechapel: the world flag turned the frame magenta,
    because that scene's environment map is missing, and the lights flag turned it purple,
    because it swapped the studio light for the scene's cold lamps.

    On cost: under software OpenGL, MATERIAL preview takes about a minute a frame while
    SOLID with TEXTURE takes a few seconds and keeps the textures. `SCOPE_SHADING=SOLID`
    with `SCOPE_SHADING_COLOR=TEXTURE` is the practical choice for a long headless run.
    """
    import os as _os

    C = bpy.context
    win = C.window
    scene = C.scene

    area = next(a for a in C.screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    space = area.spaces.active
    space.region_3d.view_perspective = 'CAMERA'
    space.camera = scene.camera

    _shading = _os.environ.get("SCOPE_SHADING", "").strip().upper()
    if _shading in ("SOLID", "MATERIAL", "RENDERED", "WIREFRAME"):
        space.shading.type = _shading
    _color = _os.environ.get("SCOPE_SHADING_COLOR", "").strip().upper()
    if _color and space.shading.type == 'SOLID':
        try:
            space.shading.color_type = _color
        except (TypeError, AttributeError):
            pass
    # Lighting is left exactly as the .blend saved it unless asked otherwise. Forcing
    # either of these flags on produced a badly wrong picture on whitechapel:
    # use_scene_world pulled in a missing environment map and turned the frame magenta,
    # and use_scene_lights replaced the studio light with the scene's own cold lamps and
    # turned it purple. Neither is a lighting choice this function should be making on
    # the author's behalf.
    def _flag(name):
        v = _os.environ.get(name, "").strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off"):
            return False
        return None

    # A note about the four scenes this benchmark ships with, because it decides the
    # default above and it is not recoverable from the code.
    #
    # book-nook, city-street, postwar-city and whitechapel were all labelled from captures
    # taken in Material Preview with STUDIO lighting, which means the scene world was NOT
    # drawn. The published answers describe those images. Turning the world on gives a
    # better-looking picture that no longer matches what a labeller saw: an outdoor scene
    # gains a sky where the labeller had flat grey, and a 360 degree sweep changes most of
    # its area. So for these four, leave the world off, and read the grey as a property of
    # the dataset rather than a fault to correct.
    #
    # A scene added later is not bound by that. If its answers are labelled from captures
    # taken with the world on, then SCOPE_SHADING_WORLD=1 is the right setting for it, and
    # the thing to be careful about is only that one run should not mix the two. Record
    # which was used alongside the results.
    _lights = _flag("SCOPE_SHADING_LIGHTS")
    if _lights is not None:
        space.shading.use_scene_lights = _lights
    _world = _flag("SCOPE_SHADING_WORLD")
    if _world is not None:
        space.shading.use_scene_world = _world

    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = out_path

    override = {'window': win, 'screen': win.screen, 'area': area, 'region': region, 'space': space}

    # Fit the camera frame to the region before rendering.
    #
    # render.opengl(view_context=True) renders the 3D view, and in camera view the camera
    # rectangle is drawn inset inside the region with the out-of-frame area around it. That
    # background is part of the render. Measured on postwar-city, the camera frame was 23% of
    # the image and the other 77% was flat viewport grey, so every capture threw away most of
    # its resolution and handed the vision model a small picture in a large empty field.
    #
    # view_center_camera sets the camera-view zoom so the frame fits the region, which takes
    # that 23% to 89%. Pushing the zoom further gains 0.6% more, so this is effectively the
    # whole win.
    #
    # The alternative, view_context=False, renders the camera exactly and fills the frame, but
    # falls back to Workbench shading: flat grey massing with no textures at all. That is
    # unusable here, which is why the fix is a zoom rather than a different render call.
    with C.temp_override(**override):
        try:
            bpy.ops.view3d.view_center_camera()
        except (RuntimeError, AttributeError):
            pass
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

#!/usr/bin/env python3
"""
helper_funcs.py — Blender camera helpers for SCOPE simulation.

Provides fast camera screenshots, FOV zoom, and panorama capture.
"""

import bpy
from PIL import Image
import numpy as np
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


# ─── Panorama ─────────────────────────────────────────────────────────────────
#
# A 360-degree sweep, driven one frame per call so the caller can yield progress
# between frames. `blender_tools._iter_full_panorama` is the caller. Its contract
# decides everything below, so it is written out here:
#
#   ts        = "%Y%m%d_%H%M%S"
#   output    = <PANOS_DIR>/<ts>_panorama.png      <- passed to start_panorama_capture
#   run_dir   = <PANOS_DIR>/_panorama_<ts>         <- where the caller looks afterwards
#   the caller loops capture_panorama_step() until it returns None, then waits up
#   to 5 seconds for run_dir/panorama_stitched.png to exist.
#
# The stitch therefore happens inside the final step, before it returns None. The
# file is already on disk when the caller starts waiting, so the 5 second budget is
# never spent.
#
# Why the stitch does not search for features: this code drives the camera, so the
# angle of every frame is known exactly. A known-geometry reprojection is
# deterministic and repeatable, which a benchmark needs. Feature matching would
# give a different panorama for the same scene on different runs, and it would need
# OpenCV, which this project does not depend on.

_PANO_STATE = {}

# Seconds to wait after a camera move before the frame is captured. The viewport
# draws asynchronously, so a capture taken too early returns the previous frame or
# a partly drawn one. Override with SCOPE_PANO_SETTLE when a scene is heavy.
_PANO_SETTLE_DEFAULT = 0.35


def _pano_run_dir(output_path: str) -> str:
    """Derive the directory the caller will look in from the path it passed in.

    The caller passes <dir>/<ts>_panorama.png and then looks in <dir>/_panorama_<ts>.
    Deriving it here keeps the two in agreement without a second argument.
    """
    import os
    d = os.path.dirname(output_path) or "."
    base = os.path.basename(output_path)
    ts = base.split("_panorama")[0] if "_panorama" in base else os.path.splitext(base)[0]
    return os.path.join(d, f"_panorama_{ts}")


def start_panorama_capture(output_path: str, overlap_ratio: float = 0.1):
    """Initialize a 360-degree panorama sweep.

    overlap_ratio is the fraction of each frame that repeats in the next one. Some
    overlap is required: it is what the blend uses to hide the seam. Too little
    leaves a visible edge, too much wastes frames. The step is then rounded so the
    sweep closes on itself exactly, because a sweep that does not close leaves one
    seam that no amount of blending can hide.
    """
    import os, math

    cam_obj = bpy.context.scene.camera
    if cam_obj is None:
        raise RuntimeError("No active scene camera set; cannot capture a panorama.")

    overlap_ratio = min(max(float(overlap_ratio), 0.0), 0.9)
    h_fov = float(cam_obj.data.angle_x)

    # Round the step down so an integer number of frames covers exactly 360 degrees.
    # The resulting overlap is >= the requested one, never less.
    n_frames = int(math.ceil((2.0 * math.pi) / (h_fov * (1.0 - overlap_ratio))))
    step = (2.0 * math.pi) / n_frames

    run_dir = _pano_run_dir(output_path)
    os.makedirs(run_dir, exist_ok=True)

    _PANO_STATE.clear()
    _PANO_STATE.update({
        "started": True,
        "output_path": output_path,
        "run_dir": run_dir,
        "overlap": overlap_ratio,
        "cam": cam_obj,
        "orig_rotation": tuple(cam_obj.rotation_euler),
        "orig_rotation_mode": cam_obj.rotation_mode,
        "h_fov": h_fov,
        "v_fov": float(cam_obj.data.angle_y),
        "step": step,
        "n_frames": n_frames,
        "index": 0,
        "frames": [],
        "settle": float(os.environ.get("SCOPE_PANO_SETTLE", _PANO_SETTLE_DEFAULT)),
    })


def capture_panorama_step():
    """Capture the next frame in the panorama sweep.

    Returns the path of the frame just written, or None when the sweep is finished.
    The call that returns None has already written panorama_stitched.png and has
    already put the camera back where it was.
    """
    import os

    st = _PANO_STATE
    if not st.get("started"):
        return None

    i = st["index"]
    if i >= st["n_frames"]:
        try:
            _stitch_panorama(st)
        finally:
            _restore_camera(st)
            st["started"] = False
        return None

    cam_obj = st["cam"]
    cam_obj.rotation_mode = 'XYZ'
    rx, ry, _rz = st["orig_rotation"]
    cam_obj.rotation_euler = (rx, ry, st["orig_rotation"][2] + i * st["step"])

    # Push the new transform through the dependency graph before drawing. Without
    # this the viewport can redraw from the previous transform and the frame is a
    # duplicate of the last one, which the blend then treats as a real observation.
    _force_scene_update()

    frame_path = os.path.join(st["run_dir"], f"frame_{i:03d}.png")
    screenshot_camera_view(frame_path, wait=st["settle"])

    st["frames"].append(frame_path)
    st["index"] = i + 1
    return frame_path


def _restore_camera(st):
    cam_obj = st.get("cam")
    if cam_obj is None:
        return
    cam_obj.rotation_mode = st.get("orig_rotation_mode", 'XYZ')
    cam_obj.rotation_euler = st["orig_rotation"]
    _force_scene_update()


def _force_scene_update():
    """Make the pending transform visible to whatever draws next.

    view_layer.update() evaluates the dependency graph. tag_redraw marks every 3D
    viewport dirty. Neither one blocks until the draw has finished, which is why the
    caller still waits afterwards.
    """
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
    try:
        for area in bpy.context.window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
    except Exception:
        pass


def _stitch_panorama(st):
    """Reproject every frame onto one cylinder and blend the overlaps.

    Why reproject rather than paste side by side: each frame is a pinhole image, so
    a feature away from the frame centre sits further from that centre than its
    angle alone would put it, by a factor of 1/cos(angle). Pasting two such frames
    edge to edge steps the scale at the join. That step is what reads as the seam,
    and the stretch near each frame edge is what reads as distortion. Mapping every
    output pixel through the angle it represents removes both, because the output is
    angle-linear by construction.

    Why blend rather than butt-join: even an exact reprojection differs a little
    between two frames that see the same angle, because the viewport shades from a
    different camera position in the sweep. A weight that peaks at each frame centre
    and falls to zero at its edge spreads the crossover over the whole overlap, so no
    single column carries the join.
    """
    import os, math
    import numpy as np
    from PIL import Image

    frames = st["frames"]
    if not frames:
        raise RuntimeError("panorama: no frames were captured")

    h_fov, v_fov = st["h_fov"], st["v_fov"]
    step = st["step"]

    first = Image.open(frames[0]).convert("RGB")
    fw, fh = first.size

    # Vertical span of the output cylinder.
    #
    # A pinhole frame does not cover the same elevation everywhere. At an angle dd
    # away from its centre it reaches only atan(tan(v_fov/2) * cos(dd)). The worst
    # served azimuth is the midpoint between two frame centres, where the nearest
    # frame is step/2 away. Sizing the cylinder to full v_fov therefore leaves
    # unfilled wedges along the top and bottom edges, which is a part of the scene
    # that never comes back. Sizing it to what every azimuth can actually supply
    # gives a complete rectangle instead. The cost is a small vertical crop, which
    # is preferable to a hole: a hole is indistinguishable from dark scenery to
    # whatever reads the panorama afterwards.
    v_out = 2.0 * math.atan(math.tan(v_fov * 0.5) * math.cos(step * 0.5))

    # Keep the horizontal angular resolution of the source frames, so the panorama
    # neither invents detail nor throws any away.
    pano_w = int(round(fw * (2.0 * math.pi) / h_fov))
    pano_h = fh

    acc = np.zeros((pano_h, pano_w, 3), dtype=np.float64)
    wsum = np.zeros((pano_h, pano_w, 1), dtype=np.float64)

    # Output pixel centres expressed as angles.
    az = (np.arange(pano_w) + 0.5) * (2.0 * math.pi / pano_w)      # 0 .. 2pi
    el = (0.5 - (np.arange(pano_h) + 0.5) / pano_h) * v_out        # up positive
    tan_h = math.tan(h_fov * 0.5)
    tan_v = math.tan(v_fov * 0.5)

    for i, path in enumerate(frames):
        im = Image.open(path).convert("RGB")
        if im.size != (fw, fh):
            im = im.resize((fw, fh))
        img = np.asarray(im, dtype=np.float64)

        # Offset of every output column from this frame's centre, wrapped to [-pi, pi).
        d_az = (az - i * step + math.pi) % (2.0 * math.pi) - math.pi
        cols = np.nonzero(np.abs(d_az) < (h_fov * 0.5))[0]
        if cols.size == 0:
            continue
        dd = d_az[cols]

        # Pinhole mapping. The 1/cos(dd) term on the vertical axis is the correction
        # that keeps horizontal lines straight away from the frame centre.
        x = (0.5 + np.tan(dd) / (2.0 * tan_h)) * fw
        y = (0.5 - (np.tan(el)[:, None] / np.cos(dd)[None, :]) / (2.0 * tan_v)) * fh

        xi = np.clip(np.rint(x).astype(np.int64), 0, fw - 1)
        yr = np.rint(y).astype(np.int64)
        valid = (yr >= 0) & (yr < fh)
        yi = np.clip(yr, 0, fh - 1)

        # Feather weight: 1 at the frame centre, 0 at its horizontal edge.
        w = np.cos(dd / (h_fov * 0.5) * (math.pi * 0.5))
        w = np.clip(w, 1e-6, None)[None, :] * valid

        patch = img[yi, np.broadcast_to(xi, yi.shape)]
        acc[:, cols, :] += patch * w[..., None]
        wsum[:, cols, :] += w[..., None]

    covered = wsum[..., 0] > 1e-6
    out = np.zeros_like(acc)
    out[covered] = acc[covered] / wsum[covered]
    pano = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    dest = os.path.join(st["run_dir"], "panorama_stitched.png")
    pano.save(dest)
    st["stitched"] = dest
    st["coverage"] = float(covered.mean())
    return dest


# Re-export for convenience
list_presets = None
apply_preset = None
create_preset = None

try:
    from ..blender.preset_helpers import list_presets, apply_preset, create_preset
except ImportError:
    # When running inside Blender, preset_helpers may be imported differently
    pass

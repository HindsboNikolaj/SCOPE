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


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _euler_xyz_matrix(rx, ry, rz):
    """Blender's default XYZ Euler order, as a world-from-camera matrix.

    Blender applies X, then Y, then Z, so the composed matrix is Rz . Ry . Rx.
    """
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _stitch_panorama(st):
    """Reproject and blend a sweep. `st` is the state dict start_panorama_capture builds.

    Required keys: frames, h_fov, v_fov, step, run_dir.
    Optional: orig_rotation, the camera's Euler at the start of the sweep. Without it the
    camera is assumed level, which reproduces the old behaviour rather than failing.
    """
    # Imported here rather than at module scope, because helper_funcs.py does not import
    # os or numpy at the top and this function must drop into it without changing that.
    import os
    import math
    import numpy as np
    from PIL import Image

    frames = st["frames"]
    if not frames:
        raise RuntimeError("panorama: no frames were captured")

    h_fov = float(st["h_fov"])
    v_fov = float(st["v_fov"])
    step = float(st["step"])

    first = Image.open(frames[0]).convert("RGB")
    fw, fh = first.size

    rot = st.get("orig_rotation")
    R0 = _euler_xyz_matrix(*rot) if rot else np.eye(3)
    R0T = R0.T

    # Vertical span of the output. A frame reaches less elevation away from its centre, so
    # sizing the cylinder to the full vertical field of view leaves unfilled corners at the
    # midpoint between two frames. Size it to what every azimuth can supply, which fills the
    # rectangle completely. The cost is a small crop, which is preferable to a hole: a hole
    # is indistinguishable from dark scenery to whatever reads the panorama.
    v_out = 2.0 * math.atan(math.tan(v_fov * 0.5) * math.cos(step * 0.5))

    pano_w = int(round(fw * (2.0 * math.pi) / h_fov))
    pano_h = fh

    acc = np.zeros((pano_h, pano_w, 3), dtype=np.float64)
    wsum = np.zeros((pano_h, pano_w, 1), dtype=np.float64)

    az = (np.arange(pano_w) + 0.5) * (2.0 * math.pi / pano_w)
    el = (0.5 - (np.arange(pano_h) + 0.5) / pano_h) * v_out
    tan_h = math.tan(h_fov * 0.5)
    tan_v = math.tan(v_fov * 0.5)

    # Directions in camera space, one per output row, before the sweep rotation.
    v_e = np.stack([np.zeros_like(el), np.sin(el), -np.cos(el)], axis=1)   # (H, 3)
    base = v_e @ R0.T                                                      # R0 . v(e)

    half = h_fov * 0.5
    for i, path in enumerate(frames):
        im = Image.open(path).convert("RGB")
        if im.size != (fw, fh):
            # The frustum crop can land a pixel differently between frames. Normalise
            # rather than fail; one pixel is not worth losing a panorama over.
            im = im.resize((fw, fh))
        img = np.asarray(im, dtype=np.float64)

        # i * step - az, not az - i * step. The sweep turns the camera by rz0 - i * step, so
        # the output azimuth runs against the frame index rather than with it. Getting this
        # backwards mirrors the whole panorama, and nothing about the result looks wrong: the
        # buildings still join, the seam metric is unchanged, the coverage is unchanged. Only
        # text gives it away, and most frames contain none.
        #
        # It survived a synthetic round trip that scored 0.67 mean absolute error out of 255,
        # because the frame generator in that test shared this convention. A reference built
        # by the code under test cannot detect a mirror. The check that settled it was nine
        # real captures of a scene with a shop sign in it, stitched under each candidate, read
        # by eye: only this one spells "The Book Nook" forwards.
        d_az = (i * step - az + math.pi) % (2.0 * math.pi) - math.pi
        # A generous column range. Which pixels are really visible is decided below by the
        # projection itself, not by this bound.
        cols = np.nonzero(np.abs(d_az) < half * 1.30)[0]
        if cols.size == 0:
            continue
        dd = d_az[cols]                                                    # (C,)

        # d_cam = R0^T . Rz(dd) . R0 . v(e), for every (row, column) pair.
        Rz = np.zeros((dd.size, 3, 3))
        c, s = np.cos(dd), np.sin(dd)
        Rz[:, 0, 0] = c; Rz[:, 0, 1] = -s
        Rz[:, 1, 0] = s; Rz[:, 1, 1] = c
        Rz[:, 2, 2] = 1.0
        M = np.einsum('ij,cjk->cik', R0T, Rz)          # (C,3,3)
        d_cam = np.einsum('cij,rj->rci', M, base)      # (H,C,3)

        x, y, z = d_cam[..., 0], d_cam[..., 1], d_cam[..., 2]
        in_front = z < -1e-9
        zz = np.where(in_front, -z, 1.0)
        u = 0.5 + (x / zz) / (2.0 * tan_h)
        w = 0.5 - (y / zz) / (2.0 * tan_v)

        px = np.rint(u * fw).astype(np.int64)
        py = np.rint(w * fh).astype(np.int64)
        valid = in_front & (px >= 0) & (px < fw) & (py >= 0) & (py < fh)
        if not valid.any():
            continue
        px = np.clip(px, 0, fw - 1)
        py = np.clip(py, 0, fh - 1)

        # Feather across the frame, so no single column carries a join. Weight is measured
        # from the projected position rather than from the angle, which keeps it correct
        # when the tilt makes the two differ.
        wgt = np.cos(np.clip((u - 0.5) * math.pi, -math.pi / 2, math.pi / 2))
        wgt = np.where(valid, np.clip(wgt, 1e-6, None), 0.0)

        patch = img[py, px]
        acc[:, cols, :] += patch * wgt[..., None]
        wsum[:, cols, :] += wgt[..., None]

    covered = wsum[..., 0] > 1e-6
    out = np.zeros_like(acc)
    out[covered] = acc[covered] / wsum[covered]
    pano = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    dest = os.path.join(st["run_dir"], "panorama_stitched.png")
    pano.save(dest)
    st["stitched"] = dest
    st["coverage"] = float(covered.mean())
    return dest

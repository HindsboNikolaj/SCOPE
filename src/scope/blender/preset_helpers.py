# preset_helpers.py
# Utility functions for creating, listing, and applying Blender camera presets.

import bpy
import os
import textwrap


def create_preset(name: str, include_transform: bool = True, include_focal_length: bool = True) -> None:
    """
    Create or override a camera preset by writing a .py file into the user presets folder.
    - name: preset filename (without .py)
    - include_transform: include camera location/rotation
    - include_focal_length: include cam.lens value
    """
    # 1) Determine user presets directory
    user_scripts = bpy.utils.user_resource('SCRIPTS')
    preset_dir = os.path.join(user_scripts, "presets", "camera")
    os.makedirs(preset_dir, exist_ok=True)

    # 2) Build file path
    file_path = os.path.join(preset_dir, f"{name}.py")
    cam_obj = bpy.context.scene.camera
    cam = cam_obj.data

    # 3) Collect lines
    lines = [
        "# Auto-generated camera preset",  
        "import bpy",
        "cam_obj = bpy.context.scene.camera",
        "cam = cam_obj.data",
        f"cam.type = '{cam.type}'",
        f"cam.clip_start = {cam.clip_start}",
        f"cam.clip_end = {cam.clip_end}",
    ]
    if include_focal_length:
        lines.append(f"cam.lens = {cam.lens}")
    if include_transform:
        loc = cam_obj.location
        rot_mode = cam_obj.rotation_mode
        rot = cam_obj.rotation_euler
        lines += [
            f"cam_obj.location = ({loc.x:.6f}, {loc.y:.6f}, {loc.z:.6f})",
            f"cam_obj.rotation_mode = '{rot_mode}'",
            f"cam_obj.rotation_euler = ({rot.x:.6f}, {rot.y:.6f}, {rot.z:.6f})",
        ]

    # 4) Write file
    with open(file_path, 'w') as f:
        f.write(textwrap.dedent("""
        %s
        """ % '\n'.join(lines)))


def _preset_dirs():
    """Where to look for camera presets, widest first.

    SCOPE_PRESET_DIR comes first because it is an explicit instruction. Blender's own preset
    paths follow, which is where scripts/04_install_presets.py puts them on a normal install.

    This exists because the two disagreed. scripts/07_smoke_test.py and the capture scripts read
    SCOPE_PRESET_DIR, apply_preset did not, and nothing reconciled them. In a container with the
    presets mounted somewhere other than Blender's user directory, the scripts found them and
    the tool did not: go_to_preset returned "Preset not found" for a preset sitting on disk, and
    the planner, reasonably, told the user it did not exist.
    """
    import os as _os
    dirs = []
    d = _os.environ.get("SCOPE_PRESET_DIR")
    if d:
        dirs.append(d)
    dirs.extend(bpy.utils.preset_paths("camera"))
    return dirs


def list_presets() -> list[str]:
    """
    Return a sorted list of all camera preset names (without .py) found in system+user dirs.
    """
    preset_dirs = _preset_dirs()
    names = set()
    for pd in preset_dirs:
        if os.path.isdir(pd):
            for fn in os.listdir(pd):
                # Skip dotfiles. A preset directory that has ever been copied through a
                # macOS archive carries a binary AppleDouble side file named ._<name>.py
                # beside every real preset. Those end in .py, so a plain suffix test
                # returns them as presets and apply_preset then fails on the first one
                # with a UnicodeDecodeError that names no preset.
                if fn.startswith("."):
                    continue
                if fn.lower().endswith(".py"):
                    names.add(os.path.splitext(fn)[0])
    return sorted(names)


# Which preset this process last applied, and the camera pose it produced.
#
# The pose is stored alongside the name because the name on its own is not trustworthy. It is a
# module global, so it survives a file open: open a second scene without applying a preset and
# the process still claims the camera is at the previous scene's viewpoint. Anything keyed on
# that claim, such as the panorama cache, would then serve a picture of somewhere else.
#
# So the name is only ever returned when the camera is still where applying it put it. A
# handler clears the record on file load as well, which covers the common case; the pose check
# covers the rest, including code that moves the camera directly.
_LAST_PRESET = None
_LAST_PRESET_POSE = None
_LAST_PRESET_BLEND = None

# Tolerance for "the camera has not moved". Five decimals on position and degrees, three on the
# focal length, which is the precision the preset files themselves are written to.
_POSE_TOL = 1e-4


def _pose_now():
    cam = bpy.context.scene.camera
    if cam is None:
        return None
    import math
    return (tuple(round(v, 5) for v in cam.location),
            tuple(round(math.degrees(v), 5) for v in cam.rotation_euler),
            round(cam.data.lens, 3))


def _same_pose(a, b):
    if a is None or b is None:
        return False
    for x, y in zip(a[0], b[0]):
        if abs(x - y) > _POSE_TOL:
            return False
    for x, y in zip(a[1], b[1]):
        if abs(x - y) > _POSE_TOL:
            return False
    return abs(a[2] - b[2]) <= _POSE_TOL


def last_applied_preset():
    """The most recently applied preset name, or None.

    Returns None if the camera has since moved, or if a different .blend is open than the one
    the preset was applied in. A caller gets a name only when that name still describes where
    the camera actually is.
    """
    if _LAST_PRESET is None:
        return None
    if _LAST_PRESET_BLEND != bpy.data.filepath:
        return None
    if not _same_pose(_LAST_PRESET_POSE, _pose_now()):
        return None
    return _LAST_PRESET


def clear_last_preset():
    """Forget the last preset. Call after moving the camera deliberately."""
    global _LAST_PRESET, _LAST_PRESET_POSE, _LAST_PRESET_BLEND
    _LAST_PRESET = _LAST_PRESET_POSE = _LAST_PRESET_BLEND = None


@bpy.app.handlers.persistent
def _clear_on_load(*_args):
    clear_last_preset()


# Registered once. Opening a file must not leave the previous scene's viewpoint on record.
if _clear_on_load not in bpy.app.handlers.load_post:
    bpy.app.handlers.load_post.append(_clear_on_load)


def apply_preset(name: str) -> bool:
    """
    Execute the named preset script to set camera data + transform.
    Returns True if applied, False if not found.

    Note that a missing preset returns False rather than raising. That is deliberate for
    callers that probe, and it has cost real time here: a script that ignores the return value
    carries on with the camera wherever it happened to be, and every log line afterwards says
    the preset was applied. If you are not checking the return, raise instead.
    """
    global _LAST_PRESET, _LAST_PRESET_POSE, _LAST_PRESET_BLEND
    for pd in _preset_dirs():
        path = os.path.join(pd, f"{name}.py")
        if os.path.isfile(path):
            # errors="replace" rather than a bare open. Any unreadable byte then becomes a
            # replacement character and the compile fails with a message naming this file,
            # instead of the decode raising somewhere inside the codecs module with no
            # indication of which preset caused it.
            with open(path, encoding="utf-8", errors="replace") as f:
                code = f.read()
            exec(compile(code, path, 'exec'), { 'bpy': bpy })
            _LAST_PRESET = name
            _LAST_PRESET_POSE = _pose_now()
            _LAST_PRESET_BLEND = bpy.data.filepath
            return True
    return False

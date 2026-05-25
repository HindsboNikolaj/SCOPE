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


def list_presets() -> list[str]:
    """
    Return a sorted list of all camera preset names (without .py) found in system+user dirs.
    """
    preset_dirs = bpy.utils.preset_paths("camera")
    names = set()
    for pd in preset_dirs:
        if os.path.isdir(pd):
            for fn in os.listdir(pd):
                if fn.lower().endswith(".py"):
                    names.add(os.path.splitext(fn)[0])
    return sorted(names)


def apply_preset(name: str) -> bool:
    """
    Execute the named preset script to set camera data + transform.
    Returns True if applied, False if not found.
    """
    for pd in bpy.utils.preset_paths("camera"):
        path = os.path.join(pd, f"{name}.py")
        if os.path.isfile(path):
            with open(path) as f:
                code = f.read()
            exec(compile(code, path, 'exec'), { 'bpy': bpy })
            return True
    return False

"""
setup_presets.py — Install SCOPE benchmark camera presets into Blender.

Usage:
    blender --background --python scripts/setup_presets.py

Reads benchmark/presets/presets.json relative to the repo root and writes
each preset as a .py file to Blender's user camera presets folder.
"""

import os
import sys
import json

# ---------------------------------------------------------------------------
# Ensure bpy is available
# ---------------------------------------------------------------------------

try:
    import bpy
except ImportError:
    print(
        "\nError: 'bpy' module not found.\n"
        "This script must be run from within Blender:\n"
        "\n"
        "    blender --background --python scripts/setup_presets.py\n"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Locate presets.json relative to the repo root
# ---------------------------------------------------------------------------

# scripts/setup_presets.py  →  repo root is one level up
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
PRESETS_JSON = os.path.join(REPO_ROOT, "benchmark", "presets", "presets.json")

if not os.path.isfile(PRESETS_JSON):
    print(f"\nError: Could not find presets file at:\n    {PRESETS_JSON}\n")
    sys.exit(1)

with open(PRESETS_JSON, "r") as fh:
    presets_data = json.load(fh)

# ---------------------------------------------------------------------------
# Determine Blender's user camera presets directory
# ---------------------------------------------------------------------------

preset_dir = os.path.join(bpy.utils.user_resource('SCRIPTS'), "presets", "camera")
os.makedirs(preset_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Write individual preset .py files
# ---------------------------------------------------------------------------

PRESET_TEMPLATE = """\
# Auto-generated camera preset — SCOPE benchmark
import bpy
cam_obj = bpy.context.scene.camera
cam = cam_obj.data
cam.type           = '{cam_type}'
cam.clip_start     = {clip_start}
cam.clip_end       = {clip_end}
cam.lens           = {lens}
cam_obj.location   = ({x}, {y}, {z})
cam_obj.rotation_mode = 'XYZ'
cam_obj.rotation_euler = ({rx}, {ry}, {rz})
"""

installed = []

for scene_path, scene_presets in presets_data.items():
    for preset_name, data in scene_presets.items():
        loc = data["location"]
        rot = data["rotation_euler"]

        content = PRESET_TEMPLATE.format(
            cam_type   = data["type"],
            clip_start = data["clip_start"],
            clip_end   = data["clip_end"],
            lens       = data["lens"],
            x  = loc[0], y  = loc[1], z  = loc[2],
            rx = rot[0], ry = rot[1], rz = rot[2],
        )

        out_path = os.path.join(preset_dir, f"{preset_name}.py")
        with open(out_path, "w") as fh:
            fh.write(content)

        installed.append(preset_name)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

count = len(installed)

print(f"""
\u2713 Installed {count} camera presets for the SCOPE benchmark.

  To verify in Blender:
    Properties \u2192 Camera \u2192 Presets dropdown  (or use the Presets panel addon)

  To add your own presets:
    1. Install scope/blender/presets_banner.py as a Blender addon
    2. Open View3D \u2192 Sidebar (N key) \u2192 "Presets" tab
    3. Position your camera, enter a name, click "Create/Override Preset"
    4. Name convention: use a scene prefix (e.g. "myworld-home") to avoid
       name collisions across scenes

  Presets are stored at:
    {preset_dir}
""")

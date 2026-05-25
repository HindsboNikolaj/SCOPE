"""
presets_banner.py — Blender UI panel for managing SCOPE camera presets.

Provides a View3D → Sidebar → "Presets" panel with three operations:
  - Create/Override Preset: saves current camera position to a named preset
  - List Presets: prints all installed presets to the Blender console
  - Apply Preset: moves camera to a named preset by name

Usage: Install as a Blender addon or run as a script in Blender's Text Editor.
To add to Blender: Edit → Preferences → Add-ons → Install → select this file.

Custom presets can be created interactively. Name them with a scene prefix
(e.g. "myworld-home") to avoid collisions across scenes.
"""

bl_info = {
    "name": "Camera Presets Panel",
    "author": "Armada AI",
    "version": (1, 4, 0),
    "blender": (4, 4, 0),
    "location": "View3D › Sidebar › Presets",
    "description": "Create, list, and apply camera presets (including transform) from a UI panel",
    "category": "Camera",
}

import bpy, os, textwrap

# -------------------------------------------------------------------
#  Operators
# -------------------------------------------------------------------

class PRESETS_OT_create(bpy.types.Operator):
    bl_idname = "presets.create"
    bl_label = "Create/Override Preset"
    bl_description = "Save current camera settings and transform as a preset"

    def execute(self, context):
        wm = context.window_manager
        name = wm.preset_name_create.strip()
        if not name:
            self.report({'ERROR'}, "Preset name cannot be empty")
            return {'CANCELLED'}

        # Write to user presets folder
        user_scripts = bpy.utils.user_resource('SCRIPTS')
        preset_dir = os.path.join(user_scripts, "presets", "camera")
        os.makedirs(preset_dir, exist_ok=True)

        file_path = os.path.join(preset_dir, f"{name}.py")
        cam_obj = bpy.context.scene.camera
        cam = cam_obj.data

        # Capture transform and data-block properties
        loc = cam_obj.location
        rot_mode = cam_obj.rotation_mode
        rot = cam_obj.rotation_euler

        lines = [
            "# Auto-generated camera preset",
            "import bpy",
            "cam_obj = bpy.context.scene.camera",
            "cam = cam_obj.data",
            f"cam.type           = '{cam.type}'",
            f"cam.clip_start     = {cam.clip_start}",
            f"cam.clip_end       = {cam.clip_end}",
            f"cam.lens           = {cam.lens}",
            f"cam_obj.location   = ({loc.x:.6f}, {loc.y:.6f}, {loc.z:.6f})",
            f"cam_obj.rotation_mode = '{rot_mode}'",
            f"cam_obj.rotation_euler = ({rot.x:.6f}, {rot.y:.6f}, {rot.z:.6f})",
        ]

        with open(file_path, 'w') as f:
            f.write(textwrap.dedent("""
            %s
            """ % '\n'.join(lines)))

        self.report({'INFO'}, f"Preset '{name}' created/overridden (including location/rotation)")
        return {'FINISHED'}


class PRESETS_OT_list(bpy.types.Operator):
    bl_idname = "presets.list"
    bl_label = "List Presets"
    bl_description = "Print all camera presets to the Console"

    def execute(self, context):
        preset_dirs = bpy.utils.preset_paths("camera")
        names = set()
        for pd in preset_dirs:
            if os.path.isdir(pd):
                for fn in os.listdir(pd):
                    if fn.lower().endswith(".py"):
                        names.add(os.path.splitext(fn)[0])

        print("\n--- Camera Presets ---")
        for n in sorted(names):
            print("  ", n)
        print("----------------------\n")

        self.report({'INFO'}, f"{len(names)} presets listed in Console")
        return {'FINISHED'}


class PRESETS_OT_apply(bpy.types.Operator):
    bl_idname = "presets.apply"
    bl_label = "Apply Preset"
    bl_description = "Apply a named camera preset (sets data and transform)"

    def execute(self, context):
        wm = context.window_manager
        name = wm.preset_name_apply.strip()
        if not name:
            self.report({'ERROR'}, "Preset name cannot be empty")
            return {'CANCELLED'}

        for pd in bpy.utils.preset_paths("camera"):
            path = os.path.join(pd, f"{name}.py")
            if os.path.isfile(path):
                # read and execute preset script
                with open(path) as f:
                    code = f.read()
                exec(compile(code, path, 'exec'), {'bpy': bpy})
                self.report({'INFO'}, f"Applied preset '{name}' (data + transform)")
                return {'FINISHED'}

        self.report({'ERROR'}, f"Preset '{name}' not found")
        return {'CANCELLED'}


# -------------------------------------------------------------------
#  Panel
# -------------------------------------------------------------------

class PRESETS_PT_panel(bpy.types.Panel):
    bl_label = "Presets"
    bl_idname = "PRESETS_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Presets"

    def draw(self, context):
        wm = context.window_manager
        layout = self.layout

        layout.label(text="▶ Create / Override:")
        layout.prop(wm, "preset_name_create", text="Name")
        layout.operator("presets.create", icon="ADD")
        layout.separator()

        layout.label(text="▶ List all presets:")
        layout.operator("presets.list", icon="VIEWZOOM")
        layout.separator()

        layout.label(text="▶ Apply preset:")
        layout.prop(wm, "preset_name_apply", text="Name")
        layout.operator("presets.apply", icon="IMPORT")


# -------------------------------------------------------------------
#  Registration
# -------------------------------------------------------------------

classes = (
    PRESETS_OT_create,
    PRESETS_OT_list,
    PRESETS_OT_apply,
    PRESETS_PT_panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.WindowManager.preset_name_create = bpy.props.StringProperty(
        name="Preset Name",
        description="Name to create/override"
    )
    bpy.types.WindowManager.preset_name_apply = bpy.props.StringProperty(
        name="Preset Name",
        description="Name to apply"
    )


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    del bpy.types.WindowManager.preset_name_create
    del bpy.types.WindowManager.preset_name_apply

register()

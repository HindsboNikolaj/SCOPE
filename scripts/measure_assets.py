"""Measure what a scene actually contains: images, packed, missing, and the world texture.

Written to be run per scene in background Blender, printing one JSON line, so the results can
be collected without trusting anything remembered from a previous run.
"""
import bpy, os, json, re


def anonymise(path):
    """Strip personal names out of a texture path that points off this machine.

    Two of these scenes were authored on somebody else's Windows machine and reference
    textures under that person's home directory, their own documents folder, and a paid
    Blender addon installed there. The paths matter, because they are why those files are
    absent and unrecoverable, and which addon a reader would need to buy. The names do not,
    and this repository is public.

    Relative paths are left exactly as they are: they point inside the scene folder and
    contain nothing personal. Absolute ones are reduced to their last few segments, which is
    the part that says what the file was.
    """
    if not path:
        return path
    # "//" is Blender's own prefix for "relative to this .blend", not an absolute path.
    if path.startswith("//") or not re.match(r"^([A-Za-z]:[\\/]|/)", path):
        return path                                    # relative, e.g. //../../texture/route/
    parts = [q for q in re.split(r"[\\/]+", path) if q and not re.match(r"^[A-Za-z]:$", q)]
    tail = "/".join(parts[-3:])
    return "<original author's machine>/.../" + tail + ("/" if path.endswith(("/", "\\")) else "")


tot = len(bpy.data.images)
packed, missing, generated = 0, [], 0
for im in bpy.data.images:
    if im.packed_file:
        packed += 1
    elif im.source == 'GENERATED':
        generated += 1
    elif im.filepath:
        try:
            p = bpy.path.abspath(im.filepath)
        except Exception:
            p = im.filepath
        if not os.path.exists(p):
            missing.append(im.filepath)

world_tex, world_ok = None, False
w = bpy.context.scene.world
if w and w.use_nodes:
    for n in w.node_tree.nodes:
        if n.type == 'TEX_ENVIRONMENT' and getattr(n, 'image', None):
            world_tex = n.image.filepath
            world_ok = bool(n.image.packed_file) or (
                n.image.filepath and os.path.exists(bpy.path.abspath(n.image.filepath)))

meshes = sum(1 for o in bpy.context.scene.objects if o.type == 'MESH')
print("[A] " + json.dumps({
    "scene": os.path.splitext(os.path.basename(bpy.data.filepath))[0],
    "images": tot, "packed": packed, "generated": generated,
    "missing": len(missing), "missing_files": sorted(missing),
    "world_texture": world_tex, "world_usable": world_ok,
    "mesh_objects": meshes,
    "resolution": [bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y],
}), flush=True)

"""Measure what a scene actually contains: images, packed, missing, and the world texture.

Written to be run per scene in background Blender, printing one JSON line, so the results can
be collected without trusting anything remembered from a previous run.
"""
import bpy, os, json

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

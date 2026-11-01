"""Compose one SCOPE scene from several .blend files + a Poly Haven HDRI.

  blender --background --python scripts/kitbash_scene.py -- --inspect base.blend
  blender --background --python scripts/kitbash_scene.py -- --manifest scene.json

manifest.json:
{ "out": "/path/out.blend",
  "hdri": "/path/abandoned_tank_farm_01_4k.exr", "hdri_strength": 1.0, "hdri_rot_z": 0,
  "parts": [ {"file": "base.blend", "collection": "Building"},
             {"file": "props.blend", "objects": ["Prop"], "loc": [40,-12,0], "rot_z": 95, "scale": 1.0} ] }

Each part is appended (link=False, so the result is self-contained) and instanced at loc/rot/scale.
Give "collection" or "objects", not both. Omitting both appends every collection in the file.
"""
import bpy, json, sys, os, math

def argv():
    a = sys.argv
    return a[a.index("--") + 1:] if "--" in a else []

def inspect(path):
    with bpy.data.libraries.load(path, link=True) as (src, _):
        print(f"\n{path}\n  collections: {sorted(src.collections)}\n  objects({len(src.objects)}): {sorted(src.objects)[:60]}")

def instance(coll, name, loc, rot_z, scale):
    e = bpy.data.objects.new(name, None)
    e.instance_type, e.instance_collection = 'COLLECTION', coll
    e.location, e.rotation_euler.z, e.scale = loc, math.radians(rot_z), (scale,) * 3
    bpy.context.scene.collection.objects.link(e)
    return e

def add_part(p, n):
    f = p["file"]
    if not os.path.exists(f):
        raise SystemExit(f"missing part file: {f}")
    want_c = [p["collection"]] if "collection" in p else None
    want_o = p.get("objects")
    with bpy.data.libraries.load(f, link=False) as (src, dst):
        if want_o:
            miss = [o for o in want_o if o not in src.objects]
            if miss: raise SystemExit(f"{f}: objects not found: {miss} (use --inspect)")
            dst.objects = want_o
        else:
            names = want_c if want_c else list(src.collections)
            miss = [c for c in names if c not in src.collections]
            if miss: raise SystemExit(f"{f}: collections not found: {miss} (use --inspect)")
            dst.collections = names
        loaded_c, loaded_o = dst.collections, dst.objects
    tag = p.get("name") or f"part{n}_{os.path.basename(f).replace('.blend','')}"
    holder = bpy.data.collections.new(tag)                 # wrap loose objects so one empty moves them
    for o in loaded_o or []:
        if o: holder.objects.link(o)
    for c in loaded_c or []:
        if c: holder.children.link(c)
    return instance(holder, tag, p.get("loc", (0, 0, 0)), p.get("rot_z", 0), p.get("scale", 1.0))

def set_hdri(path, strength, rot_z):
    if not os.path.exists(path): raise SystemExit(f"missing hdri: {path}")
    w = bpy.data.worlds.new("SCOPE_World"); w.use_nodes = True
    bpy.context.scene.world = w
    nt = w.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg  = nt.nodes.new("ShaderNodeBackground");  bg.inputs["Strength"].default_value = strength
    env = nt.nodes.new("ShaderNodeTexEnvironment"); env.image = bpy.data.images.load(path)
    map_= nt.nodes.new("ShaderNodeMapping"); map_.inputs["Rotation"].default_value[2] = math.radians(rot_z)
    tex = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(tex.outputs["Generated"], map_.inputs["Vector"])
    nt.links.new(map_.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

a = argv()
if "--inspect" in a:
    for p in a[a.index("--inspect") + 1:]: inspect(p)
    sys.exit(0)
if "--manifest" not in a:
    sys.exit(__doc__)
m = json.load(open(a[a.index("--manifest") + 1]))
bpy.ops.wm.read_homefile(use_empty=True)
for n, p in enumerate(m["parts"]): print("  +", add_part(p, n).name)
if m.get("hdri"): set_hdri(m["hdri"], m.get("hdri_strength", 1.0), m.get("hdri_rot_z", 0))
out = os.path.abspath(m["out"]); os.makedirs(os.path.dirname(out), exist_ok=True)
missing = []                                               # pack per image: pack_all aborts on the first missing file
for img in bpy.data.images:
    if img.packed_file or img.source not in {"FILE", "SEQUENCE", "TILED"}:
        continue
    try:
        img.pack()
    except RuntimeError:
        missing.append(bpy.path.abspath(img.filepath))
for p in sorted(set(missing)):
    print(f"  missing texture (left unpacked): {p}")
bpy.ops.wm.save_as_mainfile(filepath=out)
if missing:
    print(f"  {len(set(missing))} texture(s) missing from the sources; see docs/MISSING_TEXTURES.md")
print(f"\nwrote {out}  ({os.path.getsize(out)/1e6:.1f} MB, {len(bpy.data.objects)} objects)")

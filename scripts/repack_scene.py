"""Repack a scene's textures into the .blend.

Usage:  blender --background --python scripts/repack_scene.py -- <blend_path>
"""
import bpy, os, sys, argparse

argv = sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument('blend')
parser.add_argument('--search-root', default=None)
parser.add_argument('--out', default=None)
args = parser.parse_args(argv)

blend_abs = os.path.abspath(args.blend)
search_root = args.search_root or os.path.dirname(blend_abs)
bpy.ops.wm.open_mainfile(filepath=blend_abs)

# Index nearby textures
idx = {}
for root, _, files in os.walk(search_root):
    for fn in files:
        if fn.lower().endswith(('.png','.jpg','.jpeg','.tga','.tif','.tiff','.exr','.hdr','.bmp','.dds')):
            idx.setdefault(fn, []).append(os.path.join(root, fn))

relinked = 0
for img in bpy.data.images:
    if img.source != 'FILE' or not img.filepath:
        continue
    resolved = bpy.path.abspath(img.filepath, library=img.library)
    if os.path.exists(resolved):
        continue
    base = os.path.basename(img.filepath.replace('\\','/'))
    cands = idx.get(base) or idx.get(base.lower()) or []
    if not cands:
        for k, v in idx.items():
            if k.lower() == base.lower():
                cands = v; break
    if cands:
        cands.sort(key=lambda p: p.count(os.sep))
        chosen = cands[0]
        img.filepath = bpy.path.relpath(chosen, start=os.path.dirname(blend_abs))
        img.filepath_raw = img.filepath
        try: img.reload()
        except Exception: pass
        if os.path.exists(bpy.path.abspath(img.filepath)):
            relinked += 1

bpy.ops.file.pack_all()
out = args.out or (os.path.splitext(blend_abs)[0] + '.packed.blend')
bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
print(f"[repack] relinked={relinked} size={os.path.getsize(out)/1024/1024:.1f}MB out={out}")

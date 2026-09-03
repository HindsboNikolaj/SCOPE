"""Check that this installation can actually produce good captures, and prove it with images.

Why this exists. Every setup failure met while getting SCOPE running on a new machine was
silent. A capture came back blank and the run continued. A preset from another scene put the
camera inside a wall and the run continued. A missing sky texture tinted every frame magenta
and the run continued. A scene was photographed four seconds after opening, before its
textures had finished streaming, and the run continued. In each case the benchmark produced
numbers, and the numbers were about an image nobody had looked at.

The failure is not loud. Shown a panorama that was mostly empty viewport background, a vision
model described "a street scene under a dark sky". There was no sky in that image at all.
Nothing raised, and the answer was confidently wrong.

So this looks. It runs the real capture path on the real scene, checks what comes back, and
leaves the images beside the report so a person can confirm the verdict.

Two tiers, because they answer different questions at different cost.

  Tier one is arithmetic on the pixels. Free, no network, a few seconds a view. It catches
  every failure listed above, all of which are visible in the pixels or in Blender's state.

  Tier two asks a vision model, one call an image, using the model the benchmark already
  needs. It answers what arithmetic cannot: does this look like a coherent place, seen from a
  sensible position, with its surfaces intact.

Usage:
  blender <scene.blend> --python scripts/06_verify_setup.py -- <outdir> [--vlm]

  SCOPE_VERIFY_VLM=1  also run the tier two judge
  SCOPE_SETTLE=15     seconds to wait after opening before the first capture
  SCOPE_CSV=<path>    the benchmark CSV, used to find this scene's own presets
  SCOPE_PRESET_DIR=<path>  where the installed camera presets are
"""
import bpy, sys, os, json, time, csv

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = argv[0] if argv else "verify_out"
USE_VLM = ("--vlm" in argv) or os.environ.get("SCOPE_VERIFY_VLM", "").strip() in ("1", "true", "yes")
os.makedirs(OUT, exist_ok=True)

SETTLE = float(os.environ.get("SCOPE_SETTLE", "15"))
PRESET_DIR = os.environ.get("SCOPE_PRESET_DIR", "")
CSV_PATH = os.environ.get("SCOPE_CSV", "")

# What "good enough" means, as numbers, so a verdict can be argued with rather than trusted.
MIN_LONG_EDGE = 1280       # below this a vision model loses small objects and lettering
MIN_SHARPNESS = 50.0       # mean squared Laplacian; a blank or half-drawn frame is near zero
MIN_COLOURFULNESS = 8.0    # mean channel spread; an untextured grey model sits near three
MAGENTA_MARGIN = 20.0      # red and blue this far above green is Blender's missing-texture pink


def log(m):
    print("[V] " + m, flush=True)


def measure(path):
    from PIL import Image
    import numpy as np
    im = Image.open(path).convert("RGB")
    a = np.asarray(im, dtype=float)
    g = np.asarray(im.convert("L"), dtype=float)
    k = g[1:-1, 1:-1]
    sharp = float(((-4 * k + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]) ** 2).mean())
    mx, mn = a.max(axis=2), a.min(axis=2)
    r, gg, b = (float(a[..., i].mean()) for i in range(3))
    return {"size": list(im.size), "long_edge": max(im.size),
            "aspect": round(im.size[0] / max(im.size[1], 1), 3),
            "mean": round(float(a.mean()), 1), "std": round(float(a.std()), 1),
            "sharpness": round(sharp, 1),
            "colourfulness": round(float((mx - mn).mean()), 1),
            "rgb": [round(r, 1), round(gg, 1), round(b, 1)],
            "magenta_cast": bool(r > gg + MAGENTA_MARGIN and b > gg + MAGENTA_MARGIN)}


def judge(m):
    """Turn measurements into named problems, each phrased as what a reader would see."""
    bad, warn = [], []
    if m["std"] < 3:
        bad.append("the frame is blank: nothing was drawn")
    if m["sharpness"] < MIN_SHARPNESS:
        bad.append(f"almost no detail (sharpness {m['sharpness']}); the scene was probably "
                   f"still loading when it was photographed")
    if m["magenta_cast"]:
        bad.append("magenta cast: a texture or the world environment map is missing and "
                   "Blender is drawing its missing-texture colour")
    if m["colourfulness"] < MIN_COLOURFULNESS:
        warn.append(f"almost no colour (colourfulness {m['colourfulness']}); this looks like "
                    f"an untextured grey model, so check the viewport shading")
    if m["long_edge"] < MIN_LONG_EDGE:
        warn.append(f"small image ({m['size'][0]}x{m['size'][1]}); a vision model will lose "
                    f"small objects and lettering. Enlarge the window or raise the scene's "
                    f"render resolution")
    return bad, warn


C = bpy.context
scene = C.scene
report = {"blend": bpy.data.filepath, "checks": [], "views": []}

env = {"blender": bpy.app.version_string, "background": bpy.app.background,
       "windows": len(C.window_manager.windows)}
try:
    import gpu
    env["gl_renderer"] = gpu.platform.renderer_get()
    env["gl_version"] = gpu.platform.version_get()
except Exception as e:
    env["gl_renderer"] = f"unavailable: {e}"
report["environment"] = env
log(f"blender={env['blender']} windows={env['windows']} gl={env.get('gl_renderer')}")

if env["windows"] == 0 or env["background"]:
    log("FAIL: Blender has no interface, so no capture can be taken. Run with a display, "
        "or use the container in docker/. See docs/HEADLESS.md.")
    report["checks"].append({"name": "blender has a viewport", "ok": False})
    json.dump(report, open(os.path.join(OUT, "verify.json"), "w"), indent=2)
    sys.exit(0)
report["checks"].append({"name": "blender has a viewport", "ok": True})

if "llvmpipe" in str(env.get("gl_renderer", "")).lower():
    log("note: software OpenGL. Captures are correct but slow. Material Preview is about a "
        "minute a frame; Solid with textures is roughly twenty times faster and loses "
        "anything behind glass.")

r = scene.render
area = next(a for a in C.screen.areas if a.type == 'VIEW_3D')
region = next(x for x in area.regions if x.type == 'WINDOW')
space = area.spaces.active
report["setup"] = {
    "resolution": [r.resolution_x, r.resolution_y], "percentage": r.resolution_percentage,
    "aspect": round(r.resolution_x / max(r.resolution_y, 1), 3),
    "color_depth": r.image_settings.color_depth, "engine": r.engine,
    "objects": len(bpy.data.objects), "images": len(bpy.data.images),
    "workspace": C.window.workspace.name if hasattr(C.window, "workspace") else None,
    "viewport_shading": {
        "type": space.shading.type,
        "color_type": getattr(space.shading, "color_type", None),
        "light": getattr(space.shading, "light", None),
        "use_scene_lights": bool(space.shading.use_scene_lights),
        "use_scene_world": bool(space.shading.use_scene_world)}}
log(f"render {r.resolution_x}x{r.resolution_y} @{r.resolution_percentage}% "
    f"depth={r.image_settings.color_depth} "
    f"workspace={report['setup']['workspace']} shading={space.shading.type}")

# Assets the scene refers to and cannot find. This is the cause behind a magenta frame, and
# naming it here saves the reader working backwards from a colour.
missing = [im.filepath for im in bpy.data.images
           if not im.packed_file and im.filepath
           and not os.path.exists(bpy.path.abspath(im.filepath))]
report["missing_images"] = missing
report["checks"].append({"name": "every referenced texture resolves", "ok": not missing,
                         "detail": f"{len(missing)} missing" if missing else "all present"})
if missing:
    log(f"{len(missing)} referenced images are missing:")
    for m in missing[:6]:
        log(f"    {m}")

space.region_3d.view_perspective = 'CAMERA'
space.camera = scene.camera
r.image_settings.file_format = 'PNG'


def capture(path):
    scene.render.filepath = path
    override = {'window': C.window, 'screen': C.window.screen, 'area': area,
                'region': region, 'space': space}
    t0 = time.time()
    with C.temp_override(**override):
        bpy.ops.render.opengl(write_still=True, view_context=True)
    return round(time.time() - t0, 1)


import numpy as np
from PIL import Image

log(f"measuring cold start; probing until two captures agree, floor {SETTLE}s")
probe_dir = os.path.join(OUT, "_settle")
os.makedirs(probe_dir, exist_ok=True)
# The ceiling has to be counted in probes rather than seconds. A first version stopped after
# sixty seconds of wall clock, which is generous on a machine with a GPU where a probe costs
# well under a second, and useless under software rendering where one Material Preview probe
# took between twelve and forty seconds. It reported "never stable" for a scene that was
# perfectly stable, because it only managed two probes before the clock ran out.
MAX_PROBES = int(os.environ.get("SCOPE_SETTLE_PROBES", "12"))
MAX_WALL = float(os.environ.get("SCOPE_SETTLE_MAX", "600"))
t_open = time.time()
prev, agree, settled_at, probes = None, 0, None, 0
while probes < MAX_PROBES and time.time() - t_open < MAX_WALL:
    probes += 1
    p = os.path.join(probe_dir, f"probe_{probes:03d}.png")
    capture(p)
    cur = np.asarray(Image.open(p).convert("L").resize((160, 90)), dtype=float)
    if prev is not None and float(np.abs(cur - prev).mean()) <= 0.4 and cur.std() > 1.0:
        agree += 1
        if agree >= 2 and time.time() - t_open >= SETTLE:
            settled_at = round(time.time() - t_open, 1)
            break
    else:
        agree = 0
    prev = cur
report["cold_start_seconds"] = settled_at
report["cold_start_probes"] = probes
report["checks"].append({
    "name": "scene reaches a stable image", "ok": settled_at is not None,
    "detail": (f"{settled_at}s after {probes} probes" if settled_at
               else f"not stable after {probes} probes in {round(time.time()-t_open)}s")})
if settled_at is not None:
    log(f"cold start: stable at {settled_at}s, after {probes} probes")
else:
    log(f"cold start: NOT STABLE after {probes} probes. On a slow renderer this can mean the "
        f"probes cost more than the scene needs to settle; raise SCOPE_SETTLE_PROBES.")

stem = os.path.splitext(os.path.basename(bpy.data.filepath))[0].lower().replace(".packed", "")
presets = []
if CSV_PATH and os.path.exists(CSV_PATH):
    for row in csv.DictReader(open(CSV_PATH)):
        parts = row["file_location"].split("/")
        if len(parts) > 1 and parts[1].lower().replace("_", "-") == stem.replace("_", "-"):
            ps = row.get("preset_start")
            if ps and ps not in presets:
                presets.append(ps)
presets.sort()
log(f"presets for this scene: {presets or '(none found; capturing the saved camera only)'}")

cam = scene.camera
for name in ["(as opened)"] + presets:
    if name != "(as opened)":
        pf = os.path.join(PRESET_DIR, f"{name}.py") if PRESET_DIR else ""
        if not (pf and os.path.isfile(pf)):
            continue
        # errors="replace": a preset directory copied through a macOS archive carries binary
        # AppleDouble side files named ._<name>.py, and a bare open on one takes the run down.
        exec(compile(open(pf, encoding="utf-8", errors="replace").read(), pf, "exec"), {"bpy": bpy})
        bpy.context.view_layer.update()
    for a2 in C.window.screen.areas:
        if a2.type == 'VIEW_3D':
            a2.tag_redraw()
    time.sleep(0.4)
    safe = name.replace("(", "").replace(")", "").replace(" ", "-")
    path = os.path.join(OUT, f"{stem}__{safe}.png")
    secs = capture(path)
    m = measure(path)
    bad, warn = judge(m)
    report["views"].append({"scene": stem, "preset": name, "path": path, "seconds": secs,
                            "camera_location": [round(v, 4) for v in cam.location],
                            "camera_lens_mm": round(float(cam.data.lens), 2),
                            "problems": bad, "warnings": warn, **m})
    log(f"{name:18} {secs:5.1f}s {m['size'][0]}x{m['size'][1]} sharp={m['sharpness']:8.1f} "
        f"colour={m['colourfulness']:5.1f}  {'FAIL' if bad else ('warn' if warn else 'ok')}")
    for b in bad:
        log(f"      FAIL {b}")
    for w in warn:
        log(f"      warn {w}")

if USE_VLM:
    log("asking the vision model whether these look right")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
        from scope.tools.vlm_clients import create_vlm_from_env
        vlm = create_vlm_from_env()
        log(f"vision model: {getattr(vlm, 'name', '?')}")
        question = ("This is a screenshot from a 3D scene used in a robotics benchmark. "
                    "Answer in three short lines. 1) What kind of place is this. "
                    "2) Do the surfaces look like real materials, or is anything untextured, "
                    "magenta, or plainly wrong. "
                    "3) Is the camera somewhere sensible, or inside an object, or facing nothing.")
        for v in report["views"]:
            try:
                ans = vlm.query(Image.open(v["path"]), question).get("answer", "")
                v["vlm"] = " ".join(str(ans).split())
                log(f"    {v['preset']}: {v['vlm'][:150]}")
            except Exception as e:
                v["vlm_error"] = str(e)
                log(f"    {v['preset']}: vlm failed: {e}")
    except Exception as e:
        log(f"vision model unavailable, tier one results stand: {e}")
        report["vlm_error"] = str(e)

fails = sum(1 for v in report["views"] if v["problems"])
warns = sum(1 for v in report["views"] if v["warnings"])
report["summary"] = {"views": len(report["views"]), "failing": fails, "warning": warns,
                     "cold_start_seconds": settled_at, "missing_images": len(missing)}
json.dump(report, open(os.path.join(OUT, f"{stem}__verify.json"), "w"), indent=2)
cold = f"{settled_at}s" if settled_at is not None else "not measured"
log(f"SUMMARY {len(report['views'])} views, {fails} failing, {warns} with warnings, "
    f"cold start {cold}, {len(missing)} missing textures")
log(f"images and report in {OUT} — look at them")
bpy.ops.wm.quit_blender()

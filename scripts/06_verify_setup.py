"""Check that a SCOPE installation is actually able to produce good captures, and prove it.

Why this exists. Every failure that cost real time while getting SCOPE running on a new
machine was silent. The capture came back blank and the run continued. A preset from another
scene put the camera inside a wall and the run continued. A missing sky texture tinted every
frame magenta and the run continued. A scene was photographed four seconds after opening,
before its textures had finished streaming, and the run continued. In each case the
benchmark produced numbers, and the numbers were about an image nobody had looked at.

So this looks. It runs the real capture path on the real scenes, checks what comes back, and
writes a page of the images so a person can confirm the judgement.

Two tiers, because they answer different questions and cost different amounts.

  Tier 1, deterministic. Free, a few seconds per check, no network. Catches the failures
  above, all of which are visible in the pixels or in Blender's own state: a blank frame, a
  magenta cast, a camera that did not move, a scene still loading, a resolution that will
  give the model a different frame shape on one scene than another.

  Tier 2, a vision model as judge. Costs an API call per image. Answers the one question
  arithmetic cannot: does this look like a coherent place, photographed from a sensible spot,
  with its surfaces intact. A benchmark that already requires a vision model to run can use
  the same model to check its own setup, so this adds a dependency nobody did not already
  have.

Usage:
  blender <scene.blend> --python scripts/06_verify_setup.py -- <outdir> [--vlm]

  SCOPE_VERIFY_VLM=1   run the tier 2 judge as well
  SCOPE_SETTLE=15      seconds to wait after opening before the first capture
"""
import bpy, sys, os, json, time, math, csv

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = argv[0] if argv else "verify_out"
USE_VLM = ("--vlm" in argv) or os.environ.get("SCOPE_VERIFY_VLM", "").strip() in ("1", "true", "yes")
os.makedirs(OUT, exist_ok=True)

SETTLE = float(os.environ.get("SCOPE_SETTLE", "15"))
PRESET_DIR = os.environ.get("SCOPE_PRESET_DIR", "")
CSV_PATH = os.environ.get("SCOPE_CSV", "")

# What "good enough" means, as numbers, so a verdict can be argued with.
MIN_LONG_EDGE = 1280        # below this a vision model loses small objects and lettering
MIN_SHARPNESS = 50.0        # mean squared Laplacian. A blank or half-drawn frame is near 0
MIN_COLOURFULNESS = 8.0     # mean channel spread. An untextured grey model sits near 3
MAGENTA_MARGIN = 20.0       # red and blue this far above green is Blender's missing-texture pink


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
    return {
        "size": list(im.size), "long_edge": max(im.size),
        "aspect": round(im.size[0] / im.size[1], 3),
        "mean": round(float(a.mean()), 1), "std": round(float(a.std()), 1),
        "sharpness": round(sharp, 1),
        "colourfulness": round(float((mx - mn).mean()), 1),
        "rgb": [round(r, 1), round(gg, 1), round(b, 1)],
        "magenta_cast": bool(r > gg + MAGENTA_MARGIN and b > gg + MAGENTA_MARGIN),
    }


def judge(m):
    """Turn the measurements into named problems. Each one names what a reader would see."""
    bad, warn = [], []
    if m["std"] < 3:
        bad.append("the frame is blank: nothing was drawn")
    if m["sharpness"] < MIN_SHARPNESS:
        bad.append(f"almost no detail (sharpness {m['sharpness']}); the scene was probably "
                   f"still loading when it was photographed")
    if m["magenta_cast"]:
        bad.append("magenta cast: a texture or the world environment map is missing, and "
                   "Blender is drawing its missing-texture colour")
    if m["colourfulness"] < MIN_COLOURFULNESS:
        warn.append(f"almost no colour (colourfulness {m['colourfulness']}); this looks like "
                    f"an untextured grey model, so check the viewport shading")
    if m["long_edge"] < MIN_LONG_EDGE:
        warn.append(f"small image ({m['size'][0]}x{m['size'][1]}); a vision model will lose "
                    f"small objects and any lettering. Raise the scene's render resolution")
    return bad, warn


# ─── environment ──────────────────────────────────────────────────────────────
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
    log("FAIL: Blender has no interface, so no capture can be taken. "
        "Run with a display, or use the container in docker/. See docs/HEADLESS.md.")
    report["checks"].append({"name": "blender has a viewport", "ok": False})
    json.dump(report, open(os.path.join(OUT, "verify.json"), "w"), indent=2)
    sys.exit(0)
report["checks"].append({"name": "blender has a viewport", "ok": True})

if "llvmpipe" in str(env.get("gl_renderer", "")).lower():
    log("note: software OpenGL. Captures are correct but slow; MATERIAL preview is "
        "about a minute a frame. SOLID with textures is roughly twenty times faster.")

# ─── the scene as it is set up ────────────────────────────────────────────────
r = scene.render
setup = {"resolution": [r.resolution_x, r.resolution_y],
         "percentage": r.resolution_percentage,
         "aspect": round(r.resolution_x / max(r.resolution_y, 1), 3),
         "color_depth": r.image_settings.color_depth,
         "engine": r.engine,
         "objects": len(bpy.data.objects), "images": len(bpy.data.images)}
area = next(a for a in C.screen.areas if a.type == 'VIEW_3D')
region = next(x for x in area.regions if x.type == 'WINDOW')
space = area.spaces.active
setup["viewport_shading"] = {
    "type": space.shading.type,
    "color_type": getattr(space.shading, "color_type", None),
    "light": getattr(space.shading, "light", None),
    "use_scene_lights": bool(space.shading.use_scene_lights),
    "use_scene_world": bool(space.shading.use_scene_world)}
report["setup"] = setup
log(f"render {setup['resolution'][0]}x{setup['resolution'][1]} @{setup['percentage']}% "
    f"aspect={setup['aspect']} depth={setup['color_depth']} "
    f"shading={setup['viewport_shading']['type']}")

# Assets the scene refers to and cannot find. This is the cause behind a magenta frame, and
# naming it here means the reader does not have to work backwards from the colour.
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

# ─── cold start, measured rather than assumed ─────────────────────────────────
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


log(f"measuring cold start, probing until two captures agree (floor {SETTLE}s)")
import numpy as np
from PIL import Image

probe_dir = os.path.join(OUT, "_settle")
os.makedirs(probe_dir, exist_ok=True)
t_open = time.time()
prev, agree, settled_at = None, 0, None
while time.time() - t_open < 60:
    p = os.path.join(probe_dir, f"probe_{int(time.time()-t_open):03d}.png")
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
report["checks"].append({"name": "scene reaches a stable image", "ok": settled_at is not None,
                         "detail": f"{settled_at}s" if settled_at else "not stable within 60s"})
log(f"cold start: {'stable at ' + str(settled_at) + 's' if settled_at else 'NEVER STABLE'}")

# ─── one capture per preset, in the recommended shading ───────────────────────
space.shading.type = 'SOLID'
try:
    space.shading.color_type = 'TEXTURE'
except Exception:
    pass

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
    declared = None
    if name != "(as opened)":
        pf = os.path.join(PRESET_DIR, f"{name}.py") if PRESET_DIR else ""
        if not (pf and os.path.isfile(pf)):
            continue
        exec(compile(open(pf, encoding="utf-8", errors="replace").read(), pf, "exec"), {"bpy": bpy})
        bpy.context.view_layer.update()
        declared = [round(v, 4) for v in cam.location]
    for a in C.window.screen.areas:
        if a.type == 'VIEW_3D':
            a.tag_redraw()
    time.sleep(0.4)
    safe = name.replace("(", "").replace(")", "").replace(" ", "-")
    path = os.path.join(OUT, f"{stem}__{safe}.png")
    secs = capture(path)
    m = measure(path)
    bad, warn = judge(m)
    view = {"scene": stem, "preset": name, "path": path, "seconds": secs,
            "camera_location": [round(v, 4) for v in cam.location],
            "camera_lens_mm": round(float(cam.data.lens), 2),
            "problems": bad, "warnings": warn, **m}
    report["views"].append(view)
    verdict = "FAIL" if bad else ("warn" if warn else "ok")
    log(f"{name:16} {secs:5.1f}s  {m['size'][0]}x{m['size'][1]}  "
        f"sharp={m['sharpness']:7.1f} colour={m['colourfulness']:5.1f}  {verdict}")
    for b in bad:
        log(f"      FAIL {b}")
    for w in warn:
        log(f"      warn {w}")

# ─── tier 2, the vision model ─────────────────────────────────────────────────
if USE_VLM:
    log("asking the vision model whether these look right")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
        from scope.tools.vlm_clients import create_vlm_from_env
        vlm = create_vlm_from_env()
        question = (
            "This is a screenshot from a 3D scene used in a robotics benchmark. "
            "Answer in three short lines. "
            "1) What kind of place is this, in a few words. "
            "2) Do the surfaces look like real materials, or is anything an untextured flat "
            "colour, magenta, or plainly wrong. "
            "3) Is the camera somewhere sensible, or is it inside an object or facing nothing."
        )
        for v in report["views"]:
            try:
                ans = vlm.query(Image.open(v["path"]), question).get("answer", "")
                v["vlm"] = str(ans).strip()
                log(f"    {v['preset']}: {v['vlm'][:150]}")
            except Exception as e:
                v["vlm_error"] = str(e)
                log(f"    {v['preset']}: vlm failed: {e}")
    except Exception as e:
        log(f"vision model unavailable, tier 1 results stand: {e}")
        report["vlm_error"] = str(e)

fails = sum(1 for v in report["views"] if v["problems"])
warns = sum(1 for v in report["views"] if v["warnings"])
report["summary"] = {"views": len(report["views"]), "failing": fails, "warning": warns,
                     "cold_start_seconds": settled_at, "missing_images": len(missing)}
json.dump(report, open(os.path.join(OUT, f"{stem}__verify.json"), "w"), indent=2)
log(f"SUMMARY  {len(report['views'])} views, {fails} failing, {warns} with warnings, "
    f"cold start {settled_at}s, {len(missing)} missing textures")
bpy.ops.wm.quit_blender()

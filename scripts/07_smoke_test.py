"""Expected against actual, for every preset in a scene.

06_verify_setup.py asks whether this installation can produce a good capture at all. It looks
at the pixels on their own terms: not blank, not magenta, enough of the frame occupied. That
catches a broken setup but not a subtly different one, because it has nothing to compare
against.

This asks the other question. Somebody else already ran these presets on a working machine and
committed what they saw. Does your machine see the same thing? A capture that is perfectly
healthy by every measure in 06 and yet shows a different place means a preset moved, a scene
was re-exported, or the presets installed belong to another copy of the benchmark. None of
those raise anything on their own.

What is compared, and why each is loose or strict:

  pose          strict. The camera's location, rotation and focal length after applying the
                preset must match to five decimal places. This is arithmetic, not rendering,
                so there is no reason for it to differ and a difference means the preset file
                is not the one the reference was built from.

  structure     loose. A normalised correlation of edge maps at 64x64. Two machines will not
                produce identical pixels: different GPU, different driver, different sampling.
                What must survive is where things are. Above 0.55 is the same view; below 0.30
                is a different place; in between is the same place with something wrong.

  colour        loose, and a range rather than a value. Mean channel spread. Its job is to
                catch a capture that came back in the wrong shading mode, which shows up as a
                large drop, not as a small one.

  resolution    reported, not enforced. Render percentage is a local setting.

Panoramas are not swept here. They are committed under benchmark/panoramas and a run reads the
PNG, so what matters is that the file on disk is the file the sidecar describes; that is a
digest check and it needs no Blender at all. Run it with --panoramas-only.

Usage:

  blender <scene.blend> --python scripts/07_smoke_test.py -- <outdir>
  python3 scripts/07_smoke_test.py --panoramas-only

  --write-reference   record this run as the reference instead of checking against it
  SCOPE_SETTLE=15     seconds to wait after opening before the first capture
  SCOPE_PRESET_DIR    where the installed camera presets are

Exit status is 0 when every preset matched, 1 otherwise, so it can gate a merge.
"""
import base64
import json
import math
import os
import sys
import zlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE = os.path.join(REPO, "benchmark", "reference", "preset_reference.json")
PANO_DIR = os.path.join(REPO, "benchmark", "panoramas")

STRUCT_GOOD = 0.55
STRUCT_WEAK = 0.30
COLOUR_DROP = 0.45        # fraction of the reference colour below which shading is suspect
SIG = 64                  # signature is SIG x SIG greyscale


# --------------------------------------------------------------------------- signatures

def _signature_from_image(im):
    """A small greyscale thumbnail, stored compressed so the reference file stays readable."""
    import numpy as np
    g = im.convert("L").resize((SIG, SIG))
    a = np.asarray(g, dtype="uint8")
    return base64.b64encode(zlib.compress(a.tobytes(), 9)).decode("ascii")


def _signature_to_array(b64):
    import numpy as np
    raw = zlib.decompress(base64.b64decode(b64))
    return np.frombuffer(raw, dtype="uint8").reshape(SIG, SIG).astype("float64")


def _edges(a):
    import numpy as np
    gx = np.zeros_like(a); gy = np.zeros_like(a)
    gx[:, 1:-1] = a[:, 2:] - a[:, :-2]
    gy[1:-1, :] = a[2:, :] - a[:-2, :]
    g = np.hypot(gx, gy)
    g = g - g.mean()
    n = np.linalg.norm(g)
    return g / n if n > 1e-9 else g


def compare_signatures(cur_b64, ref_b64):
    a = _edges(_signature_to_array(cur_b64))
    b = _edges(_signature_to_array(ref_b64))
    score = float((a * b).sum())
    if score < STRUCT_WEAK:
        verdict = "different place"
    elif score < STRUCT_GOOD:
        verdict = "same place, something differs"
    else:
        verdict = "match"
    return round(score, 3), verdict


def colourfulness(im):
    import numpy as np
    a = np.asarray(im.convert("RGB"), dtype="float64")
    return round(float((a.max(2) - a.min(2)).mean()), 2)


# --------------------------------------------------------------------------- panoramas

def check_panoramas():
    """Every shipped panorama is the file its sidecar says it is. No Blender needed."""
    import hashlib
    import glob

    rows, bad = [], 0
    sidecars = sorted(glob.glob(os.path.join(PANO_DIR, "*.json")))
    if not sidecars:
        print(f"  no panoramas found under {PANO_DIR}")
        return 1

    for j in sidecars:
        key = os.path.basename(j)[:-5]
        png = os.path.join(PANO_DIR, key + ".png")
        meta = json.load(open(j))
        if not os.path.exists(png):
            rows.append((key, "MISSING", "the sidecar has no image beside it")); bad += 1
            continue
        h = hashlib.sha256()
        with open(png, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        got = h.hexdigest()
        want = meta.get("sha256")
        if want and got != want:
            rows.append((key, "CHANGED", f"digest {got[:12]} but the sidecar says {want[:12]}"))
            bad += 1
        else:
            rows.append((key, "ok", f"{meta.get('shading', '?')}, {os.path.getsize(png) // 1024} KB"))

    width = max(len(r[0]) for r in rows)
    for key, state, detail in rows:
        print(f"  {key:{width}}  {state:8} {detail}")
    print(f"\n  {len(rows) - bad}/{len(rows)} panoramas match their sidecar")
    return 1 if bad else 0


# --------------------------------------------------------------------------- in Blender

def _presets_for_scene(stem):
    """The preset names the benchmark starts at in this scene, read from the CSV."""
    import csv
    path = os.environ.get("SCOPE_CSV") or os.path.join(REPO, "benchmark", "scope_536.csv")
    if not os.path.exists(path):
        return set()
    want = set()
    key = stem.lower().replace("_", "-")
    for row in csv.DictReader(open(path)):
        parts = (row.get("file_location") or "").split("/")
        if len(parts) > 1 and parts[1].lower().replace("_", "-") == key:
            if row.get("preset_start"):
                want.add(row["preset_start"])
    return want


def run_in_blender(outdir, write_reference):
    import time
    import bpy
    from PIL import Image

    sys.path.insert(0, os.path.join(REPO, "src"))
    from scope.blender.helper_funcs import fast_opengl_screenshot   # noqa: E402

    settle = float(os.environ.get("SCOPE_SETTLE", "15"))
    preset_dir = os.environ.get("SCOPE_PRESET_DIR") or bpy.utils.user_resource(
        'SCRIPTS', path=os.path.join("presets", "camera"))

    stem = os.path.splitext(os.path.basename(bpy.data.filepath))[0]
    os.makedirs(outdir, exist_ok=True)

    C = bpy.context
    scene = C.scene
    area = next(a for a in C.screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    space = area.spaces.active
    space.region_3d.view_perspective = 'CAMERA'
    space.camera = scene.camera
    space.shading.type = 'MATERIAL'
    space.shading.use_scene_world = False
    for attr in ("show_overlays", "show_floor", "show_axis_x", "show_axis_y", "show_axis_z",
                 "show_cursor", "show_text", "show_stats", "show_extras",
                 "show_relationship_lines", "show_outline_selected"):
        try:
            setattr(space.overlay, attr, False)
        except (TypeError, AttributeError):
            pass

    def fit():
        try:
            with C.temp_override(window=C.window, screen=C.window.screen, area=area,
                                 region=region, space=space):
                bpy.ops.view3d.view_center_camera()
        except Exception:
            pass

    fit()
    print(f"  settling {settle}s before the first capture")
    time.sleep(settle)

    reference = {}
    if os.path.exists(REFERENCE):
        reference = json.load(open(REFERENCE))
    scene_ref = reference.get(stem, {})
    if not write_reference and not scene_ref:
        print(f"  no reference recorded for scene {stem!r}. Run with --write-reference on a "
              f"machine you trust, and commit benchmark/reference/preset_reference.json.")
        return 1

    names = sorted(n[:-3] for n in os.listdir(preset_dir) if n.endswith(".py"))
    if not names:
        print(f"  no presets installed under {preset_dir}. Run scripts/04_install_presets.py.")
        return 1

    # Only this scene's own presets. Every preset in the benchmark is installed into one
    # directory, so the list above spans all four scenes; applying whitechapel's camera to
    # book-nook puts it in empty space and recording that as a reference would be worse than
    # useless. The benchmark CSV is what says which preset belongs where.
    own = _presets_for_scene(stem)
    if own:
        names = [n for n in names if n in own]
        print(f"  {len(names)} presets belong to {stem}: {', '.join(names)}")
    else:
        print(f"  could not read the benchmark CSV, so every installed preset will be tried")

    def pose():
        cam = scene.camera
        return {"location": [round(v, 5) for v in cam.location],
                "rotation_deg": [round(math.degrees(v), 5) for v in cam.rotation_euler],
                "lens_mm": round(cam.data.lens, 3)}

    def capture(path):
        fast_opengl_screenshot(path)

    results, failures = {}, 0
    checked = 0
    for name in names:
        path = os.path.join(preset_dir, f"{name}.py")
        try:
            exec(compile(open(path, encoding="utf-8", errors="replace").read(), path, "exec"),
                 {"bpy": bpy})
        except Exception as e:
            print(f"  {name:18} preset failed to apply: {e}")
            continue
        bpy.context.view_layer.update()
        fit()

        png = os.path.join(outdir, f"{stem}__{name}.png")
        capture(png)
        im = Image.open(png)
        rec = {"pose": pose(), "signature": _signature_from_image(im),
               "colour": colourfulness(im), "size": list(im.size)}

        if write_reference:
            results[name] = rec
            print(f"  {name:18} recorded  colour {rec['colour']:5.1f}  {rec['size']}")
            continue

        ref = scene_ref.get(name)
        if not ref:
            # A preset belonging to another scene. Applying it here proves nothing.
            continue
        checked += 1

        problems = []
        for k in ("location", "rotation_deg"):
            for i, (a, b) in enumerate(zip(ref["pose"][k], rec["pose"][k])):
                if abs(a - b) > 1e-4:
                    problems.append(f"{k}[{i}] expected {a} got {b}")
        if abs(ref["pose"]["lens_mm"] - rec["pose"]["lens_mm"]) > 1e-3:
            problems.append(f"lens expected {ref['pose']['lens_mm']} got {rec['pose']['lens_mm']}")

        score, verdict = compare_signatures(rec["signature"], ref["signature"])
        if verdict != "match":
            problems.append(f"view {verdict} (edge correlation {score})")
        if ref["colour"] > 1 and rec["colour"] < ref["colour"] * COLOUR_DROP:
            problems.append(f"colour {rec['colour']} against an expected {ref['colour']}, "
                            f"which usually means the wrong shading mode")

        state = "ok" if not problems else "FAIL"
        if problems:
            failures += 1
        print(f"  {name:18} {state:5} view {score:+.3f}  colour {rec['colour']:5.1f} "
              f"(expected {ref['colour']:5.1f})")
        for p in problems:
            print(f"      {p}")

    if write_reference:
        reference[stem] = results
        os.makedirs(os.path.dirname(REFERENCE), exist_ok=True)
        json.dump(reference, open(REFERENCE, "w"), indent=1, sort_keys=True)
        print(f"\n  wrote {len(results)} presets for {stem} to {REFERENCE}")
        return 0

    print(f"\n  {checked - failures}/{checked} presets match the reference for {stem}")
    print(f"  captures in {outdir}")
    return 1 if failures else 0


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    if "--panoramas-only" in argv:
        sys.exit(check_panoramas())
    write_reference = "--write-reference" in argv
    positional = [a for a in argv if not a.startswith("--")]
    outdir = positional[0] if positional else "smoke_out"
    try:
        import bpy  # noqa: F401
    except ImportError:
        print("  this part must run inside Blender:\n"
              "    blender <scene.blend> --python scripts/07_smoke_test.py -- <outdir>\n"
              "  the panorama check does not need Blender:\n"
              "    python3 scripts/07_smoke_test.py --panoramas-only")
        sys.exit(2)
    sys.exit(run_in_blender(outdir, write_reference))


if __name__ == "__main__":
    main()

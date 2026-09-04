"""Write cache metadata for the panoramas shipped in benchmark/panoramas.

The PNGs alone are not a cache. A lookup validates the camera pose, and the pose has to come
from somewhere, so this reads it out of the preset files those panoramas were captured from and
writes the metadata beside each image.

Run it after adding or replacing a panorama by hand. It does not open Blender.

    python3 scripts/index_panorama_cache.py [--dir benchmark/panoramas]

Poses come from benchmark/presets/presets_by_scene.json, which ships with the repository, so
this works on a fresh clone before anything has been installed into Blender's user directory.
If that index is missing, run scripts/build_preset_index.py. Installed .py presets are used as
a fallback, and SCOPE_PRESET_DIR points at them if they live somewhere unusual.

The preset definitions are the source of truth for the pose. They assign the camera directly, so what
they declare is exactly what applying them produces, which is why the pose recorded here matches
what a run will present at lookup time.
"""
import argparse
import ast
import hashlib
import json
import math
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read_preset(path):
    """Pull location, rotation and lens out of a camera preset .py without executing it.

    Not executed because these files assume a live bpy, and because running arbitrary code to
    read three numbers out of it is a poor trade.
    """
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    out = {}
    for key, pat in (("lens", r"cam\.lens\s*=\s*([-\d.eE]+)"),
                     ("location", r"cam_obj\.location\s*=\s*\(([^)]*)\)"),
                     ("rotation", r"cam_obj\.rotation_euler\s*=\s*\(([^)]*)\)")):
        m = re.search(pat, txt)
        if not m:
            return None
        raw = m.group(1)
        out[key] = (float(raw) if key == "lens"
                    else [float(x) for x in ast.literal_eval("(" + raw + ")")])
    return out


def poses_from_index():
    """{scene: {preset: {location, rotation, lens}}} from the shipped index.

    Preferred over the installed .py presets because it is in the repository: a fresh clone can
    index its panoramas before running 04_install_presets.py, and a machine that never installs
    them at all still gets a working cache.
    """
    idx = ROOT / "benchmark" / "presets" / "presets_by_scene.json"
    if not idx.is_file():
        return {}
    try:
        doc = json.loads(idx.read_text())
    except (OSError, ValueError):
        return {}
    out = {}
    for scene, e in (doc.get("scenes") or {}).items():
        for name, d in (e.get("presets") or {}).items():
            if d.get("location") and d.get("rotation_euler") and d.get("lens") is not None:
                out.setdefault(scene, {})[name] = {
                    "location": d["location"],
                    "rotation": d["rotation_euler"],
                    "lens": d["lens"],
                }
    return out


def preset_dirs():
    """Where camera presets live, in the order Blender would look."""
    extra = os.environ.get("SCOPE_PRESET_DIR")
    cands = [Path(os.path.expanduser("~/.config/blender")).glob("*/scripts/presets/camera"),
             [Path(os.path.expanduser("~/Library/Application Support/Blender"))
              .joinpath(v, "scripts/presets/camera") for v in ("4.4", "4.3", "4.2")],
             [ROOT / "benchmark" / "presets" / "camera"],
             ([Path(extra)] if extra else [])]
    seen = []
    for group in cands:
        for d in group:
            if d.is_dir():
                seen.append(d)
    return seen


def find_preset(name):
    for d in preset_dirs():
        p = d / f"{name}.py"
        if p.is_file():
            return p
    return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "benchmark" / "panoramas"))
    ap.add_argument("--scenes", default=str(ROOT / "benchmark" / "scenes"))
    ap.add_argument("--shading", default="MATERIAL",
                    help="the mode these were captured in; entries are refused to runs in "
                         "another mode")
    args = ap.parse_args()

    d = Path(args.dir)
    if not d.is_dir():
        raise SystemExit(f"no such directory: {d}")

    index = poses_from_index()
    dirs = preset_dirs()
    print(f"poses from the shipped index: "
          f"{sum(len(v) for v in index.values())} across {len(index)} scenes")
    print(f"installed preset fallback: {[str(x) for x in dirs] or 'none'}")
    if not index and not dirs:
        raise SystemExit("no poses available. Run scripts/build_preset_index.py, or "
                         "scripts/04_install_presets.py.")

    written, skipped = 0, []
    for png in sorted(d.glob("*.png")):
        stem = png.stem
        if "__" not in stem:
            skipped.append((png.name, "name is not <scene>__<preset>.png")); continue
        scene, preset = stem.split("__", 1)
        # Idempotent: a file already renamed to the cache convention carries a "preset-" prefix,
        # and re-parsing it would look for a preset literally called "preset-road1".
        if preset.startswith("preset-"):
            preset = preset[len("preset-"):]
        # Entries captured in a non-default mode carry it in the name, e.g. ...__solid.
        shading = args.shading
        if preset.endswith("__solid"):
            preset, shading = preset[: -len("__solid")], "SOLID+TEXTURE"

        pose = (index.get(scene) or {}).get(preset)
        source = f"benchmark/presets/presets_by_scene.json ({scene}/{preset})"
        if pose is None:
            pf = find_preset(preset)
            if pf is None:
                skipped.append((png.name, f"no pose for {scene}/{preset!r}: not in the index "
                                          f"and no installed preset file")); continue
            pose = read_preset(pf)
            source = f"installed preset {pf.name}"
            if pose is None:
                skipped.append((png.name, f"could not read pose from {pf}")); continue

        blend = None
        for cand in Path(args.scenes).glob(f"{scene}/*.blend"):
            blend = str(cand); break

        meta = {
            "scene": scene,
            "blend": blend or f"{scene}.blend",
            "blend_mtime": None,
            "pose": {"preset": preset},
            "pose_exact": {
                "location": [round(v, 5) for v in pose["location"]],
                "rotation_deg": [round(math.degrees(v), 5) for v in pose["rotation"]],
                "lens_mm": round(pose["lens"], 3),
            },
            "shading": shading,
            "source": f"shipped with the repository, pose from {source}",
            "sha256": sha256(png),
            "written_by_pid": None,
        }
        # The name a lookup will compute, so the entry is findable.
        target = d / f"{scene}__preset-{preset}.png"
        if png != target:
            png.replace(target)
            png = target
        jf = d / f"{scene}__preset-{preset}.json"
        if jf.exists():
            try:
                prev = json.loads(jf.read_text())
                # A shading label set by hand, or by a previous run that knew better than the
                # command line default, is preserved. Getting this wrong serves a Solid
                # panorama to a Material run, which is the one thing the cache must not do.
                if prev.get("shading"):
                    meta["shading"] = prev["shading"]
                if prev.get("note"):
                    meta["note"] = prev["note"]
            except (OSError, ValueError):
                pass
        jf.write_text(json.dumps(meta, indent=2))
        written += 1
        print(f"  {png.name:46} {meta['pose_exact']['lens_mm']:6.2f}mm  {shading}")

    print(f"\n{written} entries indexed")
    for name, why in skipped:
        print(f"  skipped {name}: {why}")


main()

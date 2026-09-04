"""The shipped full views are the files their manifests describe, and carry nothing private.

A benchmark run reads these PNGs instead of sweeping, so they are data the results depend on
rather than documentation. Two things have gone wrong with them before and both are checked
here: an image was replaced without its manifest being updated, and a manifest recorded an
absolute path that named somebody's home directory in a public repository.
"""
import glob
import hashlib
import json
import os
import re

import pytest
from PIL import Image


def _keys(panorama_dir):
    return sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(panorama_dir, "*.json")))


def test_there_are_panoramas(panorama_dir):
    assert _keys(panorama_dir), "no panorama manifests found"


def test_every_manifest_has_an_image(panorama_dir):
    missing = [k for k in _keys(panorama_dir)
               if not os.path.exists(os.path.join(panorama_dir, k + ".png"))]
    assert not missing, f"manifests with no image beside them: {missing}"


def test_every_image_matches_its_recorded_digest(panorama_dir):
    """The check that catches a swapped panorama nobody re-registered."""
    wrong = []
    for k in _keys(panorama_dir):
        meta = json.load(open(os.path.join(panorama_dir, k + ".json")))
        want = meta.get("sha256")
        if not want:
            continue
        h = hashlib.sha256()
        with open(os.path.join(panorama_dir, k + ".png"), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != want:
            wrong.append(f"{k}: file is {h.hexdigest()[:12]}, manifest says {want[:12]}")
    assert not wrong, "\n".join(wrong)


def test_no_manifest_names_somebodys_home_directory(panorama_dir):
    """These ship publicly. A path here must not say whose machine made it."""
    bad = []
    home = re.compile(r"(/home/|/Users/|[A-Za-z]:[\\/]Users[\\/])", re.I)
    for k in _keys(panorama_dir):
        text = open(os.path.join(panorama_dir, k + ".json")).read()
        for m in home.finditer(text):
            bad.append(f"{k}: {text[max(0, m.start() - 20):m.start() + 60]!r}")
    assert not bad, "absolute home paths in shipped manifests:\n" + "\n".join(bad)


def test_blend_paths_are_repository_relative(panorama_dir):
    """A shipped entry has to validate on a machine that is not the one that wrote it."""
    bad = []
    for k in _keys(panorama_dir):
        meta = json.load(open(os.path.join(panorama_dir, k + ".json")))
        blend = meta.get("blend") or ""
        if os.path.isabs(blend) or re.match(r"^[A-Za-z]:", blend):
            bad.append(f"{k}: {blend}")
    assert not bad, f"absolute .blend paths, which cannot match on another checkout: {bad}"


def test_every_manifest_records_the_pose_and_shading(panorama_dir):
    """Without a pose the lookup cannot verify anything; without shading it can serve a grey one."""
    bad = []
    for k in _keys(panorama_dir):
        meta = json.load(open(os.path.join(panorama_dir, k + ".json")))
        pose = meta.get("pose_exact") or {}
        if not all(x in pose for x in ("location", "rotation_deg", "lens_mm")):
            bad.append(f"{k}: pose_exact is incomplete")
        if not meta.get("shading"):
            bad.append(f"{k}: no shading recorded")
    assert not bad, "\n".join(bad)


def test_no_panorama_is_a_blank_image(panorama_dir):
    """A flat image is what a dead capture looks like, and it would be served silently."""
    import numpy as np
    flat = []
    for k in _keys(panorama_dir):
        a = np.asarray(Image.open(os.path.join(panorama_dir, k + ".png")).convert("L"))
        if int(a.max()) - int(a.min()) == 0:
            flat.append(k)
    assert not flat, f"panoramas with no variation at all: {flat}"


@pytest.mark.parametrize("key", ["city-street__preset-hotel-m", "city-street__preset-mailbox"])
def test_city_street_is_in_colour(panorama_dir, key):
    """These two shipped as grey Solid captures once. They are not allowed to again.

    Material Preview on this scene measures around 17 on mean channel spread; the Solid
    captures measured about 2. The threshold sits well between the two rather than near either,
    so it catches the mode being wrong without being sensitive to a re-capture.
    """
    import numpy as np
    p = os.path.join(panorama_dir, key + ".png")
    if not os.path.exists(p):
        pytest.skip(f"{key} is not in this checkout")
    a = np.asarray(Image.open(p).convert("RGB"), dtype=float)
    colour = float((a.max(2) - a.min(2)).mean())
    assert colour > 8.0, (
        f"{key} has a mean channel spread of {colour:.2f}, which is what a Solid capture looks "
        f"like. It should be Material Preview, around 17.")

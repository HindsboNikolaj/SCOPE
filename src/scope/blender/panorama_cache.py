"""Reuse a panorama that has already been captured, instead of sweeping again.

Why this exists. A 360 sweep is nine or ten viewport captures and a reprojection. Under
software OpenGL that is 60 to 190 seconds, and on `city-street` it is hours, because that
scene costs roughly 700 seconds a frame whatever the resolution. Meanwhile 93 of the
benchmark's 541 rows ask for the full view, and they ask for it from ten fixed camera
positions. The same ten sweeps were being recomputed over and over for answers that cannot
differ: nothing in the scene moves between rows.

So a sweep is captured once per viewpoint and stored, and a full-view request loads it.

What makes a cache entry valid, and the rule that keeps it safe: **the key is a hint, the pose
is the guarantee**. Entries are named after the preset when one was applied, because that reads
well in a directory listing, but a name is never enough to serve on. Every hit is checked
against the camera's actual location, rotation and focal length as recorded in the entry, plus
the .blend path and the shading mode. If any of those disagree with the process's current
state, the hit is discarded and the sweep runs.

That check is what makes concurrent runs safe. Several benchmarks over several scenes may share
one cache directory, and a preset name like `road1` can exist in more than one scene. Names are
scoped by scene file, but the pose comparison is what actually prevents one run's question being
answered with another run's picture, because it does not depend on any of the bookkeeping being
right.

Writes are atomic, so a reader never sees a half-written panorama from a writer in another
process: both files go to a uniquely named temporary and are renamed into place.

  SCOPE_PANO_CACHE      directory to read and write. Defaults to <output>/pano_cache.
  SCOPE_PANO_CACHE_MODE use    read the cache, and write to it after a fresh sweep (default)
                        write  ignore what is there, sweep, and overwrite the entry
                        off    ignore the cache entirely
"""
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import bpy

# The last lookup or store, for the caller to record against whatever it is doing. A benchmark
# row that was answered from a full view should be able to say which picture answered it, and
# "the cache" is not an answer: entries are shared between runs and can be overwritten.
LAST_PROVENANCE = {}

MODE = os.environ.get("SCOPE_PANO_CACHE_MODE", "use").strip().lower()
_VALID_MODES = ("use", "write", "off")


def cache_dir(default_root=None):
    d = os.environ.get("SCOPE_PANO_CACHE")
    if d:
        return Path(d)
    root = default_root or os.environ.get("SCOPE_OUTPUT_DIR") or "."
    return Path(root) / "pano_cache"


def _scene_stem():
    return os.path.splitext(os.path.basename(bpy.data.filepath))[0] or "unsaved"


def _pose_key(cam):
    """Preset name where one was applied, otherwise a hash of the actual pose.

    The preset name is preferred because it is exact and a human can read it in a filename.
    The hash is quantised, to five decimal places on position and rotation and three on the
    focal length, so that a pose restored from the same preset file hashes the same way rather
    than missing the cache on floating point noise.
    """
    try:
        from . import preset_helpers
        name = preset_helpers.last_applied_preset()
    except Exception:
        name = None
    if name:
        return f"preset-{name}", {"preset": name}
    loc = tuple(round(v, 5) for v in cam.location)
    rot = tuple(round(math.degrees(v), 5) for v in cam.rotation_euler)
    lens = round(cam.data.lens, 3)
    raw = json.dumps([loc, rot, lens], sort_keys=True)
    return "pose-" + hashlib.sha1(raw.encode()).hexdigest()[:12], {
        "location": loc, "rotation_deg": rot, "lens_mm": lens}


def _shading():
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            for sp in area.spaces:
                if sp.type == "VIEW_3D":
                    t = sp.shading.type
                    if t == "SOLID":
                        try:
                            return f"SOLID+{sp.shading.color_type}"
                        except AttributeError:
                            return "SOLID"
                    return t
    return "unknown"


def entry_paths(root=None):
    """(image, metadata) paths for the current scene and camera pose."""
    cam = bpy.context.scene.camera
    key, _ = _pose_key(cam)
    d = cache_dir(root)
    base = f"{_scene_stem()}__{key}"
    return d / f"{base}.png", d / f"{base}.json"


POSE_TOL_LOC = 1e-4      # metres
POSE_TOL_ROT = 1e-3      # degrees
POSE_TOL_LENS = 1e-3     # millimetres


def _actual_pose():
    cam = bpy.context.scene.camera
    if cam is None:
        return None
    return {"location": [round(v, 5) for v in cam.location],
            "rotation_deg": [round(math.degrees(v), 5) for v in cam.rotation_euler],
            "lens_mm": round(cam.data.lens, 3)}


def _pose_matches(stored):
    """Is the camera now where the cached panorama was taken from?

    This is the check that makes the cache safe rather than merely fast. Everything else is
    bookkeeping that can go stale: a preset name recorded in one scene and read in another, a
    filename that happens to collide, a process that moved the camera without saying so. The
    pose cannot go stale, because it is read from the camera at the moment of the lookup.
    """
    now = _actual_pose()
    if not stored or not now:
        return False
    try:
        for a, b in zip(stored["location"], now["location"]):
            if abs(a - b) > POSE_TOL_LOC:
                return False
        for a, b in zip(stored["rotation_deg"], now["rotation_deg"]):
            if abs(a - b) > POSE_TOL_ROT:
                return False
        return abs(stored["lens_mm"] - now["lens_mm"]) <= POSE_TOL_LENS
    except (KeyError, TypeError):
        return False


def lookup(root=None, require_shading=True):
    """A cached panorama for where the camera is now, or None.

    Returns None rather than raising for every miss, including a corrupt or half-written entry,
    because a miss only costs a sweep and a wrong hit costs a wrong answer.
    """
    if MODE in ("off", "write"):
        return None
    img, meta = entry_paths(root)
    if not (img.exists() and meta.exists()):
        return None
    try:
        m = json.loads(meta.read_text())
    except (OSError, ValueError):
        return None

    # Scene identity by filename, not by absolute path, so that a cache captured on one machine
    # and shipped with the repository still works on another. The absolute path is kept in the
    # metadata for provenance but cannot be a validity condition: it differs on every checkout.
    # Cross-scene confusion is still prevented, because the filename is in the key and checked
    # here, and because the pose check below would fail anyway.
    if m.get("blend"):
        if os.path.basename(m["blend"]) != os.path.basename(bpy.data.filepath):
            return None

    # The guarantee. Everything above this line is a hint.
    if not _pose_matches(m.get("pose_exact")):
        return None

    if require_shading and m.get("shading") and m["shading"] != _shading():
        # A panorama captured in Solid is not a substitute for one captured in Material
        # Preview: Solid draws glass as an opaque surface, and on some scenes it loses the
        # texture colour as well. Better to sweep again than to answer from the wrong mode.
        return None
    # Modification time only means something on the machine that wrote the entry. A shipped
    # cache is checked out with whatever mtime git gives it, which is not the mtime the scene
    # had when the panorama was captured, so comparing them would reject every shipped entry.
    same_machine = m.get("blend") == bpy.data.filepath
    if same_machine and m.get("blend_mtime") and os.path.exists(bpy.data.filepath):
        if abs(m["blend_mtime"] - os.path.getmtime(bpy.data.filepath)) > 1.0:
            return None
    _record(str(img), m, hit=True)
    return str(img)


def store(stitched_path, root=None, extra=None):
    """Copy a freshly stitched panorama into the cache. Returns the cached path, or None."""
    if MODE == "off":
        return None
    if not stitched_path or not os.path.exists(stitched_path):
        return None
    img, meta = entry_paths(root)
    img.parent.mkdir(parents=True, exist_ok=True)
    cam = bpy.context.scene.camera
    _, pose = _pose_key(cam)
    m = {"scene": _scene_stem(), "blend": bpy.data.filepath,
         "blend_mtime": (os.path.getmtime(bpy.data.filepath)
                         if os.path.exists(bpy.data.filepath) else None),
         "pose": pose, "pose_exact": _actual_pose(),
         "shading": _shading(), "source": stitched_path,
         "written_by_pid": os.getpid()}
    if extra:
        m.update(extra)

    # Atomic, because several benchmark runs may share one cache directory. A reader in another
    # process must see either the previous entry or the new one, never a half-copied PNG.
    # Unique suffixes keep two simultaneous writers from clobbering each other's temporary.
    tag = f".tmp{os.getpid()}"
    tmp_img, tmp_meta = img.with_suffix(img.suffix + tag), meta.with_suffix(meta.suffix + tag)
    try:
        shutil.copy2(stitched_path, tmp_img)
        tmp_meta.write_text(json.dumps(m, indent=2))
        # Image first: a reader requires both, so an image without metadata is a miss rather
        # than a wrong hit.
        os.replace(tmp_img, img)
        os.replace(tmp_meta, meta)
    except OSError:
        for t in (tmp_img, tmp_meta):
            try:
                t.unlink()
            except OSError:
                pass
        return None
    _record(str(img), m, hit=False)
    return str(img)


def file_digest(path, limit=None):
    """SHA-256 of a file, so a recorded panorama can be identified later by content.

    A path is not identification. Cache entries are shared between concurrent runs and can be
    overwritten by a run using SCOPE_PANO_CACHE_MODE=write, so a result that cites only a path
    can silently come to refer to a different picture than the one that produced it.
    """
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            read = 0
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
                read += len(chunk)
                if limit and read >= limit:
                    break
    except OSError:
        return None
    return h.hexdigest()


def _record(path, meta, hit):
    """Remember where the panorama just served came from."""
    global LAST_PROVENANCE
    try:
        from . import preset_helpers
        preset = preset_helpers.last_applied_preset()
    except Exception:
        preset = None
    LAST_PROVENANCE = {
        "path": path,
        "sha256": file_digest(path),
        "from_cache": bool(hit),
        "scene": _scene_stem(),
        "blend": bpy.data.filepath,
        "preset": preset,
        "pose": _actual_pose(),
        "shading": _shading(),
        "captured_by_pid": (meta or {}).get("written_by_pid"),
        "captured_ts": (meta or {}).get("captured_ts"),
        "reader_pid": os.getpid(),
    }


def take_provenance():
    """The provenance of the last panorama served, and clear it.

    Cleared on read so that a row which did not ask for a full view cannot inherit the record
    of one that did. A stale provenance attached to the wrong row is worse than none: it
    reads as evidence.
    """
    global LAST_PROVENANCE
    p, LAST_PROVENANCE = LAST_PROVENANCE, {}
    return p

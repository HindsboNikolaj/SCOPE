"""Is this frame safe to send to the vision model?

Three questions, answered from one 64x64 greyscale copy:

  blank        nothing was drawn at all
  under-drawn  geometry is there but the textures have not finished loading
  fine         send it

A fourth statistic, colour, is measured but deliberately does NOT produce a per-frame verdict.
A single frame cannot tell "the world rendered grey" from "the model looked at something grey".
In aggregate it can, and run_colour_health() is where that question gets asked -- it is what
catches a scene whose materials never loaded, which is how a flat grey render of city-street
came to ship in benchmark/preset_views and be used as evidence about model failures.

The distinction between the middle case and the others is the point. Material Preview loads
textures asynchronously, so a capture taken too soon after a camera move shows correct geometry
with flat, untextured surfaces — brick without brick, signage without letters. It is not blurry
in the optical sense; there is no lens and no defocus in an OpenGL viewport. It is unfinished,
and unlike a blank frame it fixes itself if you wait a moment and look again.

That is why the two get different treatment. A blank frame is redrawn immediately, because a
dropped draw does not improve with time. An under-drawn frame is waited on, briefly, because
that is exactly what it needs.

COST. The maths is 0.06 ms on a 64x64 array. What is expensive is decoding the PNG: a
3240x3240 frame costs 510 ms, and the previous guard paid that twice per frame. So the probe
is computed from the PIL image the capture path already holds in memory and cached by path;
nothing re-decodes a file it just wrote. Call `probe_image` if you have the image, `probe_path`
only if you do not.
"""
from __future__ import annotations

import os

_N = 64  # side of the analysis copy; 64x64 is enough to separate all three cases
_C = 24  # side of the colour sample; colour is a whole-frame property, so it can be coarser

# Blank: near-zero variation, or only a handful of distinct greys.
BLANK_STD = float(os.environ.get("SCOPE_BLANK_STD", "3.0"))
BLANK_LEVELS = int(os.environ.get("SCOPE_BLANK_LEVELS", "4"))
# Under-drawn: geometry present, texture detail absent. Measured as mean absolute gradient,
# which is high for brick, foliage and lettering and low for flat-shaded surfaces.
DETAIL_MIN = float(os.environ.get("SCOPE_DETAIL_MIN", "3.0"))
# A frame can also be legitimately plain (a blank wall filling the view). Only treat low detail
# as unfinished when there is enough tonal range to imply a real scene behind it.
DETAIL_STD_MIN = float(os.environ.get("SCOPE_DETAIL_STD_MIN", "8.0"))
# What counts as "no colour" in one frame. Calibrated by measuring mean channel spread over a
# 600-frame random sample of the 2,502 frames the campaign captured: median 12.1, p25 7.8,
# p05 3.0, and for frames independently confirmed healthy the 5th percentile is 5.7. The floor
# sits at 3.0, below the healthy 5th percentile, so only unambiguous frames count as grey.
# This feeds the run-level aggregate, not a per-frame verdict -- see classify().
COLOUR_MIN = float(os.environ.get("SCOPE_COLOUR_MIN", "3.0"))
# Run level. One grey frame is noise; a whole scene rendering grey means the materials never
# loaded, which is what happened to the shipped benchmark/preset_views -- city-street__mailbox
# was published at a colourfulness of 3.1 while the same scene captured live measures 21.7.
# The floor is half the observed median, which no healthy run has come close to.
RUN_COLOUR_FLOOR = float(os.environ.get("SCOPE_RUN_COLOUR_FLOOR", "6.0"))
RUN_GREY_FRACTION = float(os.environ.get("SCOPE_RUN_GREY_FRACTION", "0.25"))

_CACHE: dict[str, tuple] = {}
_CACHE_MAX = 512


def probe_image(img):
    """(std, distinct_levels, detail, colour) from a PIL image. ~0.1 ms. Never raises."""
    try:
        import numpy as np
        # The luma path is unchanged, deliberately, down to the resample filter: DETAIL_MIN and
        # BLANK_STD were calibrated against real captured frames through exactly this call, and
        # computing the same statistic a cheaper way would silently move every threshold.
        a = np.asarray(img.convert("L").resize((_N, _N)), dtype=np.float32)
        sd = float(a.std())
        levels = int(np.count_nonzero(np.bincount(a.astype(np.uint8).ravel(), minlength=256)))
        detail = float((abs(a[:, 1:] - a[:, :-1]).mean() + abs(a[1:] - a[:-1]).mean()) / 2)
        # Colour comes off its own, much smaller, nearest-neighbour sample. Resizing three
        # channels the way the luma path resizes one cost 1.7 ms on its own and blew the 1 ms
        # budget; at 24x24 with NEAREST the cost is negligible. Sub-sampling is sound here
        # because "did anything render in colour" is a whole-frame property -- unlike detail,
        # it does not depend on neighbouring pixels, so no averaging is needed to measure it.
        s = np.asarray(img.resize((_C, _C), 0).convert("RGB"), dtype=np.float32)
        colour = float((s.max(axis=2) - s.min(axis=2)).mean())
        return sd, levels, detail, colour
    except Exception:
        # numpy absent or the image is unreadable. Say "fine" rather than block a run on the
        # probe itself; a guard that can fail closed is worse than no guard.
        return None, None, None, None


def remember(path, stats):
    """Cache a probe against the path the capture path is about to write."""
    if stats and stats[0] is not None:
        if len(_CACHE) > _CACHE_MAX:
            _CACHE.clear()
        _CACHE[str(path)] = stats


def probe_path(path):
    """Cached probe, decoding only if the capture path did not leave one."""
    p = str(path)
    if p in _CACHE:
        return _CACHE[p]
    try:
        from PIL import Image
        with Image.open(p) as im:
            s = probe_image(im)
    except Exception:
        return None, None, None, None
    remember(p, s)
    return s


def classify(stats):
    """'blank' | 'under-drawn' | 'fine' | 'unknown'."""
    sd, levels, detail, colour = (tuple(stats) + (None,))[:4]
    if sd is None:
        return "unknown"
    if sd < BLANK_STD or levels <= BLANK_LEVELS:
        return "blank"
    if detail < DETAIL_MIN and sd >= DETAIL_STD_MIN:
        return "under-drawn"
    # NO per-frame colour verdict, deliberately. It was written and then removed, because it
    # earned nothing and cost accuracy: over the 2,502 frames the campaign captured it changed
    # exactly one verdict, while flagging a legitimately dark night scene -- lit windows, kerb
    # edges, real structure -- as a fault. A single frame simply cannot distinguish "the world
    # rendered grey" from "the model looked at something grey", and a zoom onto a grey wall is
    # a correct picture of what was asked for.
    #
    # The colour statistic is still measured and returned, because it IS decisive in aggregate.
    # That check is run_colour_health() below, and it is where the real defect was caught.
    return "fine"


def colour_of(stats):
    """Channel spread from a probe tuple, or None for older 3-wide cached entries."""
    return stats[3] if len(stats) > 3 else None


def run_colour_health(colours):
    """Did this run render in colour at all?  -> (ok, detail_string)

    The check that matters, and the one nothing was doing. A per-frame verdict cannot tell a
    grey wall from a grey world, but a whole run can: if most of a run's frames carry no
    material colour, the scene loaded without its textures and every answer from it is being
    graded against a picture the benchmark did not intend to show.

    This is exactly how the shipped preset_views slipped through -- one scene at colourfulness
    3.1, published, and used as evidence on the audit page for weeks.
    """
    vals = [c for c in colours if c is not None]
    if not vals:
        return True, "no colour statistics recorded"
    vals.sort()
    median = vals[len(vals) // 2]
    grey = sum(1 for c in vals if c < COLOUR_MIN) / len(vals)
    ok = median >= RUN_COLOUR_FLOOR and grey <= RUN_GREY_FRACTION
    return ok, (f"n={len(vals)} median_colour={median:.1f} "
                f"(floor {RUN_COLOUR_FLOOR}) grey_fraction={grey:.1%} "
                f"(max {RUN_GREY_FRACTION:.0%})")


# --------------------------------------------------------------------------- spot-check samples
#
# The full frames are 3240x3240, about 23 MB each, and 2,506 of them had accumulated to 59 GB.
# Keeping every frame is not an option and keeping none makes a run unauditable after the fact.
# So: keep a thumbnail of every Nth frame plus EVERY frame the probe was unhappy with, which is
# the sample anyone spot-checking actually wants. At the default of one in 25, a 550-frame run
# leaves about 22 thumbnails of roughly 60 KB, so a whole run is auditable for under 2 MB.
SAMPLE_EVERY = int(os.environ.get("SCOPE_SAMPLE_EVERY", "25"))
SAMPLE_EDGE = int(os.environ.get("SCOPE_SAMPLE_EDGE", "512"))
SAMPLE_DIR = os.environ.get("SCOPE_SAMPLE_DIR", "")
_seen = [0]


def maybe_sample(path, img, stats):
    """Write a small JPEG copy if this frame is worth being able to look at later.

    Always samples anything the probe did not call 'fine', because a blank or under-drawn frame
    is precisely the one somebody will want to see. Never raises: losing a thumbnail must not
    lose a run.
    """
    if SAMPLE_EVERY <= 0:
        return
    try:
        _seen[0] += 1
        kind = classify(stats)
        if kind == "fine" and _seen[0] % SAMPLE_EVERY:
            return
        import pathlib as _pl
        src = _pl.Path(path)
        out = _pl.Path(SAMPLE_DIR) if SAMPLE_DIR else src.parent / "samples"
        out.mkdir(parents=True, exist_ok=True)
        thumb = img.convert("RGB").copy()
        thumb.thumbnail((SAMPLE_EDGE, SAMPLE_EDGE))
        tag = "fine" if kind == "fine" else kind
        thumb.save(out / f"{src.stem}__{tag}.jpg", "JPEG", quality=80, optimize=True)
    except Exception:
        pass

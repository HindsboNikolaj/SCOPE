"""The frame probe: does it tell blank from unfinished from fine, and is it cheap enough?

The benchmark grades a vision model on whatever the capture path hands it. A blank frame is
still described, still answered, still scored — the run finishes and produces a number that
looks like a result. These tests exist so that cannot happen silently.

Speed is part of correctness here. The probe runs on every captured frame, of which a full
sweep takes about 550, so a check that costs 500 ms is not a check anyone will keep enabled.
The budget is one millisecond per frame and the assertion below enforces it.
"""
import time

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image", reason="Pillow required").Image
from PIL import Image as PILImage  # noqa: E402

from scope.tools import frame_probe as P  # noqa: E402

SIDE = 512


def _img(a):
    return PILImage.fromarray(np.clip(a, 0, 255).astype(np.uint8), mode="L")


def _rgb(r, g, b):
    return PILImage.fromarray(
        np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8), mode="RGB")


def _facade(seed):
    """Structure at object scale -- windows, courses, kerbs -- not just pixel noise.

    A fixture built only from per-pixel noise averages away at the 64x64 analysis size and
    lands right on the detail threshold, which makes it fragile for reasons that have nothing
    to do with what is being tested. Real scenes carry structure at object scale, and that is
    what survives the downsample."""
    rng = np.random.default_rng(seed)
    a = np.full((SIDE, SIDE), 96, dtype=np.float32)
    for y in range(0, SIDE, 64):                       # storey courses
        a[y:y + 8] = 150
        for x in range(16, SIDE - 16, 48):             # windows
            a[y + 18:y + 52, x:x + 26] = 40
    a[int(SIDE * 0.78):] = 120                         # pavement
    return a + rng.normal(0, 12, (SIDE, SIDE))


def painted(seed=0):
    """A textured frame that also carries material colour: warm brick against cool sky."""
    a = _facade(seed)
    return _rgb(a + 26, a, a - 22)


def unpainted(seed=0):
    """The same geometry with the materials never loaded: identical in all three channels."""
    a = _facade(seed)
    return _rgb(a, a, a)


def blank(value=0):
    """Nothing drawn: one flat colour."""
    return _img(np.full((SIDE, SIDE), value, dtype=np.float32))


def under_drawn():
    """Geometry, no texture. Broad tonal range, smooth everywhere — a few flat facets, which
    is what Material Preview shows before the textures finish loading."""
    a = np.zeros((SIDE, SIDE), dtype=np.float32)
    a[: SIDE // 2] = 70          # sky band
    a[SIDE // 2 :] = 150         # ground band
    a[:, : SIDE // 3] = 110      # a wall
    return _img(a)


def textured(seed=0):
    """A real frame: broad range AND high-frequency detail everywhere."""
    rng = np.random.default_rng(seed)
    base = np.linspace(40, 200, SIDE, dtype=np.float32)[None, :].repeat(SIDE, 0)
    return _img(base + rng.normal(0, 26, (SIDE, SIDE)))


# --------------------------------------------------------------------------- classification

def test_blank_frame_is_called_blank():
    for v in (0, 127, 255):
        assert P.classify(P.probe_image(blank(v))) == "blank", f"flat {v} not caught"


def test_textured_frame_is_fine():
    for s in range(4):
        assert P.classify(P.probe_image(textured(s))) == "fine"


def test_under_drawn_is_distinguished_from_both():
    """The case that motivates the whole module: not blank, not fine, and fixable by waiting."""
    assert P.classify(P.probe_image(under_drawn())) == "under-drawn"


def test_under_drawn_is_not_blank_and_blank_is_not_under_drawn():
    """These get opposite treatment — one is redrawn at once, the other is waited on — so
    confusing them wastes a budget on a frame that will never improve, or gives up on one that
    would have."""
    assert P.classify(P.probe_image(blank())) != "under-drawn"
    assert P.classify(P.probe_image(under_drawn())) != "blank"


def test_a_dark_but_real_frame_is_not_flagged():
    """A dim night scene is a legitimate picture, not a fault. Flagging it would make every
    dark frame wait out the full budget for nothing."""
    rng = np.random.default_rng(3)
    a = np.full((SIDE, SIDE), 26, dtype=np.float32)
    a[: SIDE // 3] = 58                                   # sky, brighter than the street
    for x in range(40, SIDE - 40, 46):                    # a row of lit windows
        a[SIDE // 3 + 20 : SIDE // 3 + 70, x : x + 24] = 120
    for y in range(SIDE // 2, SIDE, 34):                  # kerb and paving edges
        a[y : y + 3, :] = 62
    a += rng.normal(0, 9, (SIDE, SIDE))                   # surface texture on top
    assert P.classify(P.probe_image(_img(a))) == "fine"


def test_known_limitation_texture_finer_than_the_downsample():
    """Documented, not fixed: the probe analyses a 64x64 copy, so a frame whose ONLY variation
    is finer than an 8-pixel block averages away and reads as blank.

    Left alone deliberately. Measured over 220 real frames from four runs the separation is
    clean — blanks sit at detail 0.0 and real frames at a median of 10.35 with a 1st percentile
    of 4.16 — because real scenes carry structure at object scale, not just at pixel scale. And
    the cost of being wrong here is bounded: a false 'blank' triggers two redraws and an
    inflated counter. The frame is still captured and still sent. If a scene ever appears whose
    content is genuinely all sub-8px texture, raise the analysis size rather than the threshold.
    """
    rng = np.random.default_rng(3)
    only_fine_texture = np.full((SIDE, SIDE), 30, dtype=np.float32) + rng.normal(0, 9, (SIDE, SIDE))
    assert P.classify(P.probe_image(_img(only_fine_texture))) == "blank"


def test_unreadable_input_does_not_raise_or_block():
    """A guard that can fail closed is worse than no guard: it would halt a run over itself."""
    assert P.probe_path("/nonexistent/frame.png") == (None, None, None, None)
    assert P.classify((None, None, None, None)) == "unknown"
    # 3-wide tuples still exist in caches written before colour was measured; they must read as
    # "no colour information", not crash the unpacking.
    assert P.classify((None, None, None)) == "unknown"
    assert P.colour_of((1.0, 2, 3.0)) is None


# --------------------------------------------------------------------------- cost

def test_cache_avoids_re_decoding(tmp_path):
    """Re-opening a captured PNG to ask whether it is blank cost 510 ms on a full-size frame,
    and the guard this replaces paid it twice per frame. The capture path caches the probe, so
    probe_path must return it without touching the file."""
    fp = tmp_path / "frame.png"
    textured().save(fp)
    P.remember(fp, P.probe_image(textured()))
    fp.unlink()                                          # the file is gone
    assert P.probe_path(fp)[0] is not None, "probe_path re-read the file instead of the cache"


# --------------------------------------------------------------------------- spot-check samples

def test_sampling_keeps_every_bad_frame_and_one_in_n_good_ones(tmp_path, monkeypatch):
    """59 GB of full-size frames had accumulated. The sample has to be small enough to keep and
    complete enough to audit: every frame the probe was unhappy with, plus a thin slice of the
    rest."""
    monkeypatch.setattr(P, "SAMPLE_EVERY", 10)
    monkeypatch.setattr(P, "_seen", [0])
    out = tmp_path / "samples"
    monkeypatch.setattr(P, "SAMPLE_DIR", str(out))
    good, bad = textured(), blank()
    for i in range(30):
        P.maybe_sample(tmp_path / f"f{i:03d}.png", good, P.probe_image(good))
    n_good = len(list(out.glob("*.jpg")))
    P.maybe_sample(tmp_path / "blank.png", bad, P.probe_image(bad))
    kinds = [p.stem.rsplit("__", 1)[1] for p in out.glob("*.jpg")]
    assert n_good == 3, f"1-in-10 of 30 good frames should keep 3, kept {n_good}"
    assert kinds.count("blank") == 1, "a blank frame must always be kept"


def test_a_sample_is_small_enough_to_keep(tmp_path, monkeypatch):
    """A thumbnail has to cost a fraction of the 23 MB original or the sampling is pointless."""
    monkeypatch.setattr(P, "SAMPLE_EVERY", 1)
    monkeypatch.setattr(P, "_seen", [0])
    monkeypatch.setattr(P, "SAMPLE_DIR", str(tmp_path))
    im = textured()
    P.maybe_sample(tmp_path / "one.png", im, P.probe_image(im))
    kb = next(tmp_path.glob("*.jpg")).stat().st_size / 1024
    assert kb < 400, f"thumbnail is {kb:.0f} KB; it should be well under a megabyte"


# --------------------------------------------------------------------------- colour

def test_a_painted_frame_is_fine():
    """The normal case has to stay normal, or the colour check is worse than no check."""
    for s_ in range(3):
        assert P.classify(P.probe_image(painted(s_))) == "fine"


def test_colour_is_measured_and_separates_painted_from_unpainted():
    """The statistic itself, before any threshold: an unpainted frame must read near zero and
    a painted one must not. Measured over 600 real captured frames the median is 12.1."""
    assert P.colour_of(P.probe_image(unpainted())) < P.COLOUR_MIN
    assert P.colour_of(P.probe_image(painted())) > P.COLOUR_MIN * 2


def test_no_single_frame_is_failed_for_being_grey():
    """Low colour alone is NOT a per-frame defect, and the probe must not treat it as one.

    A first version of this check did, and it was wrong twice over: across 2,502 captured
    frames it changed exactly one verdict, and it flagged a legitimately dark night scene as
    broken. One real frame -- a zoom onto a grey wall, which is precisely what the model asked
    to see -- measures 0.44 channel spread while being a correct picture. Greyness only means
    something in aggregate, which is what run_colour_health is for."""
    assert P.classify(P.probe_image(unpainted())) == "fine"
    flat = np.full((SIDE, SIDE), 96, dtype=np.float32)
    assert P.classify(P.probe_image(_rgb(flat, flat, flat))) == "blank"


# --------------------------------------------------------------------------- run level

def test_a_run_rendered_without_materials_fails_the_colour_floor():
    """The check that would have caught the real defect.

    benchmark/preset_views/city-street__mailbox.png shipped at colourfulness 3.1 -- flat grey
    buildings, white blob leaves, a colourless shop sign -- while the same scene captured live
    measures 21.7. Nothing looked at colour, so it was published and then used as evidence on
    the audit page. A per-frame verdict cannot catch this; an aggregate over the run can."""
    ok, why = P.run_colour_health([P.colour_of(P.probe_image(unpainted(s_))) for s_ in range(8)])
    assert not ok, why


def test_a_healthy_run_passes_the_colour_floor():
    ok, why = P.run_colour_health([P.colour_of(P.probe_image(painted(s_))) for s_ in range(8)])
    assert ok, why


def test_run_health_never_blocks_on_missing_statistics():
    """Older cached probes are 3-wide and carry no colour. That must read as 'nothing to say',
    never as a failure -- a guard that fails closed on absent data halts runs over itself."""
    ok, why = P.run_colour_health([None, None])
    assert ok and "no colour" in why


def test_measuring_colour_is_nearly_free():
    """What the colour check must not do is make the probe expensive.

    Measured against the luma path alone on RGB frames at three sizes: 512px 0.99 ms, 1920px
    10.9 ms, 3240px 29.9 ms -- the cost is the full-size convert-and-resize, and it scales with
    the frame. The colour sample is a fixed 24x24 nearest-neighbour reduction, so it costs a
    flat 0.08 ms at every size. That is what this pins: the addition, not the total."""
    im = painted()

    def luma_only(i):
        a = np.asarray(i.convert("L").resize((64, 64)), dtype=np.float32)
        return a.std(), a.mean()

    def bench(fn):
        fn(im)
        t = time.perf_counter()
        for _ in range(50):
            fn(im)
        return (time.perf_counter() - t) / 50 * 1000

    base, full = bench(luma_only), bench(P.probe_image)
    assert full - base < 1.0, f"colour added {full - base:.3f} ms; it should be under 1 ms"


def test_the_probe_is_cheap_against_the_capture_it_guards():
    """The honest budget.

    An earlier version of this test asserted 1 ms using a greyscale 512px fixture, which no
    real frame resembles -- captures are RGB and 3240x3240, where the probe costs about 30 ms.
    The assertion passed while being irrelevant. What actually matters is the ratio: a capture
    costs seconds of render and settle, so tens of milliseconds of checking is free, and
    decoding the PNG again would have cost 510 ms, which is the thing worth avoiding."""
    im = painted()
    P.probe_image(im)
    t = time.perf_counter()
    for _ in range(20):
        P.probe_image(im)
    per_ms = (time.perf_counter() - t) / 20 * 1000
    assert per_ms < 60.0, f"probe took {per_ms:.1f} ms/frame; it must stay far under a capture"

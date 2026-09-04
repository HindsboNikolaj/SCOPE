"""Which stored full view gets served, and how it is re-centred.

This is the arithmetic behind a decision that has to be right on a shared machine: given where
the camera is now, is a panorama on disk the correct answer, and if so how much does it need
turning to face the way the camera faces.

The rule being tested is that the full view belongs to the camera's **position**. A PTZ camera
is bolted to a wall; it pans, tilts and zooms but does not move. So a sweep taken from a
position stays correct however the camera has since been turned or zoomed. Matching the whole
pose modelled a camera that teleports and made a question that panned thirty degrees pay for a
fresh sweep with the right answer already on disk.
"""
import numpy as np
import pytest
from PIL import Image


BASE = {"location": [1.83471, 2.63802, 3.40196],
        "rotation_deg": [91.54141, 0.0, 228.61675],
        "lens_mm": 23.68}


def moved(**kw):
    d = {k: list(v) if isinstance(v, list) else v for k, v in BASE.items()}
    d.update(kw)
    return d


class TestPositionMatching:
    def test_same_pose_matches(self, pano_cache):
        assert pano_cache._same_position(BASE, moved())

    @pytest.mark.parametrize("yaw", [228.61675 + 30, 228.61675 - 45, 228.61675 + 179, 0.0])
    def test_turning_the_camera_still_matches(self, pano_cache, yaw):
        """The whole point: a sweep covers every heading from that spot."""
        assert pano_cache._same_position(BASE, moved(rotation_deg=[91.54141, 0.0, yaw]))

    @pytest.mark.parametrize("pitch", [86.5, 100.0, 45.0])
    def test_tilting_the_camera_still_matches(self, pano_cache, pitch):
        """A tilt does not move the camera, so the full view from there is unchanged."""
        assert pano_cache._same_position(BASE, moved(rotation_deg=[pitch, 0.0, 228.61675]))

    @pytest.mark.parametrize("lens", [47.0, 12.0, 200.0])
    def test_zooming_still_matches(self, pano_cache, lens):
        assert pano_cache._same_position(BASE, moved(lens_mm=lens))

    @pytest.mark.parametrize("delta", [0.01, -0.01, 1.0])
    def test_moving_the_camera_does_not_match(self, pano_cache, delta):
        """The one thing that does invalidate it. A centimetre is enough."""
        loc = list(BASE["location"]); loc[0] += delta
        assert not pano_cache._same_position(BASE, moved(location=loc))

    def test_missing_fields_do_not_match_rather_than_raise(self, pano_cache):
        assert not pano_cache._same_position({}, moved())
        assert not pano_cache._same_position(BASE, {})


class TestYawDelta:
    @pytest.mark.parametrize("yaw,expect", [
        (228.61675, 0.0),
        (258.61675, 30.0),
        (183.61675, -45.0),
        (218.61675, -10.0),
    ])
    def test_delta_is_the_turn_since_the_sweep(self, pano_cache, yaw, expect):
        d = pano_cache._yaw_delta(BASE, moved(rotation_deg=[91.54141, 0.0, yaw]))
        assert d == pytest.approx(expect, abs=1e-6)

    def test_delta_takes_the_short_way_round(self, pano_cache):
        """350 degrees clockwise is 10 degrees anticlockwise, and the roll must agree."""
        d = pano_cache._yaw_delta(BASE, moved(rotation_deg=[91.54141, 0.0, 228.61675 + 350]))
        assert -180 < d <= 180
        assert d == pytest.approx(-10.0, abs=1e-6)

    def test_delta_of_exactly_half_a_turn_is_bounded(self, pano_cache):
        """It comes back as -180, not +180. Recorded because it is easy to assume otherwise."""
        d = pano_cache._yaw_delta(BASE, moved(rotation_deg=[91.54141, 0.0, 228.61675 + 180]))
        assert -180 <= d < 180
        assert abs(d) == pytest.approx(180.0)


class TestRoll:
    """The roll that re-centres a stored panorama on the camera's current heading."""

    @staticmethod
    def roll(a, deg):
        return np.roll(a, -int(round(a.shape[1] * (deg / 360.0))), axis=1)

    @pytest.fixture
    def strip(self):
        # a 360-wide strip whose columns are all distinct, so a roll is measurable exactly
        a = np.zeros((8, 360, 3), dtype=np.uint8)
        a[:, :, 0] = np.arange(360, dtype=np.uint8)[None, :]
        return a

    @pytest.mark.parametrize("deg", [30.0, -45.0, 90.0, 180.0, -179.0])
    def test_rolling_is_reversible(self, strip, deg):
        """Exact rather than approximate, because a full sweep closes on itself."""
        assert np.array_equal(self.roll(self.roll(strip, deg), -deg), strip)

    def test_a_full_turn_is_the_identity(self, strip):
        assert np.array_equal(self.roll(strip, 360.0), strip)

    def test_the_shift_is_proportional_to_the_angle(self, strip):
        """90 degrees of a 360-column strip is 90 columns."""
        rolled = self.roll(strip, 90.0)
        assert rolled[0, 0, 0] == strip[0, 90, 0]

    def test_half_a_turn_rolls_the_same_either_way(self, strip):
        """Which is why the sign of a 180 degree delta does not matter."""
        assert np.array_equal(self.roll(strip, 180.0), self.roll(strip, -180.0))

    def test_no_pixels_are_lost(self, strip):
        rolled = self.roll(strip, 47.0)
        assert sorted(rolled[0, :, 0].tolist()) == sorted(strip[0, :, 0].tolist())

    def test_rolling_a_real_panorama_preserves_it(self, panorama_dir):
        import glob, os
        p = sorted(glob.glob(os.path.join(panorama_dir, "*.png")))[0]
        a = np.asarray(Image.open(p).convert("RGB"))
        back = self.roll(self.roll(a, 30.0), -30.0)
        assert np.array_equal(back, a)

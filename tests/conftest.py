"""Shared fixtures, and a stub Blender so the pure logic can be tested without one.

Most of what can go wrong in SCOPE is not inside Blender. Whether a stored panorama matches
where the camera is, whether two captures show the same place, whether a shipped file is the
file its manifest describes: all of that is arithmetic over data, and arithmetic can be tested
in a second on a laptop with no GPU and no 3D application installed.

The parts that genuinely need Blender are marked `needs_blender` and skip when `bpy` cannot be
imported, so the suite is honest about what it did and did not check rather than silently
covering less than it appears to.

The stub exists for one reason: `scope.blender.panorama_cache` imports `bpy` at module scope,
so its pose arithmetic is unreachable from a plain interpreter even though none of that
arithmetic touches Blender. Rather than restructure shipping code to suit a test, the test
supplies a `bpy` that is enough to import against.
"""
import os
import sys
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))


@pytest.fixture(scope="session")
def repo():
    return REPO


@pytest.fixture(scope="session")
def panorama_dir(repo):
    d = os.path.join(repo, "benchmark", "panoramas")
    if not os.path.isdir(d):
        pytest.skip("benchmark/panoramas is not present in this checkout")
    return d


def _install_bpy_stub():
    """A `bpy` with just enough shape to import against. Never used to simulate Blender."""
    if "bpy" in sys.modules:
        return
    bpy = types.ModuleType("bpy")

    class _Data:
        filepath = ""
        images = []

    class _Camera:
        location = (0.0, 0.0, 0.0)
        rotation_euler = (0.0, 0.0, 0.0)

        class data:
            lens = 50.0

    class _Scene:
        camera = _Camera()

    class _Screen:
        areas = []

    class _Context:
        scene = _Scene()
        screen = _Screen()

    bpy.data = _Data()
    bpy.context = _Context()
    bpy.app = types.SimpleNamespace(handlers=types.SimpleNamespace(load_post=[], persistent=lambda f: f))
    bpy.utils = types.SimpleNamespace(preset_paths=lambda _k: [], user_resource=lambda *a, **k: "")
    bpy.path = types.SimpleNamespace(abspath=lambda p: p)
    bpy.ops = types.SimpleNamespace()
    sys.modules["bpy"] = bpy


@pytest.fixture(scope="session")
def pano_cache():
    """scope.blender.panorama_cache, imported against the stub."""
    _install_bpy_stub()
    from scope.blender import panorama_cache
    return panorama_cache


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_blender: requires a real bpy, skipped without one")


def pytest_runtest_setup(item):
    if "needs_blender" in item.keywords:
        if "bpy" in sys.modules and getattr(sys.modules["bpy"], "__file__", None) is None:
            pytest.skip("only a stub bpy is available")
        try:
            import bpy  # noqa: F401
        except ImportError:
            pytest.skip("requires Blender's bpy")

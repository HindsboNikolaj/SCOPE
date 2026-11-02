#!/usr/bin/env python3
"""Install the runner's Python packages into Blender's own Python, then check them.

Blender runs the benchmark (src/scope/eval/runner.py) with the Python it ships with, not the
virtual environment you installed SCOPE into, so `pip install -e .` alone leaves the runner
without openai, yaml and friends. This script asks Blender where its interpreter is
(`sys.executable` inside Blender), so it works the same on Linux, macOS and Windows.

Packages go into Blender's user "modules" folder (`bpy.utils.user_resource('SCRIPTS',
path='modules')`), which Blender always puts on sys.path and which needs no write access to
the Blender install. Only the imports that fail are installed; Blender's bundled numpy and
requests are left alone.

    python scripts/install_blender_deps.py                 # install what is missing, then check
    python scripts/install_blender_deps.py --check         # check only (exit 1 if missing)
    python scripts/install_blender_deps.py --moondream     # also the Moondream client
    python scripts/install_blender_deps.py --blender /path/to/blender

Blender is found from --blender, then $BLENDER_BIN, then `blender` on PATH, then the usual
macOS and Windows install locations.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys

# import name -> pip requirement, for everything the runner imports inside Blender.
PACKAGES = {
    "openai": "openai>=1.0.0",
    "yaml": "pyyaml>=6.0",
    "requests": "requests>=2.31.0",
    "PIL": "Pillow>=10.0.0",
    "numpy": "numpy>=1.24.0",
}
MOONDREAM = {"moondream": "moondream"}

PROBE = r"""
import importlib, sys, bpy
missing = []
for name in sys.argv[sys.argv.index('--') + 1:]:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
print('SCOPE_PY=' + sys.executable)
print('SCOPE_MODULES=' + bpy.utils.user_resource('SCRIPTS', path='modules', create=True))
print('SCOPE_MISSING=' + ','.join(missing))
"""


def find_blender(explicit: str | None) -> str | None:
    for cand in (explicit, os.environ.get("BLENDER_BIN"), shutil.which("blender")):
        if cand and (os.path.isfile(cand) or shutil.which(cand)):
            return cand
    patterns = ["/Applications/Blender.app/Contents/MacOS/Blender",
                r"C:\Program Files\Blender Foundation\Blender*\blender.exe"]
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


def probe(blender: str, names: list[str]) -> dict[str, str]:
    # No --factory-startup: it drops the user modules folder from sys.path, so the probe would
    # not see what this script installs (the runner, started without it, would).
    cmd = [blender, "--background", "--python-exit-code", "1",
           "--python-expr", PROBE, "--", *names]
    out = subprocess.run(cmd, capture_output=True, text=True)
    info = {}
    for line in out.stdout.splitlines():
        if line.startswith("SCOPE_") and "=" in line:
            key, _, value = line.partition("=")
            info[key] = value.strip()
    if "SCOPE_PY" not in info:
        sys.stderr.write(out.stdout[-2000:] + out.stderr[-2000:])
        raise SystemExit(f"Could not ask {blender} for its Python (exit {out.returncode}).")
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--blender", help="Blender executable (default: $BLENDER_BIN, then PATH)")
    ap.add_argument("--check", action="store_true", help="only check the imports")
    ap.add_argument("--moondream", action="store_true",
                    help="also install the moondream client (Moondream Cloud or Photon)")
    args = ap.parse_args()

    blender = find_blender(args.blender)
    if not blender:
        print("Blender not found. Set BLENDER_BIN to the Blender executable, or pass --blender.")
        return 1
    wanted = dict(PACKAGES, **(MOONDREAM if args.moondream else {}))
    info = probe(blender, list(wanted))
    missing = [m for m in info.get("SCOPE_MISSING", "").split(",") if m]
    print(f"Blender:        {blender}")
    print(f"Blender Python: {info['SCOPE_PY']}")
    if not missing:
        print("OK: Blender's Python imports " + ", ".join(wanted) + ".")
        return 0
    if args.check:
        print("MISSING in Blender's Python: " + ", ".join(missing))
        print("Fix: python scripts/install_blender_deps.py")
        return 1

    py, target = info["SCOPE_PY"], info["SCOPE_MODULES"]
    if subprocess.run([py, "-m", "pip", "--version"], capture_output=True).returncode != 0:
        subprocess.run([py, "-m", "ensurepip", "--upgrade"], check=False)
    reqs = [wanted[m] for m in missing]
    print(f"Installing {', '.join(reqs)} into {target}")
    rc = subprocess.run([py, "-m", "pip", "install", "--upgrade", "--target", target, *reqs]).returncode
    if rc != 0:
        print(f"pip failed (exit {rc}). Try by hand: {py} -m pip install --target {target} {' '.join(reqs)}")
        return rc
    still = [m for m in probe(blender, list(wanted)).get("SCOPE_MISSING", "").split(",") if m]
    if still:
        print("Still missing after install: " + ", ".join(still))
        return 1
    print("OK: Blender's Python imports " + ", ".join(wanted) + ".")
    return 0


if __name__ == "__main__":
    sys.exit(main())

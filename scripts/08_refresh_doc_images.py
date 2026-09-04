"""Regenerate the full-view pictures in docs/ from the panoramas that actually ship.

These two had drifted apart, which is the failure worth preventing. docs/img/scenes/ held
thumbnails made during one capture session and benchmark/panoramas/ held the files a run reads,
and nothing tied them together, so a panorama could be replaced without the picture of it
changing. A reader would then be looking at a full view that no run would ever produce.

So the pictures are derived rather than authored. Run this after replacing anything under
benchmark/panoramas/ and commit what it writes.

  python3 scripts/08_refresh_doc_images.py [--check]

  --check   report drift and exit non-zero instead of writing, for CI
"""
import os
import sys
import glob

from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANO = os.path.join(REPO, "benchmark", "panoramas")
DOCS = os.path.join(REPO, "docs", "img", "scenes")

LONG_EDGE = 2000
QUALITY = 86
CHECK = "--check" in sys.argv


def target_name(key):
    """book-nook__preset-road1 -> book-nook__road1__pano.jpg"""
    scene, preset = key.split("__preset-", 1)
    return f"{scene}__{preset}__pano.jpg"


def main():
    os.makedirs(DOCS, exist_ok=True)
    srcs = sorted(glob.glob(os.path.join(PANO, "*.png")))
    if not srcs:
        print(f"  no panoramas under {PANO}")
        return 1

    drift, wrote = [], 0
    for src in srcs:
        key = os.path.basename(src)[:-4]
        dst = os.path.join(DOCS, target_name(key))
        im = Image.open(src).convert("RGB")
        im.thumbnail((LONG_EDGE, LONG_EDGE), Image.LANCZOS)

        if CHECK:
            if not os.path.exists(dst):
                drift.append(f"{target_name(key)} is missing")
                continue
            cur = Image.open(dst).convert("RGB")
            if cur.size != im.size:
                drift.append(f"{target_name(key)} is {cur.size}, the panorama gives {im.size}")
            continue

        im.save(dst, quality=QUALITY)
        wrote += 1
        print(f"  {target_name(key):44} {str(im.size):>14}  from {os.path.basename(src)}")

    if CHECK:
        for d in drift:
            print(f"  DRIFT  {d}")
        print(f"\n  {len(srcs) - len(drift)}/{len(srcs)} doc pictures match the shipped panoramas")
        return 1 if drift else 0

    print(f"\n  refreshed {wrote} pictures from {PANO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Write benchmark/scope_v2_smoke.csv: 24 v2 tasks for a first run, chosen by a fixed rule.

    python scripts/make_smoke_set.py           # rewrite the file
    python scripts/make_smoke_set.py --check   # exit 1 if the file differs from the rule

The rule uses only the dataset, never any model's results. Each of the eight categories
(`eval_category`, in sorted order) gets three tasks, one in each of three worlds; category i
leaves out world i mod 4 (worlds in sorted order), so every world gets six tasks. Within a
world and category the task is the one whose question_id has the smallest SHA-256. Rows keep
the order of benchmark/scope_v2.csv and every column, so any runner takes the file as
QUESTIONS_CSV or --questions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmark/scope_v2.csv"
TARGET = ROOT / "benchmark/scope_v2_smoke.csv"


def world(row):
    return row["file_location"].split("/")[1]


def select(rows):
    worlds = sorted({world(r) for r in rows})
    categories = sorted({r["eval_category"] for r in rows})
    chosen = set()
    for i, category in enumerate(categories):
        for w in worlds:
            if w == worlds[i % len(worlds)]:
                continue
            cell = [r["question_id"] for r in rows if r["eval_category"] == category and world(r) == w]
            chosen.add(min(cell, key=lambda q: hashlib.sha256(q.encode()).hexdigest()))
    return [r for r in rows if r["question_id"] in chosen]


def render():
    with SOURCE.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header, rows = reader.fieldnames, list(reader)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(select(rows))
    return out.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare instead of writing")
    args = ap.parse_args(argv)
    text = render()
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != text:
            print(f"{TARGET.relative_to(ROOT)} does not match the rule; run without --check")
            return 1
        print(f"{TARGET.relative_to(ROOT)} matches the rule")
        return 0
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

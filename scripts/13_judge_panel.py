#!/usr/bin/env python3
"""Majority-vote several judges' verdicts on one run.

    # judge the same run with three different models first
    python scripts/11_judge.py -i run.csv -o judged/run__gpt-oss-120b.csv --model gpt-oss-120b ...
    python scripts/11_judge.py -i run.csv -o judged/run__muse.csv          --model muse-glimmer ...
    python scripts/11_judge.py -i run.csv -o judged/run__genrm.csv         --model qwen3-genrm ...

    # then vote
    python scripts/13_judge_panel.py judged/run__*.csv -o judged/run__panel.csv

Why different models and not one model three times: repeating one judge averages sampling
noise but keeps its systematic bias, because a model's mistakes correlate with themselves.
Judges from different families make less correlated errors, so the majority is right more
often than any member (Verga et al. 2024, "Replacing Judges with Juries", arXiv 2404.18796).
On this benchmark one judge repeated three times at temperature 0 changed 1 verdict in 50,
while two different judges on the same answers differed by 6.9 points.

Why majority and not unanimity: "correct only if all agree" does not reduce noise, it moves
the operating point towards marking things wrong. With an even number of seats there is no
majority, so the strict rule is used and the report says so; read the all-agree to any-agree
band instead.

A seat whose coverage is short of the others (below 98% of the fullest seat) is dropped with a
warning. Voting on the intersection would silently shrink the run to that seat's rows: it once
produced a "majority accuracy" over 25 of 536 rows that looked like a real number.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

GRADED = {"true", "1", "yes", "false", "0", "no"}
TRUE = {"true", "1", "yes"}


def seat_name(path: Path) -> str:
    stem = path.stem
    return stem.split("__", 1)[1] if "__" in stem else stem


def load(path: Path):
    rows = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for r in csv.DictReader(f):
            # Count rows that carry a verdict, not rows that exist: a row the judge could not
            # grade is still written, with an empty judge_is_correct.
            if (r.get("judge_is_correct") or "").strip().lower() in GRADED:
                rows[(r["question_id"], r.get("repeat_idx", "1") or "1")] = r
    return rows


def main():
    ap = argparse.ArgumentParser(description="Majority-vote judged CSVs from several judges.")
    ap.add_argument("judged", nargs="+", type=Path, help="one judged CSV per seat")
    ap.add_argument("-o", "--out", type=Path, required=True, help="panel CSV to write")
    ap.add_argument("--summary", type=Path, help="JSON summary (default: <out>.json)")
    a = ap.parse_args()

    seats = {seat_name(p): load(p) for p in a.judged}
    sizes = {n: len(d) for n, d in seats.items()}
    full = max(sizes.values())
    short = [n for n, k in sizes.items() if k < 0.98 * full]
    for n in short:
        print(f"WARNING: dropping {n}: {sizes[n]} graded rows against {full}", file=sys.stderr)
        seats.pop(n)
    if not seats:
        sys.exit("no complete seat")
    names = list(seats)
    n_seats = len(names)
    keys = sorted(set.intersection(*(set(d) for d in seats.values())))

    out_rows, split, disagree = [], Counter(), 0
    for k in keys:
        votes = {n: seats[n][k]["judge_is_correct"].strip().lower() in TRUE for n in names}
        yes = sum(votes.values())
        verdict = (2 * yes > n_seats) if n_seats % 2 else (yes == n_seats)
        disagree += 0 < yes < n_seats
        split[f"{yes}/{n_seats}"] += 1
        row = dict(seats[names[0]][k])
        row["judge_is_correct"] = verdict
        row["panel_yes"] = yes
        row["panel_votes_json"] = json.dumps(votes)
        for n in names:
            row[f"judge_{n}"] = int(votes[n])
            row[f"reason_{n}"] = seats[n][k].get("judge_reason", "")
        out_rows.append(row)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    n = len(out_rows)
    pct = lambda x: round(100 * x / n, 1)
    summary = {
        "rows": n,
        "seats": names,
        "dropped_seats": short,
        "rule": (f"majority of {n_seats}" if n_seats % 2 else f"all {n_seats} must agree")
        if n_seats > 1 else f"single seat ({names[0]}), not a panel",
        "accuracy_pct": pct(sum(r["judge_is_correct"] for r in out_rows)),
        "per_seat_accuracy_pct": {nm: pct(sum(r[f"judge_{nm}"] for r in out_rows)) for nm in names},
        "disagreement_pct": pct(disagree),
        "band_all_agree_pct": pct(sum(r["panel_yes"] == n_seats for r in out_rows)),
        "band_any_agree_pct": pct(sum(r["panel_yes"] > 0 for r in out_rows)),
        "vote_split": dict(split),
    }
    (a.summary or a.out.with_suffix(".json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

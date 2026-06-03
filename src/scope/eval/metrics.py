#!/usr/bin/env python3
"""
SCOPE evaluation metrics -- aggregation and accuracy computation.

Provides three core functions:

* ``aggregate_csvs``      -- merge per-world result CSVs into one unified CSV
* ``compute_accuracy``    -- overall and per-category accuracy from a judged CSV
* ``error_mode_breakdown``-- count error modes from a judged CSV
"""

import csv
import glob
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# CSV aggregation                                                              #
# --------------------------------------------------------------------------- #

def aggregate_csvs(
    csv_paths: List[str],
    output_path: str,
    extra_cols: Optional[Dict[str, str]] = None,
) -> int:
    """Merge multiple per-world result CSVs into one unified CSV.

    Parameters
    ----------
    csv_paths : list[str]
        Paths to individual result CSVs.
    output_path : str
        Destination path for the merged CSV.
    extra_cols : dict[str, str], optional
        Additional columns to inject into every row (e.g.
        ``{"model_id_eval": "llama3", "vlm_model": "qwen-vl"}``).

    Returns
    -------
    int
        Total number of rows written.
    """
    if not csv_paths:
        print("[aggregate] No CSV paths provided.")
        return 0

    extra_cols = extra_cols or {}
    fieldnames: List[str] = []
    rows_all: List[Dict[str, Any]] = []

    for fp in csv_paths:
        if not os.path.isfile(fp):
            print(f"[aggregate] WARNING: file not found, skipping: {fp}")
            continue
        with open(fp, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            if not fieldnames:
                fieldnames = list(r.fieldnames or [])
            else:
                for h in (r.fieldnames or []):
                    if h not in fieldnames:
                        fieldnames.append(h)
            for row in r:
                for k, v in extra_cols.items():
                    row.setdefault(k, v)
                rows_all.append(row)

    # Ensure extra_cols appear in the header
    for col in extra_cols:
        if col not in fieldnames:
            fieldnames.append(col)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows_all:
            for h in fieldnames:
                row.setdefault(h, "")
            w.writerow(row)

    print(f"[aggregate] Wrote {output_path} with {len(rows_all)} rows from {len(csv_paths)} files.")
    for fp in csv_paths:
        print(f"[aggregate]   + {fp}")
    return len(rows_all)


def aggregate_by_pattern(
    pattern: str,
    output_path: str,
    extra_cols: Optional[Dict[str, str]] = None,
) -> int:
    """Convenience wrapper: find CSVs matching a glob *pattern* and merge them.

    Example::

        aggregate_by_pattern("results/run1__*.csv", "results/run1_merged.csv")

    Returns the total number of rows written.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[aggregate] No files matched pattern: {pattern}")
        return 0
    return aggregate_csvs(files, output_path, extra_cols=extra_cols)


# --------------------------------------------------------------------------- #
# Accuracy computation                                                         #
# --------------------------------------------------------------------------- #

def _parse_bool_cell(val: Any) -> Optional[bool]:
    """Parse a CSV cell into a boolean, returning None for blank/unknown."""
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return None


def compute_accuracy(csv_path: str) -> Dict[str, Any]:
    """Compute overall and per-category accuracy from a judged CSV.

    Looks for the ``judge_is_correct`` column.  Rows where that column
    is blank or unparseable are excluded from the totals.

    Returns
    -------
    dict
        ``{"overall": float, "by_category": {cat: float, ...},
           "total_rows": int, "correct_rows": int,
           "per_category_detail": {cat: {"total": int, "correct": int}, ...}}``
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    total = 0
    correct = 0
    cat_total: Counter = Counter()
    cat_correct: Counter = Counter()

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            verdict = _parse_bool_cell(row.get("judge_is_correct"))
            if verdict is None:
                continue  # row not yet judged
            total += 1
            cat = (row.get("eval_category") or "unknown").strip().lower()
            cat_total[cat] += 1
            if verdict:
                correct += 1
                cat_correct[cat] += 1

    overall = (correct / total) if total > 0 else 0.0
    by_category = {}
    per_category_detail = {}
    for cat in sorted(cat_total):
        ct = cat_total[cat]
        cc = cat_correct[cat]
        by_category[cat] = (cc / ct) if ct > 0 else 0.0
        per_category_detail[cat] = {"total": ct, "correct": cc}

    return {
        "overall": round(overall, 4),
        "by_category": {k: round(v, 4) for k, v in by_category.items()},
        "total_rows": total,
        "correct_rows": correct,
        "per_category_detail": per_category_detail,
    }


# --------------------------------------------------------------------------- #
# Error mode breakdown                                                         #
# --------------------------------------------------------------------------- #

def error_mode_breakdown(csv_path: str) -> Counter:
    """Count error modes from a judged CSV.

    Reads the ``judge_error_mode`` column.  Rows marked correct
    (``judge_is_correct == True``) are counted under the key ``"correct"``.

    Returns
    -------
    Counter
        Mapping from error mode string to count.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    counts: Counter = Counter()

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            verdict = _parse_bool_cell(row.get("judge_is_correct"))
            if verdict is None:
                continue
            if verdict:
                counts["correct"] += 1
            else:
                mode = (row.get("judge_error_mode") or "unknown").strip()
                if mode.lower() in ("", "none"):
                    mode = "unknown"
                counts[mode] += 1

    return counts


# --------------------------------------------------------------------------- #
# Pretty printing                                                             #
# --------------------------------------------------------------------------- #

def print_report(csv_path: str) -> None:
    """Print a human-readable accuracy and error report for a judged CSV."""
    acc = compute_accuracy(csv_path)
    errs = error_mode_breakdown(csv_path)

    print("=" * 60)
    print(f"SCOPE Evaluation Report: {csv_path}")
    print("=" * 60)
    print(f"Overall accuracy: {acc['correct_rows']}/{acc['total_rows']} = {acc['overall']:.2%}")
    print()

    if acc["by_category"]:
        print("Per-category accuracy:")
        for cat in sorted(acc["by_category"]):
            detail = acc["per_category_detail"][cat]
            pct = acc["by_category"][cat]
            print(f"  {cat:40s}  {detail['correct']:3d}/{detail['total']:3d}  ({pct:.2%})")
        print()

    if errs:
        print("Error mode breakdown:")
        for mode, count in errs.most_common():
            print(f"  {mode:40s}  {count:4d}")
    print("=" * 60)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="SCOPE evaluation metrics: accuracy, error breakdown, CSV aggregation.",
    )
    sub = ap.add_subparsers(dest="command")

    # -- report subcommand --
    rpt = sub.add_parser("report", help="Print accuracy report for a judged CSV.")
    rpt.add_argument("-i", "--input", required=True, help="Path to judged CSV.")

    # -- accuracy subcommand --
    acc = sub.add_parser("accuracy", help="Compute accuracy (JSON output).")
    acc.add_argument("-i", "--input", required=True, help="Path to judged CSV.")

    # -- errors subcommand --
    err = sub.add_parser("errors", help="Error mode breakdown (JSON output).")
    err.add_argument("-i", "--input", required=True, help="Path to judged CSV.")

    # -- aggregate subcommand --
    agg = sub.add_parser("aggregate", help="Merge multiple CSVs into one.")
    agg.add_argument("-p", "--pattern", required=True,
                     help="Glob pattern for input CSVs (e.g. 'results/run1__*.csv').")
    agg.add_argument("-o", "--output", required=True, help="Output merged CSV path.")

    args = ap.parse_args()

    if args.command == "report":
        print_report(args.input)

    elif args.command == "accuracy":
        import json
        result = compute_accuracy(args.input)
        print(json.dumps(result, indent=2))

    elif args.command == "errors":
        import json
        result = error_mode_breakdown(args.input)
        print(json.dumps(dict(result.most_common()), indent=2))

    elif args.command == "aggregate":
        aggregate_by_pattern(args.pattern, args.output)

    else:
        ap.print_help()


if __name__ == "__main__":
    main()

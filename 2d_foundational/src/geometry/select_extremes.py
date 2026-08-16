#!/usr/bin/env python3
import csv
import sys
import shutil
import argparse
from pathlib import Path
from collections import Counter


def load_results(path):
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("classification") != "ERROR"]
    for r in rows:
        r["complexity_score"] = float(r["complexity_score"])
    return rows


def dedup(rows):
    """Keep one representative per identical geometric fingerprint."""
    seen, kept = set(), []
    for r in rows:
        fp = (
            r["face_count"], r["edge_count"], r["solid_count"],
            r["surface_types"], r["inner_loops"],
            round(float(r["volume"]), 3), round(float(r["surface_area"]), 3),
        )
        if fp in seen:
            continue
        seen.add(fp)
        kept.append(r)
    return kept


def histogram(scores, bins=20, width=50):
    lo, hi = 0.0, 100.0
    counts = Counter(min(int((s - lo) / (hi - lo) * bins), bins - 1) for s in scores)
    peak = max(counts.values()) if counts else 1
    lines = []
    for b in range(bins):
        c = counts.get(b, 0)
        bar = "█" * max(1 if c else 0, round(c / peak * width))
        lines.append(f"  {b * 5:>3}–{b * 5 + 5:<3} | {bar} {c}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Select N simplest + N most complex samples.")
    p.add_argument("--results", required=True, help="results.csv from step_complexity.py")
    p.add_argument("--n", type=int, default=1000, help="samples per class (default 1000)")
    p.add_argument("--selection", default="selection.csv", help="output CSV")
    p.add_argument("--dedup", action="store_true",
                   help="collapse files with identical geometric fingerprints first")
    p.add_argument("--copy_to", default=None,
                   help="also copy selected STEP files into <dir>/simple and <dir>/complex")
    p.add_argument("--link", action="store_true",
                   help="symlink instead of copy (saves disk)")
    args = p.parse_args()

    rows = load_results(args.results)
    print(f"Loaded {len(rows)} successfully-analyzed files from {args.results}")

    if args.dedup:
        before = len(rows)
        rows = dedup(rows)
        print(f"Dedup: {before} -> {len(rows)} unique geometric fingerprints "
              f"({before - len(rows)} near-duplicates collapsed)")

    if len(rows) < 2 * args.n:
        sys.exit(f"FATAL: only {len(rows)} usable files, need {2 * args.n}.")

    rows.sort(key=lambda r: r["complexity_score"])
    simple = rows[:args.n]
    complex_ = rows[-args.n:]

    s_lo, s_hi = simple[0]["complexity_score"], simple[-1]["complexity_score"]
    c_lo, c_hi = complex_[0]["complexity_score"], complex_[-1]["complexity_score"]
    margin = c_lo - s_hi

    print(f"\nSIMPLE  (n={args.n}): scores {s_lo:.2f} – {s_hi:.2f}")
    print(f"COMPLEX (n={args.n}): scores {c_lo:.2f} – {c_hi:.2f}")
    print(f"Margin between classes: {margin:.2f} score points "
          f"({'clean separation' if margin > 10 else 'NARROW — consider a smaller n or manual review near the cutoffs'})")

    print("\nFull score distribution (all analyzed files):")
    print(histogram([r["complexity_score"] for r in rows]))

    # --- write selection CSV -------------------------------------------------
    out_fields = ["path", "label", "complexity_score", "face_count", "edge_count",
                  "curved_ratio", "freeform_ratio", "inner_loops", "reasons"]
    with open(args.selection, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        for r in simple:
            w.writerow({**r, "label": "simple"})
        for r in complex_:
            w.writerow({**r, "label": "complex"})
    print(f"\nWrote {args.selection} ({2 * args.n} rows)")

    # --- optionally materialize the dataset on disk --------------------------
    if args.copy_to:
        root = Path(args.copy_to)
        for label, group in (("simple", simple), ("complex", complex_)):
            d = root / label
            d.mkdir(parents=True, exist_ok=True)
            for r in group:
                src = Path(r["path"])
                dst = d / src.name
                if dst.exists():
                    continue
                if args.link:
                    dst.symlink_to(src.resolve())
                else:
                    shutil.copy2(src, dst)
        print(f"Materialized dataset under {root}/simple and {root}/complex "
              f"({'symlinks' if args.link else 'copies'})")


if __name__ == "__main__":
    main()
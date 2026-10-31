#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

VIEWS = ["front", "back", "left", "right", "top", "bottom"]
OPPOSING = {frozenset(p) for p in [("front", "back"), ("left", "right"), ("top", "bottom")]}


def load_mask(path: Path, dilate: int) -> np.ndarray:
    m = np.array(Image.open(path).convert("L")) < 128  # ink = dark
    if dilate > 0:
        m = binary_dilation(m, iterations=dilate)
    return m


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def pair_similarity(masks: dict[str, np.ndarray], v1: str, v2: str) -> float:
    a, b = masks[v1], masks[v2]
    s = iou(a, b)
    if frozenset((v1, v2)) in OPPOSING:
        s = max(s, iou(a, np.fliplr(b)))
    # left/right and top/bottom can also coincide under a vertical flip
    if frozenset((v1, v2)) in OPPOSING:
        s = max(s, iou(a, np.flipud(b)))
    return s


def find_view_files(part_dir: Path, pattern: re.Pattern, exclude_dim: bool,
                    order: list[str] | None = None) -> dict[str, Path]:
    """Map view name -> png. Tries the name regex first; if --order is given and
    the regex finds nothing, assigns the sorted PNGs to the views in that order."""
    pngs = [p for p in sorted(part_dir.glob("*.png"))
            if not (exclude_dim and "dim" in p.stem.lower())]
    files = {}
    for p in pngs:
        m = pattern.search(p.stem)
        if m and m.group(1).lower() in VIEWS:
            files[m.group(1).lower()] = p
    if not files and order:
        # numeric-aware sort so view_10 doesn't precede view_2
        def key(p):
            nums = re.findall(r"\d+", p.stem)
            return int(nums[-1]) if nums else p.stem
        pngs = sorted(pngs, key=key)
        if len(pngs) == len(order):
            files = dict(zip(order, pngs))
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drawings_roots", type=Path, nargs="+",
                    help="one or more roots with one sub-directory per part (e.g. png_out/complex png_out/simple)")
    ap.add_argument("--subdir", default="normal",
                    help="sub-directory inside each part folder holding the clean drawings (default: normal)")
    ap.add_argument("--out", type=Path, default=Path("symmetry_out"))
    ap.add_argument("--thr", type=float, default=0.98, help="IoU threshold for 'indistinguishable'")
    ap.add_argument("--dilate", type=int, default=1, help="binary dilation iterations before IoU")
    ap.add_argument("--pattern", default=r"(?:^|[_\-])(front|back|left|right|top|bottom)(?:$|[_\-])",
                    help="regex with one group capturing the view name from the PNG stem")
    ap.add_argument("--order", default=None,
                    help="comma-separated view order to use when filenames carry no view name, "
                         "e.g. front,back,left,right,top,bottom  (files are sorted numerically)")
    ap.add_argument("--parts", type=Path, default=None,
                    help="optional text file with one part_id per line (e.g. the 1,000-model VLM set)")
    ap.add_argument("--task1", type=Path, default=None, help="Task-1 manifest CSV")
    ap.add_argument("--task2", type=Path, default=None, help="Task-2 manifest CSV")
    args = ap.parse_args()

    pattern = re.compile(args.pattern, re.IGNORECASE)
    order = [v.strip().lower() for v in args.order.split(",")] if args.order else None
    if order and (len(order) != 6 or set(order) != set(VIEWS)):
        sys.exit(f"--order must list exactly these six views: {VIEWS}")
    args.out.mkdir(parents=True, exist_ok=True)

    part_dirs = []
    for root in args.drawings_roots:
        if args.parts:
            ids = [l.strip() for l in args.parts.read_text().splitlines() if l.strip()]
            cand = [root / pid for pid in ids]
        else:
            cand = sorted(d for d in root.iterdir() if d.is_dir())
        part_dirs += [(root.name, d) for d in cand if d.is_dir()]

    rows = []
    pair_counter: Counter[str] = Counter()
    n_missing = 0
    for i, (group, d) in enumerate(part_dirs, 1):
        view_dir = d / args.subdir if args.subdir else d
        files = find_view_files(view_dir, pattern, exclude_dim=True, order=order)
        if len(files) < 6:
            n_missing += 1
            print(f"[skip] {d.name}: found views {sorted(files)}", file=sys.stderr)
            continue
        masks = {v: load_mask(files[v], args.dilate) for v in VIEWS}
        amb, sims = [], {}
        for v1, v2 in itertools.combinations(VIEWS, 2):
            s = pair_similarity(masks, v1, v2)
            sims[f"{v1}-{v2}"] = s
            if s >= args.thr:
                amb.append(f"{v1}-{v2}")
                pair_counter[f"{v1}-{v2}"] += 1
        rows.append({"part_id": d.name, "group": group, "n_ambiguous_pairs": len(amb),
                     "ambiguous_pairs": ";".join(amb),
                     **{f"iou_{k}": f"{v:.4f}" for k, v in sims.items()}})
        if i % 100 == 0:
            print(f"{i}/{len(part_dirs)} parts", file=sys.stderr)

    if not rows:
        first = part_dirs[0][1] / args.subdir if part_dirs else None
        listing = sorted(p.name for p in first.glob("*")) if first and first.exists() else "(dir not found)"
        sys.exit(f"No parts matched. Filenames in {first}: {listing}\n"
                 f"Adjust --pattern, or pass --order if files are numbered.")

    # ---- per-part CSV
    with open(args.out / "symmetry_flags.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    amb_lookup = {r["part_id"]: set(r["ambiguous_pairs"].split(";")) - {""} for r in rows}

    # ---- summary
    n = len(rows)
    n_any = sum(1 for r in rows if r["n_ambiguous_pairs"] > 0)
    lines = [
        f"threshold IoU >= {args.thr}, dilation = {args.dilate}px",
        f"parts analysed: {n}   (skipped, missing views: {n_missing})",
        f"parts with >=1 indistinguishable view pair: {n_any} ({100*n_any/n:.1f}%)",
        "",
        "per view pair (count, % of parts):",
    ] + [f"  {k:15s} {v:6d}  {100*v/n:5.1f}%" for k, v in sorted(pair_counter.items())]
    for g in sorted({r["group"] for r in rows}):
        gr = [r for r in rows if r["group"] == g]
        ga = sum(1 for r in gr if r["n_ambiguous_pairs"] > 0)
        lines += [f"  group {g:10s}: {len(gr)} parts, {ga} with ambiguity ({100*ga/len(gr):.1f}%)"]

    # ---- affected benchmark items
    affected = []
    if args.task1:
        n_hard = n_hard_amb = 0
        with open(args.task1) as f:
            for r in csv.DictReader(f):
                if r["label"].strip().lower() != "hard":
                    continue
                n_hard += 1
                key = f"{r['render_view']}-{r['drawing_view']}".lower()
                key_rev = f"{r['drawing_view']}-{r['render_view']}".lower()
                if key in amb_lookup.get(r["part_id"], ()) or key_rev in amb_lookup.get(r["part_id"], ()):
                    n_hard_amb += 1
                    affected.append({"task": "task1_hard", **r})
        lines += ["", f"Task 1 hard negatives: {n_hard}, ambiguous: {n_hard_amb} ({100*n_hard_amb/max(n_hard,1):.1f}%)"]
    if args.task2:
        n_t2 = n_t2_amb = 0
        with open(args.task2) as f:
            for r in csv.DictReader(f):
                n_t2 += 1
                tv = r["target_view"].lower()
                # ambiguous if the target view coincides with ANY other non-front option,
                # or with the front reference itself
                amb_pairs = amb_lookup.get(r["part_id"], set())
                hit = any(tv in p.split("-") for p in amb_pairs)
                if hit:
                    n_t2_amb += 1
                    affected.append({"task": "task2", **r})
        lines += [f"Task 2 items: {n_t2}, ambiguous: {n_t2_amb} ({100*n_t2_amb/max(n_t2,1):.1f}%)"]

    if affected:
        keys = sorted({k for a in affected for k in a})
        with open(args.out / "affected_items.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(affected)

    (args.out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
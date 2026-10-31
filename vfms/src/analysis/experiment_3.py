"""
experiment_3.py -- Tab. 3: annotation ablation, clean (normal) vs dimensioned (dim) drawings,
Al-mean, drawing -> render, paired per query. Renders are the same gallery for both.

Needs a second drawing feature tree extracted from data/drawings/<id>/dim/ with the same
view order (make a manifest with  .../dim/{id}_{view}_dim.png  and run the extractor
into <feat>/drawings_dim).

  python experiment_3.py --feat features/dinov3 --ids ids.txt --key cls --name dinov3
"""
import argparse
import numpy as np
from retrieval_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--feat", required=True)
ap.add_argument("--normal", default="drawings")
ap.add_argument("--dim", default="drawings_dim")
ap.add_argument("--renders", default="renders")
ap.add_argument("--ids", default="ids.txt")
ap.add_argument("--key", default="cls")
ap.add_argument("--name", required=True)
ap.add_argument("--center", action="store_true", help="per-modality mean-centering of descriptors")
a = ap.parse_args()

tag = a.name + ("_centered" if a.center else "")
start_log(tag, "exp3")
ids = read_ids(a.ids)
Rv = normalise(load_tree(f"{a.feat}/{a.renders}", ids, a.key, f"{a.name}_{a.renders}_cache.npz"))
if a.center: Rv = center(Rv)
N = len(ids)

ranks = {}
for variant in (a.normal, a.dim):
    Dv = normalise(load_tree(f"{a.feat}/{variant}", ids, a.key, f"{a.name}_{variant}_cache.npz"))
    if a.center: Dv = center(Dv)
    check_view_order(Dv, Rv)
    ranks[variant] = ranks_from_similarity(sim_aligned(Dv, Rv, range(6)), N)
    evaluate(ranks[variant], f"[{variant}] Al-mean, drawing -> render", N)

ra, rb = ranks[a.normal], ranks[a.dim]
delta = rb - ra                                   # > 0: dim ranks worse than normal
n_worse, n_better, n_tie = int((delta > 0).sum()), int((delta < 0).sum()), int((delta == 0).sum())
print(f"\n=== Paired comparison ({a.dim} vs {a.normal}), N={N} ===")
print(f"dim WORSE : {n_worse:6d} ({100*n_worse/N:.1f}%)   <- 'Worse' column of Tab. 3")
print(f"dim BETTER: {n_better:6d} ({100*n_better/N:.1f}%)")
print(f"ties      : {n_tie:6d} ({100*n_tie/N:.1f}%)")
m = n_worse + n_better
if m:
    z = (n_worse - m / 2) / np.sqrt(m / 4)
    print(f"median |delta rank| over non-ties: {np.median(np.abs(delta[delta != 0])):.0f}")
    print(f"sign test z = {z:.2f} (|z| > 1.96 -> p < 0.05)")
ha, hb = ra == 1, rb == 1
print(f"R@1 hits: {a.normal} only {int((ha & ~hb).sum())}, {a.dim} only {int((hb & ~ha).sum())}, both {int((ha & hb).sum())}")
for k in RECALL_KS:
    print(f"R@{k:<2} gap (normal - dim): {100*((ra <= k).mean() - (rb <= k).mean()):6.2f} points")

np.savez(f"{tag}_exp3_results.npz", ids=np.array(ids), ranks_normal=ra, ranks_dim=rb)
print(f"\nSaved per-query ranks to {tag}_exp3_results.npz")

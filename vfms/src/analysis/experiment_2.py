"""
experiment_2.py -- Tab. 2 / Tab. 12: per-view analysis (Al-mean, drawing -> render).
  2A single views, both directions
  2B view pairs: orthogonal axes vs opposing views
  2C cumulative fusion in a FIXED, stated order (the paper's Tab. 2 order by default)

  python experiment_2.py --feat features/dinov3 --ids ids.txt --key cls --name dinov3
"""
import argparse
import numpy as np
from retrieval_common import *

PAIRS = [("front", "top"), ("top", "left"), ("front", "left"),      # orthogonal axes
         ("front", "back"), ("left", "right"), ("top", "bottom")]  # opposing views, same axis

ap = argparse.ArgumentParser()
ap.add_argument("--feat", required=True)
ap.add_argument("--drawings", default="drawings")
ap.add_argument("--renders", default="renders")
ap.add_argument("--ids", default="ids.txt")
ap.add_argument("--key", default="cls")
ap.add_argument("--name", required=True)
ap.add_argument("--center", action="store_true", help="per-modality mean-centering of descriptors")
ap.add_argument("--order", default="front,back,bottom,left,top,right",
                help="cumulative fusion order for 2C; 'auto' = descending single-view R@1")
a = ap.parse_args()

tag = a.name + ("_centered" if a.center else "")
start_log(tag, "exp2")
ids = read_ids(a.ids)
Dv = normalise(load_tree(f"{a.feat}/{a.drawings}", ids, a.key, f"{a.name}_{a.drawings}_cache.npz"))
if a.center: Dv = center(Dv)
Rv = normalise(load_tree(f"{a.feat}/{a.renders}",  ids, a.key, f"{a.name}_{a.renders}_cache.npz"))
if a.center: Rv = center(Rv)
N = len(ids)
print(f"{N} models | dim {Dv.shape[2]}")
check_view_order(Dv, Rv)
V = {n: i for i, n in enumerate(VIEW_NAMES)}
out = {"ids": np.array(ids)}

print("\n" + "=" * 60 + "\n2A: single-view retrieval\n" + "=" * 60)
singles = {}
for name, v in V.items():
    r = ranks_from_similarity(sim_aligned(Dv, Rv, [v]), N)
    singles[name] = evaluate(r, f"[{name}] drawing -> render", N); out[f"2A_{name}_d2r"] = r
    r = ranks_from_similarity(sim_aligned(Rv, Dv, [v]), N)
    evaluate(r, f"[{name}] render -> drawing", N); out[f"2A_{name}_r2d"] = r

print("\n" + "=" * 60 + "\n2B: two-view fusion, drawing -> render\n" + "=" * 60)
for p, q in PAIRS:
    r = ranks_from_similarity(sim_aligned(Dv, Rv, [V[p], V[q]]), N)
    evaluate(r, f"[{p} + {q}]", N); out[f"2B_{p}_{q}_d2r"] = r

print("\n" + "=" * 60 + "\n2C: cumulative view fusion, drawing -> render\n" + "=" * 60)
order = (sorted(singles, key=lambda n: singles[n]["R@1"], reverse=True)
         if a.order == "auto" else a.order.split(","))
assert sorted(order) == sorted(VIEW_NAMES), f"--order must list all six views once: {order}"
print(f"Cumulative order: {order}")
cum = []
for name in order:
    cum.append(V[name])
    r = ranks_from_similarity(sim_aligned(Dv, Rv, cum), N)
    evaluate(r, "[" + " + ".join(VIEW_NAMES[i] for i in cum) + "]", N)
    out[f"2C_{len(cum)}_d2r"] = r

np.savez(f"{tag}_exp2_results.npz", **out)
print(f"\nSaved per-query ranks to {tag}_exp2_results.npz")

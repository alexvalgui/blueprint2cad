"""
experiment_1.py -- Tab. 1: Max / Mean (view-agnostic), Al-max / Al-mean (view-aligned),
both directions, full gallery.

  python experiment_1.py --feat features/dinov3 --ids ids.txt --key cls --name dinov3
"""
import argparse
import numpy as np
from retrieval_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--feat", required=True, help="dir with drawings/ and renders/ feature trees")
ap.add_argument("--drawings", default="drawings", help="subdir of --feat (use drawings_dim for Tab. 3)")
ap.add_argument("--renders", default="renders")
ap.add_argument("--ids", default="ids.txt")
ap.add_argument("--key", default="cls", help="descriptor key inside each .npz")
ap.add_argument("--name", required=True, help="backbone tag for output files")
ap.add_argument("--center", action="store_true", help="per-modality mean-centering of descriptors")
a = ap.parse_args()

tag = a.name + ("_centered" if a.center else "")
start_log(tag, "exp1")
ids = read_ids(a.ids)
Dv = normalise(load_tree(f"{a.feat}/{a.drawings}", ids, a.key, f"{a.name}_{a.drawings}_cache.npz"))
if a.center: Dv = center(Dv)
Rv = normalise(load_tree(f"{a.feat}/{a.renders}",  ids, a.key, f"{a.name}_{a.renders}_cache.npz"))
if a.center: Rv = center(Rv)
N = len(ids)
print(f"{N} models | 6 views | dim {Dv.shape[2]} | chance R@1 {100/N:.4f}%  MedR ~{N//2}")
check_view_order(Dv, Rv)

out = {"ids": np.array(ids)}
for tag, fn in [("max",     lambda Q, G: sim_max(Q, G)),
                ("mean",    lambda Q, G: sim_mean(Q, G)),
                ("al_max",  lambda Q, G: sim_aligned(Q, G, range(6), "max")),
                ("al_mean", lambda Q, G: sim_aligned(Q, G, range(6), "mean"))]:
    r = ranks_from_similarity(fn(Dv, Rv), N); evaluate(r, f"{tag}  drawing -> render", N); out[f"{tag}_d2r"] = r
    r = ranks_from_similarity(fn(Rv, Dv), N); evaluate(r, f"{tag}  render -> drawing", N); out[f"{tag}_r2d"] = r

np.savez(f"{tag}_exp1_results.npz", **out)
print(f"\nSaved per-query ranks to {tag}_exp1_results.npz")

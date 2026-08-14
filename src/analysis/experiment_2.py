"""Experiment 2: Drawing-type / per-view analysis.

Which projection (front, back, left, right, top, bottom) is the most
retrievable as a single-view query, and how do views combine?

Reuses the view cache written by experiment_1.py. All comparisons are
view-aligned via PERM (drawing view i vs the render depicting the same
projection). Evaluates:
  2A: each single view as a query (drawing -> render and render -> drawing)
  2B: view-pair fusion (mean over 2 aligned views) for selected pairs
  2C: cumulative fusion (best single view, +2nd, +3rd, ... all 6),
      views added in descending single-view performance order
"""

import sys
from datetime import datetime

import numpy as np
from pathlib import Path

RESULTS = Path("/home/ec2-user/cad-2d-vision-foundational/results/ijepa")
DRAWING_SUBDIR = "normal"
RENDER_SUBDIR = "2d_embeds"
CHUNK = 256
OUT_PREFIX = RESULTS.name


class Tee:
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "w")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


LOGFILE = f"{OUT_PREFIX}_exp2_{datetime.now():%Y%m%d_%H%M}.log"
sys.stdout = Tee(LOGFILE)
sys.stderr = sys.stdout
print(f"Logging to {LOGFILE}")

# Drawing views load alphabetically; PERM maps each drawing view index to the
# render index depicting the same projection (see experiment_1.py).
PERM = [3, 5, 2, 0, 1, 4]
VIEW_NAMES = ["back", "bottom", "front", "left", "right", "top"]

# View pairs for 2B: complementary axes vs mirror pairs
PAIRS_2B = [
    ("front", "top"),
    ("front", "left"),
    ("top", "left"),
    ("front", "back"),    # mirror pair (expected weak gain)
    ("left", "right"),    # mirror pair
    ("top", "bottom"),    # mirror pair
]


def load_embedding(npz_path):
    d = np.load(npz_path)
    key = list(d.keys())[0]
    v = d[key].squeeze().astype(np.float32)
    return v.mean(axis=0) if v.ndim > 1 else v


def load_views(folder):
    files = sorted(folder.glob("view_*_features.npz"),
                   key=lambda p: int(p.stem.split("_")[1]))
    return [load_embedding(f) for f in files]


cache = Path(f"{OUT_PREFIX}_views_cache.npz")
if cache.exists():
    z = np.load(cache, allow_pickle=False)
    Dv, Rv = z["Dv"].astype(np.float32), z["Rv"].astype(np.float32)
    samples = [str(s) for s in z["samples"]]
    print(f"Loaded cache {cache}")
else:
    samples, draw_views, rend_views = [], [], []
    n_dv, n_rv = None, None
    for sample_dir in sorted(RESULTS.iterdir()):
        if not sample_dir.is_dir():
            continue
        dv = load_views(sample_dir / DRAWING_SUBDIR)
        rv = load_views(sample_dir / RENDER_SUBDIR)
        if not dv or not rv:
            continue
        if n_dv is None:
            n_dv, n_rv = len(dv), len(rv)
        if len(dv) != n_dv or len(rv) != n_rv:
            continue
        samples.append(sample_dir.name)
        draw_views.append(np.stack(dv))
        rend_views.append(np.stack(rv))
    Dv = np.stack(draw_views)
    Rv = np.stack(rend_views)
    np.savez(cache, Dv=Dv, Rv=Rv, samples=np.array(samples))
    print(f"Saved view cache to {cache}")

N, Vd, dim = Dv.shape
print(f"{N} paired samples | {Vd} views | dim = {dim}")

Dv = Dv / np.linalg.norm(Dv, axis=2, keepdims=True)
Rv = Rv / np.linalg.norm(Rv, axis=2, keepdims=True)
Rv = Rv[:, PERM]      # render view i now depicts the same projection as drawing view i


RECALL_KS = [1, 2, 3, 5, 10]


def evaluate(ranks, name, n_gallery, quiet=False):
    metrics = {f"R@{k}": (ranks < k).mean() * 100 for k in RECALL_KS}
    metrics["MRR"] = (1.0 / (ranks + 1)).mean()
    metrics["MedRank"] = float(np.median(ranks))
    if not quiet:
        print(f"\n=== {name} ===")
        recall_str = "  ".join(f"R@{k}: {metrics[f'R@{k}']:6.2f}" for k in RECALL_KS)
        print(f"{recall_str}  MRR: {metrics['MRR']:.4f}  "
              f"MedRank: {metrics['MedRank']:.0f} / {n_gallery}")
    return metrics


def ranks_from_similarity(sim_fn, n_query):
    ranks = np.empty(n_query, dtype=np.int64)
    for start in range(0, n_query, CHUNK):
        end = min(start + CHUNK, n_query)
        S = sim_fn(start, end)
        true_sim = S[np.arange(end - start), np.arange(start, end)]
        ranks[start:end] = (S > true_sim[:, None]).sum(1)
    return ranks


def mean_sim_over_views(Q, G, view_ids):
    """Similarity fn: mean cosine over the given aligned view indices."""
    G_t = {v: G[:, v, :].T.copy() for v in view_ids}

    def sim(s, e):
        S = np.zeros((e - s, N), dtype=np.float32)
        for v in view_ids:
            S += Q[s:e, v, :] @ G_t[v]
        return S / len(view_ids)
    return sim


results = {}

# ---- 2A: single-view queries ----
print("\n" + "=" * 60)
print("2A: single-view retrieval per projection")
print("=" * 60)
singles_d2r = {}
for v, name in enumerate(VIEW_NAMES):
    r = ranks_from_similarity(mean_sim_over_views(Dv, Rv, [v]), N)
    m = evaluate(r, f"[{name}] drawing -> render", N)
    singles_d2r[name] = m
    results[f"2A_{name}_d2r"] = r

    r = ranks_from_similarity(mean_sim_over_views(Rv, Dv, [v]), N)
    evaluate(r, f"[{name}] render -> drawing", N)
    results[f"2A_{name}_r2d"] = r

# ---- 2B: view-pair fusion ----
print("\n" + "=" * 60)
print("2B: two-view fusion (mean over aligned views), drawing -> render")
print("=" * 60)
for a, b in PAIRS_2B:
    ids = [VIEW_NAMES.index(a), VIEW_NAMES.index(b)]
    r = ranks_from_similarity(mean_sim_over_views(Dv, Rv, ids), N)
    evaluate(r, f"[{a} + {b}]", N)
    results[f"2B_{a}_{b}_d2r"] = r

# ---- 2C: cumulative fusion in descending single-view order ----
print("\n" + "=" * 60)
print("2C: cumulative view fusion, drawing -> render")
print("=" * 60)
order = sorted(singles_d2r, key=lambda n: singles_d2r[n]["R@1"], reverse=True)
print(f"View order by single-view R@1: {order}")
cum_ids = []
for name in order:
    cum_ids.append(VIEW_NAMES.index(name))
    r = ranks_from_similarity(mean_sim_over_views(Dv, Rv, cum_ids), N)
    evaluate(r, f"[{' + '.join(VIEW_NAMES[i] for i in cum_ids)}]", N)
    results[f"2C_top{len(cum_ids)}_d2r"] = r

np.savez(f"{OUT_PREFIX}_exp2_results.npz",
         samples=np.array(samples),
         **{k: v for k, v in results.items()})
print(f"\nSaved per-query ranks to {OUT_PREFIX}_exp2_results.npz")
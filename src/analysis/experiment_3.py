"""Experiment 3: Preprocessing / annotation ablation.

Does dimension annotation (auto-generated linear + diameter dims, see
render_png.py --with-dims) help or hurt cross-modal retrieval?

Compares two drawing variants against the SAME render gallery under the
best frozen protocol (1D-mean: mean cosine over the 6 view-aligned pairs):
  - 'normal': clean drawings, no dimensions
  - 'dim':    identical drawings + auto dimensions

Both variants are evaluated on the intersection of samples that have all
6 views in BOTH drawing folders (so the comparison is paired), and the
per-query rank deltas are analyzed directly, including a sign test.

Reuses/creates per-variant view caches. Renders are loaded once.
"""

import sys
from datetime import datetime

import numpy as np
from pathlib import Path

RESULTS = Path("/home/ec2-user/cad-2d-vision-foundational/results/ijepa")
RENDER_SUBDIR = "2d_embeds"
VARIANTS = ["normal", "dim"]
CHUNK = 256
OUT_PREFIX = RESULTS.name

PERM = [3, 5, 2, 0, 1, 4]
VIEW_NAMES = ["back", "bottom", "front", "left", "right", "top"]

RECALL_KS = [1, 2, 3, 5, 10]


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


LOGFILE = f"{OUT_PREFIX}_exp3_{datetime.now():%Y%m%d_%H%M}.log"
sys.stdout = Tee(LOGFILE)
sys.stderr = sys.stdout
print(f"Logging to {LOGFILE}")
print(f"RESULTS={RESULTS}  VARIANTS={VARIANTS}  PERM={PERM}")


def load_embedding(npz_path):
    d = np.load(npz_path)
    key = list(d.keys())[0]
    v = d[key].squeeze().astype(np.float32)
    return v.mean(axis=0) if v.ndim > 1 else v


def load_views(folder):
    files = sorted(folder.glob("view_*_features.npz"),
                   key=lambda p: int(p.stem.split("_")[1]))
    return [load_embedding(f) for f in files]


def build_cache(drawing_subdir):
    """Cache per drawing variant: samples, drawing views, render views."""
    cache = Path(f"{OUT_PREFIX}_{drawing_subdir}_views_cache.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return ([str(s) for s in z["samples"]],
                z["Dv"].astype(np.float32), z["Rv"].astype(np.float32))
    samples, draw_views, rend_views = [], [], []
    n_dv, n_rv = None, None
    for sample_dir in sorted(RESULTS.iterdir()):
        if not sample_dir.is_dir():
            continue
        dv = load_views(sample_dir / drawing_subdir)
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
    print(f"Saved cache {cache}")
    return samples, Dv, Rv


# The 'normal' cache may already exist from experiment_1 under the old name.
legacy = Path(f"{OUT_PREFIX}_views_cache.npz")
target = Path(f"{OUT_PREFIX}_normal_views_cache.npz")
if legacy.exists() and not target.exists():
    target.symlink_to(legacy.name)
    print(f"Linked {target} -> {legacy}")

data = {}
for variant in VARIANTS:
    s, Dv, Rv = build_cache(variant)
    data[variant] = {"samples": s, "Dv": Dv, "Rv": Rv}
    print(f"[{variant}] {len(s)} samples | views {Dv.shape[1]} | dim {Dv.shape[2]}")

# ---- pair the variants on common samples ----
common = sorted(set(data[VARIANTS[0]]["samples"]) & set(data[VARIANTS[1]]["samples"]))
print(f"\nPaired samples across variants: {len(common)}")
for variant in VARIANTS:
    idx = {name: i for i, name in enumerate(data[variant]["samples"])}
    sel = np.array([idx[name] for name in common])
    d = data[variant]
    d["Dv"] = d["Dv"][sel]
    d["Rv"] = d["Rv"][sel]

N = len(common)
# Renders are the same files for both variants; use one gallery.
Rv = data[VARIANTS[0]]["Rv"]
Rv = Rv / np.linalg.norm(Rv, axis=2, keepdims=True)
Rv = Rv[:, PERM]
Rv_t = [Rv[:, v, :].T.copy() for v in range(Rv.shape[1])]


def evaluate(ranks, name, n_gallery):
    print(f"\n=== {name} ===")
    for k in RECALL_KS:
        print(f"Recall@{k:<3}: {(ranks < k).mean() * 100:6.2f}")
    print(f"MRR        : {(1.0 / (ranks + 1)).mean():.4f}")
    print(f"Median rank: {np.median(ranks):.0f} / {n_gallery}   "
          f"(chance ~ {n_gallery / 2:.0f})")
    return ranks


def aligned_mean_ranks(Dv):
    Vd = Dv.shape[1]
    ranks = np.empty(N, dtype=np.int64)
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        S = np.zeros((end - start, N), dtype=np.float32)
        for v in range(Vd):
            S += Dv[start:end, v, :] @ Rv_t[v]
        S /= Vd
        true_sim = S[np.arange(end - start), np.arange(start, end)]
        ranks[start:end] = (S > true_sim[:, None]).sum(1)
    return ranks


ranks = {}
for variant in VARIANTS:
    Dv = data[variant]["Dv"]
    Dv = Dv / np.linalg.norm(Dv, axis=2, keepdims=True)
    ranks[variant] = evaluate(
        aligned_mean_ranks(Dv),
        f"[{variant}] 1D-mean aligned, drawing -> render", N)

# ---- paired per-query comparison ----
a, b = VARIANTS
ra, rb = ranks[a], ranks[b]
delta = rb - ra                       # >0: 'dim' ranks worse than 'normal'
n_worse = int((delta > 0).sum())
n_better = int((delta < 0).sum())
n_tie = int((delta == 0).sum())

print(f"\n=== Paired comparison ({b} vs {a}) ===")
print(f"Queries where {b} ranks WORSE : {n_worse}  ({n_worse / N * 100:.1f}%)")
print(f"Queries where {b} ranks BETTER: {n_better}  ({n_better / N * 100:.1f}%)")
print(f"Ties                          : {n_tie}  ({n_tie / N * 100:.1f}%)")
print(f"Median |delta rank|           : {np.median(np.abs(delta[delta != 0])):.0f}"
      if n_worse + n_better > 0 else "All ties")

# Sign test (two-sided, normal approximation) on non-tied pairs
m = n_worse + n_better
if m > 0:
    z = (n_worse - m / 2) / np.sqrt(m / 4)
    print(f"Sign test z = {z:.2f}  "
          f"(|z| > 1.96 -> difference significant at p < 0.05)")

# R@1 flips: queries that hit top-1 in one variant but not the other
hit_a, hit_b = ra == 0, rb == 0
print(f"R@1 hits: {a} only = {int((hit_a & ~hit_b).sum())}, "
      f"{b} only = {int((hit_b & ~hit_a).sum())}, "
      f"both = {int((hit_a & hit_b).sum())}")

np.savez(f"{OUT_PREFIX}_exp4_results.npz",
         samples=np.array(common),
         **{f"ranks_{v}": ranks[v] for v in VARIANTS})
print(f"\nSaved per-query ranks to {OUT_PREFIX}_exp4_results.npz")
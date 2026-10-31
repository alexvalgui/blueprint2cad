"""
retrieval_common.py -- shared loading, view-order check, scoring and metrics for
experiment_1/2/3. Features are expected in the layout produced by the manifest-based
extraction, both modalities in canonical view order:

    <feat_root>/<id>/view_1_features.npz ... view_6_features.npz
    view_1..6 = front, back, left, right, top, bottom

Ranks are 1-based (rank 1 = top hit); R@k uses ranks <= k; MedR is therefore directly
comparable to the paper's chance value (~N/2).
"""
import sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np

VIEW_NAMES = ["front", "back", "left", "right", "top", "bottom"]
RECALL_KS = [1, 2, 3, 5, 10]
CHUNK = 256


# ------------------------------------------------------------------ logging
class Tee:
    def __init__(self, logfile):
        self.terminal, self.log = sys.__stdout__, open(logfile, "w")
    def write(self, msg):
        self.terminal.write(msg); self.log.write(msg); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()


def start_log(prefix, tag):
    logfile = f"{prefix}_{tag}_{datetime.now():%Y%m%d_%H%M}.log"
    sys.stdout = Tee(logfile)
    print(f"Logging to {logfile}")
    return logfile


# ------------------------------------------------------------------ loading
def load_embedding(path, key):
    with np.load(path) as d:
        if key not in d.files:
            raise KeyError(f"{path}: key '{key}' not in {d.files}; pass --key")
        v = d[key].squeeze().astype(np.float32)
    return v.mean(0) if v.ndim > 1 else v           # 2-D only if a patch grid was stored


def load_tree(root, ids, key, cache, workers=None):
    """[N, 6, D] float32, l2-normalised, in ids order. Cached to `cache` (.npz)."""
    root, cache = Path(root), Path(cache)
    if cache.exists():
        z = np.load(cache)
        if list(z["ids"]) == ids:
            print(f"Loaded cache {cache}")
            return z["X"].astype(np.float32)
        print(f"Cache {cache} has a different id list; rebuilding")

    def one(mid):
        return np.stack([load_embedding(root / mid / f"view_{v}_features.npz", key)
                         for v in range(1, 7)])

    with ThreadPoolExecutor(max_workers=workers or os.cpu_count() * 2) as ex:
        X = np.stack(list(ex.map(one, ids)))
    np.savez(cache, X=X.astype(np.float16), ids=np.array(ids))
    print(f"Saved cache {cache}  shape={X.shape}")
    return X


def normalise(X):
    n = np.linalg.norm(X, axis=2, keepdims=True)
    if (n == 0).any():
        print(f"WARNING: {(n == 0).sum()} zero-norm descriptors; they will never match")
    return np.nan_to_num(X / np.maximum(n, 1e-12))


def read_ids(path):
    return [l.strip() for l in open(path) if l.strip()]


def center(X, report=True):
    """Per-modality mean-centering: subtract the mean descriptor over all models and views,
    then re-normalise. Label-free; removes a shared component (e.g. the blank canvas) that
    otherwise dominates cosine similarity for encoders without a global token."""
    mu = X.reshape(-1, X.shape[-1]).mean(0)
    Xc = normalise(X - mu)
    if report:
        rng = np.random.default_rng(0); i, j = rng.integers(0, len(X), (2, 3000))
        print(f"  random-pair cosine before/after centering: "
              f"{(X[i, 0] * X[j, 0]).sum(1).mean():.3f} -> {(Xc[i, 0] * Xc[j, 0]).sum(1).mean():.3f}")
    return Xc


# ------------------------------------------------------------------ view-order check
def check_view_order(Dv, Rv, n=2000):
    """Print the 6x6 single-view R@1 matrix (rows = drawing view, cols = render view) as a
    diagnostic. Informational only: it never stops the run."""
    n = min(n, len(Dv))
    M = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            S = Dv[:n, i] @ Rv[:n, j].T
            t = S[np.arange(n), np.arange(n)]
            M[i, j] = ((S > t[:, None]).sum(1) == 0).mean() * 100
    print("View-order check, single-view R@1 on first %d models (rows=drawing, cols=render):" % n)
    print("          " + "".join(f"{v:>8s}" for v in VIEW_NAMES))
    for i, v in enumerate(VIEW_NAMES):
        print(f"{v:8s}  " + "".join(f"{M[i, j]:8.1f}" for j in range(6)))
    perm = M.argmax(1)
    if (perm == np.arange(6)).all():
        print("View order: diagonal is the row-wise maximum everywhere.\n")
    else:
        print(f"View order note: best render view per drawing view is {perm.tolist()} "
              f"(identity expected; off-diagonal maxima within an axis pair are mirror ambiguity).\n")
    return M


# ------------------------------------------------------------------ scoring
def ranks_from_similarity(sim_fn, n_query):
    """1-based rank of the ground-truth item for each query."""
    ranks = np.empty(n_query, dtype=np.int64)
    for s in range(0, n_query, CHUNK):
        e = min(s + CHUNK, n_query)
        S = sim_fn(s, e)                                         # (chunk, N)
        t = S[np.arange(e - s), np.arange(s, e)]
        ranks[s:e] = 1 + (S > t[:, None]).sum(1)
    return ranks


def sim_max(Q, G):
    """Max over all V x V view pairs (view-agnostic)."""
    N, V, D = G.shape
    Gf = G.reshape(N * V, D).T.copy()
    def f(s, e):
        S = (Q[s:e].reshape(-1, D) @ Gf).reshape(e - s, Q.shape[1], N, V)
        return S.max(axis=(1, 3))
    return f


def sim_mean(Q, G):
    """Cosine between re-normalised mean descriptors (view-agnostic)."""
    Qm = normalise(Q.mean(1, keepdims=True))[:, 0]
    Gm = normalise(G.mean(1, keepdims=True))[:, 0].T.copy()
    return lambda s, e: Qm[s:e] @ Gm


def sim_aligned(Q, G, views, agg="mean"):
    """Compare only matching view indices; agg over views is mean (Al-mean) or max (Al-max)."""
    Gt = {v: G[:, v].T.copy() for v in views}
    def f(s, e):
        parts = [Q[s:e, v] @ Gt[v] for v in views]
        return np.mean(parts, 0) if agg == "mean" else np.max(parts, 0)
    return f


# ------------------------------------------------------------------ metrics
def metrics(ranks):
    m = {f"R@{k}": (ranks <= k).mean() * 100 for k in RECALL_KS}
    m["MRR"] = (1.0 / ranks).mean()
    m["MedR"] = float(np.median(ranks))
    return m


def evaluate(ranks, name, n_gallery):
    m = metrics(ranks)
    print(f"\n=== {name} ===")
    print("  ".join(f"R@{k}: {m[f'R@{k}']:6.2f}" for k in RECALL_KS)
          + f"  MRR: {m['MRR']:.3f}  MedR: {m['MedR']:.0f} / {n_gallery}")
    return m
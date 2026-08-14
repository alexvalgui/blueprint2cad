import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os

RESULTS = Path("/home/ec2-user/cad-2d-vision-foundational/results/ijepa")
DRAWING_SUBDIR = "normal"
RENDER_SUBDIR = "2d_embeds"
CHUNK = 256
OUT_PREFIX = RESULTS.name

RUN = {"1A": True, "1B": True, "1C": True}

RECALL_KS = [1, 2, 3, 5, 10]


def load_embedding(npz_path):
    with np.load(npz_path) as d:
        v = d[d.files[0]].squeeze().astype(np.float32)
    return v.mean(axis=0) if v.ndim > 1 else v


def load_views(folder):
    files = sorted(folder.glob("view_*_features.npz"),
                   key=lambda p: int(p.stem.split("_")[1]))
    return [load_embedding(f) for f in files]


def load_sample(sample_dir):
    if not sample_dir.is_dir():
        return None
    dv = load_views(sample_dir / DRAWING_SUBDIR)
    rv = load_views(sample_dir / RENDER_SUBDIR)
    if not dv or not rv:
        return None
    return sample_dir.name, np.stack(dv), np.stack(rv)


cache = Path(f"{OUT_PREFIX}_views_cache.npz")
if cache.exists():
    z = np.load(cache, allow_pickle=False)
    Dv, Rv = z["Dv"], z["Rv"]
    samples = [str(s) for s in z["samples"]]
    N, Vd, _ = Dv.shape
    Vr = Rv.shape[1]
    print(f"Loaded cache {cache}: {N} samples | {Vd} drawing views, "
          f"{Vr} render views, dim = {Dv.shape[2]}")
else:
    sample_dirs = sorted(RESULTS.iterdir())
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as ex:
        results = list(ex.map(load_sample, sample_dirs))

    samples, draw_views, rend_views = [], [], []
    n_dv, n_rv = None, None
    for r in results:
        if r is None:
            continue
        name, dv, rv = r
        if n_dv is None:
            n_dv, n_rv = dv.shape[0], rv.shape[0]
        if dv.shape[0] != n_dv or rv.shape[0] != n_rv:
            continue
        samples.append(name)
        draw_views.append(dv)
        rend_views.append(rv)

    Dv = np.stack(draw_views)   # (N, Vd, dim)
    Rv = np.stack(rend_views)   # (N, Vr, dim)
    N, Vd, _ = Dv.shape
    Vr = Rv.shape[1]
    print(f"Loaded {N} paired samples | {Vd} drawing views, "
          f"{Vr} render views, dim = {Dv.shape[2]}")
    np.savez(cache, Dv=Dv, Rv=Rv, samples=np.array(samples))
    print(f"Saved view cache to {cache}")

Dv = Dv / np.linalg.norm(Dv, axis=2, keepdims=True)
Rv = Rv / np.linalg.norm(Rv, axis=2, keepdims=True)


PERM = [3, 5, 2, 0, 1, 4]
Rv = Rv[:, PERM]
DRAWING_VIEW_NAMES = ["back", "bottom", "front", "left", "right", "top"]


def evaluate(ranks, name, n_gallery):
    print(f"\n=== {name} ===")
    for k in RECALL_KS:
        print(f"Recall@{k:<3}: {(ranks < k).mean():.4f}")
    print(f"MRR        : {(1.0 / (ranks + 1)).mean():.4f}")
    print(f"Mean rank  : {ranks.mean():.1f} / {n_gallery}")
    print(f"Median rank: {np.median(ranks):.0f} / {n_gallery}   "
          f"(chance ~ {n_gallery/2:.0f})")
    return ranks


def ranks_from_similarity(sim_fn, n_query):
    ranks = np.empty(n_query, dtype=np.int64)
    for start in range(0, n_query, CHUNK):
        end = min(start + CHUNK, n_query)
        S = sim_fn(start, end)                      # (chunk, N)
        true_sim = S[np.arange(end - start), np.arange(start, end)]
        ranks[start:end] = (S > true_sim[:, None]).sum(1)
    return ranks



# ---- Variant 1A: max similarity over all view pairs ----
if RUN["1A"]:
    Rflat = Rv.reshape(N * Vr, -1)
    Dflat = Dv.reshape(N * Vd, -1)

    def sim_1b_d2r(s, e):
        q = Dv[s:e].reshape(-1, Dv.shape[2])        # (chunk*Vd, dim)
        S = q @ Rflat.T                             # (chunk*Vd, N*Vr)
        S = S.reshape(e - s, Vd, N, Vr)
        return S.max(axis=(1, 3))

    def sim_1b_r2d(s, e):
        q = Rv[s:e].reshape(-1, Rv.shape[2])
        S = q @ Dflat.T
        S = S.reshape(e - s, Vr, N, Vd)
        return S.max(axis=(1, 3))

    r1a_d2r = evaluate(
        ranks_from_similarity(sim_1b_d2r, N),
        "1A multi-view max-pooled similarity: drawing -> render", N)
    r1a_r2d = evaluate(
        ranks_from_similarity(sim_1b_r2d, N),
        "1A multi-view max-pooled similarity: render -> drawing", N)

# ---- Variant 1B: mean-pooled embeddings ----
if RUN["1B"]:
    Dm = Dv.mean(axis=1)
    Dm = Dm / np.linalg.norm(Dm, axis=1, keepdims=True)
    Rm = Rv.mean(axis=1)
    Rm = Rm / np.linalg.norm(Rm, axis=1, keepdims=True)

    def sim_1c_d2r(s, e): return Dm[s:e] @ Rm.T
    def sim_1c_r2d(s, e): return Rm[s:e] @ Dm.T

    r1b_d2r = evaluate(
        ranks_from_similarity(sim_1c_d2r, N),
        "1B mean-pooled embeddings: drawing -> render", N)
    r1b_r2d = evaluate(
        ranks_from_similarity(sim_1c_r2d, N),
        "1B mean-pooled embeddings: render -> drawing", N)

# ---- Variant 1C: view-aligned matching (view i vs view i only) ----
if RUN["1C"]:
    Rv_t = [Rv[:, v, :].T.copy() for v in range(Vr)]
    Dv_t = [Dv[:, v, :].T.copy() for v in range(Vd)]

    def sim_1d_d2r_mean(s, e):
        S = np.zeros((e - s, N), dtype=np.float32)
        for v in range(Vd):
            S += Dv[s:e, v, :] @ Rv_t[v]
        return S / Vd

    def sim_1d_d2r_max(s, e):
        S = np.full((e - s, N), -np.inf, dtype=np.float32)
        for v in range(Vd):
            np.maximum(S, Dv[s:e, v, :] @ Rv_t[v], out=S)
        return S

    def sim_1d_r2d_mean(s, e):
        S = np.zeros((e - s, N), dtype=np.float32)
        for v in range(Vr):
            S += Rv[s:e, v, :] @ Dv_t[v]
        return S / Vr

    def sim_1d_r2d_max(s, e):
        S = np.full((e - s, N), -np.inf, dtype=np.float32)
        for v in range(Vr):
            np.maximum(S, Rv[s:e, v, :] @ Dv_t[v], out=S)
        return S

    r1d_mean_d2r = evaluate(
        ranks_from_similarity(sim_1d_d2r_mean, N),
        "1C view-aligned MEAN over views: drawing -> render", N)
    r1d_mean_r2d = evaluate(
        ranks_from_similarity(sim_1d_r2d_mean, N),
        "1C view-aligned MEAN over views: render -> drawing", N)
    r1d_max_d2r = evaluate(
        ranks_from_similarity(sim_1d_d2r_max, N),
        "1C view-aligned MAX over views: drawing -> render", N)
    r1d_max_r2d = evaluate(
        ranks_from_similarity(sim_1d_r2d_max, N),
        "1C view-aligned MAX over views: render -> drawing", N)

# ---- Save whatever ran ----
out = {"samples": np.array(samples)}
if RUN["1A"]:
    out.update(r1a_d2r=r1a_d2r, r1a_r2d=r1a_r2d)
if RUN["1B"]:
    out.update(r1b_d2r=r1b_d2r, r1b_r2d=r1b_r2d)
if RUN["1C"]:
    out.update(r1d_mean_d2r=r1d_mean_d2r, r1d_mean_r2d=r1d_mean_r2d,
               r1d_max_d2r=r1d_max_d2r, r1d_max_r2d=r1d_max_r2d)


tag = "".join(k for k, v in RUN.items() if v)
outfile = f"{OUT_PREFIX}_exp1_{tag}_results.npz"
np.savez(outfile, **out)
print(f"\nSaved per-query ranks to {outfile}")
import numpy as np
from pathlib import Path

RESULTS = Path("/home/ec2-user/cad-2d-vision-foundational/results/dinov3")
DRAWING_SUBDIR = "normal"
RENDER_SUBDIR = "2d_embeds"
SINGLE_VIEW_ID = 1
CHUNK = 256
OUT_PREFIX = RESULTS.name

def load_embedding(npz_path):
    d = np.load(npz_path)
    key = list(d.keys())[0]
    v = d[key].squeeze().astype(np.float32)
    return v.mean(axis=0) if v.ndim > 1 else v

def load_views(folder):
    files = sorted(folder.glob("view_*_features.npz"),
                   key=lambda p: int(p.stem.split("_")[1]))
    return [load_embedding(f) for f in files]

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

Dv = np.stack(draw_views)   # (N, Vd, dim)
Rv = np.stack(rend_views)   # (N, Vr, dim)
N, Vd, _ = Dv.shape
Vr = Rv.shape[1]
print(f"Loaded {N} paired samples | {Vd} drawing views, {Vr} render views, dim = {Dv.shape[2]}")

Dv = Dv / np.linalg.norm(Dv, axis=2, keepdims=True)
Rv = Rv / np.linalg.norm(Rv, axis=2, keepdims=True)

def evaluate(ranks, name, n_gallery):
    print(f"\n=== {name} ===")
    for k in (1, 5, 10, 100):
        print(f"Recall@{k:<3}: {(ranks < k).mean():.4f}")
    print(f"MRR        : {(1.0 / (ranks + 1)).mean():.4f}")
    print(f"Mean rank  : {ranks.mean():.1f} / {n_gallery}")
    print(f"Median rank: {np.median(ranks):.0f} / {n_gallery}   (chance ~ {n_gallery/2:.0f})")
    return ranks

def ranks_from_similarity(sim_fn, n_query):
    ranks = np.empty(n_query, dtype=np.int64)
    for start in range(0, n_query, CHUNK):
        end = min(start + CHUNK, n_query)
        S = sim_fn(start, end)                      # (chunk, N)
        true_sim = S[np.arange(end - start), np.arange(start, end)]
        ranks[start:end] = (S > true_sim[:, None]).sum(1)
    return ranks

# ---- Variant 1A: single view each side ----
D1 = Dv[:, SINGLE_VIEW_ID - 1]
R1 = Rv[:, SINGLE_VIEW_ID - 1]

def sim_1a_d2r(s, e): return D1[s:e] @ R1.T
def sim_1a_r2d(s, e): return R1[s:e] @ D1.T

r = ranks_from_similarity(sim_1a_d2r, N)
evaluate(r, f"1A single view (view {SINGLE_VIEW_ID}): drawing -> render", N)
r1a_d2r = r
r = ranks_from_similarity(sim_1a_r2d, N)
evaluate(r, f"1A single view (view {SINGLE_VIEW_ID}): render -> drawing", N)
r1a_r2d = r

# ---- Variant 1B: max similarity over all view pairs ----
Rflat = Rv.reshape(N * Vr, -1)
Dflat = Dv.reshape(N * Vd, -1)

def sim_1b_d2r(s, e):
    q = Dv[s:e].reshape(-1, Dv.shape[2])            # (chunk*Vd, dim)
    S = q @ Rflat.T                                 # (chunk*Vd, N*Vr)
    S = S.reshape(e - s, Vd, N, Vr)
    return S.max(axis=(1, 3))

def sim_1b_r2d(s, e):
    q = Rv[s:e].reshape(-1, Rv.shape[2])
    S = q @ Dflat.T
    S = S.reshape(e - s, Vr, N, Vd)
    return S.max(axis=(1, 3))

r = ranks_from_similarity(sim_1b_d2r, N)
evaluate(r, "1B multi-view max-pooled similarity: drawing -> render", N)
r1b_d2r = r
r = ranks_from_similarity(sim_1b_r2d, N)
evaluate(r, "1B multi-view max-pooled similarity: render -> drawing", N)
r1b_r2d = r

# ---- Variant 1C: mean-pooled embeddings ----
Dm = Dv.mean(axis=1)
Dm = Dm / np.linalg.norm(Dm, axis=1, keepdims=True)
Rm = Rv.mean(axis=1)
Rm = Rm / np.linalg.norm(Rm, axis=1, keepdims=True)

def sim_1c_d2r(s, e): return Dm[s:e] @ Rm.T
def sim_1c_r2d(s, e): return Rm[s:e] @ Dm.T

r = ranks_from_similarity(sim_1c_d2r, N)
evaluate(r, "1C mean-pooled embeddings: drawing -> render", N)
r1c_d2r = r
r = ranks_from_similarity(sim_1c_r2d, N)
evaluate(r, "1C mean-pooled embeddings: render -> drawing", N)
r1c_r2d = r

np.savez(f"{OUT_PREFIX}_exp1_results.npz",
         samples=np.array(samples),
         r1a_d2r=r1a_d2r, r1a_r2d=r1a_r2d,
         r1b_d2r=r1b_d2r, r1b_r2d=r1b_r2d,
         r1c_d2r=r1c_d2r, r1c_r2d=r1c_r2d)
print(f"\nSaved per-query ranks to {OUT_PREFIX}_exp1_results.npz")
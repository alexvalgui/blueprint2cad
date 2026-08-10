import numpy as np
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (adjusted_rand_score,
                             normalized_mutual_info_score,
                             silhouette_score)

RESULTS = Path("/home/ec2-user/cad-2d-vision-foundational/results/sam3")
DRAWING_SUBDIR = "normal"

def load_embedding(npz_path):
    d = np.load(npz_path)
    key = list(d.keys())[0]
    v = d[key].squeeze()
    return v.mean(axis=0) if v.ndim > 1 else v

samples, draw_embs, rend_embs = [], [], []
for sample_dir in sorted(RESULTS.iterdir()):
    if not sample_dir.is_dir():
        continue
    rend_f = sample_dir / "2d_embeds" / "aggregated_features.npz"
    draw_f = sample_dir / DRAWING_SUBDIR / "aggregated_features.npz"
    if not (draw_f.exists() and rend_f.exists()):
        continue
    samples.append(sample_dir.name)
    draw_embs.append(load_embedding(draw_f))
    rend_embs.append(load_embedding(rend_f))

D = np.stack(draw_embs)
R = np.stack(rend_embs)
N = len(samples)
print(f"Loaded {N} paired samples, dim = {D.shape[1]}")

D = D / np.linalg.norm(D, axis=1, keepdims=True)
R = R / np.linalg.norm(R, axis=1, keepdims=True)

X = np.vstack([D, R])
obj_labels = np.concatenate([np.arange(N), np.arange(N)])
mod_labels = np.concatenate([np.zeros(N), np.ones(N)])

agg = AgglomerativeClustering(n_clusters=N, metric="cosine",
                              linkage="average")
labels = agg.fit_predict(X)

agg2 = AgglomerativeClustering(n_clusters=2, metric="cosine",
                               linkage="average")
labels2 = agg2.fit_predict(X)

pair_agreement = np.mean(labels[:N] == labels[N:])

print("\n=== Agglomerative clustering (k = num objects) ===")
print(f"ARI vs object identity : {adjusted_rand_score(obj_labels, labels):.4f}")
print(f"NMI vs object identity : {normalized_mutual_info_score(obj_labels, labels):.4f}")
print(f"ARI vs modality        : {adjusted_rand_score(mod_labels, labels):.4f}")
print(f"Pair agreement         : {pair_agreement:.4f}   (chance ~ {1/N:.4f})")

print("\n=== Diagnostic (k = 2) ===")
print(f"ARI vs modality        : {adjusted_rand_score(mod_labels, labels2):.4f}")
print(f"Cluster sizes (k=2)    : {np.bincount(labels2)}")

print("\n=== Clustering-free corroboration ===")
print(f"Modality silhouette    : {silhouette_score(X, mod_labels, metric='cosine'):.4f}")
print(f"Centroid distance      : {np.linalg.norm(D.mean(0) - R.mean(0)):.4f}")
print(f"Mean intra-drawing sim : {(D @ D.T).mean():.4f}")
print(f"Mean intra-render sim  : {(R @ R.T).mean():.4f}")

print("\n=== Cross-modal retrieval: drawing -> render ===")
S = D @ R.T
ranks = (S > np.diag(S)[:, None]).sum(1)
for k in (1, 5, 10, 100):
    print(f"Recall@{k:<3}: {(ranks < k).mean():.4f}")
print(f"Mean rank  : {ranks.mean():.1f} / {N}")
print(f"Median rank: {np.median(ranks):.0f} / {N}   (chance ~ {N/2:.0f})")

print("\n=== Cross-modal retrieval: render -> drawing ===")
S_T = S.T
ranks_r = (S_T > np.diag(S_T)[:, None]).sum(1)
for k in (1, 5, 10, 100):
    print(f"Recall@{k:<3}: {(ranks_r < k).mean():.4f}")
print(f"Mean rank  : {ranks_r.mean():.1f} / {N}")
print(f"Median rank: {np.median(ranks_r):.0f} / {N}   (chance ~ {N/2:.0f})")

np.savez("dinov3_agglo_results.npz",
         samples=np.array(samples), labels=labels, labels_k2=labels2,
         obj_labels=obj_labels, mod_labels=mod_labels,
         draw_ranks=ranks, rend_ranks=ranks_r)
print("\nSaved results to dinov3_agglo_results.npz")
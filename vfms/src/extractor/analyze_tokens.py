#!/usr/bin/env python3
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# ---------------------------------------------------------------- helpers

def infer_grid(n_tokens: int):
    """Infer (h, w) grid from a token count. Returns None if ambiguous."""
    r = int(math.isqrt(n_tokens))
    if r * r == n_tokens:
        return (r, r)
    # maybe a CLS token (or register tokens) is included
    for extra in (1, 4, 5):  # CLS / 4 registers / CLS+4 registers
        m = n_tokens - extra
        r = int(math.isqrt(m))
        if m > 0 and r * r == m:
            return (r, r, extra)  # third element = leading tokens to strip
    return None


def is_patch_tokens(arr: np.ndarray) -> bool:
    """Heuristic: 2D array (N_tokens, D) with N_tokens > 4."""
    return arr.ndim == 2 and arr.shape[0] > 4


def pca_rgb(tokens: np.ndarray) -> np.ndarray:
    """Project (N, D) tokens to 3 PCA components, normalize to [0,1] RGB."""
    x = tokens.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    # SVD-based PCA (no sklearn dependency)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    proj = x @ vt[:3].T
    lo = np.percentile(proj, 2, axis=0)
    hi = np.percentile(proj, 98, axis=0)
    rgb = np.clip((proj - lo) / np.maximum(hi - lo, 1e-8), 0, 1)
    return rgb


def load_view_features(npz_path: Path) -> dict:
    with np.load(npz_path) as z:
        return {k: z[k] for k in z.files}


def try_find_image(dataset_root, object_id, view_idx, num_views, view_manifest=None):
    """Best-effort lookup of the original view image (manifest first, then pipeline naming)."""
    if view_manifest and object_id in view_manifest:
        paths = view_manifest[object_id]
        if len(paths) >= view_idx:
            p = Path(paths[view_idx - 1])
            return p if p.exists() else None
    if dataset_root is None:
        return None
    obj_dir = Path(dataset_root) / object_id
    if not obj_dir.is_dir():
        return None
    candidates = [
        obj_dir / f"view_{view_idx}.jpg", obj_dir / f"view_{view_idx}.png",
        obj_dir / f"view_{view_idx:02d}.jpg",
        obj_dir / f"{view_idx - 1}.jpg", obj_dir / f"{view_idx - 1}.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    imgs = sorted(f for f in obj_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
    if len(imgs) >= view_idx:
        return imgs[view_idx - 1]
    return None


# ---------------------------------------------------------------- checks

def check_array(name, arr, report):
    issues = []
    a = arr.astype(np.float32) if arr.dtype == np.float16 else arr
    if np.isnan(a).any():
        issues.append(f"NaN values in '{name}' ({int(np.isnan(a).sum())} elems)")
    if np.isinf(a).any():
        issues.append(f"Inf values in '{name}'")
    if a.size and np.allclose(a, 0):
        issues.append(f"'{name}' is entirely zero")
    if is_patch_tokens(a):
        norms = np.linalg.norm(a, axis=-1)
        n_zero = int((norms < 1e-6).sum())
        if n_zero:
            issues.append(f"{n_zero}/{len(norms)} zero-norm tokens in '{name}'")
        # collapsed features: all tokens nearly identical
        if len(a) > 1:
            an = a / np.maximum(norms[:, None], 1e-8)
            mean_cos = float((an @ an.mean(axis=0)).mean())
            if mean_cos > 0.999:
                issues.append(f"'{name}' looks collapsed (mean cos-to-mean {mean_cos:.4f})")
    report.extend(issues)
    return issues


# ---------------------------------------------------------------- per-run analysis

def analyze_run(run_dir: Path, out_dir: Path, dataset_root, max_objects_plot: int,
                view_manifest=None, save_panels_opts=None):
    run_name = run_dir.name
    run_out = out_dir / run_name
    run_out.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "dataset_manifest.json"
    run_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    model_label = run_manifest.get("model", run_name)

    obj_dirs = sorted(d for d in run_dir.iterdir() if d.is_dir())
    report = {"run": run_name, "model": model_label, "objects": {}, "global_issues": []}
    shape_registry = defaultdict(set)          # key -> set of shapes
    norm_samples = defaultdict(list)           # key -> flat list of token norms (sampled)
    obj_mean_embeddings = {}                   # object_id -> mean patch-token embedding

    for oi, obj_dir in enumerate(obj_dirs):
        obj_id = obj_dir.name
        obj_report = []
        view_files = sorted(obj_dir.glob("view_*_features.npz"))
        if not view_files:
            report["objects"][obj_id] = ["no view feature files found"]
            continue

        view_feats = []
        for vf in view_files:
            try:
                feats = load_view_features(vf)
            except Exception as e:
                obj_report.append(f"failed to load {vf.name}: {e}")
                continue
            view_feats.append((vf, feats))
            for k, arr in feats.items():
                shape_registry[k].add(tuple(arr.shape))
                check_array(f"{vf.name}:{k}", arr, obj_report)
                if is_patch_tokens(arr):
                    norms = np.linalg.norm(arr.astype(np.float32), axis=-1)
                    norm_samples[k].append(norms)

        # cross-view consistency: same keys / shapes on every view
        keysets = {tuple(sorted(f.keys())) for _, f in view_feats}
        if len(keysets) > 1:
            obj_report.append(f"inconsistent feature keys across views: {keysets}")

        # cross-view similarity on the main patch-token key
        patch_key = next((k for _, f in view_feats for k in f if is_patch_tokens(f[k])), None)
        if patch_key:
            means = []
            for _, f in view_feats:
                arr = f.get(patch_key)
                if arr is not None and is_patch_tokens(arr):
                    m = arr.astype(np.float32).mean(axis=0)
                    means.append(m / max(np.linalg.norm(m), 1e-8))
            if len(means) > 1:
                M = np.stack(means)
                sim = M @ M.T
                off = sim[~np.eye(len(M), dtype=bool)]
                if off.min() < 0.2:
                    obj_report.append(
                        f"low cross-view similarity (min {off.min():.3f}) — possible wrong image / bad view")
                obj_mean_embeddings[obj_id] = M.mean(axis=0)

        # grid inference check
        for vf, f in view_feats:
            for k, arr in f.items():
                if is_patch_tokens(arr) and infer_grid(arr.shape[0]) is None:
                    obj_report.append(
                        f"{vf.name}:{k} token count {arr.shape[0]} not mappable to a square grid "
                        f"(grid_size was not saved by the pipeline)")

        # aggregated file check
        agg = obj_dir / "aggregated_features.npz"
        if agg.exists():
            try:
                afeats = load_view_features(agg)
                for k, arr in afeats.items():
                    check_array(f"aggregated:{k}", arr, obj_report)
            except Exception as e:
                obj_report.append(f"failed to load aggregated_features.npz: {e}")

        if obj_report:
            report["objects"][obj_id] = obj_report

        # ---------- plots (limited to first N objects) ----------
        if oi < max_objects_plot and patch_key:
            plot_object(view_feats, patch_key, obj_id, run_out, dataset_root,
                        num_views=len(view_feats), model_label=model_label,
                        view_manifest=view_manifest)
            if save_panels_opts:
                save_panels(view_feats, patch_key, obj_id, run_out, dataset_root,
                            model_label, view_manifest=view_manifest,
                            fmt=save_panels_opts["fmt"], dpi=save_panels_opts["dpi"])

    # dataset-level shape consistency
    for k, shapes in shape_registry.items():
        if len(shapes) > 1:
            report["global_issues"].append(
                f"key '{k}' has inconsistent shapes across dataset: {sorted(shapes)}")

    # dataset-level outliers: objects whose mean embedding is far from everyone else's
    if len(obj_mean_embeddings) > 3:
        ids = list(obj_mean_embeddings)
        E = np.stack([obj_mean_embeddings[i] for i in ids])
        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
        sim_to_centroid = E @ (E.mean(axis=0) / max(np.linalg.norm(E.mean(axis=0)), 1e-8))
        mu, sd = sim_to_centroid.mean(), sim_to_centroid.std()
        for i, s in zip(ids, sim_to_centroid):
            if sd > 0 and (s - mu) / sd < -3:
                report["global_issues"].append(
                    f"object '{i}' is a 3-sigma embedding outlier (sim {s:.3f} vs mean {mu:.3f})")

    # norm histograms
    if norm_samples:
        fig, axes = plt.subplots(1, len(norm_samples), figsize=(5 * len(norm_samples), 3.5),
                                 squeeze=False)
        for ax, (k, chunks) in zip(axes[0], norm_samples.items()):
            allnorms = np.concatenate([c.ravel() for c in chunks])
            allnorms = allnorms[np.isfinite(allnorms)]
            if allnorms.size == 0:
                continue
            ax.hist(allnorms, bins=80)
            ax.set_title(f"{model_label}\ntoken L2 norms — {k}", fontsize=9)
            ax.set_xlabel("norm"); ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(run_out / "token_norm_histograms.png", dpi=140)
        plt.close(fig)

    (run_out / "error_report.json").write_text(json.dumps(report, indent=2))
    n_issues = len(report["global_issues"]) + sum(len(v) for v in report["objects"].values())
    print(f"[{run_name}] {len(obj_dirs)} objects checked, {n_issues} issues -> {run_out/'error_report.json'}")
    return report, norm_samples, model_label


def plot_object(view_feats, patch_key, obj_id, run_out, dataset_root, num_views, model_label,
                view_manifest=None):
    """PCA-RGB map + norm heatmap (+ original image if found) per view."""
    n_rows = 3 if (dataset_root or view_manifest) else 2
    fig, axes = plt.subplots(n_rows, num_views,
                             figsize=(3.2 * num_views, 3.2 * n_rows), squeeze=False)
    for col, (vf, f) in enumerate(view_feats):
        arr = f.get(patch_key)
        view_idx = col + 1
        row = 0
        if dataset_root or view_manifest:
            img_path = try_find_image(dataset_root, obj_id, view_idx, num_views, view_manifest)
            ax = axes[0][col]
            if img_path:
                ax.imshow(plt.imread(str(img_path)))
            ax.set_title(f"view {view_idx}", fontsize=9)
            ax.axis("off")
            row = 1
        ax_pca, ax_norm = axes[row][col], axes[row + 1][col]
        for ax in (ax_pca, ax_norm):
            ax.axis("off")
        if arr is None or not is_patch_tokens(arr):
            continue
        g = infer_grid(arr.shape[0])
        if g is None:
            ax_pca.set_title("grid not inferable", fontsize=8)
            continue
        strip = g[2] if len(g) == 3 else 0
        h = w = g[0] if len(g) == 2 else int(math.isqrt(arr.shape[0] - strip))
        toks = arr[strip:].astype(np.float32)
        rgb = pca_rgb(toks).reshape(h, w, 3)
        ax_pca.imshow(rgb)
        ax_pca.set_title("PCA-RGB tokens" + (f" (stripped {strip})" if strip else ""), fontsize=8)
        norms = np.linalg.norm(toks, axis=-1).reshape(h, w)
        im = ax_norm.imshow(norms, cmap="magma")
        ax_norm.set_title("token norms", fontsize=8)
        fig.colorbar(im, ax=ax_norm, fraction=0.046)
    fig.suptitle(f"{model_label} — {obj_id} — {patch_key}", fontsize=10)
    fig.tight_layout()
    fig.savefig(run_out / f"{obj_id}_tokens.png", dpi=140)
    plt.close(fig)


def save_panels(view_feats, patch_key, obj_id, run_out, dataset_root, model_label,
                view_manifest=None, fmt="png", dpi=300):
    """Save each subplot as its own clean file for paper composition.

    Layout: <run_out>/panels/<obj_id>/view{i}_{image|pca|norms}.{fmt}
    Norm maps share one color scale across the object's views; the scale and
    a standalone colorbar are saved alongside so figures stay comparable.
    """
    panel_dir = run_out / "panels" / obj_id
    panel_dir.mkdir(parents=True, exist_ok=True)

    def _save(fig, path):
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.01,
                    transparent=True)
        plt.close(fig)

    # shared norm scale across views of this object
    all_norm_maps = {}
    for col, (vf, f) in enumerate(view_feats):
        arr = f.get(patch_key)
        if arr is None or not is_patch_tokens(arr):
            continue
        g = infer_grid(arr.shape[0])
        if g is None:
            continue
        strip = g[2] if len(g) == 3 else 0
        h = w = g[0] if len(g) == 2 else int(math.isqrt(arr.shape[0] - strip))
        toks = arr[strip:].astype(np.float32)
        all_norm_maps[col] = (toks, h, w)
    if not all_norm_maps:
        return
    vmin = min(np.linalg.norm(t, axis=-1).min() for t, _, _ in all_norm_maps.values())
    vmax = max(np.linalg.norm(t, axis=-1).max() for t, _, _ in all_norm_maps.values())

    for col, (toks, h, w) in all_norm_maps.items():
        view_idx = col + 1
        # source image copy (so all paper assets live together)
        img_path = try_find_image(dataset_root, obj_id, view_idx, len(view_feats),
                                  view_manifest)
        if img_path:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(plt.imread(str(img_path)))
            ax.axis("off")
            _save(fig, panel_dir / f"view{view_idx}_image.{fmt}")

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(pca_rgb(toks).reshape(h, w, 3), interpolation="nearest")
        ax.axis("off")
        _save(fig, panel_dir / f"view{view_idx}_pca.{fmt}")

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(np.linalg.norm(toks, axis=-1).reshape(h, w),
                  cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.axis("off")
        _save(fig, panel_dir / f"view{view_idx}_norms.{fmt}")

    # standalone colorbar for the shared norm scale
    fig, ax = plt.subplots(figsize=(0.35, 3))
    fig.colorbar(plt.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), cmap="magma"),
        cax=ax)
    _save(fig, panel_dir / f"norm_colorbar.{fmt}")

    meta = {"model": model_label, "object": obj_id, "patch_key": patch_key,
            "norm_vmin": float(vmin), "norm_vmax": float(vmax)}
    (panel_dir / "panel_meta.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------- cross-model comparison

def cross_model_figure(all_norms, out_dir):
    """Overlay token-norm distributions of every model on shared axes per key."""
    keys = sorted({k for norms in all_norms.values() for k in norms})
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 4), squeeze=False)
    for ax, key in zip(axes[0], keys):
        for model_label, norms in all_norms.items():
            if key not in norms:
                continue
            allnorms = np.concatenate([c.ravel() for c in norms[key]])
            allnorms = allnorms[np.isfinite(allnorms)]
            if allnorms.size == 0:
                continue
            # normalize per-model so scales are comparable
            allnorms = allnorms / max(np.median(allnorms), 1e-8)
            ax.hist(allnorms, bins=80, histtype="step", density=True, label=model_label)
        ax.set_title(f"token norms (median-normalized) — {key}", fontsize=9)
        ax.set_xlabel("norm / median"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "cross_model_norms.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Diagnose token features across model runs")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="One or more run output dirs (one per model)")
    ap.add_argument("--dataset", type=str, default=None,
                    help="Original dataset root (to show images next to token maps)")
    ap.add_argument("--manifest", type=str, default=None,
                    help="View manifest JSON ({object_id: [view paths]}) for image lookup")
    ap.add_argument("--out", type=str, default="analysis",
                    help="Where to write reports and figures")
    ap.add_argument("--max-objects-plot", type=int, default=8,
                    help="Cap per-object figures per run (checks still run on everything)")
    ap.add_argument("--save-panels", action="store_true",
                    help="Also save each subplot as an individual file for paper figures")
    ap.add_argument("--panel-format", default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--panel-dpi", type=int, default=300)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    view_manifest = None
    if args.manifest:
        view_manifest = json.loads(Path(args.manifest).read_text())

    all_norms = {}
    summary = {}
    for run in args.runs:
        run_dir = Path(run)
        if not run_dir.is_dir():
            print(f"Skipping {run}: not a directory")
            continue
        spo = ({"fmt": args.panel_format, "dpi": args.panel_dpi}
               if args.save_panels else None)
        report, norms, model_label = analyze_run(
            run_dir, out_dir, args.dataset, args.max_objects_plot, view_manifest,
            save_panels_opts=spo)
        all_norms[model_label] = norms
        summary[model_label] = {
            "objects_with_issues": len(report["objects"]),
            "global_issues": report["global_issues"],
        }

    cross_model_figure(all_norms, out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nAll reports and figures in {out_dir}/")


if __name__ == "__main__":
    main()
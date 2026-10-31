import argparse, json, os, sys
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scipy import ndimage

VIEWS = ["front", "back", "left", "right", "top", "bottom"]


# descriptor
def load_gray(path):
    """Grayscale on a white canvas; transparent PNGs are composited onto white first."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im)
    return np.asarray(im.convert("L"))


def silhouette_from_image(path, is_drawing, grid, thresh, debug_path=None):
    img = load_gray(path)
    fg = img < thresh
    if is_drawing:
        fg = ndimage.binary_closing(fg, iterations=2)
        fg = ndimage.binary_fill_holes(fg)
        fg = ndimage.binary_opening(fg, iterations=1)       
    if debug_path:
        Image.fromarray((fg * 255).astype(np.uint8)).save(debug_path)
    ys, xs = np.where(fg)
    if ys.size < 10:
        return np.zeros(grid * grid + 2, np.float32)
    crop = fg[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    h, w = crop.shape
    m = ndimage.zoom(crop, (grid / h, grid / w), order=1).ravel()
    extra = np.array([np.log(h / w), 10.0 * fg.mean()], np.float32)
    v = np.concatenate([m, extra]).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _work(job):
    mid, drawing_pat, render_pat, render_idx, grid, thresh, debug_dir = job
    d = np.zeros((len(VIEWS), grid * grid + 2), np.float32)
    r = np.zeros_like(d)
    for vi, view in enumerate(VIEWS):
        dp = drawing_pat.format(id=mid, view=view)
        rp = render_pat.format(id=mid, view=view, idx=render_idx[view])
        dbg_d = dbg_r = None
        if debug_dir:
            dbg_d = os.path.join(debug_dir, f"{mid}_{view}_drawing.png")
            dbg_r = os.path.join(debug_dir, f"{mid}_{view}_render.png")
        d[vi] = silhouette_from_image(dp, True, grid, thresh, dbg_d)
        r[vi] = silhouette_from_image(rp, False, grid, thresh, dbg_r)
    return d, r


def build_descriptors(ids, args):
    order = args.render_order.split(",")
    assert sorted(order) == sorted(VIEWS), f"--render-order must list all of {VIEWS}"
    render_idx = {v: i for i, v in enumerate(order)}          # view -> file index
    jobs = [(mid, args.drawing_pattern, args.render_pattern, render_idx, args.grid, args.thresh,
             args.debug_dir if i < 20 else None) for i, mid in enumerate(ids)]
    with Pool(args.workers) as pool:
        out = []
        for i, res in enumerate(pool.imap(_work, jobs, chunksize=16)):
            out.append(res)
            if i % 1000 == 0:
                print(f"  {i}/{len(ids)}", file=sys.stderr)
    d = np.stack([o[0] for o in out]); r = np.stack([o[1] for o in out])
    return d, r


# retrieval
def query_ranks(dq, rg, how, block=256):
    import torch, torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rg_t = torch.as_tensor(rg, device=dev)
    if how == "mean":
        gm = F.normalize(rg_t.mean(1), dim=-1)
    ranks = np.empty(len(dq), np.int64)
    for s in range(0, len(dq), block):
        q = torch.as_tensor(dq[s:s + block], device=dev)
        if how == "mean":
            sc = F.normalize(q.mean(1), dim=-1) @ gm.T
        else:
            S = torch.einsum("qid,gjd->qgij", q, rg_t)
            sc = {"max": S.amax(dim=(2, 3)),
                  "al_max": S.diagonal(dim1=2, dim2=3).amax(-1),
                  "al_mean": S.diagonal(dim1=2, dim2=3).mean(-1)}[how]
        gt = sc[torch.arange(len(q)), torch.arange(s, s + len(q), device=dev)][:, None]
        ranks[s:s + len(q)] = (1 + (sc > gt).sum(1)).cpu().numpy()
    return ranks


COLS = ["R@1", "R@2", "R@3", "R@5", "R@10", "MRR", "MedR"]


def metrics(ranks):
    out = {f"R@{k}": round(100 * float((ranks <= k).mean()), 2) for k in (1, 2, 3, 5, 10)}
    out["MRR"] = round(float((1.0 / ranks).mean()), 3)
    out["MedR"] = float(np.median(ranks))
    return out


def fmt(c, v):
    return f"{v:8.3f}" if c == "MRR" else f"{v:8.0f}" if c == "MedR" else f"{v:8.2f}"


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="text file, one model id per line")
    ap.add_argument("--drawing-pattern", required=True, help="e.g. data/drawings/{id}/normal/{id}_{view}.png")
    ap.add_argument("--render-pattern", required=True,
                    help="e.g. data/renders/{id}/{idx:03d}.png  ({idx} = position in --render-order)")
    ap.add_argument("--render-order", default="front,back,left,right,top,bottom",
                    help="view name for render index 0,1,2,3,4,5")
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--thresh", type=int, default=248, help="pixel < thresh is foreground")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="use only the first N ids")
    ap.add_argument("--cache", default="silhouette_desc.npz")
    ap.add_argument("--debug-dir", default=None, help="save masks for the first 20 models")
    args = ap.parse_args()

    ids = [l.strip() for l in open(args.ids) if l.strip()]
    if args.limit:
        ids = ids[:args.limit]
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    if os.path.exists(args.cache) and not args.limit:
        z = np.load(args.cache); d, r = z["drawings"], z["renders"]
        print(f"loaded cached descriptors {d.shape}")
    else:
        print(f"building descriptors for {len(ids)} models ...", file=sys.stderr)
        d, r = build_descriptors(ids, args)
        if not args.limit:
            np.savez(args.cache, drawings=d, renders=r, ids=np.array(ids))

    empty = int(((np.abs(d).sum(-1) == 0) | (np.abs(r).sum(-1) == 0)).sum())
    print(f"N = {len(d)} models; empty masks: {empty} views "
          f"(chance R@1 = {100/len(d):.4f}%, chance MedR ~ {len(d)/2:.0f})\n")

    results = {}
    for how in ("max", "mean", "al_max", "al_mean"):
        results[f"D->R {how}"] = metrics(query_ranks(d, r, how))
    results["R->D al_mean"] = metrics(query_ranks(r, d, "al_mean"))

    w = max(len(k) for k in results)
    print(f"{'rule':{w}s}  " + "".join(f"{c:>8s}" for c in COLS))
    for k, m in results.items():
        print(f"{k:{w}s}  " + "".join(fmt(c, m[c]) for c in COLS))
    with open("silhouette_results.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
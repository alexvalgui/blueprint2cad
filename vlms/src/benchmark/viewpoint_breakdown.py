import json, collections, sys, argparse

VIEWS = ["front", "back", "left", "right", "top", "bottom"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--items-file", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--dim", action="store_true", help="use the _dim_ result filenames")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    path_to_view = {}
    for m in manifest:
        for v in VIEWS:
            path_to_view[m["views"][v]] = v

    def render_view(item):
        return path_to_view.get(item["images"][0], "UNKNOWN")

    items = {it["id"]: it for it in json.load(open(args.items_file))["items"]}

    prefix = "task1_dim_" if args.dim else "task1_"
    models = {
        "Claude Sonnet 5":   f"{prefix}anthropic_claude-sonnet-5.json",
        "Gemini 3.6 Flash":  f"{prefix}gemini_gemini-3.6-flash.json",
        "GPT-5.6 Luna":      f"{prefix}openai_gpt-5.6-luna.json",
        "Mistral-S 3.2 24B": f"{prefix}qwen_mistralai_Mistral-Small-3.2-24B-Instruct-2506.json",
        "InternVL3.5 14B":   f"{prefix}qwen_OpenGVLab_InternVL3_5-14B.json",
        "Qwen3-VL 8B":       f"{prefix}qwen_Qwen_Qwen3-VL-8B-Instruct.json",
    }

    def load_results(fname):
        path = f"{args.results_dir}/{fname}"
        return {r["id"]: r for r in json.load(open(path))["results"] if r["pred"] in ("yes", "no")}

    unknown_total = 0
    for name, fname in models.items():
        res = load_results(fname)
        pos_by_view = collections.defaultdict(list)
        neg_by_view = collections.defaultdict(list)
        n_unknown = 0
        for rid, r in res.items():
            it = items.get(rid)
            if it is None:
                continue
            v = render_view(it)
            if v == "UNKNOWN":
                n_unknown += 1
                continue
            correct = (r["pred"] == r["answer"])
            (pos_by_view if r["meta"]["tier"] == "positive" else neg_by_view)[v].append(correct)
        unknown_total += n_unknown
        print(f"\n{name}  (unmatched: {n_unknown})")
        for v in VIEWS:
            p, n = pos_by_view[v], neg_by_view[v]
            pacc = sum(p) / len(p) if p else float("nan")
            nacc = sum(n) / len(n) if n else float("nan")
            print(f"  {v:8s} pos={pacc:.2f} (n={len(p)})  neg={nacc:.2f} (n={len(n)})")

    print(f"\nTOTAL unmatched render paths across all models: {unknown_total}")

if __name__ == "__main__":
    main()

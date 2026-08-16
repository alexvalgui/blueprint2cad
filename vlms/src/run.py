import argparse
import json
import re
import time
from pathlib import Path

from prompts import PROMPTS
from tasks import BUILDERS
from vlm import get_vlm
import metrics

NUMERIC_TASKS = {6, 7}


def normalize(raw, task):
    raw = raw.strip()
    if task == 1:
        m = re.search(r"\b(yes|no)\b", raw, re.I)
        return m.group(1).lower() if m else raw.lower()
    if task in (2, 3, 5):
        m = re.search(r"\b([A-Y])\b", raw.upper())
        return m.group(1) if m else raw.upper()
    if task == 4:
        m = re.search(r"\b(simple|complex)\b", raw, re.I)
        return m.group(1).lower() if m else raw.lower()
    m = re.search(r"-?\d+\.?\d*", raw)
    return m.group(0) if m else raw


def run(args):
    manifest = json.loads(Path(args.manifest).read_text())
    vlm = get_vlm(args.provider, args.model, base_url=args.base_url)

    builder_kwargs = {"seed": args.seed}
    if args.task == 3:
        builder_kwargs["n_candidates"] = args.n_candidates
    if args.task in (1, 3):
        builder_kwargs["dim"] = args.dimensioned
    items = BUILDERS[args.task](manifest, args.n_items, **builder_kwargs)

    results = []
    for i, item in enumerate(items):
        prompt = PROMPTS[item["prompt_key"]].format(**item["prompt_vars"])
        try:
            raw = vlm.ask(prompt, item["images"])
        except Exception as e:
            raw = f"ERROR: {e}"
        pred = normalize(raw, args.task)
        results.append({**{k: item[k] for k in ("id", "answer", "group", "meta")},
                        "raw": raw, "pred": pred})
        print(f"[{i+1}/{len(items)}] {item['id']} -> {pred} (gt: {item['answer']})")
        time.sleep(args.delay)

    out = Path(args.out) / f"task{args.task}_{args.provider}_{args.model.replace('/', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    y_true = [str(r["answer"]).lower() for r in results]
    y_pred = [str(r["pred"]).lower() for r in results]
    groups = [r["group"] for r in results]

    if args.task in NUMERIC_TASKS:
        report = {
            "overall": metrics.numeric_metrics(y_true, y_pred),
            "by_complexity": metrics.by_group(y_true, y_pred, groups,
                                              metric_fn=metrics.numeric_metrics),
        }
    else:
        labels = sorted(set(y_true))
        report = {
            "overall": metrics.classification_metrics(y_true, y_pred, labels=labels),
            "by_complexity": metrics.by_group(y_true, y_pred, groups, labels=labels),
        }
        if args.task == 1:
            tiers = [r["meta"]["tier"] for r in results]
            report["by_tier"] = metrics.by_group(y_true, y_pred, tiers, labels=labels)

    out.write_text(json.dumps({"config": vars(args), "results": results,
                               "metrics": report}, indent=2, default=str))
    print(json.dumps(report["overall"], indent=2, default=str))
    print(f"Saved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, required=True, choices=range(1, 8))
    p.add_argument("--manifest", required=True)
    p.add_argument("--provider", required=True,
                   choices=["openai", "qwen", "gemini", "anthropic"])
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default=None)
    p.add_argument("--n-items", type=int, default=500)
    p.add_argument("--n-candidates", type=int, default=4)
    p.add_argument("--dimensioned", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--delay", type=float, default=0.5)
    p.add_argument("--out", default="results")
    run(p.parse_args())

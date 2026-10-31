import argparse
import concurrent.futures
import json
import re
import time
from pathlib import Path

from prompts import PROMPTS
from tasks import BUILDERS
from vlm import get_vlm
import metrics

NUMERIC_TASKS = {6, 7}
FATAL_MARKERS = ("insufficient_quota", "credit", "model_not_supported",
                 "401", "invalid_api_key")


def normalize(raw, task, cot=False):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if task == 1:
        if cot:
            # score only the final non-empty line; unparseable -> UNPARSEABLE (excluded)
            lines = [l for l in raw.splitlines() if l.strip()]
            last = lines[-1].strip() if lines else ""
            # the verdict must open the final line: "Yes", "**No**", "Answer: Yes",
            # "No, the holes differ" -- but not a sentence that merely contains the
            # word somewhere ("there is no match here")
            m = re.match(r"(?:(?:final\s+)?answer\s*[:\-]?\s*)?[*_\"'(]*(yes|no)\b",
                         last, re.I)
            return m.group(1).lower() if m else "UNPARSEABLE"
        m = re.search(r"\b(yes|no)\b", raw, re.I)
        return m.group(1).lower() if m else raw.lower()
    if task in (2, 3, 5):
        up = raw.upper()
        for pat in (r"ANSWER[:\s]*\(?([A-Y])\)?\b", r"\b([A-Y])\b(?!.*\b[A-Y]\b)"):
            m = re.search(pat, up, re.S)
            if m:
                return m.group(1)
        return up
    if task == 4:
        m = re.search(r"\b(simple|complex)\b", raw, re.I)
        return m.group(1).lower() if m else raw.lower()
    if task in (6, 7):
        nums = re.findall(r"-?\d+\.?\d*", raw)
        return ",".join(nums[:2]) if nums else raw
    m = re.search(r"-?\d+\.?\d*", raw)
    return m.group(0) if m else raw


def prepare(args):
    manifest = json.loads(Path(args.manifest).read_text())
    builder_kwargs = {"seed": args.seed}
    if args.task == 3:
        builder_kwargs["n_candidates"] = args.n_candidates
    if args.task in (1, 3):
        builder_kwargs["dim"] = args.dimensioned
    items = BUILDERS[args.task](manifest, args.n_items, **builder_kwargs)

    out = Path(args.items_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "items": items}, indent=2))

    groups = {}
    tiers = {}
    for it in items:
        groups[it["group"]] = groups.get(it["group"], 0) + 1
        t = it["meta"].get("tier")
        if t:
            tiers[t] = tiers.get(t, 0) + 1
    print(f"Items: {len(items)}  by class: {groups}")
    if tiers:
        print(f"By tier: {tiers}")
    print(f"Saved: {out}")


def _one(vlm, item, task, delay, cot=False):
    key = item["prompt_key"]
    if cot:
        key = key + "_cot"  # e.g. task1_verification -> task1_verification_cot
    prompt = PROMPTS[key].format(**item["prompt_vars"])
    try:
        raw = vlm.ask(prompt, item["images"])
        pred = normalize(raw, task, cot=cot)
    except Exception as e:
        raw = f"ERROR: {e}"
        pred = "ERROR"
    if delay:
        time.sleep(delay)
    return {**{k: item[k] for k in ("id", "answer", "group", "meta")},
            "raw": raw, "pred": pred}


def run(args):
    data = json.loads(Path(args.items_file).read_text())
    items = data["items"]
    if args.limit:
        items = items[:args.limit]
    task = data["config"]["task"]

    vlm_kwargs = {"max_tokens": args.max_tokens}
    if args.base_url:
        vlm_kwargs["base_url"] = args.base_url
    if args.adapter:
        vlm_kwargs["adapter"] = args.adapter
    vlm = get_vlm(args.provider, args.model, **vlm_kwargs)

    results = []
    if args.concurrency > 1:
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as ex:
            futs = [ex.submit(_one, vlm, it, task, args.delay, args.cot) for it in items]
            for i, f in enumerate(futs):
                r = f.result()
                results.append(r)
                print(f"[{i+1}/{len(items)}] {r['id']} -> {r['pred']} (gt: {r['answer']})")
    else:
        for i, item in enumerate(items):
            r = _one(vlm, item, task, args.delay, args.cot)
            results.append(r)
            print(f"[{i+1}/{len(items)}] {r['id']} -> {r['pred']} (gt: {r['answer']})")
            if r["pred"] == "ERROR" and any(s in r["raw"] for s in FATAL_MARKERS):
                print(f"FATAL: {r['raw']}\nStopping run — fix the issue and rerun.")
                break

    stem = Path(args.items_file).stem.replace("_items", "")
    suffix = ("_cot" if args.cot else "") + (f"_limit{args.limit}" if args.limit else "")
    model_tag = args.model.replace("/", "_")
    if args.adapter:
        model_tag += f"_{Path(args.adapter.rstrip('/')).name}"
    out = Path(args.out) / f"{stem}_{args.provider}_{model_tag}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    scored = [r for r in results if r["pred"] not in ("ERROR", "UNPARSEABLE")]
    n_errors = sum(r["pred"] == "ERROR" for r in results)
    n_unparseable = sum(r["pred"] == "UNPARSEABLE" for r in results)

    y_true = [str(r["answer"]).lower() for r in scored]
    y_pred = [str(r["pred"]).lower() for r in scored]
    groups = [r["group"] for r in scored]

    if not scored:
        report = {"overall": None, "n_errors": n_errors, "n_unparseable": n_unparseable}
    elif task in NUMERIC_TASKS:
        report = {
            "overall": metrics.numeric_metrics(y_true, y_pred),
            "by_complexity": metrics.by_group(y_true, y_pred, groups,
                                              metric_fn=metrics.numeric_metrics),
            "n_errors": n_errors,
        }
    else:
        labels = sorted(set(y_true))
        report = {
            "overall": metrics.classification_metrics(y_true, y_pred, labels=labels),
            "by_complexity": metrics.by_group(y_true, y_pred, groups, labels=labels),
            "n_errors": n_errors,
            "n_unparseable": n_unparseable,
        }
        if task == 1:
            tiers = [r["meta"]["tier"] for r in scored]
            report["by_tier"] = metrics.by_group(y_true, y_pred, tiers, labels=labels)

    out.write_text(json.dumps({"config": vars(args), "items_file": args.items_file,
                               "results": results, "metrics": report},
                              indent=2, default=str))
    print(json.dumps(report["overall"], indent=2, default=str))
    if n_errors:
        print(f"WARNING: {n_errors} items errored and were excluded from metrics")
    if n_unparseable:
        print(f"WARNING: {n_unparseable} CoT responses had no Yes/No on the final line "
              f"and were excluded from metrics")
    print(f"Saved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--task", type=int, required=True, choices=range(1, 8))
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--n-items", type=int, default=500)
    prep.add_argument("--n-candidates", type=int, default=4)
    prep.add_argument("--dimensioned", action="store_true")
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--items-file", default="items/task1_items.json")

    runp = sub.add_parser("run")
    runp.add_argument("--items-file", required=True)
    runp.add_argument("--provider", required=True,
                      choices=["openai", "qwen", "gemini", "anthropic", "local"])
    runp.add_argument("--model", required=True,
                      help="API model name, or HF base checkpoint for --provider local "
                           "('auto' = read it from the adapter config)")
    runp.add_argument("--adapter", default=None,
                      help="path to a LoRA/PEFT adapter dir (--provider local only)")
    runp.add_argument("--limit", type=int, default=None)
    runp.add_argument("--base-url", default=None)
    runp.add_argument("--max-tokens", type=int, default=2000)
    runp.add_argument("--delay", type=float, default=0.5)
    runp.add_argument("--concurrency", type=int, default=1)
    runp.add_argument("--out", default="results")
    runp.add_argument("--cot", action="store_true",
                      help="use the <prompt_key>_cot prompt and score only the final line "
                           "(task 1); output file gets a _cot suffix")

    args = p.parse_args()
    prepare(args) if args.cmd == "prepare" else run(args)
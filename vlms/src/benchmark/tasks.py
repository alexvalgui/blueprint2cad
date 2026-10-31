"""Each build_task* function yields items:
{"id": str, "images": [paths], "prompt_key": str, "prompt_vars": dict,
 "answer": str, "group": "simple"|"complex", "meta": dict}

Expected manifest (JSON list), one entry per model:
{
  "id": "model_0001",
  "complexity": "simple",                    # or "complex"
  "views": {"front": "...", "back": "...", "left": "...",
            "right": "...", "top": "...", "bottom": "..."},
  "drawing_dim": "path.png",
  "drawing_nodim": "path.png"
}
"""
import random

VIEWS = ["front", "back", "left", "right", "top", "bottom"]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXY"


def _drawing(m, dim):
    return m["drawing_dim"] if dim else m["drawing_nodim"]


def build_task1(manifest, n_items, dim=True, seed=0):
    """Task 1: drawing-render verification (Yes/No)."""
    rng = random.Random(seed)
    simple = [m for m in manifest if m["complexity"] == "simple"]
    complx = [m for m in manifest if m["complexity"] == "complex"]
    k = min(n_items // 2, len(simple), len(complx))
    models = rng.sample(simple, k) + rng.sample(complx, k)
    rng.shuffle(models)
    items = []
    for i, m in enumerate(models):
        view = rng.choice(VIEWS)
        r = i % 4
        if r < 2:  # positive
            drawing, answer, tier = _drawing(m, dim), "yes", "positive"
        elif r == 2:  # easy negative: another model's drawing
            other = rng.choice([x for x in manifest if x["id"] != m["id"]])
            drawing, answer, tier = _drawing(other, dim), "no", "easy_neg"
        else:  # hard negative: another model of the same complexity class
            other = rng.choice([x for x in manifest if x["id"] != m["id"]
                                and x["complexity"] == m["complexity"]])
            drawing, answer, tier = _drawing(other, dim), "no", "hard_neg"
        items.append({
            "id": f"t1_{m['id']}_{i}",
            "images": [m["views"][view], drawing],
            "prompt_key": "task1_verification", "prompt_vars": {},
            "answer": answer, "group": m["complexity"], "meta": {"tier": tier},
        })
    return items


def build_task2(manifest, n_items, n_candidates=4, dim=False, seed=0):
    """Task 2: drawing-render retrieval among n_candidates renders (one letter)."""
    rng = random.Random(seed)
    simple = [m for m in manifest if m["complexity"] == "simple"]
    complx = [m for m in manifest if m["complexity"] == "complex"]
    k = min(n_items // 2, len(simple), len(complx))
    models = rng.sample(simple, k) + rng.sample(complx, k)
    rng.shuffle(models)
    items = []
    for m in models:
        same = [x for x in manifest if x["id"] != m["id"]
                and x["complexity"] == m["complexity"]]
        other = [x for x in manifest if x["id"] != m["id"]]
        n_hard = (n_candidates - 1) // 2
        distractors = rng.sample(same, n_hard) + rng.sample(other, n_candidates - 1 - n_hard)
        candidates = distractors[:n_candidates - 1] + [m]
        rng.shuffle(candidates)
        answer = LETTERS[[c["id"] for c in candidates].index(m["id"])]
        labels = ", ".join(LETTERS[:n_candidates])
        items.append({
            "id": f"t2_{m['id']}_n{n_candidates}",
            "images": [_drawing(m, dim)] + [c["views"]["front"] for c in candidates],
            "prompt_key": "task2_retrieval",
            "prompt_vars": {"n": n_candidates, "labels": labels},
            "answer": answer, "group": m["complexity"],
            "meta": {"n_candidates": n_candidates},
        })
    return items


def build_task3(manifest, n_items, seed=0):
    """Task 3: unguided complexity probe (Simple/Complex from six views)."""
    rng = random.Random(seed)
    simple = [m for m in manifest if m["complexity"] == "simple"]
    complx = [m for m in manifest if m["complexity"] == "complex"]
    k = min(n_items // 2, len(simple), len(complx))
    models = rng.sample(simple, k) + rng.sample(complx, k)
    rng.shuffle(models)
    return [{
        "id": f"t3_{m['id']}",
        "images": [m["views"][v] for v in VIEWS],
        "prompt_key": "task3_complexity", "prompt_vars": {},
        "answer": m["complexity"],
        "group": m["complexity"], "meta": {},
    } for m in models]


BUILDERS = {1: build_task1, 2: build_task2, 3: build_task3}
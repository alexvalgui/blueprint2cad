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
  "drawing_nodim": "path.png",
  "symmetric": false,                        # optional, for tasks 2/5 filtering
  "dimensions": [{"question": "...", "value": 12.5}],   # optional, task 6
  "counts": [{"question": "...", "value": 4}]           # optional, task 7
}
"""
import random

VIEWS = ["front", "back", "left", "right", "top", "bottom"]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXY"


def _drawing(m, dim):
    return m["drawing_dim"] if dim else m["drawing_nodim"]


def build_task1(manifest, n_items, dim=True, seed=0):
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
        else:  # hard negative: assumes per-view drawings; falls back to easy neg
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


def build_task2(manifest, n_items, seed=0):
    rng = random.Random(seed)
    pool = [m for m in manifest if not m.get("symmetric", False)]
    models = rng.sample(pool, min(n_items, len(pool)))
    options = ["back", "left", "right", "top", "bottom"]
    items = []
    for i, m in enumerate(models):
        view = rng.choice(options)
        items.append({
            "id": f"t2_{m['id']}",
            "images": [m["views"]["front"], m["views"][view]],
            "prompt_key": "task2_view_id", "prompt_vars": {},
            "answer": LETTERS[options.index(view)],
            "group": m["complexity"], "meta": {"true_view": view},
        })
    return items


def build_task3(manifest, n_items, n_candidates=4, dim=False, seed=0):
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
            "id": f"t3_{m['id']}_n{n_candidates}",
            "images": [_drawing(m, dim)] + [c["views"]["front"] for c in candidates],
            "prompt_key": "task3_retrieval",
            "prompt_vars": {"n": n_candidates, "labels": labels},
            "answer": answer, "group": m["complexity"],
            "meta": {"n_candidates": n_candidates},
        })
    return items


def build_task4(manifest, n_items, seed=0):
    rng = random.Random(seed)
    simple = [m for m in manifest if m["complexity"] == "simple"]
    complx = [m for m in manifest if m["complexity"] == "complex"]
    k = min(n_items // 2, len(simple), len(complx))
    models = rng.sample(simple, k) + rng.sample(complx, k)
    rng.shuffle(models)
    return [{
        "id": f"t4_{m['id']}",
        "images": [m["views"][v] for v in VIEWS],
        "prompt_key": "task4_complexity", "prompt_vars": {},
        "answer": m["complexity"],
        "group": m["complexity"], "meta": {},
    } for m in models]


def build_task5(manifest, n_items, seed=0):
    rng = random.Random(seed)
    pool = [m for m in manifest if not m.get("symmetric", False)]
    models = rng.sample(pool, min(n_items, len(pool)))
    items = []
    for m in models:
        similar = rng.choice([x for x in pool if x["id"] != m["id"]
                              and x["complexity"] == m["complexity"]])
        candidates = [
            ("correct", m["views"]["right"]),
            ("similar_model", similar["views"]["right"]),
            ("mirror", m["views"]["left"]),      # left view stands in for mirrored right
            ("same_model_other", m["views"]["back"]),
        ]
        rng.shuffle(candidates)
        answer = LETTERS[[c[0] for c in candidates].index("correct")]
        items.append({
            "id": f"t5_{m['id']}",
            "images": [m["views"]["front"], m["views"]["top"]] + [c[1] for c in candidates],
            "prompt_key": "task5_third_view", "prompt_vars": {},
            "answer": answer, "group": m["complexity"], "meta": {},
        })
    return items


def build_task6(manifest, n_items, seed=0):
    rng = random.Random(seed)
    pool = [m for m in manifest if m.get("drawing_dim")]
    simple = [m for m in pool if m["complexity"] == "simple"]
    complx = [m for m in pool if m["complexity"] == "complex"]
    k = min(n_items // 2, len(simple), len(complx))
    models = rng.sample(simple, k) + rng.sample(complx, k)
    rng.shuffle(models)
    return [{
        "id": f"t6_{m['id']}",
        "images": [m["drawing_dim"]],
        "prompt_key": "task6_dimension", "prompt_vars": {},
        "answer": "", "group": m["complexity"], "meta": {},
    } for m in models]


def build_task7(manifest, n_items, seed=0):
    rng = random.Random(seed)
    pool = [m for m in manifest if m.get("counts")]
    rng.shuffle(pool)
    items = []
    for m in pool:
        for j, q in enumerate(m["counts"]):
            view = q.get("view", "front")
            items.append({
                "id": f"t7_{m['id']}_{j}",
                "images": [m["views"][view]],
                "prompt_key": "task7_counting",
                "prompt_vars": {"question": q["question"]},
                "answer": q["value"], "group": m["complexity"], "meta": {},
            })
            if len(items) >= n_items:
                return items
    return items


BUILDERS = {
    1: build_task1, 2: build_task2, 3: build_task3, 4: build_task4,
    5: build_task5, 6: build_task6, 7: build_task7,
}

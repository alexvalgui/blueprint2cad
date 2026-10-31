import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, mean_absolute_error)


def classification_metrics(y_true, y_pred, labels=None):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": p,
        "recall_macro": r,
        "f1_macro": f1,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
        "n": len(y_true),
    }


def numeric_metrics(y_true, y_pred, tolerance=0.01):
    y_true = np.array(y_true, dtype=float)
    parsed, valid = [], []
    for v in y_pred:
        try:
            parsed.append(float(v))
            valid.append(True)
        except (ValueError, TypeError):
            parsed.append(np.nan)
            valid.append(False)
    parsed = np.array(parsed)
    valid = np.array(valid)
    exact = valid & (parsed == y_true)
    within = valid & (np.abs(parsed - y_true) <= tolerance * np.abs(y_true))
    return {
        "exact_match": exact.mean(),
        f"within_{int(tolerance*100)}pct": within.mean(),
        "parse_rate": valid.mean(),
        "mae_on_parsed": (mean_absolute_error(y_true[valid], parsed[valid])
                          if valid.any() else None),
        "n": len(y_true),
    }


def by_group(y_true, y_pred, groups, metric_fn=classification_metrics, **kwargs):
    out = {}
    for g in sorted(set(groups)):
        idx = [i for i, x in enumerate(groups) if x == g]
        out[g] = metric_fn([y_true[i] for i in idx], [y_pred[i] for i in idx], **kwargs)
    return out

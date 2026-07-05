"""Evaluation metrics — computed from scratch so you can see exactly what they mean.

Given the true labels and a diagnoser's predictions, we compute:
  - accuracy       : fraction correct (the headline number)
  - per-class P/R/F1: precision, recall, F1 for each root-cause class
  - macro F1       : average F1 across classes (fair when classes are balanced)
  - confusion      : which classes get mistaken for which

We use plain Python (not sklearn) here so the math is transparent and the module
is easy to test. The definitions:
  precision = of the times we PREDICTED class C, how often were we right?
  recall    = of the ACTUAL class-C incidents, how many did we catch?
  F1        = harmonic mean of precision and recall.
"""

from __future__ import annotations


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Return accuracy, per-class precision/recall/F1, macro F1, and a confusion map."""
    assert len(y_true) == len(y_pred), "true and pred must be the same length"
    n = len(y_true)
    labels = sorted(set(y_true) | set(y_pred))

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / n if n else 0.0

    per_class = {}
    f1s = []
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        precision, recall, f1 = _prf(tp, fp, fn)
        support = sum(1 for t in y_true if t == c)
        per_class[c] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": support,
        }
        if support:  # only average over classes that actually occur in the truth
            f1s.append(f1)

    macro_f1 = round(sum(f1s) / len(f1s), 3) if f1s else 0.0

    # confusion: confusion[true][pred] = count
    confusion = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1

    return {
        "n": n,
        "accuracy": round(accuracy, 3),
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion": confusion,
    }


def print_report(name: str, metrics: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"  n={metrics['n']}  accuracy={metrics['accuracy']:.3f}  macro_f1={metrics['macro_f1']:.3f}")
    print("  per-class (precision / recall / f1 / support):")
    for c, m in metrics["per_class"].items():
        print(f"    {c:<28} {m['precision']:.2f} / {m['recall']:.2f} / {m['f1']:.2f}  (n={m['support']})")

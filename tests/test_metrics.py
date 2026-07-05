"""Tests for the metrics math — no data or models needed, pure logic."""

from evaluation.metrics import compute_metrics


def test_perfect_predictions():
    y = ["a", "b", "a", "c"]
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_all_wrong():
    m = compute_metrics(["a", "a"], ["b", "b"])
    assert m["accuracy"] == 0.0


def test_known_precision_recall():
    # class 'x': truth has 2, we predict 'x' twice, one right one wrong
    y_true = ["x", "x", "y"]
    y_pred = ["x", "y", "x"]
    m = compute_metrics(y_true, y_pred)
    # accuracy: only the first is correct -> 1/3 (rounded to 3 dp -> 0.333)
    assert abs(m["accuracy"] - 1 / 3) < 0.01
    px = m["per_class"]["x"]
    # predicted x twice (idx0 correct, idx2 wrong) -> precision 1/2
    assert abs(px["precision"] - 0.5) < 1e-9
    # actual x twice (idx0 caught, idx1 missed) -> recall 1/2
    assert abs(px["recall"] - 0.5) < 1e-9


def test_confusion_counts():
    m = compute_metrics(["a", "a", "b"], ["a", "b", "b"])
    assert m["confusion"]["a"]["a"] == 1
    assert m["confusion"]["a"]["b"] == 1
    assert m["confusion"]["b"]["b"] == 1

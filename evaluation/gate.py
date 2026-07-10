"""Regression gate — fails (exit code 1) if eval metrics drop below a threshold.

This is what turns the evaluation harness into a *gate*: CI (or a pre-deploy check)
runs this, and a non-zero exit fails the pipeline, so a quality regression blocks
shipping — not just a code bug.

Why it's a separate, scheduled/on-demand job (not every push): a live LLM eval is
slow, costs quota, and is non-deterministic. Running it on every commit would be
flaky and expensive. So the honest pattern is to gate on the held-out test set on a
schedule (nightly) or on demand, with configurable thresholds.

Usage:
    # check the latest SAVED results (fast, no LLM/key needed):
    uv run python evaluation/gate.py --diagnoser agent

    # run the eval LIVE first, then gate (needs an LLM key + quota):
    uv run python evaluation/gate.py --diagnoser agent --run

Thresholds come from flags or env (GATE_MIN_ACCURACY, GATE_MIN_MACRO_F1).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from evaluation.run_eval import RESULTS, evaluate  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Fail if eval metrics regress below a threshold.")
    p.add_argument("--diagnoser", default="agent", choices=["agent", "single_shot"])
    p.add_argument("--split", default="test")
    p.add_argument("--run", action="store_true",
                   help="run the eval live first (needs an LLM key); else read saved results")
    p.add_argument("--min-accuracy", type=float,
                   default=float(os.getenv("GATE_MIN_ACCURACY", "0.83")))
    p.add_argument("--min-macro-f1", type=float,
                   default=float(os.getenv("GATE_MIN_MACRO_F1", "0.77")))
    args = p.parse_args()

    # 1. Get the metrics — either from a fresh live run or the last saved results.
    if args.run:
        out = evaluate(args.diagnoser, args.split, fresh=True)
        if out is None:  # eval didn't finish (e.g. quota) — do not let it pass
            print("GATE FAIL: eval did not complete (quota/rate limit?). Blocking.")
            sys.exit(1)
        metrics = out["metrics"]
    else:
        path = RESULTS / f"{args.diagnoser}_{args.split}.json"
        if not path.exists():
            print(f"GATE FAIL: no saved results at {path}. Run with --run, or run the "
                  f"evaluator first.")
            sys.exit(1)
        metrics = json.loads(path.read_text())["metrics"]

    # 2. Compare to thresholds.
    acc = metrics["accuracy"]
    f1 = metrics["macro_f1"]
    passed = acc >= args.min_accuracy and f1 >= args.min_macro_f1
    status = "PASS" if passed else "FAIL"

    print(f"GATE {status} [{args.diagnoser}/{args.split}]  "
          f"accuracy={acc:.3f} (min {args.min_accuracy:.2f})  "
          f"macro_f1={f1:.3f} (min {args.min_macro_f1:.2f})")
    if not passed:
        print("  -> a metric regressed below its threshold; blocking the pipeline.")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

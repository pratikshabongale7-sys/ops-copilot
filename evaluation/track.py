"""Log evaluation results to MLflow — the experiment-tracking layer (Phase 5).

Why MLflow? Once you start tweaking things — a different model, a new prompt, RAG
on/off — you accumulate many eval runs and need to compare them systematically.
MLflow records each run's **parameters** (diagnoser, model, provider, split),
**metrics** (accuracy, macro-F1, and efficiency if present), and **artifacts**
(the results JSON), so you can line them up side by side in a UI over time.

This script does NOT call the LLM — it just reads the results JSONs you already
produced (evaluation/results/*_test.json) and logs them. So you can track what you
have now, and re-run it after future eval runs to add more.

Usage:
    uv run python evaluation/track.py                 # log all saved results
    uv run python evaluation/track.py --tag run1-8b   # add a tag to distinguish batches
    uv run mlflow ui                                  # then open http://localhost:5000
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib

import mlflow

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
EXPERIMENT = "ops-copilot-eval"


def _metric_or_skip(d: dict, path: list[str]):
    """Safely pull a nested numeric value; return None if missing/None."""
    cur = d
    for p in path:
        cur = (cur or {}).get(p) if isinstance(cur, dict) else None
    return cur if isinstance(cur, (int, float)) else None


def log_result_file(path: pathlib.Path, tag: str | None) -> None:
    d = json.loads(path.read_text())
    diagnoser = d.get("diagnoser", path.stem)
    model = d.get("model", "unknown")
    provider = d.get("provider", "unknown")
    split = d.get("split", "test")

    run_name = f"{diagnoser}-{model}"
    with mlflow.start_run(run_name=run_name):
        # parameters — the "what was this run" knobs
        mlflow.log_params({
            "diagnoser": diagnoser,
            "model": model,
            "provider": provider,
            "split": split,
        })
        if tag:
            mlflow.set_tag("batch", tag)

        # metrics — only log the ones that are actually present
        metrics = {
            "accuracy": _metric_or_skip(d, ["metrics", "accuracy"]),
            "macro_f1": _metric_or_skip(d, ["metrics", "macro_f1"]),
            "avg_latency_s": _metric_or_skip(d, ["efficiency", "avg_latency_s"]),
            "avg_llm_calls": _metric_or_skip(d, ["efficiency", "avg_llm_calls"]),
            "avg_total_tokens": _metric_or_skip(d, ["efficiency", "avg_total_tokens"]),
        }
        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(k, float(v))

        # artifact — keep the full results JSON attached to the run
        mlflow.log_artifact(str(path))

    logged = {k: v for k, v in metrics.items() if v is not None}
    print(f"  logged run '{run_name}': {logged}")


def main() -> None:
    p = argparse.ArgumentParser(description="Log eval results to MLflow.")
    p.add_argument("--tag", help="optional batch tag (e.g. 'run1-8b')")
    args = p.parse_args()

    files = sorted(glob.glob(str(RESULTS / "*_test.json")))
    if not files:
        print("No results found. Run the evaluators first "
              "(evaluation/run_eval.py --diagnoser ...).")
        return

    mlflow.set_experiment(EXPERIMENT)
    print(f"Logging {len(files)} result file(s) to MLflow experiment '{EXPERIMENT}'...")
    for f in files:
        log_result_file(pathlib.Path(f), args.tag)
    print("\nDone. View them with:  uv run mlflow ui   ->  http://localhost:5000")


if __name__ == "__main__":
    main()

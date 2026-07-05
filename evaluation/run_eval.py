"""Run an evaluator over the test set, score it, and (optionally) chart the results.

Usage:
    # score one diagnoser on the test split
    uv run python evaluation/run_eval.py --diagnoser single_shot
    uv run python evaluation/run_eval.py --diagnoser agent

    # after running both, build the comparison chart
    uv run python evaluation/run_eval.py --chart

Each run saves evaluation/results/<diagnoser>_<split>.json. The chart reads those
and produces docs/eval_results.png — the headline "single-shot vs agent" ablation
for your README.

The single_shot and agent evaluators call the LLM once and ~several times per
incident respectively, so they cost a few Groq calls each — fine on the free tier
for 6 test incidents.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.incidents import list_incidents  # noqa: E402
from evaluation.metrics import compute_metrics, print_report  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
CHART = pathlib.Path(__file__).resolve().parent.parent / "docs" / "eval_results.png"

# Nice labels for the chart
PRETTY = {"single_shot": "Single-shot LLM", "agent": "Agentic loop"}


async def _diagnose_one(which: str, incident_id: str) -> tuple[str, dict]:
    """Return (predicted_root_cause, stats) where stats has llm_calls, total_tokens."""
    if which == "agent":
        from app.agent import diagnose_verbose

        dx, stats = await diagnose_verbose(incident_id)
        return dx.root_cause, stats
    from app.baselines import single_shot_verbose  # single_shot

    return await single_shot_verbose(incident_id)


async def _run_and_cache(remaining, which, cache, cache_path) -> None:
    """Diagnose each remaining incident, saving to the cache file AS WE GO.

    Each result is written to disk immediately, so if the LLM quota runs out (or
    anything else fails) mid-run, all completed incidents are preserved. Re-running
    skips them and continues from where it stopped. We also record efficiency:
    wall-clock latency, number of LLM calls, and total tokens per diagnosis.
    """
    import time

    for inc in remaining:
        t0 = time.perf_counter()
        try:
            pred, stats = await _diagnose_one(which, inc.incident_id)
        except Exception as e:  # rate limit, network, etc. — stop but keep progress
            print(f"\n  stopped at {inc.incident_id}: {type(e).__name__}: {e}")
            print(f"  progress saved ({len(cache)} done). Re-run the same command "
                  f"later to continue from here.")
            return
        cache[inc.incident_id] = {
            "true": inc.root_cause,
            "pred": pred,
            "latency_s": round(time.perf_counter() - t0, 2),
            "llm_calls": stats.get("llm_calls"),
            "total_tokens": stats.get("total_tokens"),
        }
        cache_path.write_text(json.dumps(cache, indent=2))
        mark = "OK " if pred == inc.root_cause else "XX "
        print(f"  {mark} {inc.incident_id}: {pred}  (true {inc.root_cause})  "
              f"[{cache[inc.incident_id]['latency_s']}s, "
              f"{stats.get('llm_calls')} calls, {stats.get('total_tokens')} tok]")


def evaluate(which: str, split: str = "test", fresh: bool = False) -> dict | None:
    incidents = list_incidents(split=split)
    RESULTS.mkdir(exist_ok=True)
    cache_path = RESULTS / f"{which}_{split}_cache.json"

    cache = {}
    if cache_path.exists() and not fresh:
        cache = json.loads(cache_path.read_text())

    remaining = [i for i in incidents if i.incident_id not in cache]
    print(f"\nEvaluating '{which}' on {len(incidents)} {split} incidents "
          f"({len(cache)} cached, {len(remaining)} to run)...")

    if remaining:
        asyncio.run(_run_and_cache(remaining, which, cache, cache_path))
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else cache

    done = [i for i in incidents if i.incident_id in cache]
    if len(done) < len(incidents):
        print(f"\nPartial: {len(done)}/{len(incidents)} done. Re-run to finish, "
              f"then the final metrics + chart will be written.")
        return None

    # All incidents done -> compute metrics and write the final results file.
    y_true = [cache[i.incident_id]["true"] for i in incidents]
    y_pred = [cache[i.incident_id]["pred"] for i in incidents]
    metrics = compute_metrics(y_true, y_pred)
    efficiency = _efficiency(cache, incidents)
    import os

    provider = os.getenv("LLM_PROVIDER", "groq")
    model = os.getenv(
        {"groq": "GROQ_MODEL", "google": "GOOGLE_MODEL", "openai": "OPENAI_MODEL"}
        .get(provider, "GROQ_MODEL"),
        "unknown",
    )
    out = {
        "diagnoser": which,
        "split": split,
        "provider": provider,
        "model": model,
        "per_incident": [
            {"id": i.incident_id, "true": cache[i.incident_id]["true"],
             "pred": cache[i.incident_id]["pred"],
             "correct": cache[i.incident_id]["true"] == cache[i.incident_id]["pred"]}
            for i in incidents
        ],
        "metrics": metrics,
        "efficiency": efficiency,
    }
    (RESULTS / f"{which}_{split}.json").write_text(json.dumps(out, indent=2))
    print_report(PRETTY.get(which, which), metrics)
    print(f"  efficiency (avg per incident): {efficiency['avg_latency_s']}s, "
          f"{efficiency['avg_llm_calls']} LLM calls, "
          f"{efficiency['avg_total_tokens']} tokens")
    return out


def _efficiency(cache: dict, incidents) -> dict:
    """Average latency / LLM calls / tokens over the incidents (skips missing)."""
    def avg(key):
        vals = [cache[i.incident_id].get(key) for i in incidents]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    return {
        "avg_latency_s": avg("latency_s"),
        "avg_llm_calls": avg("llm_calls"),
        "avg_total_tokens": avg("total_tokens"),
    }


def make_chart(split: str = "test") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(RESULTS.glob(f"*_{split}.json"))
    if not files:
        print("No results yet. Run --diagnoser single_shot / agent first.")
        return

    names, acc, f1 = [], [], []
    for f in files:
        d = json.loads(f.read_text())
        names.append(PRETTY.get(d["diagnoser"], d["diagnoser"]))
        acc.append(d["metrics"]["accuracy"])
        f1.append(d["metrics"]["macro_f1"])

    x = range(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([i - w / 2 for i in x], acc, w, label="Accuracy", color="#4f46e5")
    ax.bar([i + w / 2 for i in x], f1, w, label="Macro F1", color="#0d9488")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title(f"Root-cause diagnosis — {split} set")
    for i, (a, ff) in enumerate(zip(acc, f1)):
        ax.text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, ff + 0.02, f"{ff:.2f}", ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    CHART.parent.mkdir(exist_ok=True)
    fig.savefig(CHART, dpi=140)
    print(f"Saved chart -> {CHART}")

    _make_efficiency_chart(files, plt)


def _make_efficiency_chart(files, plt) -> None:
    """Second chart: avg tokens and latency per diagnosis. Guarded so a missing
    efficiency field never breaks the run."""
    try:
        names, tokens, latency = [], [], []
        for f in files:
            d = json.loads(f.read_text())
            eff = d.get("efficiency") or {}
            if eff.get("avg_total_tokens") is None and eff.get("avg_latency_s") is None:
                continue
            names.append(PRETTY.get(d["diagnoser"], d["diagnoser"]))
            tokens.append(eff.get("avg_total_tokens") or 0)
            latency.append(eff.get("avg_latency_s") or 0)
        if not names:
            print("  (no efficiency data yet — re-run the evaluators to capture it)")
            return

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
        a1.bar(names, tokens, color="#4f46e5")
        a1.set_title("Avg tokens / diagnosis")
        a2.bar(names, latency, color="#0d9488")
        a2.set_title("Avg latency (s) / diagnosis")
        for ax, vals in ((a1, tokens), (a2, latency)):
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        out = CHART.parent / "eval_efficiency.png"
        fig.savefig(out, dpi=140)
        print(f"Saved efficiency chart -> {out}")
    except Exception as e:
        print(f"  (skipped efficiency chart: {type(e).__name__}: {e})")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a diagnoser or chart results.")
    p.add_argument("--diagnoser", choices=["single_shot", "agent"])
    p.add_argument("--split", default="test")
    p.add_argument("--chart", action="store_true", help="build the comparison chart")
    p.add_argument("--fresh", action="store_true",
                   help="ignore cached results and start this diagnoser over")
    args = p.parse_args()

    if args.diagnoser:
        evaluate(args.diagnoser, args.split, fresh=args.fresh)
    if args.chart:
        make_chart(args.split)
    if not args.diagnoser and not args.chart:
        p.print_help()


if __name__ == "__main__":
    main()

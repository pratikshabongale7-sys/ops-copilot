"""Baseline to compare the agent against (Phase 4).

Why a baseline? "The agent gets 80% right" is meaningless alone — 80% vs WHAT?
The single-shot baseline gives the comparison that makes the agent's value legible.

Single-shot LLM: the SAME model, handed ALL the telemetry in one prompt, with no
tools and no loop. Comparing the agent to this isolates the value of the agentic
loop itself — the headline ablation ("does reasoning step-by-step and choosing what
to look at beat just dumping everything into one prompt?").
"""

from __future__ import annotations


async def single_shot_verbose(incident_id: str) -> tuple[str, dict]:
    """Give the LLM ALL the telemetry at once and ask for a root cause in one call.

    No tools, no reason-act loop. Same model, same information — just no agency.
    Comparing this to the agent isolates the value of the loop. Returns the
    predicted root cause plus efficiency stats {llm_calls, total_tokens}.
    """
    import json

    from app import tools
    from app.agent import SYSTEM_PROMPT, _build_llm
    from app.schemas import Diagnosis

    # We fetch everything for it (the model doesn't get to choose what to look at).
    # The context is built COMPACTLY — compact JSON (no whitespace), error logs
    # trimmed to their fingerprint (service + message, no timestamps), and capped
    # at 8 lines — so the single request fits smaller models' per-request limits
    # (e.g. Groq's 8B tier). Works identically on larger models, just cheaper.
    overview = tools.get_incident_overview(incident_id)
    metrics = tools.query_metrics(incident_id)
    errors = tools.search_logs(incident_id, level="ERROR", limit=8)
    deploys = tools.get_deploys(incident_id)

    err_lines = [f"{e.get('service')}: {e.get('message')}" for e in errors]
    j = lambda x: json.dumps(x, separators=(",", ":"))  # noqa: E731
    context = (
        "OVERVIEW: " + j(overview) + "\n"
        "METRIC SUMMARIES: " + j(metrics) + "\n"
        "ERROR LOGS: " + j(err_lines) + "\n"
        "DEPLOYS: " + j(deploys)
    )
    # include_raw=True so we can read token usage off the raw AI message.
    llm = _build_llm().with_structured_output(Diagnosis, include_raw=True)
    out = await llm.ainvoke(
        SYSTEM_PROMPT
        + "\n\nAll telemetry for the incident is provided below at once. "
        "Give your diagnosis directly.\n\n"
        + context
    )
    dx: Diagnosis = out["parsed"]
    raw = out.get("raw")
    um = getattr(raw, "usage_metadata", None) or {}
    stats = {"llm_calls": 1, "total_tokens": um.get("total_tokens")}
    return dx.root_cause, stats


async def single_shot_diagnose(incident_id: str) -> str:
    """Predicted root cause only (thin wrapper over single_shot_verbose)."""
    pred, _ = await single_shot_verbose(incident_id)
    return pred

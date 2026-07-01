"""Tool logic — the agent's 'senses'.

These are plain Python functions that read ONE incident's telemetry in small,
targeted slices: search the logs, query a metric, list the deploys, get an
overview. In Phase 2 they're wrapped as MCP tools (app/mcp_server.py); in Phase 3
the agent calls them one at a time as it investigates.

Two deliberate design choices:
  1. The logic lives here as pure functions (no MCP, no LLM) so it's easy to test
     and reuse. The MCP server is a thin wrapper on top.
  2. Every function loads the incident with `with_label=False`, so the answer key
     is NEVER visible to the agent. The agent must reason from evidence, not peek.
"""

from __future__ import annotations

# Allow running this file directly (python app/tools.py) as well as importing it
# as part of the `app` package. When run directly there is no package context, so
# we add the project root to the import path. No-op when imported normally.
if __package__ in (None, ""):
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.incidents import load_incident, resolve_incident_folder

# Metrics we expose. Keeping a known list helps the agent (and validates input).
KNOWN_METRICS = [
    "error_rate",
    "latency_p95_ms",
    "cpu_pct",
    "memory_mb",
    "db_active_connections",
    "request_rate",
]


def _summarize(values: list[float]) -> dict:
    """Condense a 30-point series into the few numbers that matter for diagnosis.

    Returning a summary (not 30 raw numbers per call) keeps tool outputs small and
    cheap, and surfaces exactly what an engineer eyeballs: how high it got, when,
    and whether it ended elevated."""
    if not values:
        return {}
    peak = max(values)
    return {
        "min": round(min(values), 2),
        "max": round(peak, 2),
        "mean": round(sum(values) / len(values), 2),
        "first": round(values[0], 2),
        "last": round(values[-1], 2),
        "peak_minute": values.index(peak),  # which minute the max occurred
    }


# --------------------------------------------------------------------------- #
# The tools
# --------------------------------------------------------------------------- #

def get_incident_overview(incident_id: str) -> dict:
    """High-level orientation for an incident: time window, which services have
    telemetry, and how many log lines / deploys exist. Call this first to decide
    where to look. Does NOT reveal the root cause."""
    folder = resolve_incident_folder(incident_id)
    inc = load_incident(folder, with_label=False)
    window = inc.metrics.get("window", {})
    return {
        "incident_id": incident_id,
        "window": window,
        "services": inc.services,
        "available_metrics": KNOWN_METRICS,
        "log_count": len(inc.logs),
        "deploy_count": len(inc.deploys),
        "hint": "Use search_logs and query_metrics to investigate each service.",
    }


def search_logs(
    incident_id: str,
    level: str | None = None,
    service: str | None = None,
    contains: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Search an incident's logs. Filter by level ('ERROR'/'WARN'/'INFO'),
    by service, and/or by a substring in the message (case-insensitive). Returns
    up to `limit` matching log lines, oldest first. Great for finding the error
    fingerprint of a failure (e.g. search level='ERROR')."""
    folder = resolve_incident_folder(incident_id)
    inc = load_incident(folder, with_label=False)

    results = []
    for row in inc.logs:
        if level and row.get("level", "").upper() != level.upper():
            continue
        if service and row.get("service") != service:
            continue
        if contains and contains.lower() not in row.get("message", "").lower():
            continue
        results.append(row)

    results.sort(key=lambda r: r.get("ts", ""))
    return results[:limit]


def query_metrics(
    incident_id: str,
    service: str | None = None,
    metric: str | None = None,
) -> dict:
    """Query an incident's metrics.

    - service + metric  -> full time-series values for that one series, plus a summary
    - service only      -> a summary of every metric for that service
    - metric only       -> a summary of that metric across all services
    - neither           -> a summary of every metric for every service (orientation)

    Summaries report min/max/mean/first/last and the minute the peak occurred —
    enough to spot spikes, ramps, and saturation without dumping raw arrays.
    """
    folder = resolve_incident_folder(incident_id)
    inc = load_incident(folder, with_label=False)
    series = inc.metrics.get("series", {})

    if service and service not in series:
        return {"error": f"unknown service '{service}'", "available": list(series)}
    if metric and metric not in KNOWN_METRICS:
        return {"error": f"unknown metric '{metric}'", "available": KNOWN_METRICS}

    # one specific series -> return raw values too (the agent may want the shape)
    if service and metric:
        values = series[service].get(metric, [])
        return {
            "incident_id": incident_id,
            "service": service,
            "metric": metric,
            "values": values,
            "summary": _summarize(values),
        }

    # otherwise return summaries only
    out: dict = {"incident_id": incident_id, "summaries": {}}
    services = [service] if service else list(series)
    metrics = [metric] if metric else KNOWN_METRICS
    for s in services:
        out["summaries"][s] = {
            m: _summarize(series[s].get(m, [])) for m in metrics
        }
    return out


def get_deploys(incident_id: str) -> list[dict]:
    """List recent deploy and config-change events for an incident, oldest first.
    Useful for correlating a failure's onset with a release ('did something ship
    right when errors spiked?')."""
    folder = resolve_incident_folder(incident_id)
    inc = load_incident(folder, with_label=False)
    return sorted(inc.deploys, key=lambda d: d.get("ts", ""))


# Registry so the MCP server and (later) the agent can enumerate tools.
TOOLS = [get_incident_overview, search_logs, query_metrics, get_deploys]


if __name__ == "__main__":
    # Quick manual demo against the first incident.
    from app.incidents import list_incidents

    inc_id = list_incidents()[0].incident_id
    print(f"Demo on {inc_id}\n")
    print("overview:", get_incident_overview(inc_id))
    print("\nERROR logs:", search_logs(inc_id, level="ERROR", limit=3))
    print("\ndeploys:", get_deploys(inc_id))

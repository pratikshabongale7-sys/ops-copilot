"""MCP server — exposes the telemetry tools over the Model Context Protocol.

What MCP gives us
-----------------
MCP is a standard way to hand an LLM a set of tools. Instead of writing custom
glue for every model, we describe each tool once here (its name, arguments, and
docstring), and any MCP-aware client — including the LangGraph agent we build in
Phase 3 — can discover and call them.

This file is intentionally thin: the real logic lives in app/tools.py. Each
@mcp.tool() below just exposes one of those functions. The function's type hints
and docstring become the schema the LLM sees, which is why those docstrings in
tools.py are written for a reader who is deciding *when to call the tool*.

Run it:
    uv run python app/mcp_server.py          # runs over stdio (how agents connect)
    uv run mcp dev app/mcp_server.py         # opens the MCP Inspector to click around
"""

# Allow `mcp dev app/mcp_server.py` / `python app/mcp_server.py` to find the
# `app` package by adding the project root to the import path when run by file.
if __package__ in (None, ""):
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP

from app import tools

mcp = FastMCP("ops-copilot-telemetry")


@mcp.tool()
def get_incident_overview(incident_id: str) -> dict:
    """Orientation for an incident: time window, services with telemetry, and
    log/deploy counts. Call this first to decide where to investigate."""
    return tools.get_incident_overview(incident_id)


@mcp.tool()
def search_logs(
    incident_id: str,
    level: str | None = None,
    service: str | None = None,
    contains: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Search an incident's logs by level ('ERROR'/'WARN'/'INFO'), service, and/or
    a case-insensitive substring in the message. Returns up to `limit` lines,
    oldest first. Use level='ERROR' to find a failure's fingerprint."""
    return tools.search_logs(incident_id, level, service, contains, limit)


@mcp.tool()
def query_metrics(
    incident_id: str,
    service: str | None = None,
    metric: str | None = None,
) -> dict:
    """Query an incident's metrics. With both service and metric you get the full
    time series; with only one you get summaries; with neither you get a summary
    of everything. Summaries show min/max/mean/first/last and the peak minute."""
    return tools.query_metrics(incident_id, service, metric)


@mcp.tool()
def get_deploys(incident_id: str) -> list[dict]:
    """List recent deploy/config-change events (oldest first) to correlate a
    failure's onset with a release."""
    return tools.get_deploys(incident_id)


if __name__ == "__main__":
    mcp.run()

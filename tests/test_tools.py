"""Tests for the telemetry tools (the pure functions in app/tools.py).

We test the logic directly — no MCP server needed — since the MCP layer is just a
thin wrapper. These tests assume the dataset has been generated.
"""

import pytest

from app.incidents import DATA_DIR, list_incidents
from app.tools import (
    get_deploys,
    get_incident_overview,
    query_metrics,
    search_logs,
)

pytestmark = pytest.mark.skipif(
    not DATA_DIR.exists(),
    reason="dataset not generated — run: uv run python data/generate_incidents.py",
)


def _a_bad_deploy_id() -> str:
    """Grab one bad_deploy incident id to test against a known failure shape."""
    for inc in list_incidents():
        if inc.root_cause == "bad_deploy":
            return inc.incident_id
    pytest.skip("no bad_deploy incident found")


@pytest.mark.parametrize("inc", list_incidents() if DATA_DIR.exists() else [],
                         ids=lambda i: i.incident_id)
def test_tools_work_and_never_leak_for_every_incident(inc):
    """Run the tools across ALL incidents (every failure class), not just one.

    This guarantees the plumbing holds for every class AND that no tool ever
    exposes the answer key — the agent must never be able to peek at root_cause.
    """
    inc_id = inc.incident_id

    overview = get_incident_overview(inc_id)
    assert overview["services"]
    assert overview["log_count"] > 0

    metrics = query_metrics(inc_id)
    assert metrics["summaries"]

    deploys = get_deploys(inc_id)
    assert isinstance(deploys, list)  # some classes have no culprit deploy — that's fine

    # the label must not leak through ANY tool's output
    blob = f"{overview} {metrics} {deploys} {search_logs(inc_id)}".lower()
    assert "root_cause" not in blob
    assert "expected_diagnosis" not in blob
    assert inc.root_cause not in blob  # e.g. 'memory_leak_oom' never appears verbatim


def test_overview_lists_services_without_leaking_answer():
    inc_id = list_incidents()[0].incident_id
    ov = get_incident_overview(inc_id)
    assert ov["services"]
    assert ov["log_count"] > 0
    # the answer key must not appear anywhere in the overview
    assert "root_cause" not in ov


def test_search_logs_filters_by_level():
    inc_id = _a_bad_deploy_id()
    errors = search_logs(inc_id, level="ERROR")
    assert errors, "a bad_deploy incident should have ERROR logs"
    assert all(row["level"] == "ERROR" for row in errors)


def test_search_logs_substring_is_case_insensitive():
    inc_id = _a_bad_deploy_id()
    # bad_deploy logs include 'HTTP 500 ...'
    hits = search_logs(inc_id, contains="http 500")
    assert all("500" in row["message"] for row in hits)


def test_query_metrics_full_series_has_values_and_summary():
    inc_id = _a_bad_deploy_id()
    # find the affected service from the overview's service list
    affected = list_incidents()  # use label only to pick a service to query
    svc = next(i for i in affected if i.incident_id == inc_id).true_service
    res = query_metrics(inc_id, service=svc, metric="error_rate")
    assert len(res["values"]) == 30
    assert res["summary"]["max"] > res["summary"]["first"]  # error rate spiked


def test_query_metrics_overview_mode_returns_summaries():
    inc_id = list_incidents()[0].incident_id
    res = query_metrics(inc_id)  # neither service nor metric
    assert res["summaries"]
    # every service should have an error_rate summary
    for svc, metrics in res["summaries"].items():
        assert "error_rate" in metrics


def test_get_deploys_sorted():
    inc_id = _a_bad_deploy_id()
    deploys = get_deploys(inc_id)
    ts = [d["ts"] for d in deploys]
    assert ts == sorted(ts)

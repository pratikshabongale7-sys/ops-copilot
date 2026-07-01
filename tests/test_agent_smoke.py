"""Smoke tests for the agent.

These do NOT call the LLM by default (that needs an API key and network). They
just check the pieces import and the schema behaves. The one test that actually
runs the agent is opt-in: set RUN_AGENT_TESTS=1 and have a valid LLM key.
"""

import os

import pytest

from app.schemas import ROOT_CAUSES, Diagnosis


def test_diagnosis_schema_validates():
    dx = Diagnosis(
        root_cause="bad_deploy",
        service="checkout",
        evidence=["error_rate jumped from 0.5 to 21 at minute 8"],
        suggested_fix="roll back v1.6.3",
        confidence=0.9,
    )
    assert dx.root_cause == "bad_deploy"
    assert 0.0 <= dx.confidence <= 1.0


def test_root_cause_taxonomy_matches_dataset_classes():
    for cause in [
        "bad_deploy",
        "db_connection_pool_exhaustion",
        "memory_leak_oom",
        "downstream_timeout",
        "bad_config",
        "slow_query_saturation",
    ]:
        assert cause in ROOT_CAUSES


def test_agent_module_imports():
    # importing the agent shouldn't require an API key
    from app.agent import diagnose  # noqa: F401


@pytest.mark.skipif(
    os.getenv("RUN_AGENT_TESTS") != "1",
    reason="set RUN_AGENT_TESTS=1 (and an LLM key) to run the live agent",
)
def test_agent_diagnoses_a_bad_deploy():
    import asyncio

    from app.agent import diagnose
    from app.incidents import list_incidents

    inc = next(i for i in list_incidents() if i.root_cause == "bad_deploy")
    dx = asyncio.run(diagnose(inc.incident_id))
    assert dx.root_cause in ROOT_CAUSES

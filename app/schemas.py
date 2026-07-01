"""Shared data shapes for the agent.

The Diagnosis model is the agent's final answer. Making it a strict schema (not
free text) is what lets Phase 4 grade the agent automatically: we compare the
agent's `root_cause` to the incident's label.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# The closed set of root causes the agent may choose from. Keeping this fixed (a)
# forces the agent to commit to a category the eval can score, and (b) matches the
# labels in the dataset one-to-one.
RootCause = Literal[
    "bad_deploy",
    "db_connection_pool_exhaustion",
    "memory_leak_oom",
    "downstream_timeout",
    "bad_config",
    "slow_query_saturation",
    "unknown",  # the agent may admit uncertainty rather than guess
]

ROOT_CAUSES: list[str] = list(RootCause.__args__)  # type: ignore[attr-defined]


class Diagnosis(BaseModel):
    """The agent's structured conclusion about one incident."""

    root_cause: RootCause = Field(
        description="The single most likely root-cause category."
    )
    service: str = Field(
        description="The service where the problem originates (the true cause, "
        "not just where symptoms appear)."
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Concrete observations from the telemetry that support the "
        "conclusion (e.g. 'error_rate jumped from 0.5 to 21 at minute 8').",
    )
    suggested_fix: str = Field(
        description="A short, actionable recommendation (e.g. 'roll back v1.6.3')."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the diagnosis, 0.0-1.0."
    )

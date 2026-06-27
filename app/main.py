"""Ops Copilot API — Phase 0 skeleton.

A FastAPI app with a single /health endpoint. Everything else gets built on top
of this in later phases (the MCP tools, the LangGraph agent, the eval pipeline).
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Ops Copilot",
    description="Agentic incident-diagnosis assistant",
    version="0.1.0",
)


class HealthResponse(BaseModel):
    """Shape of the /health response. Typed models like this are how FastAPI
    validates and documents your API automatically."""

    status: str
    service: str
    version: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check. Returns 200 with basic service info.

    This is the endpoint your tests hit and the one a load balancer / Kubernetes
    would ping to know the service is alive.
    """
    return HealthResponse(status="ok", service="ops-copilot", version="0.1.0")

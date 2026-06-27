"""Test for the /health endpoint.

FastAPI's TestClient spins up the app in-process (no running server needed) and
lets you make fake HTTP requests against it. This is a real integration test:
it exercises routing, the response model, and serialization.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "ops-copilot"
    assert body["version"] == "0.1.0"

# Ops Copilot

Agentic incident-diagnosis assistant. Give it an alert or error; it runs a
reason → act → observe → reflect loop over system telemetry (logs, metrics,
deploys) and returns a probable root cause with cited evidence and a suggested
fix.

> **Status:** Phase 0 — project skeleton (FastAPI + tests + Docker + LangSmith).

## Tech
FastAPI · LangGraph · LangChain · LangSmith · MCP · Docker · Kubernetes · AWS

## Quickstart
```bash
# 1. Install deps (uv creates a virtual env automatically)
uv sync

# 2. Run the tests
uv run pytest -v

# 3. Run the API
uv run uvicorn app.main:app --reload
# open http://localhost:8000/health  and  http://localhost:8000/docs

# 4. (optional) Run in Docker
docker compose up --build
```

## LangSmith trace
```bash
cp .env.example .env   # then add your LANGSMITH_API_KEY
uv run python scripts/hello_langsmith.py
# check https://smith.langchain.com -> project "ops-copilot"
```

## Layout
```
app/main.py              FastAPI app (/health)
tests/test_health.py     health endpoint test
scripts/hello_langsmith.py   minimal LangSmith trace
Dockerfile, docker-compose.yml
pyproject.toml           deps + ruff + pytest config
```

## Incident dataset (Phase 1)
```bash
uv run python data/generate_incidents.py   # writes 24 labeled incidents to data/incidents/
uv run python app/incidents.py             # prints a summary of them
```
Each incident folder has `logs.jsonl`, `metrics.json`, `deploys.json`, and a
`label.yaml` (the ground-truth root cause). 6 failure classes, 18 train / 6 test.

## Roadmap
- [x] Phase 0 — skeleton
- [x] Phase 1 — labeled incident dataset
- [ ] Phase 2 — MCP tool servers (logs, metrics)
- [ ] Phase 3 — LangGraph agentic loop
- [ ] Phase 4 — eval + single-shot-vs-agent ablation
- [ ] Phase 5 — deploy on AWS + CI/CD

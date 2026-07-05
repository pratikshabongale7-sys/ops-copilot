# Production image for the Ops Copilot API.
# Lean: installs only the RUNTIME deps (no mlflow/matplotlib — those are eval-only).
# Compatible with Hugging Face Spaces (non-root uid 1000, listens on 7860) and
# portable to Render / Fly / Cloud Run (they inject $PORT).

FROM python:3.11-slim

# HF Spaces requires the container to run as a non-root user with uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime dependencies only (the /diagnose path: FastAPI + the agent + MCP).
RUN pip install --no-cache-dir --user \
    "fastapi>=0.115" "uvicorn[standard]>=0.32" "pydantic>=2.9" \
    "python-dotenv>=1.0" "pyyaml>=6.0" \
    "langchain>=0.3" "langgraph>=0.3" "langchain-core>=0.3" \
    "langchain-mcp-adapters>=0.1" "langchain-groq>=0.2" \
    "mcp[cli]>=1.2" "langsmith>=0.1.140"

# Copy the data generator, then generate the incident dataset INTO the image
# (deterministic, so we don't depend on the data being committed to git).
COPY --chown=user data ./data
RUN python data/generate_incidents.py
COPY --chown=user app ./app

EXPOSE 7860
# Serve on the platform port (HF Spaces uses 7860; others set $PORT).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]

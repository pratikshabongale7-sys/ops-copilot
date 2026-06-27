# Multi-stage-ish single image for the FastAPI app.
# Uses uv (a fast Python package manager) inside the container too.
FROM python:3.11-slim

# Install uv (copied from its official image — fastest way)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency manifests first so Docker can cache the install layer
COPY pyproject.toml ./

# Install runtime dependencies into the system environment
RUN uv pip install --system --no-cache fastapi "uvicorn[standard]" pydantic python-dotenv langsmith

# Copy the application code
COPY app ./app

EXPOSE 8000

# Start the API. 0.0.0.0 so it's reachable from outside the container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

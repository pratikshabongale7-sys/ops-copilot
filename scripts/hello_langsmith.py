"""Phase 0: prove LangSmith tracing works.

LangSmith records ("traces") what your LLM/agent code does so you can debug and
measure it later. Here we don't even call an LLM yet — we just wrap a plain
function with @traceable and confirm the run shows up in your LangSmith dashboard.

Setup before running:
  1. Make a free account at https://smith.langchain.com
  2. Create an API key (Settings -> API Keys)
  3. Copy .env.example to .env and fill in LANGSMITH_API_KEY
  4. Run:  uv run python scripts/hello_langsmith.py

Then open https://smith.langchain.com -> your project "ops-copilot" and you'll
see a trace named "fake_diagnose".
"""

from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()  # reads .env so LANGSMITH_* vars are available


@traceable(run_type="chain", name="fake_diagnose")
def fake_diagnose(alert: str) -> dict:
    """Pretend to diagnose an incident. No real logic yet — this is just to make
    a trace appear in LangSmith so you know the wiring works."""
    return {
        "root_cause": "placeholder",
        "evidence": [f"received alert: {alert}"],
        "confidence": 0.0,
    }


if __name__ == "__main__":
    result = fake_diagnose("payment service error rate spiked")
    print("Result:", result)
    print("If LANGSMITH_TRACING=true and your key is set, a trace was just sent.")
    print("Check https://smith.langchain.com -> project 'ops-copilot'.")

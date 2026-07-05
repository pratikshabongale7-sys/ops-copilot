"""The Ops Copilot agent — a LangGraph reason-act loop over the MCP tools.

The flow (this IS the agentic loop):
  1. The agent is told which incident to investigate.
  2. It reasons, then CALLS a tool (search_logs / query_metrics / get_deploys /
     get_incident_overview) — served over MCP by app/mcp_server.py.
  3. It observes the tool's result and reasons again, choosing the next tool.
  4. It repeats until it can conclude — bounded by a step budget so it can't loop
     forever.
  5. A final structured step turns its conclusion into a Diagnosis object.

Every step is automatically traced to LangSmith (because LANGSMITH_TRACING=true and
LangChain is installed), so you can watch the whole investigation after the fact.

The LLM is provider-swappable via env (defaults to Groq's free tier). Run it
through scripts/diagnose.py rather than calling this module directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.schemas import ROOT_CAUSES, Diagnosis

load_dotenv()

# Path to the MCP server we built in Phase 2. The agent launches it as a
# subprocess and talks to it over stdio — genuinely consuming the tools via MCP.
SERVER_PATH = Path(__file__).resolve().parent / "mcp_server.py"

# How many reason->act steps the agent may take before we force it to conclude.
# A guardrail against runaway loops (and runaway token cost).
STEP_BUDGET = int(os.getenv("AGENT_STEP_BUDGET", "12"))

SYSTEM_PROMPT = f"""You are an on-call SRE assistant. You investigate a software \
incident and determine its root cause from telemetry.

You have tools to inspect ONE incident: get_incident_overview, search_logs, \
query_metrics, and get_deploys. Investigate methodically:
  1. Start with get_incident_overview to see the services and time window.
  2. Use query_metrics to find which service and metric is abnormal (a spike, a \
steady ramp, saturation near 100%, or connections pinned at a limit).
  3. Use search_logs (try level='ERROR') to read the failure's error messages.
  4. Use get_deploys to check whether a release or config change lines up with the \
onset time.
  5. Correlate the evidence. Note that the service showing SYMPTOMS is not always \
the ROOT cause (e.g. a caller times out because a downstream dependency is slow).

The root cause must be exactly one of these categories:
{", ".join(c for c in ROOT_CAUSES if c != "unknown")}.
If the evidence is genuinely inconclusive, use "unknown" rather than guessing.

Do not ask the user questions — you have everything you need in the tools. When \
you are confident, stop calling tools and write a final answer that states: the \
root-cause category, the originating service, the specific evidence you found \
(with numbers), and a recommended fix."""

from langchain.agents import create_agent

def _build_llm():
    """Create the chat model from env. Defaults to Groq (free tier).

    Set LLM_PROVIDER=groq|openai|google and the matching API key. Swapping
    providers is a one-line env change — no code edits."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0,
        )
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
            temperature=0,
        )
    # if provider == "openai":
    #     from langchain_openai import ChatOpenAI
    #
    #     return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}' (use groq|openai|google)")


def _sum_tokens(messages) -> int | None:
    """Sum total tokens across the AI messages, if the provider reported usage.
    Returns None if no usage metadata is available (some providers omit it)."""
    total, found = 0, False
    for m in messages:
        um = getattr(m, "usage_metadata", None)
        if um and um.get("total_tokens"):
            total += int(um["total_tokens"])
            found = True
    return total if found else None


async def diagnose_verbose(incident_id: str) -> tuple[Diagnosis, dict]:
    """Like diagnose(), but also returns efficiency stats for the eval:
    {llm_calls, total_tokens}. llm_calls counts model turns in the loop plus the
    final structured-extraction call."""
    from langchain_core.messages import AIMessage

    llm = _build_llm()

    # 1. Connect to the MCP server and load its tools as LangChain tools.
    client = MultiServerMCPClient(
        {
            "telemetry": {
                "command": sys.executable,
                "args": [str(SERVER_PATH)],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()

    # 2. Build a reason-act agent (the loop) and run it, bounded by STEP_BUDGET.
    agent = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)
    task = (
        f"Investigate incident '{incident_id}' and determine its root cause. "
        f"Use the tools; refer to the incident by this id."
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": task}]},
        config={"recursion_limit": STEP_BUDGET * 2},  # each step = model + tool node
    )
    messages = result["messages"]
    final_text = messages[-1].content

    # 3. Turn the free-text conclusion into a strict Diagnosis object.
    structured = llm.with_structured_output(Diagnosis)
    diagnosis: Diagnosis = await structured.ainvoke(
        "Convert this incident conclusion into the required structured format. "
        "Keep the same root cause, service, evidence, fix, and confidence.\n\n"
        f"CONCLUSION:\n{final_text}"
    )

    model_turns = sum(1 for m in messages if isinstance(m, AIMessage))
    stats = {"llm_calls": model_turns + 1, "total_tokens": _sum_tokens(messages)}
    return diagnosis, stats


async def diagnose(incident_id: str) -> Diagnosis:
    """Investigate one incident and return a structured Diagnosis (main entry point)."""
    diagnosis, _ = await diagnose_verbose(incident_id)
    return diagnosis

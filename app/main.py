"""Ops Copilot API — the deployable service (Phase 6).

Exposes the agent over HTTP:
  GET  /health          liveness check
  GET  /incidents       list available incident ids (+ true cause, for the demo)
  POST /diagnose        run the agent on an incident, return a structured Diagnosis
  GET  /                a minimal HTML page to try it in a browser

This is what gets containerized and deployed. The heavy lifting still lives in
app/agent.py; this file is just the web layer on top.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.agent import diagnose
from app.incidents import list_incidents
from app.schemas import Diagnosis

app = FastAPI(
    title="Ops Copilot",
    description="Agentic incident-diagnosis assistant",
    version="1.0.0",
)


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class DiagnoseRequest(BaseModel):
    incident_id: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check — what a load balancer / Kubernetes pings."""
    return HealthResponse(status="ok", service="ops-copilot", version="1.0.0")


@app.get("/incidents")
def incidents() -> list[dict]:
    """List the incident ids available to diagnose (with the true cause, so the
    demo page can show whether the agent was right)."""
    try:
        return [
            {"incident_id": i.incident_id, "true_cause": i.root_cause}
            for i in list_incidents()
        ]
    except FileNotFoundError:
        return []


@app.post("/diagnose", response_model=Diagnosis)
async def diagnose_endpoint(req: DiagnoseRequest) -> Diagnosis:
    """Run the agent on one incident and return its structured diagnosis."""
    try:
        return await diagnose(req.incident_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:  # LLM/tool failures -> 502 with the message
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """A tiny no-framework demo page: pick an incident, see the diagnosis."""
    return """<!doctype html><html><head><meta charset="utf-8">
<title>Ops Copilot</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 body{font:16px system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 16px;color:#111}
 h1{margin-bottom:4px} .sub{color:#666;margin-top:0}
 .note{color:#8a6d00;background:#fff7d6;border:1px solid #f0e3a0;padding:8px 12px;border-radius:8px;font-size:14px;margin:12px 0}
 select,button{font-size:16px;padding:8px 12px;border-radius:8px;border:1px solid #ccc}
 button{background:#4f46e5;color:#fff;border:0;cursor:pointer}
 button:disabled{opacity:.5} pre{background:#0e1116;color:#e6edf3;padding:16px;border-radius:10px;white-space:pre-wrap}
 .row{display:flex;gap:8px;align-items:center;margin:16px 0}
</style></head><body>
<h1>Ops Copilot</h1>
<p class="sub">An AI agent that diagnoses an incident's root cause from its telemetry.</p>
<p class="note">⏳ Heads up: this runs on a free LLM tier, so a diagnosis can take up to a minute (the agent makes several tool calls and waits out rate limits). It's working — give it a moment.</p>
<div class="row">
  <select id="inc"></select>
  <button id="go" onclick="run()">Diagnose</button>
</div>
<pre id="out">Pick an incident and click Diagnose.</pre>
<script>
async function load(){
  const r = await fetch('/incidents'); const items = await r.json();
  const s = document.getElementById('inc');
  s.innerHTML = items.map(i=>`<option value="${i.incident_id}">${i.incident_id} (truth: ${i.true_cause})</option>`).join('');
}
async function run(){
  const btn=document.getElementById('go'), out=document.getElementById('out');
  const id=document.getElementById('inc').value;
  btn.disabled=true; out.textContent='Investigating '+id+'… the agent is calling its tools. This can take up to a minute on the free tier — hang tight.';
  try{
    const r=await fetch('/diagnose',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({incident_id:id})});
    const d=await r.json();
    out.textContent = r.ok ? JSON.stringify(d,null,2) : ('Error: '+(d.detail||r.status));
  }catch(e){ out.textContent='Error: '+e; }
  btn.disabled=false;
}
load();
</script></body></html>"""

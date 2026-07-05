# Ops Copilot — Findings, Decisions & Talking Points

A running log of the technical findings, design decisions, and trade-offs from
building this project. Use it to (a) document the work, (b) prep for interviews —
each finding has a one-line "say this" — and (c) seed resume bullets (bottom).

> Fill in the bracketed `[...]` numbers once your full eval run finishes.

---

## 1. Agent behaviour: parallel vs sequential tool calling
**Finding.** The same agent behaves differently across LLM providers. Llama (via
Groq) emits **multiple tool calls in one turn** (parallel) — it plans all its
data-gathering up front, then concludes. Gemini calls **one tool per turn**
(sequential) — reason → act → observe one result → reason again.
**Why it matters.** Sequential is more adaptive (it sees each result before the
next call) but costs more: ~**13.2K tokens/diagnosis** on Gemini vs ~**4.3K** on
Groq, because every extra model turn re-sends the growing conversation. Parallel
is cheaper and fewer round-trips but commits before seeing results.
**Say this:** "I compared two providers and found Llama batches tool calls (cheaper,
~4K tokens/run) while Gemini reasons sequentially (more adaptive, ~13K tokens/run)
— a real cost-vs-adaptivity trade-off, not just a speed difference."

the agent's parallel mode degenerates toward single-shot when one batch of evidence suffices; the loop only earns its keep when later steps depend on earlier results or when the evidence doesn't fit in one shot.

## 2. LLM/agent engineering ≠ ML training
**Finding.** This project uses a **pre-trained** LLM as-is; no weights are trained.
It's LLM/agent *engineering* (prompting, tools, the loop, structured output,
evaluation) — the dominant paradigm for "AI engineer" roles today.
**Why it matters.** Being precise about this shows you understand the landscape:
what you built vs what "training a model" means.
**Say this:** "I built a system *around* a pretrained model — the value is in the
tool design, the agentic loop, and the evaluation, not in model training."

## 3. Evaluation without training: what "held-out" means
**Finding.** With no training, the train/test split protects *tuning*, not model
weights. The 6 test incidents are the ones you never look at while adjusting the
prompt / step budget, so the final score isn't inflated by tuning toward them. The
LLM also has no memory between runs.
**Why it matters.** Prevents overfitting your *evaluation* — reporting accuracy on
cases you optimised against is dishonestly optimistic.
**Say this:** "Held-out means I never tuned against those cases, so the number is an
honest estimate — not that no code ever read the file."

## 4. The single-shot vs agent comparison — two regimes
**Finding.** Compared the agent to a **single-shot LLM** (same model, telemetry in
one prompt, no tools/loop). The result depends on whether the telemetry fits in one
prompt:
- **Full context, easy incidents → a TIE.** When all evidence fits in one prompt
  and the answer is unambiguous, the loop adds no *accuracy* — single-shot already
  has everything, and on a strong model the parallel agent basically degenerates
  into single-shot. Both hit an accuracy ceiling.
- **Constrained context (small model / trimmed prompt) → the AGENT wins**
  (agent **0.833** vs single-shot **0.667** accuracy; macro-F1 0.778 vs 0.611, on
  Groq 8B). When the prompt must be trimmed to fit (8B's request limit), single-shot
  is starved while the agent selectively *fetches* what it needs — the real-world
  case, since production telemetry never fits in one prompt. *(Re-check on 70B with
  full context — it may tie again there.)*
**Fairness caveat (state this honestly).** The constrained-context win is NOT a
pure "same information, loop vs no-loop" result — the agent could access fuller data
(raw metric arrays, more logs) than the trimmed single-shot. So the correct claim is
*"under a constrained context, selective retrieval wins,"* NOT *"the loop alone adds
+X% accuracy."*
**Efficiency (where they always differ).** Single-shot: ~1.85K tokens, 1 LLM call,
~8s per diagnosis. Agent efficiency pending — 8B's per-minute limit blocked capture;
measure on 70B (expect more calls, more tokens, higher latency).
**Say this:** "On easy, fully-visible incidents the loop is a tie and just costs
more; the moment telemetry doesn't fit one prompt, the agent's selective retrieval
pulls ahead — which is the realistic case."

## 5. Tools model evidence sources, not answers (few tools, many failures)
**Finding.** 4 tools (overview, logs, metrics, deploys) diagnose all 6 failure
classes. You don't need one tool per failure — each failure is a different *pattern*
across the same evidence sources.
**Why it matters.** General tools force the agent to *reason and correlate*; a
`diagnose_bad_deploy` tool would do the diagnosis itself and defeat the purpose.
**Say this:** "Tools expose evidence, not answers — that's what keeps the reasoning
in the agent."

## 6. Investigation order is a broad→specific funnel
**Finding.** Prompted order: overview → metrics → logs → deploys. Metrics *localise*
(which service, what shape, when); logs give the specific error once you know where
to look; deploys only mean something once you have an onset time to correlate.
**Say this:** "Measure before you read, correlate last — same order a human SRE
uses; it also minimises wasted tool calls against the step budget."

## 7. Bounded agentic loop (the step budget)
**Finding.** A step budget (default 12) caps how many reason-act cycles the agent
can take — a guardrail against infinite loops and runaway token cost. It's an env
var, meant to be tuned from eval data (set it just above what good runs use).
**Say this:** "I bounded the loop and can tune the budget empirically from the eval
— if correct runs finish in 6 steps, I don't pay for 12."

## 8. MCP tool annotations & the safety boundary
**Finding.** Annotated all tools `read_only / non-destructive / idempotent`. An
*action* tool (e.g. `rollback_deploy`) would be `destructive`, which a host can gate
behind human approval.
**Why it matters.** Encodes the core design principle: **the agent diagnoses; a
human acts.** Annotations are hints for host UX/retry-safety, not security.
**Say this:** "My read tools are annotated read-only so a host can auto-run them; an
action tool would be destructive and require approval — diagnose, don't act."

## 9. Debugging: empty tool result broke the LLM API
**Finding.** `search_logs` returning an empty list produced an empty tool message,
which Groq's API rejects (`role:tool` content must be non-empty). Fixed by wrapping
list results in `{count, items}` so content is never empty (and clearer too).
**Why it matters.** Real integration bug + fix; shows debugging across the
agent/tool boundary.
**Say this:** "An empty tool result crashed the provider API; I made tool outputs
always non-empty and self-describing."

## 10. MCP Inspector vs LangSmith (two debugging surfaces)
**Finding.** The MCP Inspector lets *you* call tools by hand (test the tools in
isolation, no LLM). LangSmith records the *agent* using tools (observe the loop).
**Say this:** "Inspector is my workbench for the tools; LangSmith is the flight
recorder for the agent — together they isolate whether a bug is in a tool or in the
model's reasoning."

## 11. Architecture layering: LangGraph / LangChain / MCP
**Finding.** LangGraph owns the loop (state + control flow). LangChain is the
abstraction layer (model, tools, messages, structured output). MCP is the wire a
single tool call rides down to the server.
**Say this:** "LangGraph orchestrates, LangChain is the vocabulary, MCP is the
transport for a tool call."

## 12. Structured output makes eval possible
**Finding.** The agent's final answer is coerced into a strict `Diagnosis` schema
(root_cause ∈ 6 classes, service, evidence, fix, confidence). That fixed shape is
what lets the eval grade it automatically against labels.
**Say this:** "Structured output isn't cosmetic — it's what turns a chatty answer
into something measurable."

## 13. Provider swappability & the free-tier reality
**Finding.** Provider is one env var (`LLM_PROVIDER=groq|google`). Also: a consumer
"Gemini Advanced / Google AI Pro" *subscription does NOT grant API access* — the API
is billed separately; Pro's free API tier is tiny (~50 req/day).
**Say this:** "I abstracted the provider so switching LLMs is a config change — and
learned the hard way that a chat subscription isn't API access."

## 14. Scope discipline: avoiding the "kitchen-sink" project
**Finding.** Deliberately kept this project focused on agents/MCP/eval and moved
RAG + orchestration into a *separate* project, so each tool has a clear reason to
exist. Dropped a tacked-on ML baseline once it stopped serving the story.
**Say this:** "I split the work into two focused projects instead of one that lists
every buzzword — coherence reads stronger than a kitchen sink."

## 15. Reproducible dataset (MLOps hygiene)
**Finding.** The incident dataset is generated deterministically (fixed seed), so
eval numbers are reproducible and the data regenerates from code.
**Say this:** "The dataset is code, not a blob — deterministic and reproducible."

---

## Errors & fixes encountered (gotchas log)

Format: **symptom** → cause → fix → lesson. Keep appending as new ones appear.

1. **`400 ... 'role:tool' content ... must be non-empty`** (Groq)
   → a tool (`search_logs`) returned an empty list, producing an empty tool
   message, which the LLM API rejects. → wrapped list-returning tools in
   `{count, items}` so content is never empty. → *Tool outputs should always be
   non-empty and self-describing.*

2. **`400 tool_use_failed ... /evidence expected array, but got string`** (Groq, 8B)
   → the smaller model emitted `evidence` as a string when the schema demanded an
   array; the API validates the tool arguments at generation time, so our code
   never saw it. → changed the `evidence` field from `list[str]` to a plain `str`.
   → *Smaller models are unreliable with array/nested schema fields — prefer flat,
   simple types for structured output if you need cross-model robustness.*

3. **`413 Request too large for model llama-3.1-8b-instant`** (Groq)
   → the single-shot prompt dumps ALL telemetry into one request, exceeding the
   small model's per-request limit. → compacted the context: compact JSON (no
   whitespace), error logs trimmed to service+message (no timestamps) and capped at
   8 — ~24% smaller, works on any model. → *One-shot "dump everything" prompts hit
   size limits fast — itself an argument for an agent that fetches selectively.*

4. **Free-tier quota ran out mid-eval, losing all progress**
   → the harness only saved results after ALL incidents finished. → added
   per-incident checkpointing: each result is written immediately, re-runs skip
   completed incidents and resume. → *Long/expensive batch jobs must checkpoint;
   never assume a run completes in one go.*

5. **`403 Forbidden` from LangSmith** → EU account but the default US endpoint in
   `.env`. → set `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`. →
   *Region matters; a key is only valid against its own region's endpoint.*

6. **`git push` rejected — password auth not supported** → GitHub dropped password
   auth over HTTPS. → use a Personal Access Token as the password (or SSH). →
   *HTTPS git needs a PAT, not your account password.*

7. **`git push` rejected — remote has work you don't have** → the repo was created
   on GitHub with a LICENSE commit. → `git pull origin main --allow-unrelated-histories`
   then push. → *Creating a repo with a README/LICENSE makes an initial commit you
   must merge in first.*

8. **`ModuleNotFoundError: No module named 'app'` running a script directly** →
   running `python app/tools.py` puts `app/` (not the project root) on the path. →
   added a small sys.path bootstrap (or run as a module: `python -m app.tools`). →
   *Absolute package imports need the project root on `sys.path`.*

9. **Consumer "Gemini Advanced / Google AI Pro" subscription didn't unlock the API**
   → the chat subscription and the developer API are billed separately. → use an
   AI Studio API key (free Flash tier) or enable API billing. → *A chat
   subscription is not API access.*

10. **Provider model names change / "model not found"** → free providers rotate
    model IDs. → check the provider's live model list and set the model via env. →
    *Never hard-code a model name; make it configurable.*

11. **Dev bug: memory-leak feature came out negative** (during the removed ML work)
    → used `last - first`, but a leak ramps up then *drops* on the OOM restart, so
    the end value is low. → measured `peak - first` instead. → *Choose the
    aggregation that matches the signal's shape, not a generic default.*

12. **Efficiency chart showed only single-shot, not the agent** → the agent's final
    results were rebuilt from a cache created *before* the efficiency tracking was
    added, so its latency/tokens fields were empty (`None`) and the chart skipped
    them. → re-ran the agent with `--fresh` to regenerate the cache with the new
    fields. → *When you change what a cached job records, old cache entries are
    stale — invalidate them (`--fresh`), don't silently reuse them.*

13. **`413 ... tokens per minute (TPM): Limit 6000, Requested 8066`** (Groq 8B, agent)
    → an agent request re-sends the system prompt + all tool *schemas* + accumulated
    tool results every turn, so it's inherently ~8K tokens — over the 8B free tier's
    6K/min cap. Single-shot fit (one lean request); the agent can't. → run agentic
    workloads on a tier with more headroom (70B / paid); don't try to squeeze an
    agent under a tiny TPM. → *Agentic workloads need much higher per-minute token
    limits than single-shot — the growing context is the cost of tool use.*

---

## Resume bullets (edit numbers, pick 2–3)

- Built an **AI incident-diagnosis agent** (LangGraph + MCP tools + LangSmith
  tracing) that investigates system telemetry in a reason-act loop and outputs a
  structured, evidence-backed root cause across 6 failure classes.
- Designed the tool layer as **MCP servers** (logs / metrics / deploys) with
  read-only safety annotations, enforcing a "diagnose, don't act" boundary.
- Built an **evaluation harness** (accuracy, macro-F1, confusion) and ran a
  **single-shot-vs-agent ablation** to quantify the value of the agentic loop
  (`[agent __]` vs `[single-shot __]` on a held-out set).
- Benchmarked two LLM providers and surfaced a **cost/adaptivity trade-off**
  (parallel tool-calling ~4K tokens/run vs sequential ~13K tokens/run).
- Made the LLM **provider-swappable** (Groq / Gemini) via config; containerised the
  service with Docker and wired LangSmith observability.

## Skills to list
LangGraph · LangChain · LangSmith · MCP (Model Context Protocol) · Agentic AI ·
prompt engineering · LLM evaluation / ablation · structured output · FastAPI ·
Pydantic · Docker · pytest · Python

---

*Keep appending findings as you go. If you'd rather not commit the resume section
to a public repo, move the bottom two sections into a private note.*

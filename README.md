# AI Content Agent

One agent, four archetypes stacked together on purpose: **RAG**, **long-term memory**,
**MCP-style tool-calling**, and a **social media agent** as the concrete application
that ties them together. (A fifth idea — medical image analysis — was deliberately
left out: it's a computer-vision problem that doesn't architecturally belong inside
an LLM tool-calling loop. If you want it later, it's a clean addition as a *sixth
tool* the agent can call, not a redesign.)

## Why one project instead of five

A tool-calling agent that generates social content is more convincing when it can
actually *ground* its drafts (RAG over brand voice + past posts) and *remember*
what worked last time (memory), instead of generating from a blank system prompt
every call. That's not five projects bolted together — it's what a real content
agent needs to not be a toy.

## Architecture

```
                        ┌─────────────────────────┐
   User message  ─────► │   Agent loop (agent.py)  │
                        │  Gemini + automatic      │
                        │  function calling        │
                        └────────────┬─────────────┘
                                     │ model decides which tool(s) to call
              ┌──────────────┬──────┴───────┬──────────────┬─────────────────┐
              ▼              ▼              ▼              ▼                 ▼
     retrieve_brand_   recall_memory / search_trending  schedule_post   get_engagement_
       context (RAG)     save_memory    _topics (mock)     (mock)         metrics (mock)
              │              │
              ▼              ▼
      Chroma vector      SQLite facts +
      store (brand       Chroma "learnings"
      docs, chunked)     collection
```

The model itself decides, per turn, whether to pull brand context, check memory,
check trends, or schedule — this is the "agent" part, not a hardcoded
generate → post pipeline (see `backend/agent.py` system prompt for the rules
it follows). Runs on **Google Gemini's free tier** (`gemini-2.0-flash`) via the
`google-genai` SDK's automatic function calling: plain Python functions are
passed as tools, and the SDK builds their schemas from type hints + docstrings
and runs the call → execute → feed-back-in loop internally, capped at 10 calls
per turn (`MAX_TOOL_ITERATIONS`).

## The feedback loop (why this is more than a content generator)

```
draft (grounded in RAG + memory)
  → schedule_post (mocked)
  → get_engagement_metrics (mocked)
  → agent evaluates what worked
  → save_memory (durable learning, e.g. "concrete numbers beat vague claims")
  → next draft's system-prompt memory context includes that learning
```

Example: `data/brand_docs/past_posts.md` already encodes one such learning
("vague announcement posts underperform badly on this audience"). Ask the
agent to draft an announcement-style post and it should retrieve that context
and push back on vague phrasing rather than producing it — that's the
grounding actually doing something, not just decoration.

## RAG design decisions

- **Chunking strategy: document-aware, header-based** (`backend/rag.py:chunk_markdown`).
  Sections split on `## ` headers first, fixed-size-with-overlap only as a
  fallback for oversized sections. This matters concretely here: `past_posts.md`
  has one post + its engagement number + its "learning" per section — a naive
  fixed-size chunker would sometimes split a post from its own outcome, and
  retrieval would return "here's a post" with no signal on whether it worked.
- **Retrieval eval included** (`backend/eval_retrieval.py`) — a 5-query hit-rate
  test against known source documents, not just "it looks like it works."
  Run it: `python -m backend.eval_retrieval`.
- **No-match handling**: `retrieve_brand_context` returns an explicit
  "no relevant context found" string rather than empty context the model might
  paper over — the system prompt tells the agent to say so rather than invent
  brand voice details.

## Memory design decisions

- **Two stores, deliberately** (`backend/memory.py`): structured facts in SQLite
  (small, always loaded in full — e.g. explicit preferences) and freeform
  "learnings" in a separate Chroma collection (larger, retrieved by relevance —
  e.g. feedback-loop takeaways). Dumping every learning into every prompt would
  grow unbounded; only structured facts are unconditionally loaded.
- **Overwrite, not append**: `remember_fact` uses `INSERT OR REPLACE` — if a
  preference changes, the old value is gone, not left to contradict the new one.
- **Privacy**: memory is scoped by `user_id` everywhere, and `memory.forget()`
  lets a user delete one fact or wipe everything. This demo intentionally only
  stores content-strategy preferences — nothing sensitive — and that's a
  deliberate scope decision, not an oversight.

## What's mocked vs. real

| Component | Status |
|---|---|
| RAG retrieval (Chroma) | Real |
| Memory storage/recall (SQLite + Chroma) | Real |
| Tool-calling loop (Google Gemini free tier) | Real |
| `schedule_post`, `get_engagement_metrics`, `search_trending_topics` | **Mocked** — synthetic but plausible data, clearly labeled in `tools.py`. Production swap: X API v2, LinkedIn Marketing API, Meta Graph API. |

## What I'd do at scale

- Swap Chroma (local, single-process) for pgvector or a managed Pinecone/Weaviate
  index once brand doc volume or concurrent users grow past what a local
  persistent client comfortably handles.
- Replace the mocked social APIs with real platform integrations, with an
  actual async job queue for scheduling rather than an in-memory dict.
- Add hybrid (BM25 + embedding) search once brand docs include things like
  exact product names or ticket IDs that pure embedding similarity tends to miss.
- Rate-limit and cap the tool-calling loop per user session (already capped at
  10 iterations per turn here) to bound cost from a confused agent looping.
- The Gemini free tier has a per-minute/per-day request cap — fine for a demo
  or personal use, but a real multi-user deployment would need a paid tier
  or a fallback/queueing strategy for 429s.

## Running it

```bash
pip install -r requirements.txt --break-system-packages   # or a venv, without the flag
cp .env.example .env                                       # then fill in GEMINI_API_KEY
export $(cat .env | xargs)

# start the backend
uvicorn backend.main:app --reload --port 8000

# in another terminal, serve the frontend
cd frontend && python3 -m http.server 5500
# open http://localhost:5500 in a browser
```

Get a free Gemini API key at **https://aistudio.google.com/apikey** (no
billing setup required for the free tier — `gemini-2.0-flash` has a generous
per-minute/per-day request quota that's more than enough for this project).

In the UI, click **"Re-index brand knowledge base"** once at the start (this
runs `/ingest`, which chunks and embeds `data/brand_docs/*.md` into Chroma —
first run downloads a small embedding model from Hugging Face, so it needs a
normal internet connection).

Then try:
- *"Draft a LinkedIn post about our new retrieval pipeline improvements"*
- *"What's trending in AI engineering right now, should I write about it?"*
- *"Schedule that post for Tuesday 9am, then check its metrics"* (mocked, but
  shows the full loop)
- *"Remember that I prefer contrarian takes over neutral explainers"* — then
  start a new conversation and ask it to draft something; it should recall
  that preference.

You can also run the retrieval eval directly:
```bash
python -m backend.eval_retrieval
```

## Repo layout

```
backend/
  main.py              # FastAPI app: /chat, /ingest, /health
  agent.py             # tool-calling loop + system prompt
  rag.py               # chunking, ingestion, retrieval
  memory.py            # SQLite facts + Chroma "learnings"
  tools.py             # tool implementations (real + mocked); schemas are built
                       # automatically by Gemini from wrapper fns in agent.py
  eval_retrieval.py     # retrieval hit-rate eval
data/brand_docs/        # sample brand voice, past posts, style guide (RAG source)
frontend/index.html     # single-file React chat UI (CDN, no build step)
```

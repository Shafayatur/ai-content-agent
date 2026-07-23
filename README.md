<div align="center">

# AI Content Agent

**RAG · Memory · Tool-calling — one agent, driving a social content workflow**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)
![Groq](https://img.shields.io/badge/LLM-Groq_(free)-F55036?style=flat-square)
![Chroma](https://img.shields.io/badge/Vector_DB-Chroma-6E56CF?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)

</div>

---

## What it does

An agent that drafts, schedules, and improves social media posts — grounded in your actual
brand voice, with memory that persists across sessions.

```
User ──► Agent loop (Groq) ──► decides which tool(s) to call
                │
     ┌──────────┼──────────┬───────────────┬──────────────────┐
     ▼          ▼          ▼               ▼                  ▼
  RAG        Memory     Trends         Schedule            Metrics
 (Chroma)  (SQLite +   (mocked)        (mocked)            (mocked)
            Chroma)
```

The model decides per turn what to call — nothing here is a hardcoded pipeline.

## Features

| | |
|---|---|
| **RAG** | Header-aware chunking over brand voice, past posts, and style guide docs |
| **Documents** | Upload, paste, or delete knowledge base docs from the UI — no code/repo access needed |
| **Memory** | Structured facts (SQLite) + freeform learnings (Chroma), persists across sessions |
| **Tool-calling** | Agent decides when to retrieve, remember, check trends, schedule, or check metrics |
| **Transparency** | Every response shows exactly which tools fired |
| **UI** | Chat / Documents / Scheduled Posts / Memory — four tabs, nothing hidden |

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # add your GROQ_API_KEY — free at console.groq.com/keys

uvicorn backend.main:app --reload --port 8000     # terminal 1
cd frontend && python3 -m http.server 5500        # terminal 2
```

Open `http://localhost:5500` and go to the **Documents** tab to upload or paste your brand
docs (sample ones are included) — indexing happens automatically. Then chat.

## What's real vs. mocked

| | |
|---|---|
| RAG retrieval, memory storage, tool-calling loop | Real |
| `schedule_post`, `get_engagement_metrics`, `search_trending_topics` | Mocked — clearly labeled, swap for real platform APIs in production |

## Stack

Python · FastAPI · Groq (`openai/gpt-oss-120b`) · ChromaDB · SQLite · React (CDN, no build step)

## Layout

```
backend/     agent.py · rag.py · memory.py · tools.py · main.py · eval_retrieval.py
data/        brand_docs/ — sample brand voice, past posts, style guide
frontend/    index.html — single-file UI
```

---

<div align="center">MIT License</div>

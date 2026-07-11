"""
Memory layer, two kinds stacked together (per the memory-assistant pattern):

1. Structured key-value facts (SQLite) -- e.g. "preferred_cta_style: vary, not
   always 'Thoughts?'". Loaded in full into the system prompt at session start;
   this table is small by design (see decay/overwrite behavior below).

2. Semantic memory over "learnings" -- freeform notes the agent writes after a
   feedback loop (e.g. "LinkedIn posts with concrete numbers outperform vague
   ones"). These accumulate over time and aren't all relevant to every request,
   so they're embedded and retrieved like RAG, not dumped wholesale.

Privacy: everything here is scoped by user_id. `forget()` lets a user delete
a specific fact or wipe everything -- nothing is retained silently. This demo
only stores content-strategy preferences, never anything sensitive by design.
"""
import os
import sqlite3
import uuid
import chromadb

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "memory_store", "memory.db")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_store")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.execute(
    """CREATE TABLE IF NOT EXISTS facts (
        user_id TEXT, key TEXT, value TEXT, updated_at TEXT,
        PRIMARY KEY (user_id, key)
    )"""
)
_conn.commit()

_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
_learnings_collection = _chroma_client.get_or_create_collection("learnings")


def remember_fact(user_id: str, key: str, value: str):
    """INSERT OR REPLACE -- updated preferences overwrite old ones rather than
    piling up as contradictory facts."""
    _conn.execute(
        "INSERT OR REPLACE INTO facts VALUES (?, ?, ?, datetime('now'))",
        (user_id, key, value),
    )
    _conn.commit()


def recall_facts(user_id: str):
    rows = _conn.execute(
        "SELECT key, value FROM facts WHERE user_id = ?", (user_id,)
    ).fetchall()
    return {k: v for k, v in rows}


def forget(user_id: str, key: str = None):
    """Delete one fact, or wipe everything for this user if key is None."""
    if key:
        _conn.execute("DELETE FROM facts WHERE user_id = ? AND key = ?", (user_id, key))
    else:
        _conn.execute("DELETE FROM facts WHERE user_id = ?", (user_id,))
        _learnings_collection.delete(where={"user_id": user_id})
    _conn.commit()


def store_learning(user_id: str, learning_text: str, context: str = ""):
    """Store a freeform learning (e.g. from a feedback-loop pass) for later
    semantic recall. Kept separate from `facts` because these are noisier and
    more numerous -- not every learning is relevant to every future request."""
    _learnings_collection.add(
        ids=[str(uuid.uuid4())],
        documents=[learning_text],
        metadatas=[{"user_id": user_id, "context": context}],
    )


def recall_relevant_learnings(user_id: str, query: str, n_results: int = 3):
    if _learnings_collection.count() == 0:
        return []
    results = _learnings_collection.query(
        query_texts=[query], n_results=n_results, where={"user_id": user_id}
    )
    docs = results.get("documents", [[]])[0]
    return docs


def build_memory_context(user_id: str, current_query: str) -> str:
    """Assembled into the system prompt: structured facts (always) + the most
    relevant past learnings for this specific request (not all of them)."""
    facts = recall_facts(user_id)
    learnings = recall_relevant_learnings(user_id, current_query)

    parts = []
    if facts:
        parts.append("Known preferences for this user:\n" + "\n".join(
            f"- {k}: {v}" for k, v in facts.items()
        ))
    if learnings:
        parts.append("Relevant past learnings from previous feedback loops:\n" + "\n".join(
            f"- {l}" for l in learnings
        ))
    return "\n\n".join(parts) if parts else "No prior memory for this user yet."

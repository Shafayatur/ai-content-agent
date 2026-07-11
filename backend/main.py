from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from . import agent, rag

app = FastAPI(title="AI Content Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    user_id: str = "demo_user"


@app.post("/chat")
def chat(req: ChatRequest):
    result = agent.run_agent_turn(req.messages, user_id=req.user_id)
    # convert content blocks to plain dicts so they're JSON-serializable back to the frontend
    serializable_messages = []
    for m in result["messages"]:
        content = m["content"]
        if isinstance(content, list):
            content = [
                c.to_dict() if hasattr(c, "to_dict") else c for c in content
            ]
        serializable_messages.append({"role": m["role"], "content": content})
    return {"messages": serializable_messages, "reply": result["final_text"]}


@app.post("/ingest")
def ingest():
    """Re-index the brand knowledge base from data/brand_docs/."""
    import os
    docs_path = os.path.join(os.path.dirname(__file__), "..", "data", "brand_docs")
    count = rag.ingest_directory(docs_path)
    return {"chunks_indexed": count}


@app.get("/health")
def health():
    return {"status": "ok"}

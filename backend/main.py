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
    return {
        "messages": result["messages"],
        "reply": result["final_text"],
        "tool_calls_made": result["tool_calls_made"],
    }


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

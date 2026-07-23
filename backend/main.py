from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from . import agent, rag, memory, tools, docs_store

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
    count = rag.ingest_directory(docs_store.DOCS_DIR)
    return {"chunks_indexed": count}


@app.get("/documents")
def list_documents():
    """Everything currently in the brand knowledge base, for the Documents
    UI tab -- lets a non-technical user see (and manage) what the agent is
    grounded in without opening the repo."""
    return {"documents": docs_store.list_documents()}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Real file upload (multipart) -- .md/.txt only, size-capped, filename
    sanitized. See docs_store.py for the actual validation logic."""
    content = await file.read()
    try:
        saved_name = docs_store.save_uploaded_bytes(file.filename, content)
    except docs_store.DocumentError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # auto-reindex so the upload is immediately usable, not just saved
    count = rag.ingest_directory(docs_store.DOCS_DIR)
    return {"saved_as": saved_name, "chunks_indexed": count}


class TextDocumentRequest(BaseModel):
    name: str
    content: str


@app.post("/documents/text")
def add_text_document(req: TextDocumentRequest):
    """Paste-text flow -- for when someone wants to add a quick note/policy
    without having a file to upload."""
    try:
        saved_name = docs_store.save_text_document(req.name, req.content)
    except docs_store.DocumentError as e:
        raise HTTPException(status_code=400, detail=str(e))
    count = rag.ingest_directory(docs_store.DOCS_DIR)
    return {"saved_as": saved_name, "chunks_indexed": count}


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    try:
        docs_store.delete_document(filename)
    except docs_store.DocumentError as e:
        raise HTTPException(status_code=400, detail=str(e))
    count = rag.ingest_directory(docs_store.DOCS_DIR)
    return {"deleted": filename, "chunks_indexed": count}


@app.get("/posts")
def list_posts():
    """All mocked scheduled/posted content, for the Scheduled Posts UI tab.
    Pulls live metrics per post so the UI can show current mock engagement
    without a separate round trip per post."""
    posts = []
    for post_id, data in tools._MOCK_POSTS.items():
        posts.append(
            {
                "id": post_id,
                "content": data["content"],
                "platform": data["platform"],
                "scheduled_time": data["scheduled_time"],
                "metrics": tools.get_engagement_metrics(post_id),
            }
        )
    return {"posts": posts}


@app.get("/memory")
def get_memory(user_id: str = "demo_user"):
    """All stored facts + learnings for a user, for the Memory viewer tab."""
    return {
        "facts": memory.recall_facts(user_id),
        "learnings": memory.list_all_learnings(user_id),
    }


class ForgetRequest(BaseModel):
    user_id: str = "demo_user"
    fact_key: Optional[str] = None
    learning_id: Optional[str] = None


@app.post("/memory/forget")
def forget_memory(req: ForgetRequest):
    """Delete one fact (by key), one learning (by id), or -- if neither is
    given -- everything for this user. Explicit per-item deletion, not just
    a single nuke button, since a memory viewer should let people correct
    one wrong thing without losing everything else."""
    if req.learning_id:
        memory.forget_learning(req.learning_id)
        return {"deleted": "learning", "id": req.learning_id}
    memory.forget(req.user_id, key=req.fact_key)
    return {"deleted": "fact" if req.fact_key else "all", "key": req.fact_key}


@app.get("/health")
def health():
    return {"status": "ok"}

"""
RAG layer: turns brand docs / past posts into a retrievable knowledge base.

Chunking strategy: document-aware recursive splitting on markdown headers first
(## sections), falling back to fixed-size w/ overlap only if a section is still
too long. This matters here specifically because past_posts.md has natural
per-post boundaries (## Post ...) we never want to split mid-post — a post and
its engagement number + learning need to stay together or retrieval returns a
post's text without knowing whether it worked.
"""
import os
import re
import uuid
import chromadb

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_store")
COLLECTION_NAME = "brand_knowledge"

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_or_create_collection(COLLECTION_NAME)
    return _collection


def chunk_markdown(text: str, source: str, max_chars: int = 900, overlap: int = 100):
    """Document-aware recursive chunking: split on '## ' headers first (natural
    section boundaries), then fall back to fixed-size w/ overlap for any section
    still over max_chars."""
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        header_match = re.match(r"^## (.+)", section)
        header = header_match.group(1) if header_match else "intro"

        if len(section) <= max_chars:
            chunks.append({"text": section, "header": header})
        else:
            # fixed-size fallback with overlap, only for oversized sections
            start = 0
            while start < len(section):
                end = start + max_chars
                chunks.append({"text": section[start:end], "header": header})
                start = end - overlap
    return [
        {
            "id": str(uuid.uuid4()),
            "text": c["text"],
            "metadata": {"source": source, "header": c["header"]},
        }
        for c in chunks
    ]


def ingest_directory(dir_path: str):
    """Read every .md file in dir_path, chunk it, and add to the vector store.
    Safe to re-run: clears and rebuilds the collection each time."""
    global _collection
    collection = _get_collection()

    # rebuild clean each run so re-ingesting doesn't duplicate chunks
    _client.delete_collection(COLLECTION_NAME)
    collection = _client.get_or_create_collection(COLLECTION_NAME)
    _collection = collection

    all_chunks = []
    for fname in os.listdir(dir_path):
        if not fname.endswith(".md"):
            continue
        with open(os.path.join(dir_path, fname), "r") as f:
            text = f.read()
        all_chunks.extend(chunk_markdown(text, source=fname))

    if not all_chunks:
        return 0

    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        metadatas=[c["metadata"] for c in all_chunks],
    )
    return len(all_chunks)


def retrieve(query: str, n_results: int = 4, source_filter: str = None):
    """Similarity search over the brand knowledge base. Returns [] (not an
    error) when nothing relevant is found -- the agent is instructed to say
    so explicitly rather than hallucinate from irrelevant context."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    where = {"source": source_filter} if source_filter else None
    results = collection.query(query_texts=[query], n_results=n_results, where=where)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {"text": d, "source": m.get("source"), "header": m.get("header"), "distance": dist}
        for d, m, dist in zip(docs, metas, distances)
    ]

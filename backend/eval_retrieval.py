"""
Tiny retrieval eval: a handful of (query, expected_source) pairs measuring
whether the right document shows up in the top-3 results. This is the kind
of thing most RAG demos skip -- it's the difference between "I built RAG"
and "I can show retrieval actually works."

Run: python -m backend.eval_retrieval
"""
import os
from . import rag

EVAL_SET = [
    ("what tone should our posts have", "brand_voice.md"),
    ("what happened with the vague announcement post", "past_posts.md"),
    ("how many characters should a linkedin post be", "style_guide.md"),
    ("best time to post on twitter", "style_guide.md"),
    ("example of a post that got debated in the comments", "past_posts.md"),
]


def run_eval():
    docs_path = os.path.join(os.path.dirname(__file__), "..", "data", "brand_docs")
    rag.ingest_directory(docs_path)

    hits = 0
    for query, expected_source in EVAL_SET:
        results = rag.retrieve(query, n_results=3)
        sources = [r["source"] for r in results]
        hit = expected_source in sources
        hits += hit
        print(f"{'HIT ' if hit else 'MISS'} | '{query}' -> expected {expected_source}, got {sources}")

    print(f"\nHit rate: {hits}/{len(EVAL_SET)} ({100*hits/len(EVAL_SET):.0f}%)")


if __name__ == "__main__":
    run_eval()

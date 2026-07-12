"""
Tool implementations the agent can call. `retrieve_brand_context` and the
memory tools are real (they hit the actual Chroma/SQLite stores built in
rag.py / memory.py). The social-platform tools (schedule_post,
get_engagement_metrics, search_trending_topics) are mocked with plausible
synthetic data, clearly labeled as such -- see README for how these would be
swapped for real X/LinkedIn API calls in production.

Note: these are the underlying implementations. The model-facing wrapper
functions (with docstrings the Gemini SDK reads as tool descriptions) live in
agent.py, since a couple of these (save_memory, recall_memory) need a
user_id bound in that isn't something the model itself should be filling in.
"""
import random
import time
import uuid

from . import rag, memory

# in-memory mock "database" of scheduled/posted content, so metrics can be
# looked up consistently within a running session
_MOCK_POSTS = {}


def retrieve_brand_context(query: str, source_filter: str = None) -> str:
    results = rag.retrieve(query, n_results=4, source_filter=source_filter)
    if not results:
        return "No relevant brand context found for this query."
    return "\n\n".join(
        f"[{r['source']} / {r['header']}]\n{r['text']}" for r in results
    )


def save_memory(user_id: str, key: str, value: str) -> str:
    memory.remember_fact(user_id, key, value)
    return f"Saved: {key} = {value}"


def recall_memory(user_id: str, query: str = "") -> str:
    return memory.build_memory_context(user_id, query)


def search_trending_topics(niche: str) -> str:
    """MOCKED -- in production this would call a real trends API
    (e.g. X API v2 trends endpoint, or a scraping/aggregation service)."""
    fake_trends = {
        "ai engineering": ["agentic RAG", "tool-calling evals", "small model fine-tuning"],
        "startups": ["seed round AI tooling", "founder burnout", "vertical SaaS"],
    }
    topics = fake_trends.get(niche.lower(), ["no strong trends detected for this niche"])
    return f"[MOCKED] Trending in '{niche}': " + ", ".join(topics)


def generate_draft_metadata(platform: str) -> dict:
    """Platform constraints the agent should respect when drafting -- pulled
    from the style guide via RAG rather than hardcoded here, but exposed as a
    quick-reference tool for simple cases."""
    limits = {
        "linkedin": {"max_chars": 1300, "hashtags": "max 3, end only"},
        "twitter": {"max_chars": 280, "hashtags": "max 1"},
        "instagram": {"max_chars": 2200, "hashtag_note": "put hashtags in first comment"},
    }
    return limits.get(platform.lower(), {"max_chars": 1000, "hashtags": "n/a"})


def schedule_post(content: str, platform: str, scheduled_time: str) -> str:
    """MOCKED -- in production this would call the real platform's post/
    schedule API (X API, LinkedIn Marketing API, Meta Graph API, etc.)."""
    post_id = str(uuid.uuid4())[:8]
    _MOCK_POSTS[post_id] = {
        "content": content,
        "platform": platform,
        "scheduled_time": scheduled_time,
        "created_at": time.time(),
    }
    return f"[MOCKED] Scheduled post {post_id} on {platform} for {scheduled_time}."


def get_engagement_metrics(post_id: str) -> str:
    """MOCKED -- synthetic engagement numbers, seeded so repeated calls for the
    same post_id are stable within a session. Real version would call the
    platform's analytics/insights API."""
    if post_id not in _MOCK_POSTS:
        return f"No record of post {post_id}."
    random.seed(post_id)
    platform = _MOCK_POSTS[post_id]["platform"]
    likes = random.randint(20, 400)
    comments = random.randint(1, 60)
    return (
        f"[MOCKED] Metrics for {post_id} ({platform}): {likes} likes, "
        f"{comments} comments."
    )


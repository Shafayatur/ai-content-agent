"""
Tool implementations the agent can call. `retrieve_brand_context` and the
memory tools are real (they hit the actual Chroma/SQLite stores built in
rag.py / memory.py). The social-platform tools (schedule_post,
get_engagement_metrics, search_trending_topics) are mocked with plausible
synthetic data, clearly labeled as such -- see README for how these would be
swapped for real X/LinkedIn API calls in production.

Runs on Groq (OpenAI-compatible tool-calling format: TOOL_SCHEMAS below is
sent as-is to chat.completions.create(tools=...)). save_memory/recall_memory
take user_id as a parameter here, but it's bound by agent.py's execution
loop, not something the model itself supplies as an argument.
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


# --- Tool schemas sent to Groq (OpenAI-compatible function-calling format) ---
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_brand_context",
            "description": (
                "Retrieve relevant brand voice guidelines, past post examples, "
                "or platform style rules from the knowledge base. Use this "
                "before drafting content, or when asked what has worked before."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "source_filter": {
                        "type": "string",
                        "description": "Optional: restrict to one doc, e.g. 'style_guide.md'",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": (
                "Save a durable fact or preference about this user/brand for "
                "future sessions (e.g. 'prefers contrarian takes on LinkedIn'). "
                "Use when the user states a preference or a feedback loop "
                "reveals a lasting pattern."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "Recall stored facts and relevant past learnings about this user before making a decision.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_trending_topics",
            "description": "Check trending topics for a niche before drafting, to decide if a post should tie into something timely. MOCKED data.",
            "parameters": {
                "type": "object",
                "properties": {"niche": {"type": "string"}},
                "required": ["niche"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_post",
            "description": "Schedule a finished draft for posting on a given platform at a given time. MOCKED -- does not really post anywhere.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "platform": {"type": "string", "enum": ["linkedin", "twitter", "instagram"]},
                    "scheduled_time": {"type": "string"},
                },
                "required": ["content", "platform", "scheduled_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_engagement_metrics",
            "description": "Get engagement metrics for a previously scheduled/posted post_id, to evaluate performance and decide what to learn from it. MOCKED data.",
            "parameters": {
                "type": "object",
                "properties": {"post_id": {"type": "string"}},
                "required": ["post_id"],
            },
        },
    },
]


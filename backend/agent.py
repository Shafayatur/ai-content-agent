"""
The agent loop, running on Google Gemini's free tier instead of a paid API.

Design note: Gemini's automatic function calling (google-genai SDK) takes
plain Python functions as tools and builds their schemas from type hints +
docstrings -- the model executes the tool-calling loop internally (call model
-> run function -> feed result back -> repeat) rather than us hand-rolling
the while-loop ourselves. We still define thin wrapper functions here (not in
tools.py) because a couple of the real tools need `user_id` bound to the
session -- that's not something the model should be filling in itself.
"""
import os
from google import genai
from google.genai import types

from . import tools

MODEL = "gemini-2.0-flash"  # free tier: generous per-minute/per-day request quota
MAX_TOOL_ITERATIONS = 10  # SDK caps automatic function-calling turns internally too

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Get a free key at "
                "https://aistudio.google.com/apikey and put it in .env"
            )
        _client = genai.Client(api_key=api_key)
    return _client


SYSTEM_PROMPT_TEMPLATE = """You are an AI content strategy agent for a technical
brand. You have tools to: retrieve brand voice/style context, recall and save
user memory, check trending topics, schedule posts, and pull engagement metrics.

Rules:
- Before drafting content, retrieve brand context (voice + relevant style guide
  for the target platform) and recall memory -- don't guess at the brand's voice.
- Respect platform constraints (character limits, hashtag conventions) exactly.
- If nothing relevant is found in brand context, say so explicitly rather than
  inventing brand voice details.
- When you learn something durable from a feedback loop (metrics + what
  worked), save it to memory so future drafts improve.
- Explain your reasoning briefly when you decide to check trends, hold a post,
  or adapt content differently per platform -- the user is evaluating your
  decision-making, not just the output.
"""


def _make_tools(user_id: str):
    """Wrapper functions the model sees. Type hints + docstrings become the
    tool schema automatically -- keep both accurate, the model relies on them
    to decide when/how to call each one."""

    def retrieve_brand_context(query: str, source_filter: str = "") -> str:
        """Retrieve relevant brand voice guidelines, past post examples, or
        platform style rules from the knowledge base. Use this before
        drafting content, or when asked what has worked before.

        Args:
            query: What to search for.
            source_filter: Optional -- restrict to one doc, e.g. 'style_guide.md'.
        """
        return tools.retrieve_brand_context(query, source_filter or None)

    def save_memory(key: str, value: str) -> str:
        """Save a durable fact or preference about this user/brand for future
        sessions (e.g. 'prefers contrarian takes on LinkedIn'). Use when the
        user states a preference or a feedback loop reveals a lasting pattern.

        Args:
            key: short name for the fact, e.g. 'preferred_cta_style'.
            value: the value to remember.
        """
        return tools.save_memory(user_id=user_id, key=key, value=value)

    def recall_memory(query: str = "") -> str:
        """Recall stored facts and relevant past learnings about this user
        before making a decision.

        Args:
            query: what the recall should be relevant to (can be empty to get
                general facts).
        """
        return tools.recall_memory(user_id=user_id, query=query)

    def search_trending_topics(niche: str) -> str:
        """Check trending topics for a niche before drafting, to decide if a
        post should tie into something timely. MOCKED data.

        Args:
            niche: content niche, e.g. 'ai engineering' or 'startups'.
        """
        return tools.search_trending_topics(niche)

    def schedule_post(content: str, platform: str, scheduled_time: str) -> str:
        """Schedule a finished draft for posting on a given platform at a
        given time. MOCKED -- does not really post anywhere.

        Args:
            content: the post text.
            platform: one of 'linkedin', 'twitter', 'instagram'.
            scheduled_time: when to post, e.g. '2026-07-15 09:00 ET'.
        """
        return tools.schedule_post(content, platform, scheduled_time)

    def get_engagement_metrics(post_id: str) -> str:
        """Get engagement metrics for a previously scheduled/posted post_id,
        to evaluate performance and decide what to learn from it. MOCKED data.

        Args:
            post_id: the id returned by schedule_post.
        """
        return tools.get_engagement_metrics(post_id)

    return [
        retrieve_brand_context,
        save_memory,
        recall_memory,
        search_trending_topics,
        schedule_post,
        get_engagement_metrics,
    ]


def run_agent_turn(conversation: list, user_id: str = "demo_user") -> dict:
    """
    conversation: list of {"role": "user"|"assistant", "content": "<text>"}
    (plain strings -- Gemini's automatic function calling handles the tool
    back-and-forth internally, so we don't need to store tool_use/tool_result
    blocks in the transcript the way a manual loop would).

    Returns the updated conversation (assistant reply appended) and the final
    text, plus which tools were actually called this turn (for transparency
    in the UI -- shows the agent's decision-making, not just its output).
    """
    client = _get_client()
    tool_funcs = _make_tools(user_id)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT_TEMPLATE,
        tools=tool_funcs,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            maximum_remote_calls=MAX_TOOL_ITERATIONS
        ),
    )

    # replay prior turns as history, send only the latest user message live
    history = []
    for m in conversation[:-1]:
        role = "model" if m["role"] == "assistant" else "user"
        history.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

    chat = client.chats.create(model=MODEL, config=config, history=history)

    latest_user_message = conversation[-1]["content"]
    response = chat.send_message(latest_user_message)

    # inspect history to see which tools got called this turn, for transparency
    tool_calls_made = []
    for content in chat.get_history()[len(history) + 1:]:
        for part in content.parts or []:
            if getattr(part, "function_call", None):
                tool_calls_made.append(part.function_call.name)

    updated_conversation = conversation + [
        {"role": "assistant", "content": response.text}
    ]
    return {
        "messages": updated_conversation,
        "final_text": response.text,
        "tool_calls_made": tool_calls_made,
    }

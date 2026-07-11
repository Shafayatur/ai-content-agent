"""
The agent loop. This is the piece that makes the whole thing "one system"
rather than four unrelated demos: RAG and memory are exposed to the model as
tools alongside the social-platform actions, so the model itself decides
when to pull brand context, when to check/save memory, when to check trends,
and when to schedule -- not a hardcoded pipeline.
"""
import anthropic
from . import tools

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"
MAX_TOOL_ITERATIONS = 10

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

user_id for this session: {user_id}
"""


def _execute_tool(name: str, tool_input: dict, user_id: str):
    if name == "retrieve_brand_context":
        return tools.retrieve_brand_context(**tool_input)
    if name == "save_memory":
        return tools.save_memory(user_id=user_id, **tool_input)
    if name == "recall_memory":
        return tools.recall_memory(user_id=user_id, query=tool_input.get("query", ""))
    if name == "search_trending_topics":
        return tools.search_trending_topics(**tool_input)
    if name == "schedule_post":
        return tools.schedule_post(**tool_input)
    if name == "get_engagement_metrics":
        return tools.get_engagement_metrics(**tool_input)
    return f"Unknown tool: {name}"


def run_agent_turn(conversation: list, user_id: str = "demo_user") -> dict:
    """
    conversation: list of {"role": "user"|"assistant", "content": ...} in
    Anthropic message format (content can be a string or content blocks).
    Returns the updated conversation (including tool_use/tool_result blocks)
    and the final assistant text.
    """
    messages = list(conversation)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(user_id=user_id)

    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=system_prompt,
            tools=tools.TOOL_SCHEMAS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [b for b in response.content if b.type == "tool_use"]
        if not tool_calls:
            final_text = "".join(
                b.text for b in response.content if b.type == "text"
            )
            return {"messages": messages, "final_text": final_text}

        tool_results = []
        for call in tool_calls:
            try:
                result = _execute_tool(call.name, call.input, user_id)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": call.id, "content": str(result)}
                )
            except Exception as e:
                # surface the error to the model so it can adapt (retry,
                # try another tool, or tell the user) instead of crashing
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": f"Error executing tool: {e}",
                        "is_error": True,
                    }
                )
        messages.append({"role": "user", "content": tool_results})

    return {
        "messages": messages,
        "final_text": "(Stopped: reached max tool-call iterations without a final answer.)",
    }

"""
The agent loop, running on Groq's free tier (chosen after Google Gemini's
free tier turned out to be region-restricted -- see README).

Groq's API is OpenAI-compatible but doesn't do automatic function calling the
way Gemini's SDK does, so this is a hand-rolled loop: call the model, execute
any tool_calls it requests, feed results back as "tool" role messages, repeat
until the model returns a plain text answer.
"""
import json
import os

from groq import Groq

from . import tools

MODEL = "llama-3.3-70b-versatile"  # free tier, supports tool calling well
MAX_TOOL_ITERATIONS = 10

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key at "
                "https://console.groq.com/keys and put it in .env"
            )
        _client = Groq(api_key=api_key)
    return _client


SYSTEM_PROMPT = """You are an AI content strategy agent for a technical
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


def _execute_tool(name: str, args: dict, user_id: str):
    if name == "retrieve_brand_context":
        return tools.retrieve_brand_context(**args)
    if name == "save_memory":
        return tools.save_memory(user_id=user_id, **args)
    if name == "recall_memory":
        return tools.recall_memory(user_id=user_id, query=args.get("query", ""))
    if name == "search_trending_topics":
        return tools.search_trending_topics(**args)
    if name == "schedule_post":
        return tools.schedule_post(**args)
    if name == "get_engagement_metrics":
        return tools.get_engagement_metrics(**args)
    return f"Unknown tool: {name}"


def run_agent_turn(conversation: list, user_id: str = "demo_user") -> dict:
    """
    conversation: list of {"role": "user"|"assistant", "content": "<text>"}
    (plain strings from the frontend). Returns the updated conversation
    (assistant reply appended), the final text, and which tools were called
    this turn (for transparency in the UI).
    """
    client = _get_client()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in conversation:
        messages.append({"role": m["role"], "content": m["content"]})

    tool_calls_made = []

    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools.TOOL_SCHEMAS,
            tool_choice="auto",
            max_tokens=1500,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            final_text = msg.content or ""
            updated_conversation = conversation + [
                {"role": "assistant", "content": final_text}
            ]
            return {
                "messages": updated_conversation,
                "final_text": final_text,
                "tool_calls_made": tool_calls_made,
            }

        # append the assistant's tool-call request to history
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        # execute each requested tool and feed the result back
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
                result = _execute_tool(tc.function.name, args, user_id)
            except Exception as e:
                result = f"Error executing tool: {e}"
            tool_calls_made.append(tc.function.name)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                }
            )

    # hit MAX_TOOL_ITERATIONS without a final answer
    final_text = "(Stopped: reached max tool-call iterations without a final answer.)"
    updated_conversation = conversation + [{"role": "assistant", "content": final_text}]
    return {
        "messages": updated_conversation,
        "final_text": final_text,
        "tool_calls_made": tool_calls_made,
    }

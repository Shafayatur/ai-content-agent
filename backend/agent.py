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
import time
import uuid

import groq
from groq import Groq

from . import tools

# openai/gpt-oss-120b (OpenAI's open-weight model, hosted free on Groq) instead
# of llama-3.3-70b-versatile: the Llama model was intermittently emitting
# malformed raw-text tool calls (<function=name>{...}</function>) that Groq's
# parser rejects with a 400 tool_use_failed -- a known reliability gap with
# Llama-family tool calling on Groq. gpt-oss is Groq's own recommendation for
# tool-call-heavy workloads and hasn't shown the same failure mode in testing.
MODEL = "openai/gpt-oss-120b"
MAX_TOOL_ITERATIONS = 10
MAX_MALFORMED_RETRIES = 2  # belt-and-suspenders: retry once or twice on tool_use_failed

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
- ANY question, request, or command that implies using an uploaded document
  -- "what's in this document", "summarize the doc I uploaded", "extract my
  skills from my resume", "pull the key points from X", "what does our style
  guide say", "list the requirements in the doc" -- must be answered by
  calling retrieve_brand_context first, every time, with no exception. This
  applies to imperative phrasing ("extract...", "summarize...", "pull...")
  just as much as literal questions ("what is..."). Do NOT default to a
  generic "I don't have a copy of that" or "I can't see uploaded files"
  response -- retrieve_brand_context is a real tool connected to a real
  knowledge base, not a hypothetical. Before ever telling the user you don't
  have access to something, you must have actually called the tool and
  gotten an empty/irrelevant result back -- never assume you lack access
  without checking first.
- STRICT SCOPE, NO EXCEPTIONS: this agent only answers from the uploaded
  knowledge base -- never from your own general/pretrained knowledge, even
  for things you're confident about (people, places, organizations,
  history, definitions, trivia, current events, anything). For ANY factual
  or informational question, call retrieve_brand_context first. If nothing
  relevant comes back, say plainly that the question is outside the scope
  of the uploaded documents and that you can only answer from what's been
  uploaded -- do not answer it anyway from what you already know. This
  restriction is about answering questions, not about using your other
  tools normally (scheduling, checking mocked trends/metrics, memory) or
  having an ordinary conversational reply to a greeting.
- Respect platform constraints (character limits, hashtag conventions) exactly.
- If nothing relevant is found in brand context, say so explicitly rather than
  inventing brand voice details.
- GROUNDING IS MANDATORY, NOT OPTIONAL: every factual claim, number, feature
  name, or detail in a draft must come from retrieved context (brand docs) or
  from something the user told you directly in this conversation. If the user
  hasn't given you specifics about what actually shipped/changed, do NOT
  invent product details, metrics, feature names, "beta tester" quotes, or
  timelines to fill the gap -- ask the user for the real specifics, or write
  around the gap explicitly (e.g. "share the concrete detail you want to lead
  with") rather than fabricating one.
- Match the retrieved voice example's actual register, not just its topic.
  If a retrieved past post is understated and admits tradeoffs, don't default
  to generic hype language (emojis, "excited to announce", "stay tuned",
  "game-changing") even if it's factually accurate -- the brand voice guide
  explicitly rules out exactly that tone. When you retrieve a past post as an
  example, treat its tone as the bar to match, not just its existence as a
  citation.
- When you learn something durable from a feedback loop (metrics + what
  worked), save it to memory so future drafts improve.
- Explain your reasoning briefly when you decide to check trends, hold a post,
  or adapt content differently per platform -- the user is evaluating your
  decision-making, not just the output.
"""


def _create_with_retry(client, messages):
    """Groq occasionally returns a 400 tool_use_failed when the model emits a
    malformed tool call (a known intermittent issue, not specific to any one
    prompt) -- retry a couple times before giving up, since a re-roll usually
    produces a well-formed call."""
    last_error = None
    for attempt in range(MAX_MALFORMED_RETRIES + 1):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools.TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0,
                max_tokens=1500,
            )
        except groq.BadRequestError as e:
            body = getattr(e, "body", None) or {}
            code = (body.get("error") or {}).get("code") if isinstance(body, dict) else None
            if code == "tool_use_failed" and attempt < MAX_MALFORMED_RETRIES:
                last_error = e
                time.sleep(0.5)
                continue
            raise
    raise last_error


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

    Grounding is auto-run, not model-decided: retrieve_brand_context and
    recall_memory are called deterministically every turn (server-side,
    before the LLM ever runs) and injected as a synthetic tool-call/
    tool-result pair, so the model always has real context in front of it
    instead of having to remember to ask for it. This replaced a
    prompt-only version that worked most of the time but not reliably --
    the model sometimes called recall_memory but skipped
    retrieve_brand_context (or vice versa) and then wrongly concluded
    "no info found" without ever actually checking. Server-side guarantee
    closes that gap entirely rather than trying to word a prompt rule
    tightly enough to prevent it. The model can still call these tools
    again mid-turn with a more specific query if it wants -- this just
    guarantees a baseline every time.
    """
    client = _get_client()
    latest_user_message = conversation[-1]["content"]

    auto_brand_context = tools.retrieve_brand_context(latest_user_message)
    auto_memory_context = tools.recall_memory(user_id=user_id, query=latest_user_message)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in conversation[:-1]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": latest_user_message})

    # synthetic tool-call/tool-result pair: makes the auto-retrieval look
    # identical to a real tool call in the message history, so the model
    # reasons over it the same way it would a self-initiated call
    auto_calls = [
        ("retrieve_brand_context", {"query": latest_user_message}, auto_brand_context),
        ("recall_memory", {"query": latest_user_message}, auto_memory_context),
    ]
    synthetic_tool_calls = []
    synthetic_tool_results = []
    for name, args, result in auto_calls:
        call_id = f"auto_{uuid.uuid4().hex[:8]}"
        synthetic_tool_calls.append({
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })
        synthetic_tool_results.append(
            {"role": "tool", "tool_call_id": call_id, "content": str(result)}
        )
    messages.append({"role": "assistant", "content": "", "tool_calls": synthetic_tool_calls})
    messages.extend(synthetic_tool_results)

    tool_calls_made = [name for name, _, _ in auto_calls]

    for _ in range(MAX_TOOL_ITERATIONS):
        response = _create_with_retry(client, messages)
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
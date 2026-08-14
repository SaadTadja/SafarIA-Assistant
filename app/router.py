"""The router: one LLM function-calling loop deciding between RAG, tools, or both.

RAG is exposed as a 5th tool (search_knowledge_base) rather than as a hand-written branch,
so one loop handles hybrid cases - "my flight is cancelled, can I get a refund?" calls
get_flight_status, then search_knowledge_base, with no special-cased code.

Uses OpenRouter (OpenAI-compatible API).
"""

import json
import logging
import time
from datetime import date

import openai

from .rag import RagIndex, load_documents
from .tools import TOOL_FUNCTIONS

SYSTEM_INSTRUCTION_TEMPLATE = """You are an airline travel assistant.

Today's date is {today}.

You have tools for live data (flight search, flight status, airport info, booking lookup)
and a tool to search internal policy documents (search_knowledge_base).

Rules:
- For questions about policies or procedures (baggage, refunds, check-in, flight changes,
  travel documents, airport services, special assistance), call search_knowledge_base.
- For questions about a specific flight, booking, or airport, call the matching tool instead.
- Some questions need both: e.g. "my flight is cancelled, can I get a refund?" requires
  checking the flight status AND searching the refund policy before answering.
- If search_knowledge_base returns found=false, or a tool returns an error, tell the user
  you don't have that information. NEVER invent details that weren't returned by a tool.
- If search_knowledge_base DOES return content, answer from it. Do not fall back on
  "contact customer service" when the policy you were given already answers the question.
- Do not stretch a policy to cover a situation it doesn't mention. If the retrieved text
  covers cancellations and the user asks about delays, say plainly that the policy you have
  covers cancellations and doesn't address delays - do not hedge with "you may also be able
  to". Refund and compensation eligibility is exactly where a guess is most damaging.
- Be direct about your own limits. You can look up flights, bookings, airports and policy,
  but you cannot make, change or cancel a booking. If asked to, say so plainly rather than
  asking for details you cannot act on.
- Only call a tool when the question actually requires it. For greetings or general
  conversation, answer directly without calling anything.
- If a required piece of information (like a flight's date) is not specified and today's
  date is a reasonable default, use today's date rather than asking the user for it.
"""


def build_system_instruction(today: date | None = None) -> str:
    """Render the system prompt for one request.

    Per call, not at import: a long-running process would otherwise keep reporting its boot
    date. Without an injected date the model has no clock and invents one.
    """
    return SYSTEM_INSTRUCTION_TEMPLATE.format(today=(today or date.today()).isoformat())


MODEL_NAME = "openai/gpt-4o-mini"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_rag_index = RagIndex(load_documents())


def search_knowledge_base(query: str) -> dict:
    return _rag_index.search(query)


ALL_TOOLS = {**TOOL_FUNCTIONS, "search_knowledge_base": search_knowledge_base}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search available flights between two cities on a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "departure city or airport, e.g. 'Paris'"},
                    "destination": {"type": "string", "description": "arrival city or airport, e.g. 'Algiers'"},
                    "departure_date": {"type": "string", "description": "date of departure, e.g. '2026-08-13' or 'tomorrow'"},
                },
                "required": ["origin", "destination", "departure_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_flight_status",
            "description": "Get the current status of a specific flight: status, departure/arrival times, terminal, gate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_number": {"type": "string", "description": "the flight number, e.g. 'AH1235'"},
                    "date": {"type": "string", "description": "the date of the flight, e.g. '2026-08-12'. Default to today's date if not specified."},
                },
                "required": ["flight_number", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_airport_info",
            "description": "Get information about an airport: name, city, terminals, timezone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "airport_code": {"type": "string", "description": "IATA airport code, e.g. 'CDG'"},
                },
                "required": ["airport_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_booking",
            "description": "Get booking details by reference number: flight, date, passengers, class, baggage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_reference": {"type": "string", "description": "the booking reference code, e.g. 'ABC123'"},
                },
                "required": ["booking_reference"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search internal airline policy documents: baggage allowance, refund conditions, "
                "check-in rules, flight changes, travel documents, airport services, special assistance. "
                "Use this for general policy or 'what are the rules' questions. "
                "Do NOT use this to look up a specific flight's status, a specific booking, or a "
                "specific airport's details - use the other tools for that."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    # Phrasing matters: "refund policy" scores 0.464 and is rejected by the
                    # confidence gate, "refund policy for cancelled flights" scores 0.862.
                    "query": {
                        "type": "string",
                        "description": (
                            "A complete natural-language policy question, e.g. 'What is the refund "
                            "policy for a cancelled flight?'. Phrase it as a full question, not as "
                            "keywords ('refund policy' retrieves poorly). Never include flight "
                            "numbers, booking references, or airport codes - policy documents don't "
                            "contain them and they degrade retrieval badly."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def build_client(api_key: str) -> openai.OpenAI:
    return openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def call_with_retry(
    fn,
    *args,
    max_retries: int = 5,
    base_delay: float = 15,
    max_total_delay: float = 60,
    **kwargs,
):
    """Retry rate limits and transient 5xx with exponential backoff.

    max_total_delay bounds the *cumulative* sleep. Unbounded this waits 465s before
    surfacing an error - fine for a batch script, useless behind an HTTP request.
    """
    slept = 0.0

    def backoff(attempt: int) -> bool:
        """Sleep before the next attempt. False if the budget is spent (stop retrying)."""
        nonlocal slept
        remaining = max_total_delay - slept
        if remaining <= 0:
            return False
        delay = min(base_delay * (2 ** attempt), remaining)
        time.sleep(delay)
        slept += delay
        return True

    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except openai.RateLimitError:
            if attempt == max_retries - 1 or not backoff(attempt):
                raise
        except openai.APIStatusError as e:
            if e.status_code < 500 or attempt == max_retries - 1 or not backoff(attempt):
                raise


logger = logging.getLogger("safaria")


def log_turn(**fields) -> None:
    """One structured JSON line per turn: routing, tools, latency, tokens, transport.

    Message and answer text are deliberately not logged - that is user content.
    """
    logger.info(json.dumps({"event": "turn", **fields}, ensure_ascii=False, default=str))


def summarize_source(calls_made: list[str]) -> str:
    """Routing label, shared by chat() and chat_stream() so the UI badge cannot depend on
    which transport produced the answer."""
    labels = ["rag" if name == "search_knowledge_base" else name for name in calls_made]
    return "+".join(dict.fromkeys(labels)) if labels else "llm"


def _accumulate_tool_call_deltas(pending: dict, delta_tool_calls) -> None:
    """Reassemble tool calls split across streamed chunks. Keyed by index, not id - only
    the first fragment carries the id."""
    for fragment in delta_tool_calls:
        slot = pending.setdefault(fragment.index, {"id": None, "name": None, "arguments": ""})
        if fragment.id:
            slot["id"] = fragment.id
        if fragment.function and fragment.function.name:
            slot["name"] = fragment.function.name
        if fragment.function and fragment.function.arguments:
            slot["arguments"] += fragment.function.arguments


def chat_stream(
    client: openai.OpenAI,
    message: str,
    messages: list[dict] | None = None,
    max_tool_calls: int = 4,
):
    """Streaming counterpart to chat(). Yields {"type": "tool"|"token"|"done", ...}.

    Every turn is streamed, tool-calling ones included - they simply carry no content
    deltas, so no lookahead is needed. chat() is left intact; tests and evals drive it.
    """
    if messages is None:
        messages = [{"role": "system", "content": build_system_instruction()}]
    messages = messages + [{"role": "user", "content": message}]

    calls_made: list[str] = []
    final_answer = ""
    started = time.perf_counter()

    for _ in range(max_tool_calls):
        stream = call_with_retry(
            client.chat.completions.create,
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS_SCHEMA,
            max_tokens=500,
            stream=True,
        )

        content_parts: list[str] = []
        pending_tool_calls: dict[int, dict] = {}

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
                yield {"type": "token", "text": delta.content}
            if delta.tool_calls:
                _accumulate_tool_call_deltas(pending_tool_calls, delta.tool_calls)

        content = "".join(content_parts)

        if not pending_tool_calls:
            final_answer = content or final_answer
            break

        tool_calls = [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
            for _index, c in sorted(pending_tool_calls.items())
        ]
        assistant_message = {"role": "assistant", "tool_calls": tool_calls}
        if content:
            assistant_message["content"] = content
        messages.append(assistant_message)

        for call in tool_calls:
            tool_name = call["function"]["name"]
            calls_made.append(tool_name)
            yield {"type": "tool", "name": tool_name}

            tool_args = json.loads(call["function"]["arguments"])
            tool_result = ALL_TOOLS[tool_name](**tool_args)
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(tool_result, ensure_ascii=False, default=str),
            })

    if not final_answer:
        final_answer = "Sorry, I couldn't complete this request."
        yield {"type": "token", "text": final_answer}

    messages.append({"role": "assistant", "content": final_answer})
    source = summarize_source(calls_made)
    # No token counts: streamed responses carry no usage block by default.
    log_turn(transport="stream", source=source, tools=calls_made,
             latency_ms=round((time.perf_counter() - started) * 1000),
             answer_chars=len(final_answer))
    yield {"type": "done", "answer": final_answer, "source": source, "messages": messages}


def chat(
    client: openai.OpenAI,
    message: str,
    messages: list[dict] | None = None,
    max_tool_calls: int = 4,
) -> dict:
    """Run one user message through the router. Pass the "messages" list returned by a
    previous call to continue that conversation with full context; omit it to start a new
    one. Returns {"answer", "source", "tool_calls", "messages"} - "messages" is the updated
    history, to be passed back in on the next turn for conversational memory."""
    if messages is None:
        messages = [{"role": "system", "content": build_system_instruction()}]
    messages = messages + [{"role": "user", "content": message}]

    calls_made: list[str] = []
    final_answer = "Sorry, I couldn't complete this request."
    started = time.perf_counter()
    prompt_tokens = completion_tokens = 0

    for _ in range(max_tool_calls):
        response = call_with_retry(
            client.chat.completions.create,
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS_SCHEMA,
            max_tokens=500,  # answers are short; also keeps cost/latency predictable
        )
        if response.usage:
            prompt_tokens += response.usage.prompt_tokens
            completion_tokens += response.usage.completion_tokens

        message_obj = response.choices[0].message
        messages.append(message_obj.model_dump(exclude_none=True))

        if not message_obj.tool_calls:
            final_answer = message_obj.content or final_answer
            break

        for tool_call in message_obj.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            calls_made.append(tool_name)

            tool_result = ALL_TOOLS[tool_name](**tool_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                # json.dumps, not str(): str() yields Python repr, not JSON.
                "content": json.dumps(tool_result, ensure_ascii=False, default=str),
            })

    source = summarize_source(calls_made)
    log_turn(transport="json", source=source, tools=calls_made,
             latency_ms=round((time.perf_counter() - started) * 1000),
             prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
             answer_chars=len(final_answer))
    return {"answer": final_answer, "source": source,
            "tool_calls": calls_made, "messages": messages}

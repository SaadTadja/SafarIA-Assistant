"""The router: one LLM function-calling loop that decides between RAG, tools, or both.

Design: RAG retrieval is exposed to the LLM as a 5th tool (search_knowledge_base) alongside
the 4 real API tools, rather than as a separate hand-written "if policy question -> RAG else
-> tools" branch. One decision loop then naturally handles hybrid cases (e.g. "my flight is
cancelled, can I get a refund?") by calling get_flight_status, seeing the result, and then
calling search_knowledge_base before answering - without any special-cased code for that
scenario.

The documents themselves are still plain text; only the retrieval function is callable.

Uses OpenRouter (OpenAI-compatible API) against a free-tier model. This project briefly
moved to Anthropic Claude direct (claude-haiku-4-5), which measured 100% routing accuracy
in testing, but the provided Anthropic key turned out to have a $0 credit balance and
Anthropic has no free tier - a request fails outright rather than degrading. Reverted to
OpenRouter's free tier so the app keeps working with no funds needed. Known tradeoff,
measured directly: the free model here (nvidia/nemotron-nano-9b-v2:free, the only free
OpenRouter model that responded reliably across several tried) scores 66.7% routing
accuracy vs. 100% for a paid model - see README.md's "LLM provider history" section for
the full comparison (Gemini -> OpenRouter paid -> OpenRouter free -> Anthropic -> OpenRouter
free again).
"""

import json
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

    The date is substituted per call rather than baked in at import time: a server
    process that stays up for days would otherwise keep telling the model it is still
    whatever day it booted on.

    Without this, the "use today's date" rule above was unfollowable - the model has no
    clock and silently invented one, emitting dates from its training era (observed:
    2023-10-31, 2023-10-12 and 2023-10-18 for three runs of the same query, in 2026).
    Harmless only while get_flight_status ignores its date argument; wrong the moment
    the mocked tools are swapped for a real API.
    """
    return SYSTEM_INSTRUCTION_TEMPLATE.format(today=(today or date.today()).isoformat())


MODEL_NAME = "openai/gpt-4o-mini"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_rag_index = RagIndex(load_documents())


def search_knowledge_base(query: str) -> dict:
    """Search internal airline policy documents for policy/procedure questions."""
    return _rag_index.search(query)


ALL_TOOLS = {**TOOL_FUNCTIONS, "search_knowledge_base": search_knowledge_base}

# OpenAI-compatible tool schema: {"type": "function", "function": {"name", "description",
# "parameters"}} - a nested "function" key, unlike Anthropic's flat "input_schema" format.
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
                    # The retrieval stage is a cross-encoder, which scores a terse keyword
                    # query markedly lower than the same intent phrased as a full question -
                    # low enough to fall under CONFIDENCE_THRESHOLD and return found=False
                    # even when the corpus does contain the answer. Measured: "refund policy"
                    # scores 0.464 (rejected) while "refund policy for cancelled flights"
                    # scores 0.862 on the same corpus. Identifiers are worse still ("AH1235
                    # refund" -> 0.013), because no policy document mentions them. Hence the
                    # explicit phrasing contract here rather than a lower threshold: the fix
                    # belongs at the query the model writes, not at the gate that judges it.
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
    """Retry on rate limits / transient server errors with exponential backoff
    (challenge bonus: "gestion correcte des erreurs API").

    max_total_delay bounds the *cumulative* sleep, not each individual one. Unbounded,
    this schedule waits 15+30+60+120+240 = 465s before surfacing the error - fine for a
    batch script, useless behind an HTTP request that a browser abandoned minutes ago
    (observed during development: a /chat call sat for over three minutes before
    returning the 429 it already knew about on the first attempt). Whatever budget is
    left is still spent retrying; only the pointless tail is cut.
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

    for _ in range(max_tool_calls):
        response = call_with_retry(
            client.chat.completions.create,
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS_SCHEMA,
            max_tokens=500,  # answers are short; also keeps cost/latency predictable
        )
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
                # json.dumps, not str(): str() on a dict yields Python repr (single
                # quotes, None/True rather than null/true), which is not the JSON the
                # model is trained to read back from a tool role.
                "content": json.dumps(tool_result, ensure_ascii=False, default=str),
            })

    labels = ["rag" if name == "search_knowledge_base" else name for name in calls_made]
    source = "+".join(dict.fromkeys(labels)) if labels else "llm"

    return {"answer": final_answer, "source": source, "tool_calls": calls_made, "messages": messages}

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

import openai

from .rag import RagIndex, load_documents
from .tools import TOOL_FUNCTIONS

SYSTEM_INSTRUCTION = """You are an airline travel assistant.

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
- Only call a tool when the question actually requires it. For greetings or general
  conversation, answer directly without calling anything.
- If a required piece of information (like a flight's date) is not specified and today's
  date is a reasonable default, use today's date rather than asking the user for it.
"""

MODEL_NAME = "nvidia/nemotron-nano-9b-v2:free"
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
                    "query": {"type": "string", "description": "the policy question to search for"},
                },
                "required": ["query"],
            },
        },
    },
]


def build_client(api_key: str) -> openai.OpenAI:
    return openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def call_with_retry(fn, *args, max_retries: int = 5, base_delay: float = 15, **kwargs):
    """Retry on rate limits / transient server errors with exponential backoff
    (challenge bonus: "gestion correcte des erreurs API")."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))
        except openai.APIStatusError as e:
            if e.status_code < 500 or attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


def chat(client: openai.OpenAI, message: str, max_tool_calls: int = 4) -> dict:
    """Run one user message through the router. Returns {"answer", "source", "tool_calls"}."""
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": message},
    ]

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
                "content": str(tool_result),
            })

    labels = ["rag" if name == "search_knowledge_base" else name for name in calls_made]
    source = "+".join(dict.fromkeys(labels)) if labels else "llm"

    return {"answer": final_answer, "source": source, "tool_calls": calls_made}

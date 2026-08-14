"""Automated checks on the RAG / Tool routing decision (bonus: "tests automatiques sur le
choix RAG / Tool"): the brief's 6 scenarios, plus small talk that must call nothing, an
edge case the confidence threshold alone cannot reject, and conversational memory.

Makes real API calls, so it skips automatically without OPENROUTER_API_KEY. All 9 pass on
the current model. A broader 31-scenario set lives in eval/routing_set.py - kept out of the
suite so `pytest` stays fast; run it with `python -m eval.run_eval`.
"""

import os
import time

import pytest

from app.router import build_client, chat

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set - skipping live routing tests",
)

SCENARIOS = [
    ("Find me a flight from Paris to Algiers tomorrow.", "search_flights"),
    ("What is the status of flight AH1235?", "get_flight_status"),
    ("What are the cabin baggage rules?", "rag"),
    ("Give me the information for booking ABC123.", "get_booking"),
    ("My flight AH1235 is cancelled. Can I get a refund?", "get_flight_status+rag"),
    ("Give me information about CDG airport.", "get_airport_info"),
]


@pytest.fixture(scope="module")
def client():
    return build_client(os.environ["OPENROUTER_API_KEY"])


@pytest.mark.parametrize("query,expected_source", SCENARIOS)
def test_routing_matches_expected_source(client, query, expected_source):
    result = chat(client, query)
    time.sleep(2)
    assert result["source"] == expected_source, (
        f"query={query!r} expected source={expected_source!r} got={result['source']!r} "
        f"answer={result['answer']!r}"
    )


def test_no_tool_called_for_small_talk(client):
    result = chat(client, "Hello, who are you?")
    assert result["source"] == "llm"


def test_router_catches_hard_edge_case_raw_rag_misses(client):
    """The second defense layer: the router should recognise this isn't a policy question
    and not call search_knowledge_base at all, independently of what retrieval would score.

    Kept as a router-level test even though the reranker now also rejects this query
    cleanly (0.025) - it guards the routing judgment, not the threshold.
    """
    result = chat(client, "What is the aircraft's maximum cruising altitude?")
    assert result["source"] == "llm", (
        f"expected no tool call (source='llm'), got source={result['source']!r} "
        f"answer={result['answer']!r}"
    )


def test_conversational_memory_uses_prior_context(client):
    """Two turns: the second message ("Can I get a refund for it?") only makes sense if
    the model remembers "it" = AH1235 from the first turn's "messages" history. Without
    memory (i.e. without passing "messages" back in), the model has no way to know which
    flight is being asked about."""
    first = chat(client, "What is the status of flight AH1235?")
    time.sleep(2)
    second = chat(client, "Can I get a refund for it?", messages=first["messages"])

    assert "rag" in second["source"], (
        f"expected search_knowledge_base to be called using context carried over from the "
        f"first turn, got source={second['source']!r} answer={second['answer']!r}"
    )
    clarifying_phrases = ["which flight", "flight number", "specify", "let me know which"]
    answer_lower = second["answer"].lower()
    assert not any(phrase in answer_lower for phrase in clarifying_phrases), (
        f"model asked for clarification instead of using context from the first turn - "
        f"conversational memory isn't working: {second['answer']!r}"
    )

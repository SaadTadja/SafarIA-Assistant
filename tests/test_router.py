"""The 6 routing scenarios from the challenge brief itself, as an automated
Routing Accuracy check. Requires a live OpenRouter API key and makes real API calls,
so it's skipped automatically when OPENROUTER_API_KEY isn't set (e.g. in CI without secrets).

Runs against a free-tier model (see app/router.py's module docstring for why) - measured
at 66.7% routing accuracy rather than 100%, so some of these are expected to fail until
there's budget for a paid model. That's a known, documented limitation, not a test bug.
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
    """app/rag.py's confidence threshold alone can't reject this query (it scores 0.619,
    above even some genuine matches - see eval/calibrate_threshold.py). This tests the
    second defense layer: the router's tool-selection judgment should recognize this isn't
    actually a policy question and not call search_knowledge_base for it at all, regardless
    of what the raw similarity score would have said."""
    result = chat(client, "What is the aircraft's maximum cruising altitude?")
    assert result["source"] == "llm", (
        f"expected no tool call (source='llm'), got source={result['source']!r} "
        f"answer={result['answer']!r}"
    )

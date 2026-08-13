"""Retrieval quality checks: does the confidence gate correctly separate relevant
from irrelevant chunks? This is Recall@K / the confidence-threshold behavior from
the notebooks, turned into an automated test instead of eyeballed scores.
"""

import pytest

from app.rag import load_documents, RagIndex

EXPECTED_SOURCES = {
    "baggage_policy",
    "refund_policy",
    "checkin_policy",
    "flight_change_policy",
    "travel_documents",
    "airport_services",
    "special_assistance",
}


@pytest.fixture(scope="module")
def index() -> RagIndex:
    return RagIndex(load_documents())


def test_all_seven_categories_loaded():
    chunks = load_documents()
    sources = {c["source"] for c in chunks}
    assert sources == EXPECTED_SOURCES


def test_chunks_are_non_empty():
    chunks = load_documents()
    assert len(chunks) >= len(EXPECTED_SOURCES)  # at least 1 section per document
    for chunk in chunks:
        assert chunk["text"].strip()


@pytest.mark.parametrize("query,expected_source", [
    ("How many bags can I bring in the cabin?", "baggage_policy"),
    ("Can I get a refund if my flight is cancelled?", "refund_policy"),
    ("How early should I arrive at the airport?", "checkin_policy"),
    ("Can I change the date of my flight?", "flight_change_policy"),
    ("Do I need a visa to travel internationally?", "travel_documents"),
    ("Is there Wi-Fi at the airport?", "airport_services"),
    ("I need wheelchair assistance, how do I request it?", "special_assistance"),
    # French queries - the reason the embedding model was switched to a multilingual one.
    ("Comment demander une assistance en fauteuil roulant ?", "special_assistance"),
    ("Puis-je voyager avec mon animal de compagnie ?", "special_assistance"),
])
def test_retrieval_finds_correct_category(index: RagIndex, query: str, expected_source: str):
    result = index.search(query)
    assert result["found"] is True
    assert expected_source in result["sources"]


def test_out_of_scope_query_is_rejected(index: RagIndex):
    # Deliberately unrelated to all 7 categories, in a language the corpus also covers.
    # NOTE: "What is the aircraft's maximum cruising altitude?" is a KNOWN hard case for
    # this multilingual model - it scores 0.619, above even some genuine matches (visa
    # question: 0.595) - see eval/calibrate_threshold.py. No threshold value gets both
    # right, so that specific query is intentionally not used here; it's covered instead
    # by a router-level test (tests/test_router.py) checking the second defense layer -
    # the LLM's tool-selection judgment - catches it even when raw similarity doesn't.
    result = index.search("Quel temps fait-il a Paris aujourd'hui ?")
    assert result["found"] is False

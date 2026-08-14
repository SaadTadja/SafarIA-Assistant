"""Retrieval quality: does the confidence gate separate relevant chunks from irrelevant?"""

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
    # Unrelated to all 7 categories, in a language the corpus also covers.
    result = index.search("Quel temps fait-il a Paris aujourd'hui ?")
    assert result["found"] is False


def test_retrieval_is_robust_to_query_form(index: RagIndex):
    """Guards the failure that shipped undetected: 100% Hit@1 on well-formed questions
    while the router sent terse queries the gate rejected.

    Floors sit below the measured values (0.923 / 0.885 / 0.833) so a real regression
    fails rather than noise; retrieval is deterministic, so nothing drifts.
    """
    from eval.robustness_eval import evaluate

    report = evaluate(index)

    assert report["overall"]["hit_rate_at_1"] >= 0.85, report["failures"]
    assert report["overall"]["gate_pass_rate"] >= 0.80, report["failures"]
    assert report["by_form"]["identifier"]["gate_pass_rate"] >= 0.70, report["failures"]
    assert report["out_of_scope"]["rejection_rate"] == 1.0  # the gate's whole purpose

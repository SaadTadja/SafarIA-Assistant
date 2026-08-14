"""Empirically determine CONFIDENCE_THRESHOLD instead of eyeballing it.

Prints the top-1 score after the full pipeline - normalization, bi-encoder, reranker - for
queries that SHOULD match and queries that should NOT, so the gap between the two clusters
can be read directly. Scores are on the reranker's scale, so re-run this whenever the
embedding model, the reranker or CANDIDATE_POOL_SIZE changes.

eval/robustness_eval.py covers the same ground on a larger set (26 in-scope, 10 out).
"""

from app.rag import RagIndex, load_documents

SHOULD_MATCH = [
    ("How many bags can I bring in the cabin?", "baggage_policy"),
    ("Can I get a refund if my flight is cancelled?", "refund_policy"),
    ("How early should I arrive at the airport?", "checkin_policy"),
    ("Can I change the date of my flight?", "flight_change_policy"),
    ("Do I need a visa to travel internationally?", "travel_documents"),
    ("Is there Wi-Fi at the airport?", "airport_services"),
    ("I need wheelchair assistance, how do I request it?", "special_assistance"),
    # French versions of a few - the point of switching models
    ("Quel est le statut du vol AH1235 ?", None),  # tool question, not RAG, just sanity
    ("Comment demander une assistance en fauteuil roulant ?", "special_assistance"),
    ("Puis-je voyager avec mon animal de compagnie ?", "special_assistance"),
    ("Quelles sont les conditions pour un enfant non accompagné ?", "special_assistance"),
]

SHOULD_NOT_MATCH = [
    "What is the aircraft's maximum cruising altitude?",
    "What movies are available on the entertainment system?",
    "Quel temps fait-il a Paris aujourd'hui ?",  # weather, French
    "Can you recommend a good restaurant in Casablanca?",
]


def main():
    index = RagIndex(load_documents())

    print("=== SHOULD MATCH (want high scores) ===")
    match_scores = []
    for query, expected in SHOULD_MATCH:
        results = index.retrieve(query, top_k=1)
        score = results[0][1] if results else 0.0
        source = results[0][0]["source"] if results else "-"
        match_scores.append(score)
        flag = "" if expected is None or source == expected else "  <-- wrong category"
        print(f"  {score:.3f}  [{source}]  {query}{flag}")

    print("\n=== SHOULD NOT MATCH (want low scores) ===")
    nomatch_scores = []
    for query in SHOULD_NOT_MATCH:
        results = index.retrieve(query, top_k=1)
        score = results[0][1] if results else 0.0
        source = results[0][0]["source"] if results else "-"
        nomatch_scores.append(score)
        print(f"  {score:.3f}  [{source}]  {query}")

    print(f"\nShould-match:    min={min(match_scores):.3f}  max={max(match_scores):.3f}")
    print(f"Should-NOT-match: min={min(nomatch_scores):.3f}  max={max(nomatch_scores):.3f}")

    if min(match_scores) > max(nomatch_scores):
        suggested = (min(match_scores) + max(nomatch_scores)) / 2
        print(f"\nClean gap. Suggested threshold: {suggested:.3f}")
    else:
        print("\nNo clean gap - overlapping distributions, threshold will trade off recall vs precision.")


if __name__ == "__main__":
    main()

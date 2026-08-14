"""Retrieval robustness across query *forms*, not just topics.

run_eval.py's 9 queries are all well-formed questions. The router doesn't write those - it
produced 'refund policy', which scores 0.464 and was gate-rejected while 59 refund chunks
sat in the corpus. Hit Rate@1 read 100% while the flagship scenario failed 4 times in 5.

Three forms over the same topics: sentence (control), terse, identifier-bearing. Plus 10
out-of-scope queries instead of run_eval.py's one.

No API key needed. Usage: python -m eval.robustness_eval
"""

import json
from collections import defaultdict

from app.rag import RagIndex, load_documents

TOP_K = 4

# (query, expected_source, form)
QUERY_SET = [
    # --- control: full sentences ---
    ("How many bags can I bring in the cabin?", "baggage_policy", "sentence"),
    ("Can I get a refund if my flight is cancelled?", "refund_policy", "sentence"),
    ("How early should I arrive at the airport?", "checkin_policy", "sentence"),
    ("Can I change the date of my flight?", "flight_change_policy", "sentence"),
    ("Do I need a visa to travel internationally?", "travel_documents", "sentence"),
    ("Is there Wi-Fi at the airport?", "airport_services", "sentence"),
    ("I need wheelchair assistance, how do I request it?", "special_assistance", "sentence"),
    ("Comment demander une assistance en fauteuil roulant ?", "special_assistance", "sentence"),
    ("Puis-je voyager avec mon animal de compagnie ?", "special_assistance", "sentence"),

    # --- terse keyword queries ---
    ("refund policy", "refund_policy", "terse"),
    ("baggage allowance", "baggage_policy", "terse"),
    ("cabin baggage rules", "baggage_policy", "terse"),
    ("check-in deadline", "checkin_policy", "terse"),
    ("flight change fees", "flight_change_policy", "terse"),
    ("visa requirements", "travel_documents", "terse"),
    ("airport lounge", "airport_services", "terse"),
    ("wheelchair assistance", "special_assistance", "terse"),
    ("unaccompanied minor", "special_assistance", "terse"),
    ("pet travel", "special_assistance", "terse"),
    ("excess baggage fee", "baggage_policy", "terse"),

    # --- identifier-bearing queries ---
    ("refund for cancelled flight AH1235", "refund_policy", "identifier"),
    ("AH1235 refund policy", "refund_policy", "identifier"),
    ("baggage allowance for booking ABC123", "baggage_policy", "identifier"),
    ("check-in time for flight AH1009", "checkin_policy", "identifier"),
    ("change flight AH1235 to another date", "flight_change_policy", "identifier"),
    ("CDG airport wheelchair assistance", "special_assistance", "identifier"),
]

OUT_OF_SCOPE = [
    "Quel temps fait-il a Paris aujourd'hui ?",
    "What is the capital of Japan?",
    "Who won the World Cup in 2022?",
    # Was a hard case pre-reranker; now scores 0.025, so it belongs here.
    "What is the aircraft's maximum cruising altitude?",
    "How do I cook a tagine?",
    "What is the stock price of Royal Air Maroc?",
    "Tell me a joke",
    "What is 15 times 42?",
    "Recommend a hotel in Marrakech",
    "How many employees does the airline have?",
]


def evaluate(index: RagIndex, top_k: int = TOP_K) -> dict:
    by_form = defaultdict(lambda: {"n": 0, "hit1": 0, "gate": 0})
    failures = []

    for query, expected, form in QUERY_SET:
        results = index.retrieve(query, top_k=top_k)
        sources = [chunk["source"] for chunk, _score in results]
        top_score = results[0][1] if results else 0.0
        hit1 = sources[0] == expected
        gate_passed = index.search(query)["found"]

        stats = by_form[form]
        stats["n"] += 1
        stats["hit1"] += hit1
        stats["gate"] += gate_passed

        if not (hit1 and gate_passed):
            failures.append({"query": query, "form": form, "expected": expected,
                             "top1": sources[0], "hit_at_1": hit1,
                             "gate_passed": gate_passed, "top_score": round(top_score, 3)})

    rejected = [q for q in OUT_OF_SCOPE if index.search(q)["found"] is False]
    n = len(QUERY_SET)

    return {
        "by_form": {f: {"n": s["n"],
                        "hit_rate_at_1": round(s["hit1"] / s["n"], 3),
                        "gate_pass_rate": round(s["gate"] / s["n"], 3)}
                    for f, s in sorted(by_form.items())},
        "overall": {
            "n": n,
            "hit_rate_at_1": round(sum(s["hit1"] for s in by_form.values()) / n, 3),
            "gate_pass_rate": round(sum(s["gate"] for s in by_form.values()) / n, 3),
        },
        "out_of_scope": {"n": len(OUT_OF_SCOPE),
                         "rejection_rate": round(len(rejected) / len(OUT_OF_SCOPE), 3)},
        "failures": failures,
    }


def main():
    print(json.dumps(evaluate(RagIndex(load_documents())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

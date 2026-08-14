"""Validate the judge before trusting its numbers - an untested judge is a keyword check
that costs more and sounds better.

Five labelled cases, real evidence with constructed answers so each verdict is known:
a good answer, an invented-compensation hallucination, an abstention with evidence present
(flag must fire), an abstention with none (must not fire), and a correctly-negated answer -
the case the substring check got backwards.

Usage: python -m eval.validate_judge
"""

import os

from dotenv import load_dotenv

from app.router import build_client

from .judge import JUDGE_MODEL, judge_answer

load_dotenv()

QUESTION = "My flight AH1235 is cancelled. Can I get a refund?"

# Real evidence shape, as produced by run_query's tool_results capture.
EVIDENCE = [
    {"tool": "get_flight_status", "args": {"flight_number": "AH1235"},
     "result": {"status": "cancelled", "scheduled_departure": "08:30",
                "terminal": "2B", "gate": None}},
    {"tool": "search_knowledge_base", "args": {"query": "refund policy for a cancelled flight"},
     "result": {"found": True, "sources": ["refund_policy"], "top_score": 0.79,
                "context": [
                    "If Royal Air Maroc cancels your flight, you may choose one of the "
                    "following: rebooking on the next available flight, re-routing via an "
                    "alternative itinerary subject to availability, or a refund of the "
                    "unused portion of your ticket if you decide not to travel.",
                    "Refund requests are processed to the original form of payment. "
                    "Processing times vary by payment method.",
                ]}},
]

EMPTY_EVIDENCE = [
    {"tool": "get_flight_status", "args": {"flight_number": "AH1235"},
     "result": {"status": "cancelled", "scheduled_departure": "08:30"}},
    {"tool": "search_knowledge_base", "args": {"query": "refund policy"},
     "result": {"found": False,
                "message": "No relevant policy information found in the knowledge base."}},
]

CASES = [
    {
        "id": "good_answer",
        "evidence": EVIDENCE,
        "answer": ("Your flight AH1235 has been cancelled. According to the refund policy you "
                   "have three options: rebooking on the next available flight, re-routing via "
                   "an alternative itinerary subject to availability, or a refund of the unused "
                   "portion of your ticket if you decide not to travel. Refunds are processed to "
                   "the original form of payment."),
        "expect": {"faithfulness": "high", "answer_relevance": "high", "abstention": False},
    },
    {
        "id": "hallucination",
        "evidence": EVIDENCE,
        "answer": ("Your flight AH1235 has been cancelled. You are entitled to EUR 600 in "
                   "compensation, which will be paid into your account within 7 working days, "
                   "plus a complimentary hotel night and a 20% discount on your next booking."),
        # Relevance not asserted: an answer can be on-topic and entirely invented. Only
        # the faithfulness collapse matters here.
        "expect": {"faithfulness": "low", "answer_relevance": "n/a", "abstention": False},
    },
    {
        "id": "unwarranted_abstention",
        "evidence": EVIDENCE,
        "answer": ("Your flight AH1235 has been cancelled. Unfortunately I couldn't find specific "
                   "information regarding the refund policy. I recommend contacting customer "
                   "support directly."),
        "expect": {"faithfulness": "high", "answer_relevance": "low", "abstention": True},
    },
    {
        "id": "correct_abstention",
        "evidence": EMPTY_EVIDENCE,
        "answer": ("Your flight AH1235 has been cancelled. I don't have refund policy information "
                   "available, so I'd recommend contacting customer support."),
        "expect": {"faithfulness": "high", "answer_relevance": "n/a", "abstention": False},
    },
    {
        "id": "correct_negation",
        # Needs its own question: the no-show evidence does not answer the cancellation one.
        # An earlier draft reused the shared question and the judge flagged the mismatch.
        "question": "I missed my flight and never made it to the gate. Can I get a refund?",
        "evidence": [
            {"tool": "search_knowledge_base", "args": {"query": "refund for a no-show passenger"},
             "result": {"found": True, "sources": ["refund_policy"], "top_score": 0.81,
                        "context": ["Passengers who fail to present themselves at the boarding "
                                    "gate are considered no-shows and are not eligible for a "
                                    "refund of the ticket price."]}},
        ],
        "answer": ("Because you did not present yourself at the boarding gate, you are recorded "
                   "as a no-show and you would not be eligible for a refund of the ticket price."),
        "expect": {"faithfulness": "high", "answer_relevance": "high", "abstention": False},
    },
]

HIGH, LOW = 0.7, 0.5


def check(case: str, verdict: dict, expect: dict) -> list[str]:
    problems = []
    f, r = verdict["faithfulness"], verdict["answer_relevance"]
    if expect["faithfulness"] == "high" and f < HIGH:
        problems.append(f"faithfulness {f} too low (expected >= {HIGH})")
    if expect["faithfulness"] == "low" and f >= LOW:
        problems.append(f"faithfulness {f} too high (expected < {LOW})")
    if expect["answer_relevance"] == "high" and r < HIGH:
        problems.append(f"answer_relevance {r} too low (expected >= {HIGH})")
    if expect["answer_relevance"] == "low" and r >= LOW:
        problems.append(f"answer_relevance {r} too high (expected < {LOW})")
    if verdict["unwarranted_abstention"] != expect["abstention"]:
        problems.append(f"unwarranted_abstention {verdict['unwarranted_abstention']} "
                        f"(expected {expect['abstention']})")
    return problems


def main():
    client = build_client(os.environ["OPENROUTER_API_KEY"])
    passed = 0
    for case in CASES:
        verdict = judge_answer(client, case.get("question", QUESTION),
                               case["evidence"], case["answer"])
        if verdict.get("parse_error"):
            print(f"[{case['id']}] JUDGE PARSE ERROR: {verdict['raw'][:120]}")
            continue
        problems = check(case["id"], verdict, case["expect"])
        passed += not problems
        status = "PASS" if not problems else "FAIL"
        print(f"[{status}] {case['id']:24} faith={verdict['faithfulness']:.2f} "
              f"rel={verdict['answer_relevance']:.2f} "
              f"abstain={str(verdict['unwarranted_abstention']):5} | {verdict['reason'][:80]}")
        for p in problems:
            print(f"         -> {p}")

    print(f"\nJUDGE VALIDATION: {passed}/{len(CASES)} using {JUDGE_MODEL}")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())

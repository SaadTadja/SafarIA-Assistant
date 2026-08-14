"""LLM-as-judge scoring, replacing a keyword grounding check that failed both ways: it
flagged "you would *not* be eligible for a refund" as a hallucination, and passed
"I couldn't find the refund policy" on a scenario whose keyword was "refund".

Three axes, scored against the evidence the model actually saw: faithfulness (claims
supported by context), answer_relevance (addresses the question vs deflects), and
unwarranted_abstention (claims ignorance while the context held the answer).

Scope: it sees (question, context, answer), so it grades generation, not retrieval. Given
empty context, "I couldn't find that" is correct and scores well - retrieval failures are
caught by eval/robustness_eval.py instead.
"""

import json

from app.router import call_with_retry

# Not MODEL_NAME: a model grading its own output is lenient toward it.
JUDGE_MODEL = "anthropic/claude-haiku-4.5"

JUDGE_SYSTEM = """You grade an airline assistant's answers. You are strict and literal.

You receive the USER QUESTION, the EVIDENCE the assistant was given (tool results and any
retrieved policy text), and the ASSISTANT ANSWER.

Score three axes independently. Read negations carefully: "you are NOT eligible for a
refund" is a different claim from "you are eligible for a refund", and an answer that
correctly reports a negative entitlement is faithful, not a hallucination.

1. faithfulness (0-1): fraction of the answer's factual claims that the EVIDENCE supports.
   1.0 = every claim supported. Penalise invented numbers, fees, deadlines or entitlements.
   An answer that makes no factual claims at all is 1.0 by default - judge relevance
   separately, do not punish it twice here.

2. answer_relevance (0-1): does the answer actually address the question?
   1.0 = directly answers it. 0.0 = deflects entirely (e.g. "contact customer support"
   when the evidence contained the answer). Redirecting the user elsewhere when the
   evidence genuinely lacks the answer is not penalised here - that is abstention, axis 3.

3. unwarranted_abstention (true/false): true ONLY if the answer claims it cannot find or
   does not have information that IS present in the EVIDENCE. If the evidence is empty or
   says found=false, an answer admitting ignorance is correct behaviour - return false.

Return ONLY a JSON object, no prose:
{"faithfulness": 0.0-1.0, "answer_relevance": 0.0-1.0,
 "unwarranted_abstention": true/false, "unsupported_claims": ["..."], "reason": "one sentence"}"""


def format_evidence(tool_results: list[dict]) -> str:
    """Render what the model saw into something the judge can read."""
    if not tool_results:
        return "(no tools were called - the assistant answered from general conversation)"
    parts = []
    for entry in tool_results:
        parts.append(
            f"--- tool: {entry['tool']}  args: {json.dumps(entry['args'], ensure_ascii=False)}\n"
            f"{json.dumps(entry['result'], ensure_ascii=False, default=str)}"
        )
    return "\n".join(parts)


def judge_answer(client, question: str, tool_results: list[dict], answer: str,
                 model: str = JUDGE_MODEL) -> dict:
    """Score one answer. Returns the three axes plus token usage."""
    user_content = (
        f"USER QUESTION:\n{question}\n\n"
        f"EVIDENCE:\n{format_evidence(tool_results)}\n\n"
        f"ASSISTANT ANSWER:\n{answer or '(empty)'}"
    )
    response = call_with_retry(
        client.chat.completions.create,
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        max_tokens=400,
    )
    raw = (response.choices[0].message.content or "").strip()
    # Judges wrap JSON in prose or fences often enough to be worth handling.
    if "```" in raw:
        raw = raw.split("```")[1].removeprefix("json").strip()
    start, end = raw.find("{"), raw.rfind("}")
    try:
        verdict = json.loads(raw[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return {"parse_error": True, "raw": raw[:300],
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens}

    return {
        "faithfulness": float(verdict.get("faithfulness", 0.0)),
        "answer_relevance": float(verdict.get("answer_relevance", 0.0)),
        "unwarranted_abstention": bool(verdict.get("unwarranted_abstention", False)),
        "unsupported_claims": verdict.get("unsupported_claims", []),
        "reason": verdict.get("reason", ""),
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }

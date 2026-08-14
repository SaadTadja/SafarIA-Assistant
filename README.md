# SafarIA Assistant

An LLM travel assistant that answers policy questions from a RAG knowledge base and
user-specific questions via tool calling, routed through a single function-calling loop.
Built on FastAPI + the OpenAI SDK directly, without LangChain or LlamaIndex. Styled after
Royal Air Maroc's brand; "Safar" (travel) + "IA" (AI).

## Architecture

```
                    POST /chat  (or /chat/stream)
                              |
                              v
              LLM via OpenRouter — system prompt + 5 tools
                              |
      +-----------+-----------+-----------+------------------+
      v           v           v           v                  v
search_flights  get_flight_  get_booking  get_airport_  search_knowledge_base
                status                    info          (wraps RAG retrieval)
      |           |           |           |                  |
      +-----------+-----------+-----------+------------------+
                              v
              Answer, possibly combining several tools
              {"answer": "...", "source": "get_flight_status+rag"}
```

**The one design decision that matters:** RAG is exposed to the model as a *fifth tool*
rather than as a hand-written "if policy question then RAG else tools" branch. One loop
handles all routing, including hybrid cases — it can call `get_flight_status` to confirm a
cancellation, then `search_knowledge_base` for the refund policy, before answering. The
model never sees document text directly, only the result of a retrieval call.

## Livrables

| Attendu | Où |
|---|---|
| API `/chat` | `app/main.py` — plus `/chat/stream` (SSE) and `/health` |
| Pipeline RAG | `app/rag.py` — loading, chunking, query normalization, two-stage retrieval, confidence gate |
| Intégration des outils | `app/tools.py` (4 mocked tools) + `app/router.py` (schemas, function-calling loop) |
| Tests | `tests/` — **39 passing** |
| Interface | `app/static/index.html` — no build step |
| Évaluation | `eval/` — retrieval, routing, LLM-judge, robustness |

## Running it

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
echo OPENROUTER_API_KEY=your-key-here > .env
uvicorn app.main:app
```

Model is `openai/gpt-4o-mini`, swappable via `MODEL_NAME` in `router.py`. UI at
http://127.0.0.1:8000/ — each answer carries a badge showing its `source`.

```bash
pytest                      # 39 tests; live ones skip without an API key
python -m eval.run_eval     # routing, judge, cost — needs a key
python -m eval.robustness_eval   # retrieval only, no key needed
```

## Comment le choix RAG / Tools est fait

The system prompt states the rule explicitly: policy and procedure questions go to
`search_knowledge_base`; questions about a specific flight, booking or airport go to the
matching tool; some questions need both, and the model is told to check dynamic state
first, then the policy. That mirrors how a human agent works the problem, and it is what
lets "cancelled flight → refund" work without any special-cased routing code.

## Comment les appels inutiles sont évités

Two independent layers, not one.

1. **Tool descriptions are precise and mutually exclusive.** `search_knowledge_base`'s
   description states what it is *not* for, which keeps it from firing on flight lookups.
2. **The confidence gate** in `RagIndex.search` rejects retrieval before it reaches the LLM
   when the reranked top score is below `0.4`, so a borderline question costs no generation
   call. Measured out-of-scope rejection: **100% on 10 queries**.

Small talk is a regression test: "Hello, who are you?" must produce `source: "llm"` with
zero tool calls. **Unnecessary-call rate: 0%.**

## Comment les hallucinations sont limitées

- The prompt forbids inventing details, forbids stretching a policy to a situation it does
  not cover, and requires the assistant to state plainly what it cannot do.
- The confidence gate means weak context never reaches the model, removing the temptation
  to stretch a semi-relevant chunk into an answer.
- Every fact the model works from is structured tool output, never free text it generated.
- **Faithfulness is measured by an LLM judge, not keyword matching** (`eval/judge.py`).
  The judge is itself validated against 5 labelled cases — including an invented-compensation
  hallucination and a correctly-negated answer — before any of its numbers are used: **5/5**.

## Métriques

Measured, not estimated. `openai/gpt-4o-mini`; judge on `claude-haiku-4.5`.

| Routing | |
|---|---|
| Routing accuracy (31 scenarios) | **96.8%** |
| The brief's 6 scenarios | **100%** |
| Tool selection / argument extraction | **100%** / **100%** |
| Unnecessary tool calls on small talk | **0%** |
| Prompt-injection leaks | **0** |

| Retrieval | |
|---|---|
| Hit Rate@1 — well-formed questions (9) | **100%** |
| Hit Rate@1 — mixed query forms (26) | **92.3%** |
| Confidence-gate pass (26) | **88.5%** |
| Out-of-scope rejection (10) | **100%** |

| Answer quality (LLM judge) | |
|---|---|
| Mean faithfulness | **0.94 – 0.99** |
| Unwarranted abstention rate | **0%** |
| Judge validation | **5/5** |

| Operations | |
|---|---|
| Cost per query | **$0.00030** |
| Latency | 2.6 – 3.2 s |
| Startup (cached embeddings) | 10.3 s, vs 18.8 s cold |
| Tests | **39/39** |

**On reading these numbers honestly:** running the same evaluation twice moves aggregate
judge scores by up to 0.166, so any single-run difference smaller than that is noise, not
signal. The routing set was expanded from 6 scenarios to 31 for exactly this reason — at
n=6 one scenario was worth 16.7%.

## Bonus

| Item | Statut |
|---|---|
| Mémoire conversationnelle | ✅ `session_id` on `/chat`; live test asserts the model resolves "it" from a prior turn |
| Gestion correcte des erreurs API | ✅ Backoff on 429/5xx, capped at 60 s; 402 and rate limits mapped to clean HTTP errors; 4 regression tests |
| Streaming | ✅ `POST /chat/stream` (SSE) — `tool`, `token`, `done` events; UI reads incrementally |
| Observabilité des appels LLM | ✅ One JSON line per turn (routing, tools, latency, tokens) + per-answer source badge in the UI |
| Tests automatiques RAG / Tool | ✅ 6 brief scenarios + 31-scenario routing set + retrieval robustness regression test |
| Interface utilisateur simple | ✅ Branded, bilingual EN/FR, markdown rendering, source badges |

## Limites

- **Faithfulness is judged by an LLM**, which has its own error rate. It is a large
  improvement over the substring check it replaced — that one scored a broken answer as
  passing — but it is not ground truth.
- **Small samples.** 31 routing scenarios, 26 retrieval queries, 10 out-of-scope. Real
  measurements, wide confidence intervals.
- **One known grounding gap:** "Do I need a visa to travel to Europe?" is answered from the
  model's own knowledge instead of the corpus. Visa rules are exactly what should come from
  documents; found by expanding the routing set, unfixed.
- **Short queries retrieve poorly.** Two-word queries score near zero under the cross-encoder
  even when ranking correctly. The router is instructed to write full questions, which avoids
  it in practice but does not fix it.
- **Tools are mocked** (`app/tools.py`), and **conversation history is in-memory** — fine for
  a demo, wrong for a deployment, which would need Redis or a database with TTL.
- **No vector database.** Brute-force cosine over 312 chunks; fine at this size, won't scale
  past a few thousand.

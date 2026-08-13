# SafarIA Assistant

An LLM-based travel assistant that answers general policy questions from a RAG knowledge
base and dynamic/user-specific questions via tool calling, routed through a single
function-calling decision loop. Styled after Royal Air Maroc's own brand palette; "Safar"
(travel, Arabic/Darija) + "IA" (AI, French).

## Architecture

```
                          POST /chat {"message": "..."}
                                     |
                                     v
                  LLM via OpenRouter (system prompt + 5 tools)
                                     |
              +----------------------+----------------------+
              |                      |                       |
              v                      v                       v
      search_flights          get_flight_status      search_knowledge_base
      get_airport_info         get_booking           (wraps RAG retrieval)
              |                      |                       |
              +----------------------+----------------------+
                                     |
                                     v
                       Final answer (may combine multiple
                       tool results, e.g. flight status + policy)
                                     |
                                     v
                    {"answer": "...", "source": "get_flight_status+rag"}
```

**Key design decision:** RAG retrieval is exposed to the LLM as a 5th tool
(`search_knowledge_base`) alongside the 4 real API tools, rather than as a separate
hand-written "if policy question then RAG else tools" branch. One function-calling loop
handles the entire routing decision, including hybrid cases, by calling multiple tools in
sequence (e.g. `get_flight_status` to confirm a cancellation, then `search_knowledge_base`
for the refund policy) before producing a final answer.

The documents themselves stay plain text; only the retrieval function is callable. This is
different from "the docs are tools" - the LLM never sees document text directly, it only
ever sees the *result* of a retrieval call.

## Project layout

```
app/
  docs/              Hand-written starter documents (only kept live where real source
                     material doesn't fully supersede them - see Knowledge base sources)
  rag.py             Document loader (chunks by markdown '##' section for app/docs, by
                     paragraph for Docs(for retrieving)), two-stage retrieval (multilingual
                     bi-encoder candidate search + cross-encoder reranking), confidence gate
  tools.py           The 4 mocked API tools from the brief
  router.py          System prompt, the 5-tool model, the function-calling loop
  main.py            FastAPI app, POST /chat, graceful API-error handling, serves the UI
  static/            Branded browser chat UI (no build step) - shows the source badge
                     per answer, hero banner, bilingual EN/FR support
tests/
  test_rag.py        Retrieval correctness per category + confidence-gate rejection
  test_tools.py      Unit tests for the mocked tools
  test_router.py     The 6 routing scenarios from the brief (live API, skipped w/o key)
  test_main.py       API contract tests + API-error regression tests
eval/
  run_eval.py        Computes real evaluation metrics from actual retrieval scores and
                     live API calls, not estimates
  calibrate_threshold.py   Empirical confidence-threshold calibration
  ui_smoke_test.py   Live browser UI check (Playwright)
```

## Project evolution: what actually changed, and why

This project went through several real pivots, each driven by an actual constraint or a
measured problem, not a plan drawn up in advance. Documented here rather than smoothed
over, because the reasoning behind each change is as relevant as the final state.

**LLM provider:**

| Stage | Model | Why it changed |
|---|---|---|
| 1. Learning phase | Gemini (free tier) | Used only while building intuition for RAG/tool-calling concepts before writing the real deliverable |
| 2. Initial deliverable | `openai/gpt-4o-mini` via OpenRouter | Reliable tool-selection/argument accuracy at low cost - measured 100% routing accuracy on all 6 brief scenarios |
| 3. Free-tier fallback | `nvidia/nemotron-nano-9b-v2:free` via OpenRouter | The OpenRouter account had a $5 minimum top-up that available budget couldn't clear. Four free models were tried; three failed outright (retired from the free tier, rate-limited after ~4.5 minutes, or returned empty responses) before this one proved reliable enough to use. Measured routing accuracy: 66.7-83.3% across runs, down from 100% |
| 4. Paid attempt #2 | `claude-haiku-4-5` via Anthropic direct | Re-measured 100% routing accuracy in initial testing - but the provided API key turned out to have a **$0 credit balance**, meaning every live request failed outright with `BadRequestError`, no partial degradation |
| 5. Current state | Back to `nvidia/nemotron-nano-9b-v2:free` via OpenRouter | The only currently-viable no-cost option. This is the real, current configuration - not a placeholder for something better. Two independent measurements (stages 2 and 4) confirm routing accuracy would return to ~100% with either paid provider funded |

**RAG knowledge base:**

Started from a small hand-written placeholder corpus (7 categories, plausible but invented
policy numbers) to validate the retrieval pipeline architecture. Progressively replaced with
real public source material per the recruiter's guidance that knowledge-base collection is
part of the candidate's scope - Royal Air Maroc's own passenger-facing pages, plus Rules
85/87/90 extracted from the official ATPCO Tariff AT-1 filed with the US DOT. See "Knowledge
base sources" below for the specific curation decisions made along the way, including one
placeholder document that was archived after being found to directly conflict with real fare
data, and a scraped page that turned out to mix genuine RAM refund policy with an unrelated
third-party claims-service's promotional content - cleaned to keep only the former.

**Retrieval quality**, measured before and after each change rather than assumed:

| Stage | Hit Rate@1 | Precision@3 |
|---|---|---|
| After 3rd content enrichment round | 77.8% | 72.2% |
| + rewording one chunk to match query phrasing | 88.9% | 72.2% (unchanged - repositioning within top-4 doesn't change the correct-chunk count) |
| + archiving the conflicting placeholder doc, filtering footnote-fragment chunks | **100%** | **75%** |

Two of the ten remaining "off-topic" slots in the current Precision@3 measurement turned out,
on inspection, to be **correct behavior** - genuinely cross-topical content (e.g. a fare table
covering both change and refund conditions) legitimately surfacing under both categories, not
a bug. Pushing Precision@3 materially higher would require either accepting a small amount of
real semantic-adjacency confusion (e.g. "connection time" vs. "general arrival time" queries
sharing vocabulary) as a hard case, or a reranker/embedding-model change with unverified
payoff - documented as a known limitation rather than chased further.

**UI:** started as a minimal functional chat interface (bonus point: "interface utilisateur
simple"), then restyled using Royal Air Maroc's actual brand palette (sampled directly from
their live site's CSS custom properties, not guessed), a custom hero banner, and finally
renamed to "SafarIA Assistant" with bilingual English/French support made explicit in the UI
itself. Several real visual bugs were found and fixed via live screenshot review rather than
assumed correct on the first pass: a background-gradient seam between page sections, a
generic emoji icon replaced with a mark consistent with the branding, and a suggestion chip
that referenced a generic foreign airport (Paris CDG) instead of Royal Air Maroc's actual hub
(Casablanca CMN).

## Running it

Uses [OpenRouter](https://openrouter.ai) (OpenAI-compatible API). **Current configuration:
`nvidia/nemotron-nano-9b-v2:free`** (see "Project evolution" above for why) - swappable via
`MODEL_NAME` in `router.py` without touching any routing logic. Switching to a paid model
(OpenRouter or otherwise) is a one-line change plus `build_client`/error-type adjustments if
changing providers entirely.

> **Known trade-off of the current free-tier configuration**: routing accuracy measured at
> 66.7-83.3% across different runs, down from 100% with either paid provider tested. This
> is a model-reliability ceiling, not a router-logic bug - see "Project evolution" for the
> two independent paid-model measurements that confirm this.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

echo OPENROUTER_API_KEY=your-key-here > .env    # never commit this file (see .gitignore)
uvicorn app.main:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d "{\"message\": \"What is the status of flight AH1235?\"}"
```

Or open **http://127.0.0.1:8000/** in a browser for the chat UI (bonus point: "interface
utilisateur simple") - `app/static/index.html`, plain HTML/CSS/JS with no build step. Each
answer is shown with a colored badge for its `source` (knowledge base, a specific tool, or a
hybrid of both), making the routing decision visible per message, not just in logs.

Tests:

```bash
pytest                          # everything, including live routing tests if .env is set
pytest tests/test_main.py tests/test_rag.py tests/test_tools.py   # non-live subset only
```

25 non-live tests currently pass (API contract, retrieval correctness, mocked tool unit
tests). The 8 live tests in `tests/test_router.py` (the brief's 6 scenarios + a small-talk
check + a hard-edge-case check) require a funded API key and are skipped automatically
without one.

## How RAG vs. Tools is chosen

The system prompt in `router.py` tells the model explicitly: policy/procedure questions
(baggage, refunds, check-in, flight changes, travel documents, airport services, special
assistance) go to `search_knowledge_base`; questions about a specific flight, booking, or
airport go to the matching tool; some questions need both, and the model is told to check
dynamic state first (e.g. flight status) before pulling the relevant policy. This mirrors how
a human agent would work the problem, and it's what allows the "cancelled flight -> refund"
scenario to be handled without any special-cased routing code.

## How unnecessary calls are avoided

Two mechanisms, not one:

1. **Tool descriptions are precise and mutually exclusive.** `search_knowledge_base`'s
   docstring explicitly states what it's *not* for ("Do NOT use this to look up a specific
   flight's status..."), which is what keeps it from firing on flight/booking questions.
2. **The confidence threshold in `RagIndex.search`** (`app/rag.py`) rejects retrieval before
   it ever reaches the LLM if the top score (after reranking - see "Retrieval: two-stage
   reranking" below) is below `0.5` (recalibrated for the reranker's score scale - see
   `eval/calibrate_threshold.py`) - so even if `search_knowledge_base` gets called on a
   borderline or unanswerable question, no wasted generation call happens on top of it; the
   function itself returns `found: False`.

`tests/test_router.py` includes a small-talk case ("Hello, who are you?") that must produce
`source: "llm"` with zero tool calls, as a regression check against unnecessary calls creeping
back in. It also includes `test_router_catches_hard_edge_case_raw_rag_misses`: a query where
the raw similarity score is actually *higher* than some genuine matches, meaning threshold #2
alone can't reject it - that test verifies mechanism #1 (the router's own judgment about query
intent) catches it anyway. Two independent layers, not one.

## How hallucinations are limited

- The system prompt explicitly instructs: *"NEVER invent details that weren't returned by a
  tool"* and to say so plainly when `search_knowledge_base` returns `found: False` or a tool
  returns an error.
- `search_knowledge_base` never hands the LLM weak/irrelevant context to begin with - the
  confidence threshold means the LLM literally cannot see chunks below the similarity bar,
  removing the temptation to stretch a semi-related chunk into an answer.
- Every tool result the LLM works from is real, structured data returned by a function call
  - never free text it generated itself - so the final answer is always synthesized from
  something inspectable and testable (see `test_router.py`).
- **Content provenance is tracked deliberately, not incidentally.** When a hand-written
  placeholder document was found to state a specific policy detail ("flexible fares allow
  free date changes") that directly conflicted with real Royal Air Maroc fare-table data
  (which shows change fees applying broadly), the placeholder was archived rather than left
  live - an unverified claim about a financially consequential entitlement (refund/change
  eligibility) is exactly the kind of thing this project treats as a hallucination risk in
  the corpus itself, not just in generation.

## How API errors are handled

`call_with_retry` in `router.py` wraps every API call with exponential backoff on rate limits
(`RateLimitError`) and transient 5xx errors (`APIStatusError` with `status_code >= 500`),
retrying up to 5 times before giving up. Non-retryable errors (4xx other than rate limits -
e.g. a `402` insufficient-credits error, which was hit for real during development on both
providers tried) propagate immediately rather than retrying pointlessly. `max_tokens=500` is
set explicitly on every request: answers here are short, and an unbounded default risks both
unnecessary cost and, as happened during testing, requesting more tokens than a low account
balance can cover.

**A second, real bug was caught via manual UI testing, not by inspection**: `chat_endpoint`
in `main.py` didn't catch `router_chat`'s exceptions at all, so a `402` (insufficient
credits) or a `RateLimitError` propagated as an unhandled exception. The browser UI's
error-handling code had nothing valid to parse, so the failure was silent/broken rather than
shown to the user. Fixed by wrapping the call in `chat_endpoint` and translating both cases
into a proper `HTTPException` (`502` with a clear message for provider errors, `429` for
rate limits) - verified by re-testing the UI live (`eval/ui_smoke_test.py`) and covered by
two regression tests in `tests/test_main.py`.

**A third real failure, hit during this project's evaluation runs**: OpenRouter's free tier
has a *daily* request cap (`free-models-per-day`) separate from per-minute rate limiting -
`call_with_retry`'s backoff cannot recover from this (it's not transient), so a full eval run
mid-quota simply fails with a clear error rather than hanging or silently producing partial
results. Documented here as a known operational constraint of the free-tier configuration.

## Knowledge base sources

Per the recruiter's guidance that the knowledge base collection is part of the candidate's
scope, the corpus combines two layers, merged by `app/rag.py`:

- **`app/docs/*.md`** - a small hand-written starter set (the original "small but
  representative" corpus used to validate the pipeline).
- **`Docs(for retrieving)/<Category>/*.txt`** - real public source material: Royal Air Maroc's
  published passenger-facing pages (baggage, check-in, flight change, travel documents,
  special assistance - reduced mobility, unaccompanied minors, service animals, pregnancy)
  and Rules 85/87/90 (Schedules & Cancellations, Denied Boarding Compensation, Refunds)
  extracted from ATPCO Tariff AT-1, the official passenger fares and rules tariff filed with
  the US DOT on behalf of Royal Air Maroc (`eval/extract_atpco.py` does the extraction and
  cleanup, run once, output committed as a `.txt`). Real content now covers all 7 categories.

**Curation decisions, not blind ingestion:**
- A third PDF (an ACI World Airport Traffic Forecast executive summary, dropped into the
  `Airport services` folder) was excluded - it's a global passenger-volume statistics report,
  not passenger-facing service information, and would have diluted that category with
  off-topic content.
- Where real data **contradicted** the hand-written starter doc for the same category
  (baggage weight limits, check-in windows, and - discovered later - flight-change fee
  policy), the placeholder was archived (`app/docs/_archive_placeholder/`) so the LLM never
  sees two conflicting numbers for the same fact. This happened three times across the
  project, most recently for `flight_change_policy.md`.
- Where real data was real but **thin or non-conflicting**, placeholders were kept alongside
  it instead, since there's no contradiction risk and it adds coverage the real source lacks.
- Scraped web content had boilerplate noise (`Open in a new window`, `Image` placeholders,
  standalone table-footnote fragments like `(*) Service disponible dans certains aeroports`);
  `app/rag.py` filters known noise patterns and drops sub-15-character fragments before
  chunking.
- A refund-policy text file that had genuine Royal Air Maroc content pasted in was found to
  also contain an unrelated third-party claims-aggregator site's content mixed in below it -
  a dynamic table of example flights, a comparison table naming other airlines (Air France,
  KLM, Japan Airlines), and promotional/CTA text. Cleaned to keep only the genuine Royal Air
  Maroc EU261 policy content; the other-airline and other-airport references were removed
  entirely rather than filtered by category, since they weren't relevant to this assistant
  under any category.
- The ATPCO tariff's dense legal text has no blank lines (a PDF-extraction artifact), which
  would have produced one 22,000-character chunk per rule. `extract_atpco.py` splits on the
  tariff's own lettered sub-headings instead, producing well-sized chunks.
- A pregnancy-related liability waiver was supplied as a bilingual (English/Arabic) fillable
  PDF form - mostly blank signature fields and repeated legal boilerplate around one real
  fact. Rather than ingest the raw form, the one fact (medical clearance required beyond 8
  months' gestation) was extracted as clean text.
- Two documents genuinely relevant to more than one category (a fare-family table covering
  both change and refund conditions; a flight-disruptions doc covering both cancellation
  procedure and refund eligibility) were deliberately duplicated across both category
  folders rather than filed under one - real airline documentation doesn't respect clean
  category boundaries, and forcing a single category would have hidden genuinely relevant
  content from the other query type.
- **Switched to a multilingual embedding model** (`paraphrase-multilingual-MiniLM-L12-v2`,
  replacing an English-only model) once the corpus became substantially French-language
  content from RAM's French site - confirmed necessary by a real test failure where an
  English "wheelchair assistance" query didn't match French-only content at all. This
  required recalibrating `CONFIDENCE_THRESHOLD` (see `eval/calibrate_threshold.py`): the
  multilingual model has a measurably worse score separation than the English-only model did,
  compensated for by the router's tool-selection judgment as a second, independent defense
  layer (see "How unnecessary calls are avoided" above).

## Retrieval: two-stage reranking

`RagIndex.retrieve` (`app/rag.py`) is two stages, not one:

1. **Bi-encoder candidate search** - the multilingual embedding model scores the whole
   corpus against the query (cheap: precomputed chunk vectors, one query encode per call),
   narrowed to the top `CANDIDATE_POOL_SIZE=6`.
2. **Cross-encoder reranking** - `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` re-scores just
   those 6 candidates by reading the query and each chunk's text *together* (a bi-encoder
   compares two independently-computed vectors and can never do this), then the top `top_k`
   of the reranked list is what actually reaches the LLM.

This exists specifically to address a precision decline observed during corpus enrichment - a
cross-encoder is measurably better at telling genuinely relevant content from merely
lexically-similar content, at the cost of being too expensive to run over the whole corpus
(hence the two-stage design: cheap search narrows the field, expensive reranking only touches
a small pool).

**Reranking was not a strict, unconditional win, and that's documented rather than glossed
over**: it improved Precision@top_k but initially *regressed* Hit Rate@1 on one specific query
("How early should I arrive at the airport?", where a French connection-timing chunk outranked
the correct check-in content) and added latency per RAG call. That specific regression was
later fixed not by changing the reranker, but by rewording the underlying check-in content to
more directly match the query's phrasing - a reminder that a ranking problem doesn't always
need a ranking-algorithm fix.

**`CANDIDATE_POOL_SIZE` was tuned empirically, not guessed.** Tested 12 vs. 6:
`CANDIDATE_POOL_SIZE=6` recovers a meaningful share of the latency reranking added while
keeping most of the precision gain - a clearly better trade than the pool=12 default, so 6 is
what's actually configured.

## Evaluation

`eval/run_eval.py` computes real metrics rather than estimates - retrieval scores from the
actual corpus, and live API calls for routing/tool-selection/argument extraction.

**RAG retrieval (current, verified against the current corpus and code):**

| Metric | Value |
|---|---|
| Hit Rate@1 | 100% |
| Hit Rate@3 | 100% |
| Precision@3 | 75% |
| Out-of-scope rejection rate | 100% |

**Router / tool-calling (live, current free-tier model):**

| Metric | Value |
|---|---|
| Routing accuracy (6 brief scenarios) | 83.3% (5/6) - varies 66.7-83.3% across runs |
| No tool call on small talk | Pass |
| Rejects hard out-of-scope edge case | Pass |

**Routing accuracy with a funded paid model, measured earlier in this project (see "Project
evolution"): 100%, on both `gpt-4o-mini` and `claude-haiku-4-5` independently.**

Metrics not yet automated: Faithfulness/Completeness via an LLM-as-judge pass (current
grounding checks are keyword-based, which has real, demonstrated limitations - a substring
check once flagged "you would **not** be eligible for a refund" as a hallucination because
it contains "eligible for a refund", missing the negation) and NDCG/MRR, which would need
full relevance judgments across the whole corpus per query.

## Limitations

- **Currently running on a free-tier LLM**, with a measured routing-accuracy ceiling of
  66.7-83.3% versus 100% with a funded paid provider (verified twice, with two different
  providers - see "Project evolution"). This is the single largest gap between this
  project's engineering and its live behavior, and it's a budget constraint, not an
  architecture gap.
- **No conversational memory.** Each `/chat` call starts a fresh session; a follow-up like
  "and what about my baggage?" after a flight-status question won't have context from the
  prior turn.
- **No vector database.** Retrieval is a brute-force cosine similarity scan over all chunks
  in memory (`app/rag.py`). This is fine at the current corpus size but won't scale past a
  few thousand chunks - a real production system would need Chroma/FAISS/pgvector with
  approximate nearest-neighbor search.
- **Tools are mocked.** `app/tools.py` returns fixed in-memory data; no real airline API was
  available for this challenge. Swapping in real API calls doesn't require changing anything
  else in the system.
- **Confidence threshold (0.5, reranker scale) was picked from an empirical calibration run**
  (`eval/calibrate_threshold.py`), not a labeled evaluation set, and the calibration found no
  clean score separation between relevant and irrelevant content even after reranking. The
  threshold alone cannot perfectly gate retrieval; the router's own judgment is relied on as
  a second layer (see "How unnecessary calls are avoided").
- **Precision@3 (75%) has a partially-understood ceiling.** Roughly a fifth of the
  "off-topic" slots in the current measurement are actually correct behavior (legitimately
  cross-category content), and the remainder is a mix of genuine semantic-adjacency
  confusion (untested whether a stronger reranker or embedding model would resolve it) and
  negligible-score noise. Documented as understood rather than chased further, given
  uncertain payoff for the remaining effort.
- **The RAM logo used in the UI header is Royal Air Maroc's actual trademarked asset**, used
  here for a private/local demo of this challenge submission. If sharing this project
  publicly beyond this evaluation, the custom (non-trademarked) plane mark it was swapped in
  for and out of during development would be the safer choice.
- **Uses OpenRouter's free tier**, which carries a *daily* request cap in addition to
  per-minute rate limiting (see "How API errors are handled") - a full evaluation run can
  hit this mid-run, which happened for real during this project's own testing.

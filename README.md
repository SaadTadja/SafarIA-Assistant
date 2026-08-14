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
                     paragraph for Docs(for retrieving)), query normalization, two-stage
                     retrieval (multilingual bi-encoder candidate search + cross-encoder
                     reranking), confidence gate
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
| 5. Free-tier fallback again | Back to `nvidia/nemotron-nano-9b-v2:free` via OpenRouter | The only viable no-cost option at the time. Two independent measurements (stages 2 and 4) suggested routing accuracy would return to ~100% with either paid provider funded |
| 6. Current state | `openai/gpt-4o-mini` via OpenRouter (funded) | The account was topped up, which settled the question the previous four stages danced around: a **full evaluation run costs $0.0019**, and a single query $0.00028. The constraint was never the inference bill - it was OpenRouter's $5 minimum top-up, a threshold problem misread as an affordability one for most of this project's life. Routing accuracy re-measured at 100%, matching stages 2 and 4 |

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

(Both columns above are measured on the original 9-query, full-sentence set. The Precision
column is Precision@4 - see the note under "Evaluation". For the expanded 26-query set that
also covers terse and identifier-bearing queries, see "Two bugs" below.)

Two of the ten remaining "off-topic" slots in the current Precision@3 measurement turned out,
on inspection, to be **correct behavior** - genuinely cross-topical content (e.g. a fare table
covering both change and refund conditions) legitimately surfacing under both categories, not
a bug. Pushing Precision@3 materially higher would require either accepting a small amount of
real semantic-adjacency confusion (e.g. "connection time" vs. "general arrival time" queries
sharing vocabulary) as a hard case, or a reranker/embedding-model change with unverified
payoff - documented as a known limitation rather than chased further.

**Two bugs that every test and metric in this project scored as passing.** Both were found
by using the assistant in a browser once the paid key made that reliable, not by reading the
code, and neither would have been caught by the evaluation as written. They are documented
at length because the way they hid is more instructive than the fixes.

*1. The retrieval test set measured a query distribution the router never produced.*

Every one of the 9 RAG evaluation queries is a well-formed sentence ("Can I get a refund if
my flight is cancelled?"). The router does not write sentences. Asked the brief's flagship
question - "My flight AH1235 is cancelled. Can I get a refund?" - it usually called
`search_knowledge_base` with the query `'refund policy'`, which the cross-encoder scores
**0.464** against the then-current 0.50 gate. Retrieval returned `found: False` while 59
refund-policy chunks sat in the corpus, and the assistant told the user to contact customer
support. **The flagship scenario failed 4 times in 5 in the live app**, against a measured
Hit Rate@1 of 100%.

Both numbers were correct. The retrieval was genuinely excellent at the queries it was
tested on, and those were not the queries it received.

The evaluation scored the broken answer as a *pass*: scenario 5's grounding check is
`answer_must_contain_any: ["refund"]`, and *"I couldn't find specific information regarding
the refund policy"* contains "refund". This is the same keyword-matching weakness already
documented under "Evaluation" below, in the far more dangerous direction - a false pass
rather than a false alarm.

Fixed at the query the model writes rather than the gate that judges it: the `query`
argument's description now states a phrasing contract (full question, never keywords or
identifiers). Lowering the threshold instead would have papered over a bad query by
weakening the only mechanism giving 100% out-of-scope rejection. Measured after: 10/10
retrieval, 5/5 correct live answers.

*2. The model was instructed to use today's date and never told what it was.*

The system prompt says to default to today's date for an unspecified flight date. Nothing
ever supplied one. The model has no clock, so it invented dates from its training era -
`2023-10-31`, `2023-10-12` and `2023-10-18` across three runs of the same query, in 2026.
Invisible because `get_flight_status` ignores its `date` argument; wrong the instant the
mocked tools are swapped for a real API, which this README elsewhere claims "doesn't require
changing anything else in the system". This was that one thing. The date is now rendered per
request rather than at import, so a process that stays up for days doesn't keep reporting
the day it booted. 5/5 correct after.

The common shape: both bugs are correct-looking, test-passing, and only observable by
running the thing. The project had been verified by reading and by unit tests, and the one
check that would have caught either - typing a question into the UI - costs three requests,
well inside even the free tier's daily cap that the four provider stages above were spent
working around.

**Retrieval robustness, measured after the above.** The 9-query set was expanded to 26,
adding the two forms it never covered - terse keyword queries and identifier-bearing
queries - and the out-of-scope set from 1 query to 10:

| | Hit Rate@1 | Confidence-gate pass | Out-of-scope rejection |
|---|---|---|---|
| Before | 88.5% | 69.2% | 100% |
| After | **92.3%** | **88.5%** | **100%** |
| *(identifier-bearing queries only)* | 66.7% -> **83.3%** | 16.7% -> **83.3%** | - |

Two changes, neither sufficient alone (76.9% and 73.1% separately, 88.5% together): query
normalization strips record-locator tokens absent from the corpus before retrieval, and
`CONFIDENCE_THRESHOLD` moved 0.50 -> 0.40. See "Retrieval" below for both.

Notably, **5 of the 8 original failures ranked the correct source first and were then
discarded by the confidence gate** - a calibration problem, not a ranking one, which is why
the fix targeted the gate's input rather than the reranker.

**One hypothesis tested and rejected**, recorded because a negative result is still a
result: 23 chunks (7.4% of the corpus) are bare headings under 40 characters
("Recommendations", "Upon Your Arrival"), and these were assumed to be noise crowding a
6-slot candidate pool. Dropping them, and separately merging them into the following chunk,
both made retrieval measurably *worse* (Hit Rate@1 92.3% -> 84.6%, Precision 77.9% ->
76.9%). The headings carry real semantic signal. The corpus was left untouched.

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
`openai/gpt-4o-mini`** - swappable via `MODEL_NAME` in `router.py` without touching any
routing logic. Switching to another paid model (OpenRouter or otherwise) is a one-line change
plus `build_client`/error-type adjustments if changing providers entirely.

> **Cost**: $0.00028 per query, $0.0019 for a complete evaluation run. Running this is
> effectively free; the earlier free-tier configuration (see "Project evolution") cost a
> measured 17-33 points of routing accuracy to avoid a $5 minimum top-up.

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

**36/36 tests pass** - 27 non-live (API contract, retrieval correctness, mocked tool unit
tests) plus the 9 live tests in `tests/test_router.py` (the brief's 6 scenarios, a small-talk
check, a hard-edge-case check, and a conversational-memory check), which require a funded API
key and are skipped automatically without one. `pytest` is scoped to `tests/` in
`pyproject.toml`, so a bare `pytest` from the repo root no longer aborts trying to collect
`eval/ui_smoke_test.py`, whose `playwright` dependency is deliberately optional.

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
   reranking" below) is below `0.4` - so even if `search_knowledge_base` gets called on a
   borderline or unanswerable question, no wasted generation call happens on top of it; the
   function itself returns `found: False`. The value was originally 0.5 (recalibrated for the
   reranker's score scale - see `eval/calibrate_threshold.py`) and moved to 0.4 on a sweep
   across a 26-query in-scope set and a 10-query out-of-scope set: 0.4 is where in-scope pass
   rate stops improving, and it sits 0.174 above the highest out-of-scope score observed
   (0.226), so rejection stays at 100%.

`tests/test_router.py` includes a small-talk case ("Hello, who are you?") that must produce
`source: "llm"` with zero tool calls, as a regression check against unnecessary calls creeping
back in. It also includes `test_router_catches_hard_edge_case_raw_rag_misses`: a query where
the raw similarity score is actually *higher* than some genuine matches, meaning threshold #2
alone can't reject it - that test verifies mechanism #1 (the router's own judgment about query
intent) catches it anyway. Two independent layers, not one.

## Conversational memory

Bonus feature: `POST /chat` accepts an optional `session_id`. Omit it to start a new
conversation; the server generates one and returns it in the response. Send it back on
follow-up requests and the router has full context from every prior turn in that session -
`router.chat()` accepts an optional `messages` history and continues it rather than always
starting fresh, and returns the updated history for the caller to persist.

`app/main.py` keeps an in-memory `dict[session_id, messages]` (`_sessions`) - the simplest
option that satisfies the bonus requirement for a demo; see "Limitations" for why this
wouldn't be the right choice for a real deployment. The browser UI (`app/static/index.html`)
generates a session on the first message and reuses it for the rest of the page's lifetime.

Tested at two levels: `tests/test_main.py` has two non-live tests verifying the session
plumbing itself (a session ID is generated and correctly round-tripped, and the exact history
returned by one call is what the next call receives) without needing a live LLM call.
`tests/test_router.py::test_conversational_memory_uses_prior_context` is a live test that
actually exercises the model: it asks about flight AH1235, then asks "Can I get a refund for
it?" in a second turn, and asserts the router correctly resolves "it" from context (routes to
`search_knowledge_base` and doesn't ask which flight) - this is the test that verifies memory
*works*, not just that it's wired up.

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
retrying up to 5 times, and **bounded to 60 seconds of cumulative sleep**. That cap was added
after watching the unbounded schedule (15+30+60+120+240 = 465 s) hold an HTTP request for
over three minutes before surfacing a 429 it already knew about on the first attempt - a
sensible policy for a batch script and a useless one behind a browser that gave up minutes
earlier. Whatever budget remains is still spent retrying; only the pointless tail is cut.
Non-retryable errors (4xx other than rate limits -
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
results. This no longer applies to the current paid configuration, but it is what the
unbounded retry schedule was originally tuned against - and tuning for a batch script is
exactly why that schedule behaved so badly behind an HTTP request (see the 60 s cap above).

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

`RagIndex.retrieve` (`app/rag.py`) is two stages, preceded by a normalization step:

0. **Query normalization** - record-locator tokens (flight numbers like `AH1235`, booking
   references like `ABC123`) are stripped before retrieval. Policy documents describe rules,
   never individual flights, so such a token can only be noise here - but a cross-encoder
   scores the query and chunk text *together*, so an unmatched token drags the whole pair's
   score down rather than being ignored. Measured: `"refund for cancelled flight AH1235"`
   scores 0.059 and is rejected; `"refund for cancelled flight"` scores 0.790 and passes.
   Only tokens **absent from the corpus** are stripped, which is what keeps `EU261` (a
   regulation the refund documents cite) and `B737`/`B787`/`E190` (aircraft the fleet
   documents name) intact - a blanket regex over identifier-shaped tokens would have caused
   exactly the failure this is meant to prevent.

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

**RAG retrieval, original 9-query set (full-sentence queries only):**

| Metric | Value |
|---|---|
| Hit Rate@1 | 100% |
| Hit Rate@3 | 100% |
| MRR | 1.000 |
| Precision@3 | 85.2% |
| Precision@4 | 75% |
| Out-of-scope rejection rate | 100% |

> `run_eval.py` reports the 75% figure under the key `precision_at_3`, but it divides by
> `len(sources)` where `top_k=4` - so that number is Precision@**4**. True Precision@3 is
> 85.2%. The metric was understating retrieval quality by 10 points.

**RAG retrieval, expanded 26-query set** (adds terse and identifier-bearing queries - the
forms the router actually emits, which the original set never covered):

| Metric | Full sentence (9) | Terse (11) | Identifier (6) | Overall (26) |
|---|---|---|---|---|
| Hit Rate@1 | 100% | 90.9% | 83.3% | **92.3%** |
| Confidence-gate pass | 100% | 81.8% | 83.3% | **88.5%** |

Out-of-scope rejection: **100%** across 10 queries. The 3 remaining in-scope failures
("check-in deadline", "flight change fees", "baggage allowance for booking") all rank
sensibly but score near zero - the cross-encoder is simply unreliable on 2-3 word queries.
That is a model property rather than a tunable defect, and the router-side phrasing contract
prevents those forms reaching retrieval in practice.

**Router / tool-calling (live, `openai/gpt-4o-mini`):**

| Metric | Value |
|---|---|
| Routing accuracy (6 brief scenarios) | **100%** |
| Tool-selection accuracy | **100%** |
| Argument extraction accuracy | **100%** |
| Answer grounding rate (keyword-based - see caveat below) | 100% |
| Unnecessary tool calls on small talk | **0%** |
| Tool selection, 11 extended scenarios (graded subset) | **100%** |
| Forbidden-content violations (incl. 2 prompt-injection probes) | **0%** |
| Avg latency | 2.6-2.9 s/query |
| Avg tokens | 1,411 in / 109 out |
| Cost | $0.00028/query |

Latency is almost entirely the LLM: ~1 s per round trip, 1-3 round trips per query
depending on routing. Retrieval contributes 0.274 s (of which the reranker is 0.254 s),
about 8% of a RAG-answering request. Index build at startup is a one-time 15 s.

> **The grounding rate above should not be read as a faithfulness measurement.** The check
> is `any(keyword in answer)`. It scored the flagship scenario 100% while that scenario was
> failing 4 times in 5 (see "Project evolution"), because the failure message contained the
> keyword. It cannot distinguish a correct answer, a retrieval failure, and a hallucination
> that happens to use the right word. Replacing it with an LLM-as-judge pass is the single
> highest-value item left, and now costs about a cent to run.

Metrics not yet automated: Faithfulness/Completeness via an LLM-as-judge pass (current
grounding checks are keyword-based, which has real, demonstrated limitations - a substring
check once flagged "you would **not** be eligible for a refund" as a hallucination because
it contains "eligible for a refund", missing the negation) and NDCG/MRR, which would need
full relevance judgments across the whole corpus per query.

## Bonus features

Honest scorecard against the brief's bonus list - claimed only where actually built and
verified, not where merely partially addressed:

| Bonus item | Status | Notes |
|---|---|---|
| Mémoire conversationnelle | ✅ Done | See "Conversational memory" above - tested at both a non-live (plumbing) and live (model actually uses context) level. |
| Gestion correcte des erreurs API | ✅ Done | See "How API errors are handled" - retry with backoff, distinct handling per error type, two real bugs found and fixed via testing. |
| Tests automatiques sur le choix RAG / Tool | ✅ Done | `tests/test_router.py` - all 6 brief scenarios parametrized and asserted against `source`, live-verified. |
| Interface utilisateur simple | ✅ Done | `app/static/index.html` - branded, source badges per answer, bilingual EN/FR, no build step. |
| Observabilité des appels LLM | 🟡 Partial | Every answer is tagged with its routing decision (`source`, visible as a UI badge, not just in logs) - real but lightweight. No structured logging, request tracing, or per-call latency/token tracking at runtime; `eval/run_eval.py` computes token counts and cost, but only as an offline evaluation script. |
| Streaming | ❌ Not done | `chat.completions.create()` is called without `stream=True`; the UI does one `fetch().then()`, not incremental reads. Not attempted. |

## Limitations

- **Faithfulness is not actually measured.** The grounding check is keyword matching, and it
  has now demonstrably failed in both directions: a false alarm (flagging "you would *not* be
  eligible for a refund" as a hallucination) and a false pass (scoring the flagship scenario
  100% while it failed 4 times in 5). Every "100% grounding" figure in this README means only
  that an expected word appeared. This is the most significant remaining gap.
- **Sample sizes are small.** Routing accuracy is 6 scenarios, so each one is worth 16.7% and
  no run-to-run difference below that is meaningful. Retrieval is 26 queries, out-of-scope is
  10. The numbers are real measurements, not estimates, but they are measurements with wide
  confidence intervals and should be read that way.
- **Conversation history is in-memory and unbounded** (`_sessions` dict in `main.py`) - fine
  for a demo/single-process deployment, but lost on restart and never evicted. A real
  deployment would need a store with TTL (Redis, a DB) instead.
- **No vector database.** Retrieval is a brute-force cosine similarity scan over all chunks
  in memory (`app/rag.py`). This is fine at the current corpus size but won't scale past a
  few thousand chunks - a real production system would need Chroma/FAISS/pgvector with
  approximate nearest-neighbor search.
- **Tools are mocked.** `app/tools.py` returns fixed in-memory data; no real airline API was
  available for this challenge. Swapping in real API calls doesn't require changing anything
  else in the system.
- **Confidence threshold (0.4, reranker scale)** was moved from 0.5 on a sweep over the
  expanded 26-query in-scope and 10-query out-of-scope sets, which is better evidence than the
  original `eval/calibrate_threshold.py` run but still not a labeled evaluation set. The
  threshold alone cannot perfectly gate retrieval - the cross-encoder's absolute score is
  strongly phrasing-dependent even when its *ranking* is correct - so the router's own
  judgment is relied on as a second layer (see "How unnecessary calls are avoided").
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
- **Retrieval degrades on very short queries.** Two- and three-word queries score near zero
  under the cross-encoder even when the correct chunk ranks first ("flight change fees" ranks
  correctly at 0.059). The router-side phrasing contract keeps such queries from being
  generated, but the underlying retrieval weakness is unaddressed and would resurface behind
  any other caller - a direct search box, for instance.

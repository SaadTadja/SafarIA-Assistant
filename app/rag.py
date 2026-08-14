"""RAG pipeline: load the policy documents, chunk them by section, embed, and retrieve.

Two document sources are merged into one corpus:
- app/docs/*.md: the original curated documents, chunked by '## Section' heading. Each
  section is already a self-contained, human-authored unit of meaning, so splitting on
  headings avoids cutting a sentence in half the way a fixed character count would.
- Docs(for retrieving)/<Category>/*.txt: supplementary material gathered from public
  sources (regulations, IATA/TSA standards, etc.) per the recruiter's guidance to build
  the knowledge base from public sources. Chunked by paragraph instead, since raw pasted
  research text won't reliably use markdown headings - if a file happens to include
  '## Heading' lines they're respected, otherwise it falls back to one chunk per
  blank-line-separated paragraph.

Both sources feed the same RagIndex, so this is purely additive: dropping a .txt file into
one of the category folders adds more retrievable chunks without touching what's already
tested and working in app/docs/.
"""

import hashlib
import math
import re
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

APP_DIR = Path(__file__).parent
MD_DOCS_DIR = APP_DIR / "docs"
EXTRA_DOCS_DIR = APP_DIR.parent / "Docs(for retrieving)"

# Multilingual: the corpus now contains substantial French source material (RAM's French
# site), and queries may legitimately arrive in French too given the real user base - a
# monolingual English model fails to match either direction. Confirmed by a real test
# failure with all-MiniLM-L6-v2 on an English "wheelchair assistance" query against
# French-only content.
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Second-stage reranker: a cross-encoder reads the query and a chunk TOGETHER (unlike the
# bi-encoder above, which compares two independently-computed vectors), which makes it much
# better at telling genuinely relevant content from merely lexically-similar content. This
# is the fix for the precision decline documented in README.md's Evaluation section - only
# used to re-score a small candidate pool from the bi-encoder, so it stays cheap.
RERANKER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CANDIDATE_POOL_SIZE = 6

# Recalibrated for the reranker's score scale using eval/calibrate_threshold.py - NOT the
# same number as the bi-encoder-only threshold used before reranking was added. Different
# scoring stages produce different distributions, so this must be re-run any time the
# embedding model, reranker model, or candidate pool size changes.
CONFIDENCE_THRESHOLD = 0.4

# Tokens shaped like a record locator: uppercase alphanumeric, 4+ chars, containing at
# least one letter AND one digit. Matches flight numbers (AH1235) and booking references
# (ABC123) - but also genuine corpus vocabulary (EU261, B737, B787, E190), which is why
# _normalize_query only strips the ones absent from the corpus rather than all of them.
IDENTIFIER_TOKEN_RE = re.compile(r"\b(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]{4,}\b")

EMBEDDING_CACHE_DIR = APP_DIR.parent / ".embedding_cache"


def _load_or_build_embeddings(model, texts: list[str], model_name: str):
    """Encode the corpus, reusing a cached matrix when nothing relevant has changed.

    Encoding 312 chunks costs most of the ~15 s startup, paid on every boot, every test
    session and every --reload. The cache key is a hash of the model name and the corpus
    text itself, so editing a document or switching embedding models misses the cache and
    re-encodes: correctness does not depend on remembering to clear anything.

    A corrupt or unreadable cache file is not fatal - it re-encodes and overwrites.
    """
    digest = hashlib.sha256(
        ("\x00".join([model_name, *texts])).encode("utf-8")
    ).hexdigest()[:16]
    cache_path = EMBEDDING_CACHE_DIR / f"{digest}.npy"

    if cache_path.exists():
        try:
            cached = np.load(cache_path)
            if cached.shape[0] == len(texts):
                return cached
        except (OSError, ValueError):
            pass  # fall through and re-encode

    embeddings = model.encode(texts, normalize_embeddings=True)
    try:
        EMBEDDING_CACHE_DIR.mkdir(exist_ok=True)
        np.save(cache_path, embeddings)
    except OSError:
        pass  # a read-only deployment still works, just without the speedup
    return embeddings


SECTION_HEADING_RE = re.compile(r"(?m)^##\s+")
BLANK_LINE_RE = re.compile(r"\n\s*\n")

# Matches a paragraph that's just a table footnote annotation, e.g. "(*) Service
# disponible dans certains aeroports" - meaningless once separated from its table.
FOOTNOTE_MARKER_RE = re.compile(r"^\(\*+\)\s")

# Boilerplate left over from copy-pasting website content (link labels, image alt-text
# placeholders) rather than genuine policy information. Filtered out before chunking so
# it doesn't become its own empty-content chunk in the index.
NOISE_LINES = {"open in a new window", "image", "image alternative text"}

# Maps the folder names under Docs(for retrieving)/ to the canonical source slugs used
# everywhere else in the system (tool descriptions, tests, the brief's 6 scenarios).
CATEGORY_FOLDER_TO_SOURCE = {
    "Baggage policy": "baggage_policy",
    "Refund policy": "refund_policy",
    "Check-in policy": "checkin_policy",
    "Flight change policy": "flight_change_policy",
    "Travel documents": "travel_documents",
    "Airport services": "airport_services",
    "Special assistance": "special_assistance",
}


def _load_markdown_docs(docs_dir: Path) -> list[dict]:
    chunks = []
    for path in sorted(docs_dir.glob("*.md")):
        source = path.stem  # e.g. "refund_policy"
        text = path.read_text(encoding="utf-8")

        lines = text.splitlines()
        title = lines[0].lstrip("#").strip() if lines else source

        sections = SECTION_HEADING_RE.split(text)
        for section in sections[1:]:  # sections[0] is the '# Title' line before the first '##'
            heading, _, body = section.partition("\n")
            body = body.strip()
            if not body:
                continue
            chunks.append({
                "id": f"{source}::{heading.strip().lower().replace(' ', '_')}",
                "source": source,
                "title": title,
                "heading": heading.strip(),
                "text": body,
            })
    return chunks


def _load_extra_txt_docs(extra_dir: Path) -> list[dict]:
    """Load supplementary .txt files from Docs(for retrieving)/<Category>/*.txt."""
    chunks = []
    if not extra_dir.exists():
        return chunks

    for folder in sorted(p for p in extra_dir.iterdir() if p.is_dir()):
        source = CATEGORY_FOLDER_TO_SOURCE.get(folder.name, folder.name.lower().replace(" ", "_"))

        for path in sorted(folder.glob("*.txt")):
            raw_text = path.read_text(encoding="utf-8").strip()
            if not raw_text:
                continue

            lines = [line for line in raw_text.splitlines() if line.strip().lower() not in NOISE_LINES]
            text = "\n".join(lines).strip()
            if not text:
                continue

            if SECTION_HEADING_RE.search(text):
                sections = SECTION_HEADING_RE.split(text)
                for i, section in enumerate(sections[1:]):
                    heading, _, body = section.partition("\n")
                    body = body.strip()
                    if not body:
                        continue
                    chunks.append({
                        "id": f"{source}::{path.stem}::{i}",
                        "source": source,
                        "title": folder.name,
                        "heading": heading.strip(),
                        "text": body,
                    })
            else:
                # Drop paragraphs under 15 chars too (leftover bullet glyphs, stray
                # single words from scraped nav menus) - too short to be a useful,
                # independently-retrievable fact. Also drop standalone footnote markers
                # (e.g. "(*) Service disponible dans certains aeroports") - a table
                # annotation with no meaning outside the table it refers to, which
                # otherwise becomes its own low-value, source-mismatched chunk.
                paragraphs = [
                    p.strip() for p in BLANK_LINE_RE.split(text)
                    if len(p.strip()) >= 15 and not FOOTNOTE_MARKER_RE.match(p.strip())
                ]
                for i, para in enumerate(paragraphs):
                    chunks.append({
                        "id": f"{source}::{path.stem}::{i}",
                        "source": source,
                        "title": folder.name,
                        "heading": path.stem,
                        "text": para,
                    })

    return chunks


def load_documents(md_docs_dir: Path = MD_DOCS_DIR, extra_docs_dir: Path = EXTRA_DOCS_DIR) -> list[dict]:
    """Load and chunk both document sources (see module docstring)."""
    chunks = _load_markdown_docs(md_docs_dir) + _load_extra_txt_docs(extra_docs_dir)

    if not chunks:
        raise ValueError(
            f"No document chunks found - check {md_docs_dir}/*.md and {extra_docs_dir}/*/*.txt"
        )

    return chunks


class RagIndex:
    """Embeds a set of chunks once at startup, then answers similarity queries against them."""

    def __init__(
        self,
        chunks: list[dict],
        model_name: str = EMBEDDING_MODEL_NAME,
        reranker_model_name: str = RERANKER_MODEL_NAME,
    ):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        self.reranker = CrossEncoder(reranker_model_name)
        texts = [c["text"] for c in chunks]
        self.embeddings = _load_or_build_embeddings(self.model, texts, model_name)
        # Identifier-shaped tokens the corpus actually uses, so _normalize_query can tell
        # real vocabulary (EU261, B737) from a caller's record locator (AH1235, ABC123).
        self.corpus_identifiers = {
            token for text in texts for token in IDENTIFIER_TOKEN_RE.findall(text)
        }

    def _normalize_query(self, query: str) -> str:
        """Drop record-locator tokens the corpus has never heard of.

        Policy documents describe rules, never individual flights or bookings, so a
        flight number in the query can only ever be noise here - but the cross-encoder
        scores the query and chunk text *together*, so an unmatched token drags the whole
        pair's relevance score down rather than being ignored. Measured on this corpus:

            "refund for cancelled flight AH1235"  0.059  ->  "refund for cancelled flight"  0.790
            "change flight AH1235 to another date" 0.075 ->  "change flight to another date" 0.520

        Both were rejected by the confidence gate before stripping and pass after it, with
        no change to what the corpus contains. Only tokens absent from the corpus are
        removed: EU261 is a real regulation the refund documents cite, and B737/B787/E190
        are aircraft the fleet documents name - stripping those would cause the very
        problem this is meant to fix.
        """
        stripped = IDENTIFIER_TOKEN_RE.sub(
            lambda m: m.group(0) if m.group(0) in self.corpus_identifiers else " ", query
        )
        return re.sub(r"\s{2,}", " ", stripped).strip() or query

    def _retrieve_candidates(self, query: str, top_k: int) -> list[tuple[dict, float]]:
        """First-pass retrieval: fast bi-encoder cosine similarity over the whole corpus."""
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_embedding
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_idx]

    def retrieve(
        self, query: str, top_k: int = 4, candidate_pool_size: int = CANDIDATE_POOL_SIZE
    ) -> list[tuple[dict, float]]:
        """Two-stage retrieval: the bi-encoder narrows the full corpus down to a small
        candidate pool, then the cross-encoder reranker re-scores just that pool by reading
        the query and each chunk together - far more accurate than comparing precomputed
        vectors, at the cost of only being feasible on a small pool, not the whole corpus."""
        query = self._normalize_query(query)
        candidates = self._retrieve_candidates(query, top_k=candidate_pool_size)
        pairs = [(query, chunk["text"]) for chunk, _bi_encoder_score in candidates]
        raw_scores = self.reranker.predict(pairs)
        # Manual sigmoid rather than relying on the model's default activation config,
        # which varies across cross-encoder checkpoints and sentence-transformers versions -
        # this guarantees a consistent 0-1 scale regardless.
        rerank_scores = [1 / (1 + math.exp(-float(s))) for s in raw_scores]

        reranked = sorted(
            zip([chunk for chunk, _score in candidates], rerank_scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [(chunk, score) for chunk, score in reranked[:top_k]]

    def search(self, query: str, top_k: int = 4, threshold: float = CONFIDENCE_THRESHOLD) -> dict:
        """Retrieve + apply the confidence gate. Returns found=False (no LLM call needed)
        when nothing in the corpus is actually relevant, instead of returning weak context
        that an LLM might be tempted to stretch into an answer."""
        results = self.retrieve(query, top_k=top_k)
        top_score = results[0][1] if results else 0.0

        if top_score < threshold:
            return {"found": False, "message": "No relevant policy information found in the knowledge base."}

        return {
            "found": True,
            "context": [c["text"] for c, _score in results],
            "sources": sorted({c["source"] for c, _score in results}),
            "top_score": top_score,
        }

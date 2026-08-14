"""RAG pipeline: load policy documents, chunk, embed, retrieve.

Two sources feed one corpus: app/docs/*.md chunked by '## Section', and
Docs(for retrieving)/<Category>/*.txt chunked by paragraph.
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

# Multilingual: the corpus is largely French, queries arrive in both languages.
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

RERANKER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CANDIDATE_POOL_SIZE = 6

# On the reranker's score scale; re-run eval/calibrate_threshold.py if either model changes.
CONFIDENCE_THRESHOLD = 0.4

# Record-locator shape: uppercase alphanumeric, 4+ chars, at least one letter and one digit.
IDENTIFIER_TOKEN_RE = re.compile(r"\b(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]{4,}\b")

EMBEDDING_CACHE_DIR = APP_DIR.parent / ".embedding_cache"


def _load_or_build_embeddings(model, texts: list[str], model_name: str):
    """Encode the corpus, reusing a cached matrix. Keyed by model name + corpus text, so
    editing a document or changing model re-encodes automatically."""
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

# Table footnote annotations, e.g. "(*) Service disponible..." - meaningless alone.
FOOTNOTE_MARKER_RE = re.compile(r"^\(\*+\)\s")

# Scraping boilerplate, filtered before chunking.
NOISE_LINES = {"open in a new window", "image", "image alternative text"}

# Folder names -> the source slugs used everywhere else.
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
                # Skip fragments too short to be a retrievable fact, and footnote markers.
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
        # Real corpus vocabulary, so _normalize_query keeps EU261 but drops AH1235.
        self.corpus_identifiers = {
            token for text in texts for token in IDENTIFIER_TOKEN_RE.findall(text)
        }

    def _normalize_query(self, query: str) -> str:
        """Drop record-locator tokens absent from the corpus (AH1235, ABC123).

        The cross-encoder scores query and chunk together, so an unmatched identifier drags
        the pair down: "refund for cancelled flight AH1235" scores 0.059, without it 0.790.
        Only unknown tokens go - EU261 and B737 are real corpus vocabulary.
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
        """Two-stage: bi-encoder narrows the corpus to a small pool, cross-encoder reranks
        it. The reranker is more accurate but only affordable on a small pool."""
        query = self._normalize_query(query)
        candidates = self._retrieve_candidates(query, top_k=candidate_pool_size)
        pairs = [(query, chunk["text"]) for chunk, _bi_encoder_score in candidates]
        raw_scores = self.reranker.predict(pairs)
        # Manual sigmoid: default activation varies across checkpoints and library versions.
        rerank_scores = [1 / (1 + math.exp(-float(s))) for s in raw_scores]

        reranked = sorted(
            zip([chunk for chunk, _score in candidates], rerank_scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [(chunk, score) for chunk, score in reranked[:top_k]]

    def search(self, query: str, top_k: int = 4, threshold: float = CONFIDENCE_THRESHOLD) -> dict:
        """Retrieve + confidence gate. Returns found=False rather than weak context an LLM
        might stretch into an answer."""
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

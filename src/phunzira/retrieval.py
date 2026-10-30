"""Retrievers: BM25 (optionally with a bilingual glossary), dense embeddings, and hybrid fusion.

Every retriever exposes ``search(query, k, lang=None)``. Passing ``lang`` restricts results to
chunks in that language, which is how cross-lingual retrieval is evaluated (query in one
language, index restricted to the other).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

from .chunking import Chunk
from .config import DEFAULT_EMBED_MODEL
from .text import analyze

RETRIEVER_KINDS = ("bm25", "bm25+glossary", "dense", "hybrid")


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 5, lang: str | None = None) -> list[Hit]: ...


# --------------------------------------------------------------------------- glossary


class Glossary:
    """A small bilingual term list used to expand queries across English and Chichewa.

    Each entry maps one English word to one or more Chichewa forms (singular/plural, etc.).
    Expansion works in both directions.
    """

    def __init__(self, pairs: Iterable[tuple[str, Sequence[str]]]):
        self._map: dict[str, list[str]] = {}
        for english, ny_forms in pairs:
            en_tok = self._single(english)
            for form in ny_forms:
                ny_tok = self._single(form)
                self._link(en_tok, ny_tok)
                self._link(ny_tok, en_tok)

    @staticmethod
    def _single(term: str) -> str:
        tokens = analyze(term)
        if len(tokens) != 1:
            raise ValueError(f"Glossary terms must be a single word, got {term!r}")
        return tokens[0]

    def _link(self, a: str, b: str) -> None:
        targets = self._map.setdefault(a, [])
        if b not in targets:
            targets.append(b)

    @classmethod
    def from_tsv(cls, path: str | Path) -> "Glossary":
        """Read lines of ``english<TAB>chichewa1, chichewa2``. Blank lines and ``#`` comments are skipped."""
        pairs: list[tuple[str, list[str]]] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            english, _, rest = line.partition("\t")
            forms = [f.strip() for f in rest.split(",") if f.strip()]
            if not forms:
                raise ValueError(f"Glossary line has no Chichewa forms: {line!r}")
            pairs.append((english.strip(), forms))
        return cls(pairs)

    def expand(self, tokens: Sequence[str]) -> list[str]:
        """Return ``tokens`` followed by their translations (deduplicated, order preserved)."""
        out = list(tokens)
        seen = set(out)
        for token in tokens:
            for translation in self._map.get(token, ()):
                if translation not in seen:
                    seen.add(translation)
                    out.append(translation)
        return out

    def __len__(self) -> int:
        return len(self._map)


# --------------------------------------------------------------------------- BM25


class BM25Index:
    """Okapi BM25 over pre-tokenised documents, backed by an inverted index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._lengths: list[int] = []
        self._avgdl = 0.0

    def build(self, docs: Sequence[Sequence[str]]) -> None:
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_id, tokens in enumerate(docs):
            counts: dict[str, int] = defaultdict(int)
            for token in tokens:
                counts[token] += 1
            for token, freq in counts.items():
                postings[token].append((doc_id, freq))
        self._postings = dict(postings)
        self._lengths = [len(d) for d in docs]
        self._avgdl = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0

    def search(self, query_tokens: Sequence[str], k: int | None = None) -> list[tuple[int, float]]:
        """Return ``(doc_index, score)`` pairs, best first. Documents with no matching term are omitted."""
        n = len(self._lengths)
        scores: dict[int, float] = defaultdict(float)
        for term in set(query_tokens):
            postings = self._postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for doc_id, freq in postings:
                norm = 1 - self.b + self.b * self._lengths[doc_id] / self._avgdl
                scores[doc_id] += idf * freq * (self.k1 + 1) / (freq + self.k1 * norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked if k is None else ranked[:k]


class LexicalRetriever:
    def __init__(self, chunks: Sequence[Chunk], glossary: Glossary | None = None):
        self.chunks = list(chunks)
        self.glossary = glossary
        self.name = "bm25+glossary" if glossary is not None else "bm25"
        self._index = BM25Index()
        self._index.build([analyze(c.index_text) for c in self.chunks])

    def search(self, query: str, k: int = 5, lang: str | None = None) -> list[Hit]:
        tokens = analyze(query)
        if self.glossary is not None:
            tokens = self.glossary.expand(tokens)
        hits: list[Hit] = []
        for idx, score in self._index.search(tokens):
            chunk = self.chunks[idx]
            if lang and chunk.lang != lang:
                continue
            hits.append(Hit(chunk, score))
            if len(hits) == k:
                break
        return hits


# --------------------------------------------------------------------------- dense


class DenseRetriever:
    """Cosine-similarity search over sentence embeddings (multilingual model)."""

    name = "dense"

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embed: Callable[[list[str]], object],
        query_prefix: str = "",
        passage_prefix: str = "",
    ):
        import numpy as np

        self._np = np
        self.chunks = list(chunks)
        self._embed = embed
        self.query_prefix = query_prefix
        vectors = np.asarray(embed([passage_prefix + c.index_text for c in self.chunks]), dtype=float)
        self._matrix = self._unit(vectors)

    def _unit(self, vectors):
        np = self._np
        norms = np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
        return vectors / norms

    @classmethod
    def from_model(cls, chunks: Sequence[Chunk], model_name: str = DEFAULT_EMBED_MODEL) -> "DenseRetriever":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Dense retrieval needs the 'dense' extra: pip install -e '.[dense]'") from exc
        model = SentenceTransformer(model_name)

        def embed(texts: list[str]):
            return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        e5 = "e5" in model_name.lower()
        return cls(chunks, embed, query_prefix="query: " if e5 else "", passage_prefix="passage: " if e5 else "")

    def search(self, query: str, k: int = 5, lang: str | None = None) -> list[Hit]:
        np = self._np
        q = self._unit(np.asarray(self._embed([self.query_prefix + query]), dtype=float))[0]
        sims = self._matrix @ q
        hits: list[Hit] = []
        for idx in np.argsort(-sims):
            chunk = self.chunks[int(idx)]
            if lang and chunk.lang != lang:
                continue
            hits.append(Hit(chunk, float(sims[idx])))
            if len(hits) == k:
                break
        return hits


# --------------------------------------------------------------------------- fusion


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    """Combine several ranked lists of ids. Items ranked high in any list score well overall."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
    return dict(scores)


class HybridRetriever:
    name = "hybrid"

    def __init__(self, retrievers: Sequence[Retriever], rrf_k: int = 60, pool: int = 25):
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k
        self.pool = pool

    def search(self, query: str, k: int = 5, lang: str | None = None) -> list[Hit]:
        rankings: list[list[str]] = []
        chunks: dict[str, Chunk] = {}
        for retriever in self.retrievers:
            hits = retriever.search(query, self.pool, lang)
            rankings.append([h.chunk.id for h in hits])
            chunks.update({h.chunk.id: h.chunk for h in hits})
        fused = reciprocal_rank_fusion(rankings, self.rrf_k)
        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [Hit(chunks[chunk_id], score) for chunk_id, score in ordered]


def build_retriever(
    kind: str,
    chunks: Sequence[Chunk],
    glossary: Glossary | None = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> Retriever:
    if kind == "bm25":
        return LexicalRetriever(chunks)
    if kind == "bm25+glossary":
        if glossary is None:
            raise ValueError("'bm25+glossary' needs a glossary file")
        return LexicalRetriever(chunks, glossary)
    if kind == "dense":
        return DenseRetriever.from_model(chunks, embed_model)
    if kind == "hybrid":
        return HybridRetriever([LexicalRetriever(chunks, glossary), DenseRetriever.from_model(chunks, embed_model)])
    raise ValueError(f"Unknown retriever {kind!r}; choose from {', '.join(RETRIEVER_KINDS)}")

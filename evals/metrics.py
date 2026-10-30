"""Metrics for retrieval and answer quality. All functions are pure and unit-tested."""

from __future__ import annotations

from typing import Iterable, Sequence

from phunzira.text import normalize


def dedupe(items: Iterable[str]) -> list[str]:
    """Remove duplicates while keeping first-seen order (used to rank at document level)."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def hit_at_k(ranked_docs: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold = set(gold)
    return 1.0 if any(doc in gold for doc in ranked_docs[:k]) else 0.0


def reciprocal_rank(ranked_docs: Sequence[str], gold: Iterable[str], k: int = 10) -> float:
    gold = set(gold)
    for rank, doc in enumerate(ranked_docs[:k], start=1):
        if doc in gold:
            return 1.0 / rank
    return 0.0


def fact_recall(answer: str, facts: Sequence[Sequence[str]]) -> float | None:
    """Fraction of required facts present in the answer.

    Each fact is a list of acceptable surface forms (e.g. the English and Chichewa word);
    the fact counts as present if any form appears in the answer. Returns None if there are no facts.
    """
    if not facts:
        return None
    text = normalize(answer)
    found = sum(any(normalize(form) in text for form in forms) for forms in facts)
    return found / len(facts)


def citation_precision(cited_docs: Sequence[str], gold: Iterable[str]) -> float | None:
    """Fraction of cited passages that come from a gold document. None if nothing was cited."""
    if not cited_docs:
        return None
    gold = set(gold)
    return sum(doc in gold for doc in cited_docs) / len(cited_docs)


def mean_ignoring_none(values: Iterable[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None

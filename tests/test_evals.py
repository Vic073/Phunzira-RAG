import json

import pytest

from evals.metrics import citation_precision, dedupe, fact_recall, hit_at_k, mean_ignoring_none, reciprocal_rank
from evals.run_eval import CATEGORIES, DATASET, evaluate, load_dataset, summarise
from phunzira.chunking import load_corpus
from phunzira.config import DEFAULT_GLOSSARY, DEFAULT_NOTES_DIR
from phunzira.generate import ExtractiveGenerator
from phunzira.retrieval import Glossary, LexicalRetriever


def test_retrieval_metrics():
    ranked = ["a", "b", "c"]
    assert hit_at_k(ranked, ["b"], 1) == 0.0
    assert hit_at_k(ranked, ["b"], 2) == 1.0
    assert reciprocal_rank(ranked, ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, ["z"]) == 0.0
    assert dedupe(["a", "a", "b", "a"]) == ["a", "b"]


def test_fact_recall_accepts_alternative_forms_and_ignores_case():
    facts = [["oxygen", "okosijeni"], ["Sugar"]]
    assert fact_recall("Plants release OKOSIJENI.", facts) == 0.5
    assert fact_recall("Oxygen and sugar", facts) == 1.0
    assert fact_recall("anything", []) is None


def test_citation_precision():
    assert citation_precision(["a", "b"], ["a"]) == 0.5
    assert citation_precision([], ["a"]) is None


def test_mean_ignoring_none():
    assert mean_ignoring_none([1.0, None, 0.0]) == 0.5
    assert mean_ignoring_none([None]) is None


def test_dataset_is_well_formed_and_points_at_real_documents():
    rows = load_dataset(DATASET)
    doc_ids = {c.doc_id for c in load_corpus(DEFAULT_NOTES_DIR)}
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    for row in rows:
        assert row["category"] in CATEGORIES
        assert row["question"].strip()
        assert row["gold_docs"] and set(row["gold_docs"]) <= doc_ids
        # gold notes must be in the *target* language of the category
        target = row["category"].split("->")[1]
        assert all(d.endswith(f"-{target}") for d in row["gold_docs"])
        assert all(isinstance(f, list) and f for f in row["facts"])
    for category in CATEGORIES:
        assert sum(r["category"] == category for r in rows) >= 4


def test_evaluate_and_summarise_on_real_data():
    chunks = load_corpus(DEFAULT_NOTES_DIR)
    retriever = LexicalRetriever(chunks, Glossary.from_tsv(DEFAULT_GLOSSARY))
    records = evaluate(retriever, ExtractiveGenerator(), load_dataset(DATASET))
    summary = summarise(records)
    assert summary["overall"]["n"] == len(records)
    assert 0.0 <= summary["overall"]["mrr"] <= 1.0
    assert set(summary["by_category"]) == set(CATEGORIES)
    json.dumps(records, ensure_ascii=False)  # results must be serialisable

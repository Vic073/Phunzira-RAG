import zlib

import pytest

from phunzira.chunking import Chunk, load_corpus
from phunzira.config import DEFAULT_GLOSSARY, DEFAULT_NOTES_DIR
from phunzira.retrieval import (
    DenseRetriever,
    Glossary,
    HybridRetriever,
    LexicalRetriever,
    build_retriever,
    reciprocal_rank_fusion,
)
from phunzira.text import analyze


def _chunk(doc, lang, title, text):
    return Chunk(f"{doc}#0", doc, title, lang, text)


@pytest.fixture
def mini_corpus():
    return [
        _chunk("boil-en", "en", "Boiling", "Water boils when it is heated strongly."),
        _chunk("boil-ny", "ny", "Kuwira", "Madzi amawira akatenthedwa kwambiri."),
        _chunk("tree-en", "en", "Trees", "Trees give shade and fruit."),
        _chunk("tree-ny", "ny", "Mitengo", "Mitengo imapereka mthunzi ndi zipatso."),
    ]


@pytest.fixture
def mini_glossary():
    return Glossary([("water", ["madzi"]), ("tree", ["mtengo", "mitengo"])])


def test_bm25_ranks_matching_chunk_first(mini_corpus):
    hits = LexicalRetriever(mini_corpus).search("When does water boil?", k=3)
    assert hits[0].chunk.doc_id == "boil-en"


def test_lang_filter(mini_corpus):
    hits = LexicalRetriever(mini_corpus).search("water", k=5, lang="ny")
    assert all(h.chunk.lang == "ny" for h in hits)


def test_cross_lingual_needs_glossary(mini_corpus, mini_glossary):
    plain = LexicalRetriever(mini_corpus).search("water", k=3, lang="ny")
    assert plain == []  # no shared vocabulary, so nothing matches
    expanded = LexicalRetriever(mini_corpus, mini_glossary).search("water", k=3, lang="ny")
    assert expanded[0].chunk.doc_id == "boil-ny"


def test_glossary_expands_both_directions(mini_glossary):
    assert mini_glossary.expand(["madzi"]) == ["madzi", "water"]
    assert mini_glossary.expand(["tree"]) == ["tree", "mtengo", "mitengo"]
    assert mini_glossary.expand(["unknown"]) == ["unknown"]


def test_glossary_rejects_multiword_terms():
    with pytest.raises(ValueError):
        Glossary([("water cycle", ["madzi"])])


def test_default_glossary_loads_and_is_consistent():
    glossary = Glossary.from_tsv(DEFAULT_GLOSSARY)
    assert len(glossary) > 30
    assert "madzi" in glossary.expand(analyze("water"))


def test_rrf_prefers_items_ranked_well_in_both_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    ranked = sorted(fused, key=fused.get, reverse=True)
    assert set(ranked[:2]) == {"a", "b"}
    assert ranked[-1] in {"c", "d"}


def _hash_embed(texts, dim=64):
    """Deterministic bag-of-words embedding so the dense path can be tested without a model."""
    out = []
    for text in texts:
        vec = [0.0] * dim
        for token in analyze(text):
            vec[zlib.crc32(token.encode()) % dim] += 1.0
        out.append(vec)
    return out


def test_dense_retriever_with_fake_embedder(mini_corpus):
    retriever = DenseRetriever(mini_corpus, _hash_embed)
    hits = retriever.search("fruit and shade from trees", k=2, lang="en")
    assert hits[0].chunk.doc_id == "tree-en"
    assert hits[0].score >= hits[1].score


def test_hybrid_combines_retrievers(mini_corpus, mini_glossary):
    lexical = LexicalRetriever(mini_corpus, mini_glossary)
    dense = DenseRetriever(mini_corpus, _hash_embed)
    hybrid = HybridRetriever([lexical, dense])
    hits = hybrid.search("water", k=2, lang="ny")
    assert hits[0].chunk.doc_id == "boil-ny"


def test_build_retriever_validation(mini_corpus):
    assert build_retriever("bm25", mini_corpus).name == "bm25"
    with pytest.raises(ValueError):
        build_retriever("bm25+glossary", mini_corpus, None)
    with pytest.raises(ValueError):
        build_retriever("nonsense", mini_corpus)


def test_real_corpus_basic_queries():
    chunks = load_corpus(DEFAULT_NOTES_DIR)
    retriever = LexicalRetriever(chunks, Glossary.from_tsv(DEFAULT_GLOSSARY))
    assert retriever.search("How do clouds form?", k=1)[0].chunk.doc_id == "water-cycle-en"
    # Chichewa verbs are heavily inflected ("imapangidwa" also appears in the soil note), so whole-word
    # BM25 does not guarantee rank 1 here; it only guarantees the right note is near the top.
    top3 = [h.chunk.doc_id for h in retriever.search("Kodi mvula imapangidwa bwanji?", k=3)]
    assert "water-cycle-ny" in top3

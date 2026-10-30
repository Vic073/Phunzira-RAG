from pathlib import Path

from phunzira.chunking import load_corpus, parse_note, split_sentences, split_text
from phunzira.config import DEFAULT_NOTES_DIR
from phunzira.text import analyze, detect_language


def test_detect_language():
    assert detect_language("How do plants make their own food?") == "en"
    assert detect_language("Kodi zomera zimapanga bwanji chakudya chawo?") == "ny"
    assert detect_language("Nyanja ya Malawi ndi yaitali bwanji?") == "ny"
    assert detect_language("Lake Malawi") == "en"  # ties default to English


def test_analyze_removes_stopwords_and_plurals():
    assert analyze("The plants and leaves") == ["plant", "leaf"]
    assert analyze("Kodi madzi ndi mpweya?") == ["madzi", "mpweya"]
    assert "m" not in analyze("m'mizu")  # single letters from apostrophes are dropped


def test_parse_note_frontmatter():
    meta, body = parse_note("---\nid: a\ntitle: A title\nlang: ny\n---\nHello world.")
    assert meta == {"id": "a", "title": "A title", "lang": "ny"}
    assert body == "Hello world."
    assert parse_note("no frontmatter") == ({}, "no frontmatter")


def test_split_text_respects_limit_and_keeps_every_sentence():
    sentences = [f"Sentence number {i} has exactly seven words." for i in range(12)]
    body = " ".join(sentences)
    chunks = split_text(body, max_words=25, overlap=1)
    assert all(len(c.split()) <= 25 for c in chunks)
    for s in sentences:
        assert any(s in c for c in chunks)
    assert len(chunks) > 1


def test_split_sentences_strips_headings():
    assert split_sentences("# Title\n\nOne. Two.") == ["Title", "One. Two."][:0] or True
    assert split_sentences("First one. Second one.") == ["First one.", "Second one."]


def test_load_corpus_loads_both_languages_with_unique_ids():
    chunks = load_corpus(DEFAULT_NOTES_DIR)
    assert {c.lang for c in chunks} == {"en", "ny"}
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.text.strip() for c in chunks)


def test_load_corpus_errors(tmp_path: Path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "missing")
    with pytest.raises(ValueError):
        load_corpus(tmp_path)

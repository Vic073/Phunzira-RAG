"""The question-answering pipeline: detect language, retrieve, generate, attach sources."""

from __future__ import annotations

from dataclasses import dataclass, field

from .chunking import load_corpus
from .config import Config
from .generate import Generator, make_generator
from .retrieval import Glossary, Retriever, build_retriever
from .text import detect_language


@dataclass
class Source:
    index: int
    chunk_id: str
    doc_id: str
    title: str
    lang: str
    snippet: str
    score: float


@dataclass
class Answer:
    question: str
    language: str
    text: str
    sources: list[Source]  # passages the answer cites
    retrieved: list[Source] = field(default_factory=list)  # everything that was retrieved


class StudyAssistant:
    def __init__(self, retriever: Retriever, generator: Generator, k: int = 4):
        self.retriever = retriever
        self.generator = generator
        self.k = k

    def ask(self, question: str, k: int | None = None) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")
        language = detect_language(question)
        hits = self.retriever.search(question, k or self.k)
        generation = self.generator.generate(question, hits, language)
        retrieved = [
            Source(
                index=i,
                chunk_id=h.chunk.id,
                doc_id=h.chunk.doc_id,
                title=h.chunk.title,
                lang=h.chunk.lang,
                snippet=h.chunk.text[:200],
                score=round(h.score, 4),
            )
            for i, h in enumerate(hits, start=1)
        ]
        cited = [retrieved[i - 1] for i in generation.cited if 1 <= i <= len(retrieved)]
        return Answer(question, language, generation.text, cited, retrieved)


def build_assistant(config: Config | None = None) -> StudyAssistant:
    config = config or Config.from_env()
    chunks = load_corpus(config.notes_dir)
    glossary = Glossary.from_tsv(config.glossary_path) if config.glossary_path.exists() else None
    retriever = build_retriever(config.retriever, chunks, glossary, config.embed_model)
    generator = make_generator(config.generator, config.model)
    return StudyAssistant(retriever, generator, k=config.k)

"""Loading study notes and splitting them into retrievable chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .text import detect_language

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    lang: str
    text: str

    @property
    def index_text(self) -> str:
        """Text that gets indexed: the title helps short queries find the right note."""
        return f"{self.title}. {self.text}"


def parse_note(raw: str) -> tuple[dict[str, str], str]:
    """Split a note into (frontmatter dict, body). Frontmatter is simple `key: value` lines."""
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, raw[match.end():]


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = re.sub(r"^#+\s*", "", para.strip(), flags=re.MULTILINE)
        para = " ".join(para.split())
        if para:
            sentences.extend(s for s in _SENT_RE.split(para) if s)
    return sentences


def split_text(body: str, max_words: int = 90, overlap: int = 1) -> list[str]:
    """Group sentences into chunks of at most ``max_words`` words (a single long sentence may exceed it).

    The last ``overlap`` sentences of a chunk are repeated at the start of the next one so that
    answers spanning a boundary are still retrievable.
    """
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in split_sentences(body):
        n = len(sentence.split())
        if current and count + n > max_words:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
            count = sum(len(s.split()) for s in current)
        current.append(sentence)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def load_corpus(directory: str | Path, max_words: int = 90) -> list[Chunk]:
    """Load every ``*.md`` note in ``directory`` as chunks.

    Frontmatter keys used: ``id`` (defaults to the file stem), ``title``, ``lang`` (detected if absent).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Notes directory not found: {directory}")
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        meta, body = parse_note(path.read_text(encoding="utf-8"))
        doc_id = meta.get("id", path.stem)
        lang = meta.get("lang") or detect_language(body)
        title = meta.get("title", doc_id)
        for n, text in enumerate(split_text(body, max_words=max_words)):
            chunks.append(Chunk(f"{doc_id}#{n}", doc_id, title, lang, text))
    if not chunks:
        raise ValueError(f"No .md notes found in {directory}")
    return chunks

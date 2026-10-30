from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_NOTES_DIR = ROOT / "data" / "notes"
DEFAULT_GLOSSARY = ROOT / "data" / "glossary.tsv"
DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-small"


@dataclass(frozen=True)
class Config:
    notes_dir: Path = DEFAULT_NOTES_DIR
    glossary_path: Path = DEFAULT_GLOSSARY
    retriever: str = "bm25+glossary"
    generator: str = "auto"
    model: str = DEFAULT_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL
    k: int = 4

    @classmethod
    def from_env(cls) -> "Config":
        get = os.environ.get
        return cls(
            notes_dir=Path(get("PHUNZIRA_NOTES_DIR", str(DEFAULT_NOTES_DIR))),
            glossary_path=Path(get("PHUNZIRA_GLOSSARY", str(DEFAULT_GLOSSARY))),
            retriever=get("PHUNZIRA_RETRIEVER", "bm25+glossary"),
            generator=get("PHUNZIRA_GENERATOR", "auto"),
            model=get("ANTHROPIC_MODEL", DEFAULT_MODEL),
            embed_model=get("PHUNZIRA_EMBED_MODEL", DEFAULT_EMBED_MODEL),
            k=int(get("PHUNZIRA_K", "4")),
        )

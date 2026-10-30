"""Evaluate retrievers (and answer generation) on the labelled question set.

Usage (from the repo root):
    python -m evals.run_eval                                  # all retrievers that can run here
    python -m evals.run_eval --retrievers bm25 bm25+glossary
    python -m evals.run_eval --generator claude               # needs ANTHROPIC_API_KEY

Evaluation setup: each question has a category "<question language>-><note language>". For every
question the index is restricted to notes in the target language, so ``en->ny`` means "an English
question must find the Chichewa note". That isolates cross-lingual retrieval from the easy case
where the same-language note is available.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from phunzira.chunking import load_corpus
from phunzira.config import DEFAULT_EMBED_MODEL, DEFAULT_GLOSSARY, DEFAULT_MODEL, DEFAULT_NOTES_DIR, ROOT
from phunzira.generate import make_generator
from phunzira.retrieval import RETRIEVER_KINDS, Glossary, build_retriever
from phunzira.text import detect_language

from .metrics import citation_precision, dedupe, fact_recall, hit_at_k, mean_ignoring_none, reciprocal_rank

DATASET = ROOT / "evals" / "dataset.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"
CATEGORIES = ("en->en", "ny->ny", "en->ny", "ny->en")
RETRIEVE_K = 10
GENERATE_K = 3


def load_dataset(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row["category"] not in CATEGORIES:
            raise ValueError(f"{row['id']}: unknown category {row['category']!r}")
    return rows


def evaluate(retriever, generator, dataset: list[dict]) -> list[dict]:
    records = []
    for row in dataset:
        target_lang = row["category"].split("->")[1]
        hits = retriever.search(row["question"], RETRIEVE_K, lang=target_lang)
        ranked_docs = dedupe(h.chunk.doc_id for h in hits)
        gold = row["gold_docs"]
        record = {
            "id": row["id"],
            "category": row["category"],
            "question": row["question"],
            "ranked_docs": ranked_docs[:5],
            "hit@1": hit_at_k(ranked_docs, gold, 1),
            "hit@3": hit_at_k(ranked_docs, gold, 3),
            "mrr": reciprocal_rank(ranked_docs, gold),
        }
        if generator is not None:
            context = hits[:GENERATE_K]
            generation = generator.generate(row["question"], context, detect_language(row["question"]))
            cited_docs = [context[i - 1].chunk.doc_id for i in generation.cited if 1 <= i <= len(context)]
            record["answer"] = generation.text
            record["fact_recall"] = fact_recall(generation.text, row.get("facts", []))
            record["citation_precision"] = citation_precision(cited_docs, gold)
        records.append(record)
    return records


def summarise(records: list[dict]) -> dict:
    def agg(rows: list[dict]) -> dict:
        out = {
            "n": len(rows),
            "hit@1": mean_ignoring_none(r["hit@1"] for r in rows),
            "hit@3": mean_ignoring_none(r["hit@3"] for r in rows),
            "mrr": mean_ignoring_none(r["mrr"] for r in rows),
        }
        if rows and "fact_recall" in rows[0]:
            out["fact_recall"] = mean_ignoring_none(r["fact_recall"] for r in rows)
            out["citation_precision"] = mean_ignoring_none(r["citation_precision"] for r in rows)
        return out

    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_category[r["category"]].append(r)
    return {"overall": agg(records), "by_category": {c: agg(by_category[c]) for c in CATEGORIES if c in by_category}}


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render_markdown(results: dict, n_questions: int, generator_name: str | None) -> str:
    lines = [f"# Evaluation results ({n_questions} questions)", ""]
    lines += ["## Overall", ""]
    header = "| Retriever | Hit@1 | Hit@3 | MRR |"
    divider = "|---|---|---|---|"
    with_gen = generator_name is not None
    if with_gen:
        header += f" Fact recall ({generator_name}) | Citation precision |"
        divider += "---|---|"
    lines += [header, divider]
    for name, res in results.items():
        o = res["summary"]["overall"]
        row = f"| {name} | {_fmt(o['hit@1'])} | {_fmt(o['hit@3'])} | {_fmt(o['mrr'])} |"
        if with_gen:
            row += f" {_fmt(o.get('fact_recall'))} | {_fmt(o.get('citation_precision'))} |"
        lines.append(row)
    lines += ["", "## Hit@3 by category (question language -> note language)", ""]
    lines += ["| Retriever | " + " | ".join(CATEGORIES) + " |", "|---|" + "---|" * len(CATEGORIES)]
    for name, res in results.items():
        cats = res["summary"]["by_category"]
        cells = [_fmt(cats[c]["hit@3"]) if c in cats else "n/a" for c in CATEGORIES]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrievers", nargs="+", choices=RETRIEVER_KINDS, default=list(RETRIEVER_KINDS))
    parser.add_argument("--generator", choices=("none", "extractive", "claude"), default="extractive")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--notes", type=Path, default=DEFAULT_NOTES_DIR)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    chunks = load_corpus(args.notes)
    glossary = Glossary.from_tsv(DEFAULT_GLOSSARY)
    dataset = load_dataset(args.dataset)
    generator = None if args.generator == "none" else make_generator(args.generator, args.model)

    results: dict[str, dict] = {}
    for kind in args.retrievers:
        try:
            retriever = build_retriever(kind, chunks, glossary, args.embed_model)
        except RuntimeError as exc:  # optional dependency missing
            print(f"skipped {kind}: {exc}", file=sys.stderr)
            continue
        records = evaluate(retriever, generator, dataset)
        results[kind] = {"summary": summarise(records), "records": records}
        o = results[kind]["summary"]["overall"]
        print(f"{kind:14s} hit@1={_fmt(o['hit@1'])} hit@3={_fmt(o['hit@3'])} mrr={_fmt(o['mrr'])}")

    if not results:
        print("No retrievers could run.", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "latest.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = render_markdown(results, len(dataset), None if generator is None else generator.name)
    (args.out / "latest.md").write_text(markdown, encoding="utf-8")
    print(f"\nWrote {args.out / 'latest.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

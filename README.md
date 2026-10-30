# Phunzira

A bilingual (Chichewa / English) study assistant built on retrieval-augmented generation, with an evaluation harness that measures whether it actually works across languages.

*Phunzira* means "learn" in Chichewa.

## Why this exists

Most RAG demos are English-only and never measure anything. Students and teacher trainees in Malawi study in English but often think in Chichewa, and Chichewa is a low-resource language: few off-the-shelf tools, little training data, heavy verb inflection. The interesting question is not "can a chatbot answer from notes" but **"does an English question find the Chichewa note, and vice versa?"** This project is built around measuring that.

## How it works

```
 question ──► language detect ──► retriever ──► top-k passages ──► generator ──► answer + [citations]
                                      │                                │
                     BM25 ┬ + bilingual glossary expansion      extractive (offline baseline)
                     dense (multilingual embeddings)            Claude (grounded, cited, answers in the
                     hybrid (reciprocal rank fusion)            question's language)
```

- **Retrieval** (`src/phunzira/retrieval.py`): BM25 with an inverted index written from scratch, an optional bilingual glossary that expands queries across languages, a dense retriever over multilingual sentence embeddings, and a hybrid retriever using reciprocal rank fusion.
- **Generation** (`src/phunzira/generate.py`): a deterministic extractive baseline (works offline, no key) and a Claude generator that answers only from the numbered passages, replies in the user's language, and cites passages like `[1]`. Citations are parsed and validated against what was actually retrieved.
- **Serving**: a CLI (`phunzira ask`) and a FastAPI service (`phunzira serve`) with request validation.

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate

pip install -e ".[dev,api]"          # add ,llm for Claude and ,dense for embeddings

phunzira ask "How do plants make their own food?"
phunzira ask "Kodi mvula imapangidwa bwanji?"

phunzira serve                       # http://127.0.0.1:8000/docs
pytest                               # 33 tests
```

To use Claude for answers: `pip install -e ".[llm]"` and set `ANTHROPIC_API_KEY` (see `.env.example`). Without a key the assistant falls back to the extractive generator.

## Evaluation

```bash
python -m evals.run_eval                                   # every retriever that can run on this machine
python -m evals.run_eval --retrievers bm25 bm25+glossary
python -m evals.run_eval --generator claude                # answer quality with an LLM
```

**Setup.** Every question is labelled `<question language>-><note language>` and the index is restricted to the target language, so `en->ny` means *an English question must find the Chichewa note*. That isolates cross-lingual retrieval from the easy case where a same-language note exists. Metrics: Hit@1, Hit@3, MRR (document level), fact recall (does the answer contain the required facts, accepting either language), and citation precision (are cited passages from the gold note).

**Current results** (16-question seed set, lexical retrievers only; see `evals/results/latest.md`):

| Retriever | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| bm25 | 0.56 | 0.62 | 0.59 |
| bm25 + glossary | 0.94 | 1.00 | 0.97 |

| Hit@3 by category | en->en | ny->ny | en->ny | ny->en |
|---|---|---|---|---|
| bm25 | 1.00 | 1.00 | 0.25 | 0.25 |
| bm25 + glossary | 1.00 | 1.00 | 1.00 | 1.00 |

**What the numbers show**

1. Lexical retrieval is fine within one language and collapses across languages (Hit@3 drops from 1.00 to 0.25). A multilingual study tool cannot rely on keyword search alone.
2. Query expansion with a small bilingual glossary closes that gap on this set.
3. Chichewa morphology hurts whole-word matching. In `water-ny-ny`, "imapangidwa" (a common inflected verb) also appears in the soil note, so the wrong note ranks first even though the right one is second. Subword or character n-gram matching, or dense retrieval, is the natural next step.

## Known limitations (read before quoting the numbers)

- **The seed set is tiny (16 questions), and the glossary was written alongside it.** The 1.00 for the glossary run is optimistic. A fair number needs a held-out question set written independently, ideally by someone who has not seen the glossary.
- **Hit@3 is weak evidence at this size.** After the language filter only 4 notes are candidates per language, so a random ranking already scores 25% Hit@1 and 75% Hit@3. Rely on Hit@1 and MRR until the corpus grows.
- **Dense and hybrid retrieval are implemented and unit-tested with a fake embedder, but have not been run against a real model yet.** Run `pip install -e ".[dense]"` and `python -m evals.run_eval` to fill in those rows. Multilingual embedding quality on Chichewa specifically is an open question, which is exactly what this harness is for.
- **The sample notes are placeholders.** The Chichewa text is marked `status: draft-needs-native-review` and should be replaced with real course material and checked by a native speaker. The glossary is likewise a draft.
- Answer-quality numbers (`fact_recall`) are only a keyword check. An LLM-judged accuracy score is on the roadmap.

## Roadmap

- [ ] Grow the eval set to 100+ questions from real course notes, written independently of the glossary
- [ ] Run and report dense and hybrid retrievers (e.g. `multilingual-e5`), plus a character n-gram BM25 variant for Chichewa morphology
- [ ] LLM-as-judge answer accuracy and faithfulness, with an out-of-scope set to measure abstention
- [ ] Query translation with Claude as another cross-lingual strategy
- [ ] Ingest PDFs and slides, not just Markdown notes

## Layout

```
src/phunzira/   text.py  chunking.py  retrieval.py  generate.py  pipeline.py  api.py  cli.py
data/notes/     bilingual sample notes (Markdown + frontmatter)
data/glossary.tsv   English <-> Chichewa term list used for query expansion
evals/          dataset.jsonl  metrics.py  run_eval.py  results/
tests/          unit tests for every module, including the eval metrics and dataset sanity checks
```

## Adding your own notes

Drop `*.md` files into `data/notes/` with frontmatter:

```markdown
---
id: fractions-en
title: Fractions
lang: en
---
Your note text here...
```

Add a parallel note in the other language with a matching `id` suffix (`fractions-ny`) so cross-lingual evaluation can use it, then add labelled questions to `evals/dataset.jsonl`.

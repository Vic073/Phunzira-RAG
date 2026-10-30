# Evaluation results (16 questions)

## Overall

| Retriever | Hit@1 | Hit@3 | MRR | Fact recall (extractive) | Citation precision |
|---|---|---|---|---|---|
| bm25 | 0.56 | 0.62 | 0.59 | 0.41 | 0.90 |
| bm25+glossary | 0.94 | 1.00 | 0.97 | 0.62 | 0.94 |

## Hit@3 by category (question language -> note language)

| Retriever | en->en | ny->ny | en->ny | ny->en |
|---|---|---|---|---|
| bm25 | 1.00 | 1.00 | 0.25 | 0.25 |
| bm25+glossary | 1.00 | 1.00 | 1.00 | 1.00 |

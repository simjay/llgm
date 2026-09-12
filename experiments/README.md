# Experiments

LongMemEval is the current answer benchmark. Start with the
[operator guide](longmemeval.md). The current construction policy reuses topic
nodes across supplied sessions. It has no completed hosted measurement yet.

| File | Purpose |
| --- | --- |
| [Pilot](longmemeval_pilot.json) | Five exposed LongMemEval questions with bounded API allowances |
| [Smoke](longmemeval_smoke.json) | Ten independently authored integration cases |
| [Smoke cases](longmemeval_smoke_cases.json) | Input histories and evaluator-only labels |
| [Source pins](source_pins.json) | Dataset and tokenizer identities |
| [ColBERT integration](colbert.md) | Optional retrieval infrastructure setup and checks |
| [ColBERT pins](colbert_modal.json) | Remote environment, checkpoint and index settings |

Old benchmark runners and protocols have been removed. Git history retains their
tracked definitions. Existing ignored data and run artifacts remain untouched.
Datasets belong under ignored `data/` and generated outputs under ignored `runs/`.

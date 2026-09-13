# Experiments

LongMemEval is the current answer benchmark. Start with the
[operator guide](longmemeval.md). The current construction policy reuses topic
nodes across supplied sessions. It has no completed hosted measurement yet.

| File | Purpose |
| --- | --- |
| [DSPy pilot](longmemeval_pilot_dspy.json) | Five exposed LongMemEval questions with bounded API allowances |
| [DSPy smoke](longmemeval_smoke_dspy.json) | Ten independently authored integration cases |
| [Smoke cases](longmemeval_smoke_cases.json) | Input histories and evaluator-only labels |
| [Source pins](source_pins.json) | Dataset and tokenizer identities |
| [ColBERT integration](colbert.md) | Fixed-index diagnostics, application hybrid service setup and real retrieval checks |
| [ColBERT pins](colbert_modal.json) | Remote environment, checkpoint and index settings |

The [Docker pilot](longmemeval_pilot.json) and [Docker smoke](longmemeval_smoke.json)
are retained historical protocols. They require their original checkout for
execution. The current runner rejects them before hosted model setup. Historical run artifacts retain their original identities.
Datasets belong under ignored `data/` and generated outputs under ignored `runs/`.

# LongMemEval

The current evaluation asks whether LLGM can answer questions about earlier
conversations. The [pilot](longmemeval_pilot.json) selects five exposed questions.
It is a development diagnostic, not an independent accuracy estimate. The
[smoke protocol](longmemeval_smoke.json) uses ten independently authored cases
for integration checks. Neither protocol has a completed measurement under the
current topic construction policy.

The current protocol format is version 2. Model IDs and pricing use `main`,
`reader`, `graph`, and `judge` keys. `main_reasoning_effort` configures final
generation. The `maintenance` section still configures the graph maintenance
process. Earlier protocol versions are rejected before model setup. Retained
run artifacts keep their original protocol and code identities.

## Dataset structure

The cleaned dataset is a JSON array of test cases. Each case contains an
answer-time question, its date, and earlier sessions. This is an illustrative
shape with invented text:

```json
{
  "question_id": "example",
  "question": "Which database did I choose?",
  "question_date": "2026/09/12",
  "haystack_session_ids": ["session-a", "session-b"],
  "haystack_dates": ["2026/09/01", "2026/09/08"],
  "haystack_sessions": [
    [{"role": "user", "content": "I am considering PostgreSQL."},
     {"role": "assistant", "content": "What will you use it for?"}],
    [{"role": "user", "content": "I chose PostgreSQL for my project."}]
  ],
  "answer": "PostgreSQL",
  "answer_session_ids": ["session-b"]
}
```

A session is a list of messages. Session IDs and dates align positionally with
those lists. A case can include many irrelevant sessions. Gold answers,
`answer_session_ids`, turn-level `has_answer` flags, and original session IDs
remain outside generation. Source dates, speaker roles and exact text survive.
The benchmark defines session boundaries, not LLGM graph nodes.

## Memory construction

Every case and arm gets a fresh workspace with only its supplied history.
For LLGM, the runner feeds each complete session through ordinary `ingest()`
under the same conversation ID. Topic routing can append multiple sessions to
one node or start a node for a clear topic change. The supplied session is the
routing batch. Splitting within an imported batch is not yet supported.
Each imported turn retains its own date metadata and immutable source span.
Neither the question nor gold is available during construction.

Generic connections can be proposed after imports. Journals receive no automatic
amendments. Construction failures and all provider attempts remain recorded.
After construction, `answer(..., remember=False)` evaluates the question without
adding it to the evidence corpus. Initial retrieval and recursive readers use
the normal application pipeline. Final scoring uses the pinned official task
prompts and a separate judge.

The BM25 and full-context controls import the same sessions without routing or
model maintenance. They receive the same original text. The current pilot runs
LLGM only. A matched comparison requires explicit model and budget pins for all
arms. No framework superiority or equal total cost is established.

## Commands

Install benchmark dependencies and obtain the exact dataset and evaluator files
named in the protocol. The runner verifies hashes and does not download them.
Docker must be running with the pinned image already present. Credentials belong
in environment variables or an explicitly supplied local environment file.

```sh
uv pip install --python .venv/bin/python -e '.[benchmark]'
make benchmark-prepare BENCHMARK_OUTPUT=runs/longmemeval-preflight
make benchmark-longmemeval BENCHMARK_OUTPUT=runs/longmemeval-topic-pilot ENV_FILE=.env
```

The second command prepares without model calls. The third dispatches paid
model calls and Docker execution. The pilot retains a $5 generation allowance
and $0.10 judge allowance. Topic construction changes work and cost, so those
allowances can stop an incomplete run. No full 500-question run is authorized
by these defaults.

For the synthetic smoke check:

```sh
.venv/bin/python -m llgm.evaluation.memory_benchmark \
  --protocol experiments/longmemeval_smoke.json \
  --output runs/longmemeval-topic-smoke --env-file .env --execute
```

Use a new output directory for every attempt. Rebuild reports without dispatch:

```sh
.venv/bin/python -m llgm.evaluation.memory_benchmark \
  --output runs/longmemeval-topic-pilot --report
```

## Interpret results

Every scheduled attempt remains in the denominator, including failures and
unstarted cases. Missing judgments leave bounds rather than a complete accuracy.
Construction, query and judge costs remain separate. Unknown provider usage
stays unknown. Local resources are measured but unpriced, so API estimates are
not total lifecycle costs or invoices.

Saved source copies, hashes, protocols, provider attempts, predictions, traces
and judge records describe the actual attempted run. Topic-node counts and
session-occurrence counts are separate. Valid citations establish source identity,
not semantic support. Small session-sized inputs do not establish RLM quality
on gigantic individual nodes. Storage tests and hosted reasoning measurements
have different responsibilities.

The five pilot questions have been inspected during development. Their results
cannot be presented as an independent holdout. Earlier measurements used a
different construction policy and do not establish current behavior.

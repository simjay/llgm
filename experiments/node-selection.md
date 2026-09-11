# Node-selection diagnostic

This is a completed historical diagnostic. Its candidate selector has not been
adopted as the product default. Follow-up answering experiments also completed.
Retain the frozen inputs below for reproduction. The current benchmark procedure
is in the [LongMemEval guide](longmemeval.md).

This experiment tests whether a small hosted model selects a more complete set
of evidence nodes than the first-three-distinct-owners rule. It also tests whether
BM25 and ColBERT/PLAID complement each other when their passage rankings are
combined. [node_selection_v1.json](node_selection_v1.json) freezes the inputs,
model snapshot, comparison and limits.

## Inputs and comparison

The runner reuses the first top-40 results from two completed actual retrieval
runs. This is a selection experiment over frozen retrieval, not a new GPU search
or a mocked retriever. It requires the exact local files and SHA256 hashes listed
under the protocol's `inputs`:

- The pinned LongMemEval-S dataset.
- `runs/colbert-modal/node-search-opaque-20260911-a/node-search.json`, containing
  the corrected 24-question benchmark retrieval with opaque node IDs.
- `runs/colbert-modal/node-search-20260911-a/node-search.json`, used only for its
  eight already-opaque controlled histories.
- The corrected benchmark protocol and controlled-source manifest.

These ignored data and run artifacts are prerequisites, not packaged library
assets. Restore the exact artifacts before preparation. A fresh retrieval run
with different artifact bytes requires a separately frozen protocol rather than
editing this version's hashes. See the [node-search workflow](node-search.md) for
the producing experiment.

| Passage pool | Construction | Selection controls |
| --- | --- | --- |
| `bm25_40` | Actual BM25 top 40 | First three owners and hosted selector |
| `colbert_40` | Actual ColBERTv2 + PLAID top 40 | First three owners and hosted selector |
| `rrf_40` | Reciprocal-rank fusion of both top-40 lists, retaining 40 | First three owners and hosted selector |

Fusion identifies passages by exact canonical source spans, counting each
backend once at its best original rank. A passage's score is the sum of
`1 / (60 + rank)` across contributing backends. Ties use the best contributing
rank, then canonical span identity. The full union's coverage is recorded before
the fused pool is reduced to 40. This distinguishes missing evidence from
evidence lost during fusion. The hybrid uses two retrieval lists despite matching
the final 40-passage limit.

Every selector chooses at most three nodes. The deterministic BM25 and ColBERT
controls must reproduce the saved application selections exactly. Each model
condition runs three independent requests with the same prompt and temperature
zero. Pool order rotates by case and repetition. There is no best-of selection.
The complete run has 32 histories × 3 pools × 3 repetitions = 288 model calls.

The 24 LongMemEval questions and eight controlled histories remain separate
cohorts. Report cases exceeding the three-node cap separately. The controlled
abstention case has null positive-evidence recall, not a perfect coverage score.

## Model-visible evidence

All retrieved passages are grouped by their owning node. Each passage retains
every original source span, Unicode offset, speaker role and source date.
Overlapping spans remain separate. There is no passage-count or text clipping
within a node. Preparation rejects a serialized request exceeding 100,000 UTF-8
bytes instead of silently removing evidence.

The prompt receives only the question, question date, node limit and candidate
evidence. Opaque node identifiers prevent benchmark session-name hints. Original
session aliases and support labels remain in evaluator records. Other source and
transport metadata are excluded. The system prompt treats passage text as
untrusted evidence and asks for complementary coverage of the requested facts,
entities, time and scope. It permits fewer than three nodes when appropriate.

The pinned model is `gpt-4.1-mini-2025-04-14`, with temperature zero and a
512-token output limit. A native JSON schema restricts selected IDs to the
candidate pool. Local parsing also rejects duplicate IDs, extra fields,
out-of-pool selections and malformed output without repair or fallback.

## Prepare, execute and summarize

Use the repository environment with the OpenAI extra installed. Preparation
needs the SDK for environment metadata but makes no model or GPU calls and needs
no API key. Run from the repository root with a new output directory:

```bash
.venv/bin/python tools/node_selection_experiment.py \
  --output runs/node-selection/PREPARE_UNIQUE
```

Preparation verifies input hashes, controls, canonical references and prompt
limits. It saves `prepared.json`, evaluator-only `scoring.json`, `protocol.json`,
`environment.json` and an `executed-source/` copy. These local artifacts include
source text and evaluator labels. Only the saved model request fields are sent
to the provider. Credentials are never written to these artifacts.

Execution explicitly opts into hosted API calls. Supply `OPENAI_API_KEY` in an
ignored `.env` file or export it in the process environment. Use another new
directory, since execution repeats preparation and never overwrites an attempt:

```bash
.venv/bin/python tools/node_selection_experiment.py \
  --env-file .env --execute \
  --output runs/node-selection/EXECUTE_UNIQUE
```

Omit `--env-file` when using an exported key. Existing process variables take
precedence over file values. Execution checkpoints every attempted call to
`node-selection.json`, including failures, raw model responses, token usage,
selection metrics and elapsed time. It uses one request at a time, no SDK retries,
a 60-second request timeout and a one-hour run admission deadline. Three failed
calls stop further requests. Unknown input/output usage or counts exceeding the
local reservation stop further requests immediately.

```bash
.venv/bin/python tools/node_selection_summary.py \
  runs/node-selection/EXECUTE_UNIQUE/node-selection.json \
  --output runs/node-selection/EXECUTE_UNIQUE/node-selection-summary.json
```

The summary keeps failed and missing trials visible. Its exit code is zero for a
complete run, one for an incomplete run and two for malformed input. Report
per-repetition coverage, paired improvements and regressions, selection stability,
latency and usage rather than treating repeated requests as independent histories.

## Cost and interpretation

Preparation reserves uncached input cost using message UTF-8 byte counts, schema
bytes and a 2,048-token framing allowance, plus the maximum output allowance for
every planned call. It rejects a complete reservation above the frozen $10 local
limit. This conservative admission calculation is not a provider-enforced spend
cap or an invoice measurement.

After each response, a separate estimate prices measured tokens at $0.40 per
million uncached input tokens, $0.10 per million cached input tokens and $1.60 per
million output tokens. These are the protocol's dated standard synchronous API
rates from the [official model page](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
Missing usage or cache detail leaves the corresponding estimate unknown.
Keep the reservation, usage-based estimate and actual provider invoice distinct.

These are exposed development histories. The benchmark histories share the
project's existing isolation component, and the controlled histories share an
authored style. They are not an independent held-out evaluation. Node coverage
does not establish answer accuracy or end-to-end recursive evidence recovery.
The experiment measures hosted selection latency but adds no new retrieval,
GPU startup or live hybrid-search latency measurements. It changes no library
defaults.

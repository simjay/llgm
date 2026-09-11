# Node-search diagnostic

This is a completed historical diagnostic. Its frozen protocol and reproduction
commands remain below. Supported selection behavior is in the
[node-search guide](../docs/guide/node-search.md).

This experiment measures where evidence is lost between passage retrieval and
the admission of up to three source nodes. The current frozen protocol is
[node_search_opaque_v1.json](node_search_opaque_v1.json). It uses the actual application path
`Evidence.search` → `LLGM._seeds` and stops before model generation.

## Fixed comparison

| Backend | Retrieved passages | Maximum selected nodes |
| --- | ---: | ---: |
| SQLite FTS5 BM25 | 12 | 3 |
| SQLite FTS5 BM25 | 40 | 3 |
| Official ColBERTv2 + PLAID | 12 | 3 |
| Official ColBERTv2 + PLAID | 40 | 3 |

Both backends receive identical canonical passages, split with the pinned
ColBERT tokenizer at a 180-token window and 32-token overlap. BM25 still uses
SQLite's `unicode61` term tokenization and BM25 scoring. Shared chunk boundaries
do not mean shared scoring tokenization. The official code, checkpoint and search
configuration remain pinned by [colbert_modal.json](colbert_modal.json).

Queries are unchanged. Selection keeps the first occurrence of each source owner
in retrieval order, up to three owners. There is no node reranker, query rewrite,
journal fusion, graph traversal or answer generation. Arms alternate order by
history, with five warm repetitions. Returned references are resolved against
the real workspace before node admission.

The current cohort has 24 complete LongMemEval-S histories, four per question-type
stratum. It repeats exactly the expanded cohort from the original protocol, with
no selection based on observed retrieval results. Selection excludes the five
diagnostic IDs listed in the protocol and ranks eligible IDs by a fixed salted
SHA256. The exact IDs are frozen. All LongMemEval-S
histories share one connected component under the project's isolation rule.
These are development diagnostics, not independent held-out observations.

Each source receives a deterministic opaque identifier without consulting gold
annotations. Retriever-visible metadata retains only its date. Original session
identifiers and support annotations remain in evaluator-only records. Source
order, speaker roles, dates and full turn text are preserved.

The original protocol also includes eight controlled synthetic histories with 96
sources each. Those sources already use opaque IDs and do not share the original
LongMemEval identifier issue. They are not rerun by the corrected protocol. Their
questions cover paraphrases, passage crowding, close scope distinctions,
multiple-source needs, capacity and abstention. Source text and separate gold
annotations were frozen before retrieval in
[node_search_controlled.json](node_search_controlled.json), using
[node_search_cases.py](../tools/node_search_cases.py). Histories have disjoint
source IDs and exact text, but share an authored generation style. They do not
establish population generalization. Gold annotations never enter source metadata
or retrieval input.

## Run and retain

Use the configured Modal account and the prerequisites in
[the ColBERT workflow](colbert.md). No hosted generation key or local GPU stack
is needed. The corrected job builds indexes from sources with opaque identifiers
and verifies their canonical passage identities.

Reproduce this historical diagnostic with the preserved runner and a new run ID:

```bash
.venv/bin/python -m modal run tools/colbert_modal.py \
  --action node-search-opaque --run-id UNIQUE
```

The worker permits one A10 container, a 3,300-second worker deadline and a
3,600-second function timeout. It stops after three failed histories. Failed and
unfinished cases remain visible in the artifacts instead of being silently
removed. These limits do not implement dollar accounting or guarantee a charge
ceiling. Use the Modal workspace spend limit. Billed cost remains unknown when
the provider has not supplied it.

Run files are retained under `runs/colbert-modal/UNIQUE/` and the persistent
`llgm-colbert-assets` Volume. Summarize the worker result locally:

```bash
.venv/bin/python tools/node_search_summary.py \
  runs/colbert-modal/UNIQUE/node-search.json \
  --output runs/colbert-modal/UNIQUE/node-search-summary.json
```

## Original protocol and identifier caveat

[node_search_v1.json](node_search_v1.json) retains the original 35-history design:
three replications, 24 expanded LongMemEval questions, and eight controlled
histories. The LongMemEval arm rendered original session IDs in its passages.
Those identifiers contain prefixes correlated with answer-source membership.
Its LongMemEval results therefore cannot support a clean retrieval comparison.
Keep those artifacts as historical evidence rather than replacing them with the
corrected run. The controlled histories already had opaque identifiers and remain
valid within their declared diagnostic scope.

For explicit reproduction of that original, caveated protocol:

```bash
.venv/bin/python -m modal run tools/colbert_modal.py \
  --action node-search --run-id ORIGINAL_UNIQUE
```

The corrected repeat is not an isolated causal estimate of identifier effects.
It also removes original session-ID metadata, rebuilds indexes and operates on
already exposed development questions.

## Read the result

Record required-node coverage among candidate passages and selected seeds
separately. A required node absent from candidates indicates a retrieval miss.
A required node present in candidates but absent from selected seeds indicates an
admission loss. Retain per-question IDs, reference provenance, unique-owner
counts, duplicate concentration, largest-owner share, unlabeled selected nodes,
timings and failures. Unlabeled does not necessarily mean irrelevant because
benchmark support annotations may be incomplete.

Report the two LongMemEval questions with more than three annotated source nodes
as annotation-capacity cases. The original controlled cohort has a separate
four-node capacity case. Complete annotated-source selection is impossible at
the fixed cap. The benchmark annotations are not necessarily minimal evidence
sets, so this does not prove that answering requires more than three seeds.
The original no-evidence controlled case has null
recall rather than a perfect score. Its admitted nodes describe retrieval
behavior, not whether a final model would abstain.

The runner compares top-12 and top-40 ranking prefixes and selected sets. With a
stable ranking prefix, retrieving more passages cannot change the first three
distinct owners if all three already appear in the first twelve. A larger
candidate pool can therefore improve candidate coverage while leaving node
admission unchanged. It may also add references to already-selected nodes.
Reference enrichment is recorded separately and does not by itself prove usefulness.

Separate index preparation, first measured search and warm repetitions. Index
construction already performs a search, so the first measured query is not a
fresh-process cold start. The integration diagnostic's separate-container reopen
serves a different measurement purpose.

These results measure retrieval and direct seed admission. They establish neither
answer accuracy nor end-to-end evidence loss, because later recursive inspection
can reach further nodes. This comparison also does not isolate PLAID's
approximation error against exact ColBERT scoring.

# Passage retrieval and node selection: September 11, 2026

**Later status:** The selector and final-answer follow-ups recommended by this
report have completed. Its original measurements and recommendations below
remain historical. Use the [current improvement order](../ROADMAP.md#node-search-improvement-order)
for the latest node-search priorities.

The corrected experiment supports improving node selection next. With opaque
source IDs, BM25 retrieves all annotated support nodes at 40 passages for all
24 benchmark questions, but the current rule selects all required nodes for only
15 of the 22 questions that fit the three-node cap. ColBERT + PLAID selects all
required nodes for 18/22, with all required nodes available for 21/22.

Use a bounded, question-aware selector over a larger candidate pool as the next
experiment. Keep both retrievers as separate controls. This is a recommendation
to test, not a measured improvement from a selector or an answer-quality claim.

Both live runs completed. The original 35-history run exposed label-correlated
benchmark IDs in passage headers. Its observations are retained with that caveat.
The same 24 additional benchmark questions were then rerun with opaque IDs.
The eight already-opaque controlled histories remain a separate diagnostic cohort.

## Question and execution

The experiment separates finding an annotated source among retrieved passages
from admitting that source as an initial node. Four arms compare SQLite FTS5
BM25 and the released ColBERTv2 checkpoint with official PLAID at 12 and 40
passages, with at most three initial nodes. Every measurement calls the actual
`LLGM._seeds` → `Evidence.search` path. It stops before LLM generation, graph
traversal, journal interpretation, or answer judging.

Both backends index exactly the same canonical passages per history, using a
180-token ColBERT-tokenizer window and 32-token overlap. BM25 still uses its own
`unicode61` scoring tokenization. Queries are unchanged. Selection retains the
first distinct owners in passage order. There are five warm repetitions after
the first measured search, and arm order alternates by history. Index preparation
is separate. The first measured query is not a fresh-process cold query.

The [operator guide](../../experiments/node-search.md) owns commands. Frozen inputs:

- Original protocol: `experiments/node_search_v1.json`, SHA256
  `21ee53d13caeb639bc3c1d7b68f46756923e82250ec35bfc056c6fa3d9b2fe3b`.
- Corrected protocol: `experiments/node_search_opaque_v1.json`, SHA256
  `ab93afec91a37174c7de6a597fef5904492cb0858c75279308937c9c4fc657fe`.
- Official ColBERT commit: `cc4f3dc91c0b45d2d08c251d9d95178285c65f1c`.
- Checkpoint revision: `c1e84128e85ef755c096a95bdb06b47793b13acf`.
  Full checkpoint SHA256: `482743edb1820fda2de676d3260032ec9f15970a110bd8e468c1c6506e3ab804`.
- LongMemEval-S SHA256: `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
- A10, four CPU cores, 16 GiB requested memory. `query_maxlen=128`,
  `doc_maxlen=180`, `nbits=2`, `ncells=2`, `ndocs=1024`, centroid threshold 0.45,
  and four k-means iterations. These are the existing pinned settings.

## Corpus and identifier audit

The original cohort contains three exposed replications, 24 additional questions
selected by a frozen salted-hash rule with four per question type, and eight
authored stress histories with 96 source nodes each. Report these cohorts
separately. All 500 LongMemEval-S questions share one history component under the
project's isolation rule. These are development diagnostics, not independent
held-out observations. The original protocol's wording about a “prior pilot”
exclusion refers to its explicit five-ID diagnostic exclusion list, not a complete
historical exposure audit. No additional question was selected from observed scores.

Independent checking found that all 43 annotated support sessions in the 24-case
cohort have an `answer_` prefix, while none of its 1,112 other sessions do.
`split_nodes` renders original node IDs in passage headers. Explicit gold answers
and `has_answer` fields were excluded, but these IDs remained a potential shortcut.
This establishes input leakage risk, not proof that either backend exploited it.
The original scores cannot support a claim of label-neutral retrieval quality.

The corrected protocol keeps the same questions, source order, dates, roles and
full text. It replaces each source ID with a deterministic opaque hash and keeps
only date metadata in the source. Original identifiers and occurrence aliases
remain evaluator-only. Its 1,155 source mappings passed pre-execution checks.
Re-rendering headers can change token counts and chunk boundaries, so the repeat
does not isolate only the semantic effect of the `answer_` string. Within each
corrected history, both backends still receive identical passages.

The synthetic histories already use opaque IDs and empty metadata. Their source
labels were independently checked without retrieval outcomes. The ns-06 adult
workshop deliberately shares a time with the children's workshop. Its scope does
not support the children's answer. The ns-08 cabinet code is absent. Shared
templates, short relevant notes, and deliberately repetitive planning threads
limit generalization from these probes.

## Corrected benchmark results

The primary comparison uses `runs/colbert-modal/node-search-opaque-20260911-a/`.
All 24 histories and 96 arms passed execution checks. The table below counts
questions with complete annotated-source coverage, using the 22 questions that
fit the fixed three-node cap.

| Arm | All required nodes available among candidates | All required nodes selected | Warm search plus selection, median |
| --- | ---: | ---: | ---: |
| BM25, 12 passages | 17/22 | 15/22 | 9.05 ms |
| BM25, 40 passages | 22/22 | 15/22 | 18.83 ms |
| ColBERT + PLAID, 12 passages | 18/22 | 18/22 | 20.11 ms |
| ColBERT + PLAID, 40 passages | 21/22 | 18/22 | 29.32 ms |

Timing is the median of per-question medians over five warm repeats of the actual
evidence/search/selection path, measured inside the worker. It excludes index
preparation, workspace import, remote RPC dispatch, and generation. It is not an
interactive end-to-end latency estimate.

Across all 24 questions, including the two four-source cases, complete candidate
coverage was 17/24 and 24/24 for BM25 at 12 and 40, and 19/24 and 23/24 for
ColBERT. Selected-node macro recall was 69.79% for BM25 and 88.54% for ColBERT
at both cutoffs. On paired capacity-feasible questions, ColBERT had higher
selected-node recall on five, BM25 on zero, and 17 tied. These are descriptive
observations from the frozen diagnostic cohort, not significance or superiority
claims over the complete benchmark.

Increasing the cutoff improved candidate coverage but did not change annotated
seed-node recall on any of these 24 questions. Selected sets did change on three
BM25 and eight ColBERT questions as previously unused slots were filled. Every
top-12 raw/canonical ranking remained an exact prefix of top 40. Each history
also gained references to already-selected nodes: 194 additional BM25 references
and 391 ColBERT references overall. Their downstream usefulness is unmeasured.

At 40, BM25 misses no annotated source in the candidate pool. ColBERT still misses
the source for `06f04340`, a homegrown-dinner preference question. Thus better
initial ordering and more complete candidate coverage favor different backends
on this cohort. A selector alone cannot recover a source absent from its input.

The source-coverage ceilings for a hypothetical perfect three-node selector are
22/22 for BM25 and 21/22 for ColBERT, compared with observed 15/22 and 18/22.
Those ceilings describe available evidence, not an expected model-selector result.

## Controlled stress results

These eight histories come from the completed original run. Six have positive
labels and fit the three-node cap. One requires four nodes. One has no positive
labels and therefore null recall.

| Arm | All required nodes among candidates, feasible cases | All required nodes selected, feasible cases |
| --- | ---: | ---: |
| BM25, 12 passages | 3/6 | 1/6 |
| BM25, 40 passages | 5/6 | 3/6 |
| ColBERT + PLAID, 12 passages | 4/6 | 2/6 |
| ColBERT + PLAID, 40 passages | 6/6 | 4/6 |

Both crowding cases retrieved twelve passages from one planning thread and
missed both required sources at 12. At 40, both backends retrieved and selected
both required sources. This supports a larger fallback pool when passages are
concentrated in too few nodes.

At 40, the aquarium and children's-workshop cases still lose available evidence
during selection. The aquarium's two required nodes rank fourth and fifth for
BM25. Neither becomes a seed. ColBERT selects one and leaves the other at fourth
place. In the workshop case, each backend selects only one of three required
sources despite having all three among its candidates. Merely deduplicating
owners does not resolve scope or question coverage.

The four-node tasting case selects three of four sources with either backend.
That is the declared direct-seed capacity limit. The no-code case returns
candidates but has no recall denominator. It does not test whether a final model
would abstain.

## Original benchmark observations: identifier caveat applies

| Arm | All required nodes among candidates, 24 cases | All required nodes selected, 22 capacity-feasible cases |
| --- | ---: | ---: |
| BM25, 12 passages | 17/24 | 16/22 |
| BM25, 40 passages | 24/24 | 16/22 |
| ColBERT + PLAID, 12 passages | 19/24 | 18/22 |
| ColBERT + PLAID, 40 passages | 22/24 | 18/22 |

The two four-node questions, `gpt4_731e37d7` and `2788b940`, remain separately
visible in the artifacts. Original selected-node macro recall over all 24 cases
was 76.04% for BM25 and 88.54% for ColBERT at both cutoffs. These are descriptive
scores for the original headers, not answer accuracy.

All 35 histories preserved exact raw and canonical top-12 prefixes at top 40.
On the 24 additional questions, selected sets changed for four BM25 and eight
ColBERT cases as additional node slots were filled. Annotated-node recall did
not change. All 24 gained references to already-selected nodes: 191 additional
BM25 references and 367 ColBERT references. Their usefulness for downstream
inference remains unmeasured.

The three original replications had complete candidate coverage for both
backends at both cutoffs. BM25 selected all annotated nodes on two of three.
ColBERT did so on three of three. They remain exposed diagnostics with the same
identifier caveat.

## Retained execution and validation

Original run: `runs/colbert-modal/node-search-20260911-a/`.
All 35 histories and 140 arms passed execution checks, with 840 actual initial
seed searches and zero hosted LLM generation calls. Five warm repeats had stable
rankings and seed selections. No canonical-hit count differed from its raw-hit
count. The run reused three indexes and built 32. Summed per-history counts were
2,068 sources and 33,316 passages. Shared histories are not globally deduplicated.

Worker time was 1,291.12 seconds. The outer remote-action sequence took 1,320.24
seconds. First persisted-index opening took 88.03 seconds, with the next two
opens taking 5.88 and 5.64 seconds. The CPU-only asset verification container's
CUDA banner is not a GPU execution failure. Native FAISS warnings about sample
size relative to centroids remain in the log. These small indexes do not support
a broad scalability or tuned-index-quality claim.

The original job used image `im-uDGyOQ1G1mqwiWmccYHjrK`, container
`ta-01M27TFCDDCC420T1QT6SFNJZR`, Python 3.11.5 and A10 driver 580.95.05.
Its source aggregate is `6af6f5bde7290925a1969a218a346c411fbcccc62110ce324811a0250fb80876`.
All 57 recorded source files matched and were retained under `executed-source/`
before the identifier correction. The app stopped with zero tasks. Actual billed
cost was not returned and remains unknown.

The original local validation is retained under
`runs/node-search-20260911/validation/`: 820 deterministic tests passed, two
skipped, 21 integration tests deselected, and 230 subtests passed. The separate
dataset-selection test passed. Statement coverage was 90.60% overall and 91.43%
for the original evaluator. Combined statement/branch coverage was 88.11%.
Ruff, docstrings, strict Sphinx, link/publication audit, build and Twine checks
passed. Built archives contained no `research/` or `agent-context/` paths.
The correction has separate focused validation. These original measurements are
not silently promoted to cover the changed code.

The corrected run rebuilt all 24 indexes over 1,155 sources and 29,039 passages.
All 576 application seed searches completed with stable warm rankings and
selections and zero hosted LLM generation calls. The worker took 1,367.68 seconds.
The outer remote-action sequence took 1,388.79 seconds. The first native build
took 111.53 seconds, including startup work. Billed cost remains unknown.

The corrected image remained `im-uDGyOQ1G1mqwiWmccYHjrK`. Its container was
`ta-01M27W9M0QWEMFXZQ146E4J8NR`, with source aggregate
`7df03737dd402585ff0872ef1039bcdc946b1ac4d132a298e05b07d58d6d192f`.
All 57 recorded source files matched locally and were copied to the corrected
run's `executed-source/`. Both experiment apps stopped, and the final container
inventory was empty. Indexes and artifacts remain on the existing Modal Volume.

Independent scoring reconstructed source occurrences and opaque hashes directly
from the raw dataset, decoded every returned owner, and recomputed all 96 arm
recalls and missing-node sets. All matched. The audit also checked 2,496
first-sample canonical reference observations against original source bounds.

Final validation lives in `runs/node-search-20260911/opaque-validation/`:
825 deterministic tests passed, two skipped, 21 integration tests deselected,
and 230 subtests passed. The focused pre-run checks also exercised the actual
dataset-selection test. Overall statement coverage is 90.62%, branch coverage
80.97%, and combined coverage 88.13%. The evaluator has 92.31% statement coverage.
Ruff, 1,869/1,869 docstrings, strict Sphinx, documentation-boundary checks,
distribution builds, Twine, and archive exclusion checks passed. Corrected
distributions are retained beside these validation artifacts.

## Limits and next decision

Source coverage measures contact with annotated sessions, not discovery of the
answer-bearing span or correctness of an answer. Support annotations are not
exhaustive relevance judgments. An unlabeled selected node is not necessarily
irrelevant. Later recursive searches or links may recover omitted initial nodes.
These runs do not test full answering, learned maintenance, independent held-out
quality, exact ColBERT versus PLAID approximation, or query length 32 versus 128.

Cold-start optimization remains deferred during research and must be reported
separately from warm timing.

The next experiment should retrieve 40 passages, group them by owning node, and
let a smaller model choose at most three distinct nodes using bounded excerpts,
question coverage, and the requested time/scope. Compare that choice with the
existing first-occurrence rule, separately for BM25 and ColBERT, while holding
queries, corpora and the three-node limit fixed. Preserve opaque IDs in every
model-visible handle. Keep scorer labels out of the selection prompt.

Measure complete selected-source coverage, regressions on previously successful
questions, added model usage/cost, latency, and operational failures. The controlled
crowding results support a larger candidate pool. The scope cases support testing
a better choice within that pool. Simply raising the passage cutoff or summing
repeated passage scores is not established as a sufficient solution.

Keep BM25 as the inexpensive candidate-retrieval control and ColBERT + PLAID as
the stronger observed initial-ordering control. Do not discard either based on
this cohort. Test full recursive answering after the selector comparison, then
evaluate frozen choices on independent data. No new selection policy or library
default was introduced by this experiment.

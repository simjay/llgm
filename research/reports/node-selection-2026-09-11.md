# Small-model node selection: September 11, 2026

**Later status:** The [frozen-seed answer comparison](seed-answers-2026-09-11.md)
and [ordinary LongMemEval pilot](longmemeval-2026-09-11.md) have completed.
Recommendations below retain this experiment's original context. Current work
follows the [node-search improvement order](../ROADMAP.md#node-search-improvement-order).

Keep the existing application seed policy for now. The tested small-model
selector improves the controlled scope cases but reduces complete annotated-source
coverage on LongMemEval for every candidate pool. Combining both retrievers with
passage-level RRF does not outperform the stronger single-backend control here.
This is a result for one frozen model, prompt and fusion policy, not a conclusion
that model-based selection or complementary retrieval cannot work.

All 288 hosted calls completed successfully. Estimated model usage cost was
$0.9514792, with no new GPU work. Independent audits verified every prompt,
selected-source score and token-cost calculation. Product defaults are unchanged.

The next evaluation should measure final-answer correctness and evidence
attribution under matched inference budgets. Keep source coverage as a diagnostic.
Do not optimize it as if every annotated source were necessary for every answer.

## Frozen comparison

The experiment reuses actual retrieval from the corrected opaque-ID
[node-search run](node-search-2026-09-11.md), plus its eight already-opaque
controlled histories. No retrieval is mocked or regenerated. This is a hosted
selection experiment over frozen real retrieval, not a fresh hybrid retrieval or
end-to-end answering benchmark. The product's existing seed policy is unchanged.

The [operator guide](../../experiments/node-selection.md) owns reproduction commands.
The frozen input is `experiments/node_selection_v1.json`, SHA256
`c69d274efcbe5e14d34515908fc7a8ead6aeba9a6fe02fe96166302b25633366`.

| Pool | Candidate construction | Controls and model selection |
| --- | --- | --- |
| BM25 | Actual top 40 passages | First three distinct owners versus hosted selection of up to three |
| ColBERT + PLAID | Actual top 40 passages | Same comparison |
| Combined RRF | Union of both top-40 lists, reciprocal-rank fusion, retain 40 | Same comparison |

Fusion deduplicates exact canonical span tuples within and across backends. A
backend contributes once per passage using its best original rank. Scores sum
`1 / (60 + rank)`. Ties use the best contributing rank and then canonical span
identity. The complete deduplicated union, before the top-40 cut, is scored
separately. The combined arm requires two underlying searches. Matching the final
passage cap does not match retrieval compute or establish live hybrid latency.

Every node group retains all retrieved spans, roles and source dates. The prompt
includes the original question and its date. It contains no benchmark case ID,
original session identifier, evaluator flag, required-source count or gold answer.
A 100,000-byte request cap rejects oversized inputs instead of truncating them.
The actual requests occupy 25,506 to 38,679 bytes. An independent preflight audit
checked all 32 cases, 96 pools and 3,840 source spans against original data. All
64 single-backend controls reproduce the prior application selections exactly.

The selector is `gpt-4.1-mini-2025-04-14`, accessed through the existing native
OpenAI adapter. It uses a fixed system prompt, temperature 0, JSON-schema output,
a 512-token output cap and no automatic retries. It chooses at most three unique
candidate handles and gives a short reason. Invalid output is retained as a
failure without replacement or repair. Each pool receives three independent
requests with the same prompt. Pool execution order rotates by case and repeat.
No prompt or ranking policy changes are made in response to observed selections.

The run uses one concurrent model call, a 60-second per-call timeout, a
3,600-second admission deadline, and a stop after three failed calls. An admitted
request can finish after the admission deadline. The complete 288-call local
reservation is $4.0103052, below the declared $10 experiment cap. It counts input
UTF-8 bytes, schema/framing allowance and maximum output at uncached rates.
Returned token usage is checked against those reservations. This is local
admission accounting, not a provider-enforced invoice ceiling.

Standard token pricing was frozen at $0.40 per million input tokens, $0.10 per
million cached input tokens, and $1.60 per million output tokens. Estimates use
provider-reported categories without charging cached input twice. Actual invoices
remain a separate measurement. [OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-4.1-mini)

## Cohorts and interpretation

The benchmark cohort repeats 24 exposed LongMemEval-S questions. Twenty-two have
at most three annotated sources and two have four. All 500 questions
share one connected history component under the project's isolation rule. This
experiment does not create an independent held-out sample.

The eight controlled histories have 96 source nodes each. Six have positive
labels and fit the cap, one lists four sources, and one has no positive label.
They probe paraphrase, crowding, scope, multiple-source needs and absent evidence.
They share an authored style and deliberately repetitive distractors. Report them
separately from LongMemEval. No-positive source recall is null, not perfect recall.

Source-node coverage measures whether annotated sessions are selected. It does
not measure whether a retrieved excerpt contains the answer, whether the later
node delegate will extract it, or whether final synthesis is correct. Unlabeled
nodes can still be relevant. The selector's short explanation is not a semantic
judge. Later recursive exploration can recover omitted nodes.

The annotated source set is not necessarily a minimal sufficient evidence set.
Some update questions label both an old and a new observation, although the new
observation directly states the requested current value. Some multi-session
histories repeat facts across nodes. Therefore an annotated-source count above
three makes complete label coverage impossible at this cap, but does not prove
the question cannot be answered from three or fewer nodes. The semantic audit
retains this distinction without changing gold labels or frozen scoring rules.

## Candidate availability before model selection

| Cohort and pool | All required nodes available, capacity-feasible questions | All required nodes selected by the existing rule |
| --- | ---: | ---: |
| LongMemEval, BM25 | 22/22 | 15/22 |
| LongMemEval, ColBERT + PLAID | 21/22 | 18/22 |
| LongMemEval, combined RRF | 21/22 | 18/22 |
| Controlled, BM25 | 5/6 | 3/6 |
| Controlled, ColBERT + PLAID | 6/6 | 4/6 |
| Controlled, combined RRF | 6/6 | 3/6 |

The complete union has all required nodes for 22/22 feasible LongMemEval
questions and 6/6 feasible controlled questions. For LongMemEval `06f04340`,
BM25 finds a required source that ColBERT misses. The 68-passage union contains
it, but RRF's top-40 truncation removes it. This limitation was independently
identified before paid execution and the frozen policy was preserved.

## Selection results

The table counts questions for which every annotated source is selected, among
questions whose annotation count fits the three-node cap. Each model column is
a separate repetition. No best-of-three outcome is used.

| Cohort and pool | First-owner control | Model repeat 1 | Model repeat 2 | Model repeat 3 |
| --- | ---: | ---: | ---: | ---: |
| LongMemEval, BM25 | 15/22 | 14/22 | 14/22 | 14/22 |
| LongMemEval, ColBERT + PLAID | 18/22 | 17/22 | 17/22 | 17/22 |
| LongMemEval, combined RRF | 18/22 | 15/22 | 15/22 | 15/22 |
| Controlled, BM25 | 3/6 | 5/6 | 5/6 | 5/6 |
| Controlled, ColBERT + PLAID | 4/6 | 6/6 | 6/6 | 6/6 |
| Controlled, combined RRF | 3/6 | 6/6 | 6/6 | 6/6 |

For LongMemEval, BM25 gains complete coverage on three questions and loses it
on four. ColBERT gains one and loses two. Combined RRF gains one and loses four.
These paired outcomes are identical across repetitions. The controlled BM25
and ColBERT selectors each gain two with no losses. The combined selector gains
three with no losses. Its controlled result matches ColBERT selection while
using both underlying retrieval runs.

The controlled gains include the aquarium and children's-workshop scope cases.
The combined pool also recovers the drink-preference case that its first-owner
control misses. BM25 cannot recover that source because it is absent from its
candidate pool. These controlled gains demonstrate useful behavior under the
authored conditions, not a general benchmark advantage.

Selected sets are identical across repeats in 93 of 96 case/pool combinations.
The exceptions are LongMemEval `06f04340` with RRF, and controlled `ns-03` and
`ns-07` with ColBERT. Primary capacity-feasible completion scores do not vary.
The no-positive controlled case selects no node in all nine calls. That is a
selector observation, not a measured final-answer abstention rate. The three
annotation-capacity cases remain in per-case artifacts and all-positive recall.

## What the semantic review changes about interpretation

A targeted, unblinded review inspected the observed regressions and one gain.
Its original sources, model-visible excerpts, selected handles and reasons are
retained under `live-a/semantic-review-initial.md` and accompanying files. It
does not change the frozen labels or substitute for an answer benchmark.

| Case | Finding | Implication |
| --- | --- | --- |
| `dad224aa` | BM25 and RRF omit explicit later Saturday wake-time statements and choose an older variable-routine source. | Clear evidence-selection concern despite all relevant statements being visible. |
| `184da446`, `72e3ee87` | The selected newer source directly states 220 pages or 50 episodes, while an older progress report is dropped. | Lower source coverage does not demonstrate a wrong answer. |
| `a4996e51` | Every selector retains an assistant-written suggestion containing 50 hours but drops the user's separate 40-hour baseline needed to verify the stated increase of 10. | Answer text remains visible, but primary evidence and attribution weaken. |
| `gpt4_59149c78` | ColBERT and RRF overlook the visible Metropolitan Museum statement and choose a source about different, older visits. | A temporal/entity selection error is credible without claiming a generated answer was wrong. |
| `2788b940` | Two full nodes can cover all five weekly classes despite four annotated source IDs. BM25's selected excerpts cover all five, while RRF omits visible schedules. | Equal source recall can hide different evidence sufficiency, and annotation capacity is not semantic capacity. |
| `gpt4_372c3eed` | All selectors add the intervening Associate's-degree source omitted by the controls. | Useful evidence gain, but the final education answer might also be inferred from endpoint dates. |

The model sometimes explains that other nodes contain no relevant information
when the supplied excerpts directly contradict that explanation. Structured
output and valid source handles do not establish sound evidence interpretation.

## Model latency, cost and validation

The run completed in 795.71 seconds with one concurrent request. Median model-call
wall time was 2.39 seconds, ranging from 1.20 to 22.56 seconds. These are hosted
selection times including transport, not complete query times. Retrieval was
replayed and no live hybrid retrieval latency was measured.

| LongMemEval pool | Median selector call | Mean estimated cost per call |
| --- | ---: | ---: |
| BM25 | 2.24 seconds | $0.003151 |
| ColBERT + PLAID | 2.42 seconds | $0.003256 |
| Combined RRF | 2.35 seconds | $0.003213 |

Across 288 calls the provider reported 2,314,050 input tokens, including 87,552
cached tokens, and 32,578 output tokens. All calls resolved to the pinned model.
Independent pricing gives $0.9514792. The per-call figures price one selection,
while the experiment pays for three repetitions per case/pool. Repeated prompts
can receive caching discounts, so these observations are not an uncached
production cost guarantee. No token categories or failed calls have unknown cost.
Actual invoice charges were not retrieved.

The final independent audit verified 288 completions, 96 reconstructed prompts
and pools, 3,840 canonical spans, all 64 original controls, and 56 executed-source
snapshots. Request and response identities are unique. The audit and its
reproducible script are retained as `live-a/final-independent-audit.json` and
`live-a/final_independent_audit.py`.

Local validation passed 906 tests, with two skipped, 21 integration tests
deselected and 230 subtests passed. The independent preflight exercised the real
dataset and retained retrieval across component boundaries. Overall statement
coverage is 90.79% and combined statement/branch coverage is 88.34%. The selector
module has 96.95% statement coverage and 95.94% combined coverage. Ruff and
1,941/1,941 docstrings passed. Strict Sphinx, documentation-boundary checks,
wheel/source builds, Twine and private-directory archive exclusions passed.

## Final recommendation

Keep both retrievers available, but retain the current seed policy as the control
and product default. This experiment does not justify an unconditional model
selector or automatic RRF fusion. The combined pool's discarded unique source
also shows that agreement-focused passage fusion can lose complementary evidence.

Next, run matched final-answer comparisons with the current policy and this
frozen selector, using identical root/delegate models and inference budgets.
Measure answer correctness, support from original user evidence, temporal errors,
latency and total cost. Report both source coverage and final answers, including
cases where they disagree. Do not remove difficult cases or relabel annotations
to favor a policy.

If the combined retriever is developed further, test preserving distinct evidence
from both backends before pruning the pool. Declare any larger pool or inspection
budget as a separate arm. A bounded RLM inspection step may be worth testing when
excerpts are ambiguous, but neither that change nor a different selector model
was measured here. Freeze choices before independent evaluation.

## Retained artifacts

- Preflight and independent audit: `runs/node-selection-20260911/preflight-a/`.
- Live prepared requests, scorer records, protocol, executed sources and results:
  `runs/node-selection-20260911/live-a/`.
- Local validation and live execution log:
  `runs/node-selection-20260911/validation/`.

Provider outputs, selected handles, token categories, model/request identities,
latencies and failures are retained per attempt. Gold records remain outside
requests. No new GPU job, deployment or library-default change is part of this run.

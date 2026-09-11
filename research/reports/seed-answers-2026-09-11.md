# Final answers from frozen seed choices

**Later status:** Root citation, maintenance-schema and a reproduced Docker
cleanup conflict were subsequently repaired and checked. The
[ordinary LongMemEval pilot](longmemeval-2026-09-11.md) then completed without
operational failures but still lost facts inside selected nodes and during
synthesis. This report preserves the earlier attempts and their original
limitations. Current priorities are in the
[roadmap](../ROADMAP.md#node-search-improvement-order).

September 11, 2026. The frozen 192-answer comparison completed. It does not
establish a useful selector winner because the chosen model and budget
configuration rarely delivered source evidence to the root. Product defaults
remain unchanged. A separate eight-case delegate qualification improved citation
delivery but still exposed root synthesis and operational failures. Fix and
qualify evidence delivery before using final-answer scores to choose a selector.

## Main result

| Initial pool | LongMemEval first-owner | LongMemEval selector | Controlled first-owner | Controlled selector |
| --- | --- | --- | --- | --- |
| BM25 top 40 | 0/24, one unknown judgment | 0/24 | 1/8 | 1/8 |
| ColBERTv2 + PLAID top 40 | 0/24 | 0/24 | 1/8 | 1/8 |
| Fused top 40 | 0/24 | 1/24 | 1/8 | 1/8 |

Each entry counts correct judgments over the full planned denominator. These are
six settings on the same exposed questions, not independent samples. All 192
answer attempts and 384 judge attempts were retained. Accuracy produced 191
usable judgments and one incomplete response, which remains unknown. All support
judgments parsed. The summary therefore labels judging incomplete while the
execution record correctly reports all planned attempts completed.

The single benchmark win, `65240037` with fused model selection, answered a
general factual question correctly without citations. Its support judgment was
unsupported. This is not evidence that the selector recovered the conversation's
source. Every controlled success was the unanswerable `ns-08` case. None of the
42 answerable controlled attempts was correct. There were no answers judged both
factually correct and supported by their cited evidence.

The runtime returned 169 partial outcomes and 23 failed outcomes. Twenty-six
answers were empty. Only one of all 192 answers retained a canonical final
citation, in `fca70973` with ColBERT first-owner selection. That answer correctly
described the irrelevant material it had read, but failed the benchmark question.
No final citation discovered a node outside the initial seeds.

## Why selection could not be assessed reliably

The post-hoc runtime audit found 433 initial seed frames and no nested frames or
recorded `query_node` operations. Models attempted 1,917 reads. Valid callbacks returned
681 read results across 74 trials, but 73 of those trials still ended without
final citations. The bridge could resolve real sources, while evidence delivery
was usually lost during model-generated operations and finalization.

The traces contain 1,236 malformed-reference errors, 815 rejections of citations
the invocation had not accessed, and 162 rejections of finishing without source
inspection. Eighteen trials exhausted the per-execution callback limit. All 23
root failures attempted a citation not returned by a node branch.

Among 953 generated Python steps, 949 were AST-parseable. Only 29 contained a
direct `print` call. Another 735 contained read syntax without a direct `print`.
This is a static descriptive heuristic, not proof that a particular statement
executed. The retained examples show successful reads whose results were never
printed, invented source contents, dropped reference fields, and node IDs used
where accessed evidence IDs were required.

For example, `184da446` with ColBERT first-owner selection supplied complete typed
spans, but generated reads omitted `type` and `turn_id`. Another branch explicitly
simulated a page count without reading. `dad224aa` with BM25 first-owner selection
did execute valid reads, then exhausted callbacks and cited unaccessed IDs. These
are failures of the model/interface/budget combination, not corrupted seed maps
or evidence of a defective ColBERT index.

The runtime source hash is identical to the preceding successful hosted
diagnostic. Models, budgets and history size changed together between studies.
The present result does not isolate the smaller model alone, establish a general
Python-RLM limitation, or justify declaring the two selectors equivalent.

The support judge also inconsistently classified uncited abstentions as supported,
unsupported or not applicable. Its raw judgments are preserved rather than
rewritten after inspection. Correctness, structural citations and semantic support
must remain separate measurements.

All three selector responses on `ns-08` were empty partial answers with no seed
nodes and no inference-model calls. The accuracy judge accepted them as
abstentions. First-owner responses gave textual absence explanations, which the
accuracy judge also accepted but the support judge called unsupported. These
scores do not establish equal or superior user-facing abstention quality.

## Main run latency and cost

| LongMemEval pool | First-owner median / p95 seconds | Selector median / p95 seconds |
| --- | --- | --- |
| BM25 | 17.85 / 25.17 | 26.13 / 63.48 |
| ColBERT | 20.83 / 30.18 | 23.51 / 30.02 |
| Fused | 19.04 / 25.06 | 24.48 / 28.90 |

These are answer-stage wall times, excluding historical selection and initial
retrieval. Fewer calls or lower cost in an arm can reflect fewer admitted seeds
or earlier failure. Neither establishes efficiency at useful answer quality.
The full run took 2,415.10 seconds, about 40.3 minutes, including workspace preparation,
judging and two-case scheduling.

The 2,552 current provider calls comprise 1,979 delegate, 189 root and 384 judge
calls. Three empty selector choices started no answer-model calls. Known token
counts total 3,937,485 input and 338,225 output tokens. Two delegate requests timed
out without usage, so these token counts are subtotals, not complete usage.
Known generation cost is $2.4860512 and judging cost is $0.3630925. The known
current subtotal is **$2.8491437**, with **$0.0134584** of unresolved reservations.
The complete estimated cost remains unknown. The reused selector repetition adds
$0.32594 of historical cost across its 96 arms, separate from these new charges.

Independent final audits reconciled all 192 trial files, 2,552 call records,
72 frozen source paths, seed choices, official prompts, blind support inputs,
usage, price calculations and summary statistics. The two unknown costs and
single unknown accuracy judgment remain explicit. Post-hoc runtime counts and
inspected examples are in `runtime-diagnostics.json` alongside the final
accounting audit and summary in the live directory.

## Separate delegate qualification

After observing the failures, an eight-case follow-up changed the per-answer
delegate from GPT-4.1 mini to full GPT-4.1. Root, prompt, temperature, sources,
BM25 first-owner handles and inference limits remained fixed. The cohort was the
first question from each benchmark stratum plus `ns-01` and `ns-03`, fixed before
the follow-up's answers. This post-hoc diagnostic does not replace the original
run, compare selectors, or supply an independent test set. Cases ran sequentially
rather than two at a time, so latency is not a controlled comparison.

| Observation | Corresponding main-run attempts | Full GPT-4.1 delegate follow-up |
| --- | --- | --- |
| Planned questions | Eight | Eight |
| Final canonical citations | 0/8 | 6/8 |
| Benchmark correct | 0/6 | 0/6, one unknown judgment |
| Controlled correct | 0/2 | 2/2 |
| Correct and fully citation-supported | 0/8 | 1/8 |

The larger delegate could deliver evidence, but this did not reliably produce a
correct benchmark answer. All 33 read callbacks succeeded, and all 25 generated
Python steps included explicit print calls. There were no malformed-reference or inaccessible
citation rejections. All six final citation records exactly matched their
canonical source text, role and date. Nevertheless, 19 of 24 seed frames read
only their first supplied reference, including nine with more references
available. Four of six benchmark cases had all annotated source nodes among
their seeds. These observations separate interface execution from evidence use.

In `21436231`, a delegate returned the required fish
count with evidence `e3`. The root's request contained that finding and quote,
yet its answer said the count was absent and cited an unrelated documentary
quote `e1`. In `1da05512`, a delegate returned relevant NAS guidance as `e6`, but
the root said the sources did not discuss NAS devices and cited phone/laptop
material as `e1`. These examples identify root synthesis failures after useful
evidence had already arrived, not just insufficient passage fetching.

In controlled `ns-03`, branches returned both carrier and departure-time evidence.
The root answered both correctly but cited only the carrier record. Its support
judgment was therefore insufficient. This was an omitted final citation, not an
unseen or guessed departure time. `ns-01` produced a correct, supported answer.

Two benchmark attempts, `dad224aa` and `cc539528`, failed during interpreter
cleanup. Each canceled an in-flight delegate request and never reached root
synthesis. These operational losses must not be attributed to answer quality.
For `dad224aa`, a branch had already generated the required wake-up time before
cleanup failed. The accuracy judge also returned an incomplete response for that
empty answer, which remains unknown. B has seven usable accuracy judgments and
eight usable support judgments.

Docker event history subsequently confirmed that all 24 qualification containers
were destroyed. In each failed-case window, one container exited normally and
was destroyed milliseconds later without an explicit kill event. Automatic
removal racing against forced removal is a plausible cause, but the original
trace omitted the removal command's stderr, so the exact cause is unproven.
A bounded follow-up of 12 actual interpreter lifecycles reproduced no cleanup
failures and left no containers. It made no hosted model calls. The diagnostic
and event evidence are retained in `qualification-a/lifecycle-diagnostic/`.
Cleanup confirmation and diagnostic detail remain follow-up work. No production
cleanup changes or replacement answer attempts were made.

The follow-up ran 49 delegate, six root and 16 judge calls in 71.39 seconds.
Known cost was $0.231578, with $0.0655 reserved for its two canceled calls whose
provider usage is unknown. Combined A and B known costs are **$3.0807217** with
**$0.0789584** of unresolved reservations. The conservative local admission
liability is $3.1596801, below the original $20 limit. Complete token usage and
total estimated cost remain unknown.

The follow-up's pre-execution budget guard was refined when A's timeouts appeared:
unknown costs retain their full reservations rather than blocking all later work
or being treated as zero. Its complete $2.50 allowance was admitted only after
the completed parent, known subtotals and unknown liabilities reconciled. This
changed no cohort, prompt, seed choice, model intervention or prior result.
The exact parent result was hashed before the follow-up and rechecked before
dispatch. An independent audit reconciled all eight trial files and 71 calls.
Artifacts are under `runs/seed-answers-20260911/qualification-a/` with their own
protocol, executed-source snapshot, accounting audit and semantic review.

## Recommendation

Keep both BM25 and ColBERT available and retain first-owner selection as the
product default. The original selector's source-coverage gains on controlled
examples have not yet translated into a dependable answer-quality advantage.
The failure-heavy comparison cannot settle that question.

The next gate is evidence delivery under the intended model allocation. Start
with the previously successful root/delegate pair on representative realistic
payloads, rather than assuming a cheaper compatible API model is qualified for
the Python interface. Measure valid source reads, selected branch returns,
whether the root uses the relevant returned facts, and whether every material
answer claim has its needed citation. Keep these checks distinct from source
coverage and structural JSON validity. Resolve operational cleanup failures
before another broad answer sweep.

Once that gate is reliable, freeze a matched repeat of the same selector
comparison. Preserve these negative runs, keep the controlled and benchmark
cohorts separate, and use independent development data before a paper-level
claim. Do not promote the selector or redesign retrieval solely to optimize
scores produced by failed evidence delivery.

## Question and design

Does the small-model selector improve final answers over the first three distinct
retrieved node owners, once the same recursive inference system can inspect and
search the complete history?

The preceding [selection experiment](node-selection-2026-09-11.md) measured source
coverage, which can penalize useful selections when annotations include older or
redundant sources. This experiment instead generates actual answers, applies the
official LongMemEval accuracy prompts, and separately judges their cited evidence.

The fixed cohort contains all 24 prior LongMemEval questions and eight controlled
histories. All questions remain included, including the two benchmark cases with
four annotated support sources. Each question receives one answer for each of
three initial passage pools and two seed policies, totaling 192 planned answers.

| Dimension | Frozen choice |
| --- | --- |
| Initial pools | Retained BM25 top 40, official ColBERTv2/PLAID top 40, and their reciprocal-rank fusion truncated to 40 |
| Seed policies | First three distinct owners, or the preceding selector's repetition 0 |
| Root | `gpt-4.1-2025-04-14` |
| Node delegates | `gpt-4.1-mini-2025-04-14` |
| Accuracy and support judge | `gpt-4o-2024-08-06` |
| Model settings | Temperature zero, no SDK retries, one answer attempt per arm |
| Per-answer limits | 13 total model calls, 12 delegate calls, eight searches, 180 seconds |
| Other runtime limits | Depth three, eight steps per frame, 96 operations, three concurrent branches |
| Context/evidence/bundle admission | 65,536 / 65,536 / 16,000 UTF-8 byte units |
| Per-call output cap | 1,536 provider tokens for generation |
| Execution order | Rotate six arms by case index, two cases concurrently, six sequential arms within each case |
| Current cost admission | $16 generation and $4 judging, conservative concurrent reservations |

The model pair was chosen for a low-cost, reproducible comparison. It differs from
the previously successful node-runtime diagnostic, which used `gpt-6-astra` and
`gpt-5.6-terra`. That earlier diagnostic also had larger call, context, bundle,
step and operation limits and smaller controlled sources. Its success does not
qualify the configuration in this experiment.

## What runs

The experimental seam overrides only initial seed admission. The ordinary
`LLGM.answer`, node runtime, isolated Docker interpreters, callback validation,
shared accounting and final synthesis execute unchanged. Frozen handles retain
every saved span belonging to the selected owners. Empty selections remain empty.
Neither gold labels nor selector reasons, ability labels, pool identities or
original benchmark aliases enter inference.

Every arm sees the same complete opaque-ID history for its case, without
prepared edges or journal edits. All subsequent searches use the ordinary local
BM25 index with `passage_chars=2048`. This continuation policy is held fixed even
when ColBERT supplied the initial seeds. Its passage construction differs from
the original 180-token, 32-token-overlap initial retrieval corpus. This is not a
comparison of different recursive search backends or learned graph maintenance.

The initial selection event explicitly records replay and charges one search
slot. The fused pool nevertheless required two original backend searches.
Both policies omit initial skipped-owner lists, so the experimental seed event
is not a byte-identical production first-owner trace. Source import and common
BM25 preparation happen before the six arms. Per-answer wall time includes exact
workspace validation and application execution. Prior retrieval and GPU startup
are excluded. The original selector's measured latency and cost are additional
historical overhead, not newly measured execution.

## Scoring

Accuracy uses the checksum-pinned official `get_anscheck_prompt` function,
extracted without executing the evaluator CLI. Benchmark ability-specific prompts
and the original question, reference answer and generated answer are preserved.
The judge uses Chat Completions, temperature zero, and `max_tokens=10`. Parsing
accepts only yes/no with an optional trailing period rather than the upstream
substring check. This is official-prompt judging, not the unmodified official
script. Controlled abilities map to its multi-session or abstention template and
are reported separately. The controlled abstention preserves its original
`None` reference explanation.

The independent support judge uses Responses with a fixed schema. It sees only
the question, date, answer, and exact records actually cited by the final root.
Canonical spans, exact text, speaker role and source date are verified before
judging. Uncited reads and unselected child findings are not promoted into
supporting evidence. The judge receives no gold, ability, selector explanation
or backend identity. Supported correctness requires both an accurate answer and
a supported verdict. Pure abstention is intended to be `not_applicable` and is
separate from supported factual correctness. Model judgments are fallible and
require inspection alongside their evidence.

All planned cases remain in denominators. Runtime failures, empty responses,
failed judgments and missing trials remain explicit. A partial runtime response
can still be judged correct. One independent generation per arm does not isolate
small stochastic differences, even at temperature zero. These exposed benchmark
histories share one isolation component, and the controlled examples share
authored templates. Neither cohort supports independent held-out claims.

## Accounting and reproducibility

Costs are synchronous API token estimates, not invoices. The frozen rates per
million input / cached input / output tokens are $2 / $0.50 / $8 for the
[root model](https://developers.openai.com/api/docs/models/gpt-4.1),
$0.40 / $0.10 / $1.60 for the
[delegate](https://developers.openai.com/api/docs/models/gpt-4.1-mini),
and $2.50 / $1.25 / $10 for the
[judge](https://developers.openai.com/api/docs/models/gpt-4o), checked September 11.

Before every request the runner reserves uncached cost using UTF-8 message/schema
bytes plus a 2,048-token framing allowance and the full output cap. Shared
reservations account for in-flight requests. Returned usage applies cache prices
once. Unknown usage keeps its reservation and remains unknown in totals. A bound
breach, exhausted monetary allowance, 16 unknown-usage calls, or the shared
four-hour admission deadline prevents further requests. These are local controls,
not provider-enforced billing caps. Historical selector, current answering and
current judging are reported separately.

The operator protocol and commands live in
[experiments/seed-answers.md](../../experiments/seed-answers.md).
The live directory is `runs/seed-answers-20260911/live-a/`. It retains the prepared
choices, evaluator-only gold, per-request inputs and responses, per-trial output,
canonical citations, traces, workspace data, source snapshot and checkpoint.
No fresh GPU run occurs and existing product defaults remain unchanged.

| Identity | SHA-256 |
| --- | --- |
| `experiments/seed_answers_v1.json` | `7a45f891877440357e4f584013ad4f978601b4e9c7923f92bf355989cdfc9dc6` |
| Runner | `e0bb33fba2deb2a907aa42a922831c3165ca253831c77723bf222357d3240ee5` |
| Summary | `052fa3ebc66482535833a32598e1a4916f8daaf2a773ebe7e76725b1f09d9623` |
| Frozen-seed helper | `57c57da1aa8f6149f7931a9fbd013a3910f38018999134ad8c0faf5e874e79ea` |
| Judge helper | `3655dca270365344ed59759c6d26e717c7bcb2fdbcb5e6b92792cce8fc9988b6` |
| Unchanged node runtime | `ea08e1bd6fc8dfc22b08a55ecbdb2e659e38ecf413eb6306d3bef5771d3ed5d1` |
| Official evaluator source | `ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251` |

The official evaluator checkout is pinned to
`9e0b455f4ef0e2ab8f2e582289761153549043fc`.
Input hashes transitively retain the original dataset, retrieval artifacts,
selection protocol and exact selector outputs. Preflight C snapshots the final
runner. The independent preflight audit under preflight B explicitly records its
older snapshot and verifies the corrected current runner with zero paid calls.

## Implementation validation

The new helpers and repository runners have deterministic tests for canonical
seed replay, exact cited evidence, full-history access, untouched real application
dispatch, official prompt equivalence, blinded support inputs, malformed outputs,
concurrent cost admission, unknown usage, cancellations and honest summary
denominators. Scripted model decisions establish these contracts only.

Actual Docker integration executed the new seam against the pinned image.
The full deterministic suite passed 1,050 tests with two optional skips, 23
integration deselections and 230 subtests. Five local pinned LongMemEval tests
passed. The separate exact official-prompt check passed all 66 tests.
The separate qualification added 35 passing contract tests. Final Ruff checks
passed across 131 Python files, and 2,108 of 2,108 definitions had docstrings.
Coverage across the package was 6,583 of 7,235 statements and 2,091 of 2,554
branches, or 88.61% combined. The frozen-seed helper had 96.12% statement coverage
and 95.42% combined coverage. The judging helper had 100% statement and branch
coverage. These coverage denominators exclude repository-root tools.

Strict Sphinx construction and the publication/link audit passed. Wheel and
source distribution builds and strict Twine metadata checks passed. Validation
records are under `runs/seed-answers-20260911/validation/`. The package and site
remain unpublished.

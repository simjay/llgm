# Full node-pipeline live diagnostic

Observed September 11, 2026. The user authorized hosted model testing. The fixed
15-case development cohort was run twice through the full LLGM Python pipeline
and a flat retrieval/root baseline. Both runs retain every scheduled attempt.

The prompt clarification improved LLGM from 7/15 to 12/15 semantically correct
answers. The flat baseline answered 12/15 correctly in both runs with fewer model
calls. This cohort does not establish an LLGM quality advantage. Live recursion
was observed, and it exposed a loss of completed child findings when a parent
exhausted its budget.

## Conditions and identities

| Item | Value |
| --- | --- |
| Cohort | `experiments/node_pipeline_cases_v1.json`, 15 fixed synthetic cases |
| Cohort SHA-256 | `31c83240f47baeb7b201e70299eec57d09343cefb5fea315d792e3b8ff964c04` |
| Configuration | `experiments/node_pipeline_v1.json` |
| Configuration SHA-256 | `c6c76fb1717470b7add0863ef12fa488d0abef22f7893e06bb2555251a457576` |
| Root | OpenAI `gpt-6-astra` |
| Node delegates | OpenAI `gpt-5.6-terra` |
| Python execution | Local Docker, image `sha256:ec7d6c95cd3692a2e2d228a8b1ca74e4025b54121fcc4c5da6f09cfa473315ad` |
| Retrieval | Warm reusable SQLite FTS5, `retrieval_k=12` |
| Branches | Up to three initial seeds, concurrency three, depth three |
| Per-answer limits | 16 total model calls, 12 sidecar calls, eight searches, 180 seconds |
| Python-step limit | 120 seconds including nested child work |
| Cohort limits | 255 model calls and 1,800 seconds per full run |
| Maintenance | Disabled, primary edges and exact journal patches curated |
| Evaluation | Fixed lexical diagnostics plus separate assistant semantic/citation review |

The flat arm uses one initial search and one root call. Both arms share a case
workspace, root model, effective journal interpretation, retrieval cutoff, and
declared context/evidence/output bounds. LLGM can spend additional calls and
follow edges. These are different amounts of computation, not matched-cost arms.
Each case uses generated opaque node IDs. Expected answers and evaluator labels
are excluded from model requests. Models see the synthetic source facts.

Model IDs identify the requested provider models, not immutable provider weights.
Each run freezes the configuration, cohort, and Python package sources before
dispatch. Provider request identities, token usage, generated Python, observations,
runtime events, actual node maps and canonical citation text are retained.
The explicit local credential file is excluded from frozen inputs.

## Full-cohort outcomes

| Measurement | Initial LLGM | Initial flat | Revised-prompt LLGM | Revised-prompt flat |
| --- | ---: | ---: | ---: | ---: |
| Attempted / scheduled | 15/15 | 15/15 | 15/15 | 15/15 |
| Semantically correct | 7 | 12 | 12 | 12 |
| Lexical diagnostic passes | 7 | 12 | 12 | 12 |
| Completed status | 0 | 11 | 8 | 11 |
| Partial status | 15 | 4 | 6 | 4 |
| Failed status | 0 | 0 | 1 | 0 |
| Model calls | 81 | 15 | 100 | 15 |
| Reported input tokens | 86,365 | 29,292 | 150,789 | 29,201 |
| Reported output tokens | 5,780 | 505 | 4,850 | 508 |
| Median trial seconds | 9.81 | 2.38 | 8.99 | 2.45 |
| Sum of trial seconds | 187.26 | 38.08 | 168.86 | 37.31 |
| Unknown-usage calls | 0 | 0 | 0 | 0 |
| Invalid returned citation spans | 0 | 0 | 0 | 0 |

Partial status and correctness measure different things. A supported answer may
retain an operational gap, and an unsupported abstention may be cautious but
still miss an answer established by the complete fixture. A run-level completed
status means all trials were attempted. It does not mean all answers succeeded.

Currency cost remains unknown. Admission accounting uses a conservative UTF-8
byte bound while the table reports provider token usage. Neither call limits nor
these measurements establish a provider-enforced dollar cap. Latencies include
local Docker orchestration and provider work after index preparation. They are
single observations, not a scaling or latency benchmark.

### Initial failures

The first run is `runs/node-pipeline-20260911/live-v1/`. Several delegates ended
with a claim that source text had not been supplied, without executing Python.
Those replies often contained both commentary and final-answer messages. The
adapter retained the final answer and counted discarded commentary. Its text
was not retained, so an intended operation in commentary remains a hypothesis.

Other delegates rebuilt reference dictionaries without the required `type`
field. They recovered using `source_info`, but the schema error remained in the
final unresolved list. The three-seed case opened all three interpreters and
performed no reads. The two-seed answer was correct, but one branch obtained
both facts while the other returned no evidence.

The initial contradiction answer selected BIRCH despite an equally authoritative
HAZEL approval. Its warning about uncertainty did not make the single-token
answer correct. Seven other LLGM errors were missed-answer abstentions. The
flat arm's three missed answers were the graph-access cases.

### Prompt correction and repeat

`src/llgm/inference/nodes.py` now tells a delegate to return one final JSON host
operation, stop for the observation, and avoid commentary/tool-call prose.
It explains that source handles are available access, demonstrates reading a
supplied reference intact, and asks delegates to contribute local findings for
the root to combine. Provider phase parsing, callback validation, allowed direct
reads, and delegation choices were unchanged.

The complete cohort was repeated once in `live-v2/` with unchanged cases, model
pair, budgets, and seeded arm order. This is development verification after
inspecting v1, not held-out evidence of improvement. No failed trial was silently
replaced. `followup-v2.json` records the declared change before dispatch.

The production/staging patches, latest overwrite, both validity boundaries,
suggestion versus approval, contradiction, missing fact, and entity/negation
answers were correct in v2. Three failures remained:

- Case 03: all three delegates read their local facts, then repeated searches
  and reads. Twelve shared sidecar calls were exhausted before any delegate
  returned findings. The reserved root returned an empty answer, rejected by
  the answer contract.
- Case 05: the two-hop branch reached the credential node at depth two, but
  spent the last sidecar call listing its source handle. It never read JASPER.
- Case 06: a child read FLINT, finished with canonical evidence, and returned
  it to its parent. The parent then exhausted its sidecar budget and cleared
  the completed child's findings. The root received no supporting evidence.

Correct answers in cases 04, 12 and 15 still carried branch-local missing-fact
notes that sound inconsistent when merged into a global unresolved list. The
correct UNKNOWN in case 14 followed node schema failures. These outcomes are
not clean operational successes.

## What the graph traces establish

V1 had no `query_node` callbacks. Case 04 followed a primary edge and read its
neighbor directly. A correct graph answer alone did not prove recursive RLM.

V2 case 05 establishes the actual chain: entry delegate calls `query_node`, a
depth-one routing child opens an interpreter and reads/discovers, then a
depth-two credential child opens another interpreter and requests source
metadata. Both child returns and parent continuations appear in the trace.
Budget exhaustion prevented the final source read and selected findings.

V2 case 06 establishes a stronger partial mechanism: a real child executed
Python, read the credential, returned the correct cited finding, and the parent
received it. The subsequent parent budget failure lost that result.

The irrelevant-edge case does not establish informed edge selection. The model
received opaque neighbor references without edge relation/provenance text and
chose the first neighbor. It did not inspect the unrelated contact source in
that realization. Exposing sufficient edge information for useful selection
remains a separate interface question.

## Preserving completed child findings

The parent exhaustion defect is fixed after v2. A branch records child-selected
findings only after complete callback admission. If it later exhausts calls,
steps, or its execution deadline, it can return those findings with the child's
identity, original selected citations, unresolved needs, and an explicit parent
exhaustion status. The same bundle and escaped root-context limits apply to
normal and retained returns. Oversized aggregate findings are omitted explicitly.
Local reads without a selected child result remain unselected. Schema failures,
undelivered payloads, programming errors, and cancellation do not activate this
budget fallback. No extra model call or controller class was added.

Regression tests reproduce successful child completion followed by parent
exhaustion, propagation through two exhausted ancestors, rejection of unselected
local reads, schema failures, undelivered results, aggregate bundle overflow and
escaped-context overflow. All 25 node tests passed.

The separate `live-child-return-v1/` follow-up reran only exposed case 06 through
both arms, with unchanged per-trial settings and a smaller cohort cap of 17 calls
and 360 seconds. It used 14 calls. LLGM returned FLINT with canonical supporting
citations, but chose direct neighbor reading and created no child. Therefore
this live realization does **not** exercise the new preservation branch. The
deterministic tests establish that boundary. The flat arm returned an empty
answer and failed schema validation. Both outcomes remain in its two-trial
denominator and are not added to the 15-case v2 score.

Across the three runs, 62 trials made 225 generation calls. Provider usage totals
are 317,423 input tokens and 12,532 output tokens, with no unknown-usage calls.
No paid judge calls were made. Final `make check` passed 737 tests and 230 subtests,
with two editable-package checks skipped and 20 integration tests deselected.
The strict documentation build and rendered navigation/API/publication audit
passed. These counts describe the whole checkout at the final checkpoint,
including concurrent work outside this task.

## Interpretation and remaining work

The runtime performs source reads, amendments and actual nested hosted/Docker
execution. Automatic planning and return scheduling are less reliable than the
structural unit tests alone suggested. The flat baseline remains the cheaper
choice on these short directly retrieved facts. No broad quality or efficiency
advantage has been shown.

The completed-child preservation defect is fixed and regression-tested.
Next prioritize budget-aware branch finalization and duplicate investigation. Keep branch-local
gaps attributed when the root combines results. Evaluate useful edge selection
with visible relationship information. Do not solve these observed defects by
adding source versions, workspace snapshots, or extra controller classes.

The cohort contains short synthetic development records and curated edges. It
does not measure large-source memory behavior, learned maintenance, distributed
storage, ColBERT retrieval, general forgetting, or independent benchmark quality.
Semantic judgments are assistant reviews, not blinded human evaluation or a paid
external judge. The reviewer also authored the fixed cohort. Lexical agreement
with those judgments does not turn substring checks into an entailment metric.

## Artifact map

Paths are relative to `runs/node-pipeline-20260911/` and ignored by Git.

| Path | Meaning |
| --- | --- |
| `preflight.json` | Account model access and local image identity, zero generation calls |
| `live-v1/`, `live-v2/` | Separate complete frozen runs, model requests/responses, predictions, usage and traces |
| `aggregate-v1.json`, `aggregate-v2.json` | Status, usage and latency summaries |
| `semantic-review.json`, `semantic-review-v2.json` | Separate per-answer semantic and canonical citation judgments |
| `recursion-review.json`, `recursion-review-v2.json` | Invocation/parent evidence paths and exact trace anchors |
| `artifact-review.json`, `artifact-review-v2.json` | Schedule, provider accounting, source and citation integrity audits |
| `post-prompt-check.log` | Local checks after the prompt edit |
| `child-return-inputs/`, `followup-child-return.json`, `live-child-return-v1/` | Declared targeted follow-up, frozen inputs and both observed outcomes |
| `semantic-review-child-return.json` | Separate semantic judgment of the targeted follow-up |
| `artifact-review-child-return.json` | Targeted request/usage/status and frozen-input integrity snapshot |
| `final-check.log`, `final-docs.log`, `final-validation.json` | Final local validation and cross-run accounting summary |

Run reconstruction and interpretation commands are in
[the experiment instructions](../../experiments/node-pipeline.md).

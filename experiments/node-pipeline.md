# Full node-pipeline diagnostic

This fixed synthetic development cohort compares `LLGM.answer()` with one
initial search followed by one root synthesis call. Both arms use the same
workspace, initial retrieval limit, effective journal reads, root model, and
declared evidence/context/output limits. LLGM can spend additional sidecar calls
and follow primary edges. Report observed usage rather than claiming equal
compute or benchmark superiority.

The 15 cases cover exact facts, two and three retrieved seeds, one and two graph
hops, irrelevant edges, scoped amendments, overwrite ordering, validity bounds,
suggestions, contradictions, missing facts, negation and entity distinctions.
Primary edges are curated and automatic maintenance is disabled. This cohort
does not evaluate learned edge quality. Expected answers and node labels stay
outside model prompts. Values are synthetic and carry no user secrets.

## Prepare without hosted calls

```sh
python -m llgm.evaluation.node_pipeline \
  --cases experiments/node_pipeline_cases_v1.json \
  --config experiments/node_pipeline_v1.json \
  --output runs/node-pipeline-preparation
```

This validates source spans, builds local workspaces/indexes, and freezes inputs
and code. It requires neither credentials nor Docker. Output must be new or
empty. A prepared run has zero attempted trials and must not be reported as a
model result.

## Execute the fixed trials

Export `OPENAI_API_KEY` in the process that launches the runner, or supply an
explicit local file with `--env-file /path/to/credentials.env`. Existing process
values take precedence. Credential values are never copied into frozen inputs.
Do not place credential values in configuration JSON. The checked-in configuration uses
`gpt-6-astra` for root synthesis and `gpt-5.6-terra` for node delegates, retaining
the pair from earlier repository checks. Account access must be available.
The adapters use native structured outputs through the Responses API, following
the [provider contract](https://developers.openai.com/api/docs/guides/structured-outputs).

Docker must be running with the configured local image. The configuration pins
the image verified in this checkout. On another machine, create a separate
configuration naming an available trusted image. No image is downloaded by the
runtime. Its 120-second Python-step deadline includes recursive child work.

```sh
python -m llgm.evaluation.node_pipeline \
  --cases experiments/node_pipeline_cases_v1.json \
  --config experiments/node_pipeline_v1.json \
  --output runs/node-pipeline-live \
  --execute
```

Each of 30 scheduled trials runs once. A seeded arm order is stored before
dispatch. The configuration limits each LLGM answer to 16 model calls and 180
seconds, the complete cohort to 255 calls and 30 minutes. Cleanup may extend the
wall deadline. These are execution limits, not a provider-enforced dollar cap.
No automatic retry or paid judging is performed. Changed inputs, prompts,
budgets, or models require a new run directory and a separately identified run.

## Interpret the artifacts

`manifest.json` and `frozen/` retain configuration, schedule, cohort/code hashes
and source copies. `workspaces/` contains the actual source IDs and graph records.
`gold/` holds evaluator-only label mappings and expectations.

Append-only `cases.jsonl`, `traces.jsonl`, `usage.jsonl` and `predictions.jsonl`
retain attempts, model inputs/responses, generated Python observations, runtime
events, answers, canonical citation text, failures, usage and timings. Never
insert credential values or evaluator labels into model requests.

The automatic answer check is a lexical diagnostic. Node coverage records which
required sources were cited, not whether a particular passage entails a claim.
Review every answer and cited passage before writing semantic judgments.
An operational `partial` result can correctly report a conflict or missing fact.
Failed and cancelled attempts remain in the attempted denominator, and unstarted
trials are counted separately. Unknown usage and cost remain unknown.

Successful graph answers do not by themselves prove recursion. Inspect
`query_node`, child invocation/parent IDs, child interpreter/model execution,
returned evidence and parent continuation. Direct reading of a discovered
neighbor is also a legal operation and must be distinguished from delegation.
`children_with_findings` counts nonempty selected evidence in admitted child
returns, including a budget-exhausted child preserving findings from its own
completed descendants. It does not count empty returns or certify answer quality.

## Focused regression and mechanism checks

`node_pipeline_cases_focused_v1.json` adds twelve autonomous paired cases and
five controlled LLGM-only cases. `node_pipeline_focused_v1.json` declares 20
sidecar calls and 24 total model calls per answer, with a 420-call, 40-minute
cohort limit. These limits differ from the original comparison. Report the
autonomous and controlled denominators separately.

```sh
python -m llgm.evaluation.node_pipeline \
  --cases experiments/node_pipeline_cases_focused_v1.json \
  --config experiments/node_pipeline_focused_v1.json \
  --output runs/node-pipeline-focused \
  --env-file .env --execute
```

The optional case `arms` field selects unique `llgm`/`flat` arms. Omission keeps
the original paired schedule. A `protocol` requires the LLGM-only arm. Its
instructions are added to the interpreter context before ordinary model context
admission. Interventions are recorded before subsequent model work can fail.

| Protocol | Controlled intervention and required verification |
| --- | --- |
| `guided_recursion` | Prescribes node-local reading and `query_node` navigation without supplying the answer. Verify child execution and cited evidence returning through every ancestor. |
| `child_return_exhaustion` | Uses guided recursion and injects one budget failure only after Python returned with an admitted, completed child finding. Verify that the root receives that finding and the exhaustion gap. |
| `transient_read_error` | Replaces the first read request with an invalid range. Verify a real model repairs the read and the final result has no stale operational gap. |
| `transient_operation_error` | Replaces the first read with an unknown operation. Verify recovery within the declared budget. |
| `relation_order` | Sorts edge descriptions by relation and ID, placing the fixture's incident contact first. Verify selection uses the authorization relation and retains its canonical evidence. |

The protocol wrapper delegates generated Python to Docker and never substitutes
a model response. Controlled checks establish the observed mechanism under the
declared intervention. They do not establish autonomous planning. A correct
answer with no triggered intervention is insufficient for its mechanism gate.
Keep expected mechanism requirements in evaluator files and inspect actual
traces before scoring them.

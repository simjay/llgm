# Non-ColBERT implementation and validation

September 10, 2026. This study covered mechanism tests and a small real-data pilot, excluding ColBERT/PLAID execution. This report separates deterministic contracts, live protocol checks and semantic measurements. It makes no benchmark superiority claim.

This is a dated run report. [Concepts and architecture](../../docs/guide/concepts.md) explains the current execution paths. [Implementation status](../../docs/reference/implementation-status.md) tracks current capability and [Coverage](coverage-2026-09-10.md) records subsequent deliberate test measurements. Later code fixes do not rerun or replace the live results below.

September 11 input-audit note: the [node-search study](node-search-2026-09-11.md)
identified support-label cues in original LongMemEval session identifiers. The
original-ID benchmark inputs used by earlier pilots need an identifier-neutral
replication before supporting label-neutral retrieval or answer-quality claims.
The measurements below remain unchanged historical diagnostics.

## Implementation delivered

`MemoryApplication` connects source ingestion, bounded model-assisted link maintenance, fresh evidence snapshots and structured recursive answering. Maintenance policies allow validated publication, proposals or disabled organization. Source publication survives maintenance failure. Provenance, partial publications and usage remain recorded. The settings factory owns its workspace and hosted clients. Direct construction leaves ownership with the caller. The default metadata/retrieval path is SQLite with SQLite FTS5.

`RLMRuntime` connects hosted model decisions to actual Docker Python execution and recursive `llm_query` callbacks. Each child receives only the selected prompt in a fresh interpreter. Persistent variables remain local to a frame. Native structured output is used when the adapter supports it. Child startup instructions explicitly identify the already-running interpreter so the smaller model can inspect external context. This is LLGM's declared Python RLM protocol, not an exact reproduction of the published RLM implementation.

The OpenAI adapter now distinguishes intermediate commentary from completed final-answer messages and preserves the phase when replaying normalized assistant text. Concatenating those messages could create multiple executable JSON operations in one response. Unknown phases or missing final output fail explicitly. Legacy unphased responses remain supported. The provider documents the significance of these phases in its [message-phase guidance](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.5).

Query scope/date reach recursive frames. Snapshot preparation is inside the application's answer deadline. Maintenance charges unique exposed passages before model dispatch. Cancellation during Python cleanup retains container ownership until cleanup completes. A child cleanup failure is run-fatal even when the callback transport reports it as an ordinary Python observation. Native schema content is included in context admission before dispatch and, for RLM, before container startup. CLI and experiment tests cover invalid matrices, prepared-data integrity, explicit backend scope, cancellation, failure denominators and physical embedding accounting. Standalone tokenizer pins no longer require pretending that a tokenizer revision is the ColBERT encoder's code revision.

## Deterministic and local-data checks

The deterministic measurement for this report recorded **350 passed, 151 subtests passed, two installed-package checks skipped and 19 integration tests deselected**. Separate local LongMemEval tests recorded **five passed**. The [coverage report](coverage-2026-09-10.md) records subsequent measurements with their own full-package line/branch denominators. Those denominators do not describe this 350-test checkpoint. External models, Docker guests and native services are not part of those coverage percentages. This report does not identify a complete source/test archive for the deterministic checkpoint. The pilot source archive described below preserves its separate iterative runner.

Ruff and the repository docstring checker passed. The docstring audit found 1,071 documented definitions across 62 Python files. Presence is not a measure of writing quality. The nested-child cleanup regression and native-schema context admission checks pass. The strict Sphinx build and wheel/sdist metadata checks pass. Fresh Python 3.11 wheel and source-distribution installations each pass 350 tests and 151 subtests outside the checkout. Their two skips are optional SDK signature checks because those clean core environments omit provider SDKs. The development measurement includes both SDKs, and its two skips are the installed-distribution probes. GitHub CI was not dispatched.

A subsequent documentation review found that the successful Sphinx command had rendered raw autodoc directives and an empty API index. The API parsing was corrected, and the documentation check now verifies rendered symbol anchors, index links and portable source links in addition to build success. This correction does not alter the live measurements below.

## Hosted mechanism checks

| Check | Observed outcome | Scope |
|---|---|---|
| OpenAI native JSON schema | Passed | `gpt-6-astra`. One schema request |
| Hosted embeddings | Passed | `text-embedding-3-large`. Three strings in two physical requests |
| Docker contracts | Four passed | Real isolation, persistent variables, callbacks, output limits and cleanup |
| Structured recursive protocol | Passed after two retained malformed-response failures | Full LongMemEval `001be529`. Oracle whole-source handles. Prescribed root → child → grandchild → parent continuation |
| Official judge on recursive answer | 1/1 accepted | Predeclared diagnostic threshold 1.0 |
| Python RLM protocol | Passed after three retained failures | 72 seeded external records. Real child inspection. Correct exact token returned through parent |
| Autonomous memory application | Passed | Full LongMemEval `001be529`. Original question, no oracle handles or forced delegation |
| Controlled memory application | Passed | Source replacement, dated production/staging corrections, distractors, restart, three predeclared answers and journal citations |
| Official judge on application answer | 1/1 accepted | Predeclared diagnostic threshold 1.0 |

The root used `gpt-6-astra`. Recursive/maintenance/child roles used `gpt-5.6-terra`. Provider IDs, response metadata, usage and attempted operations are saved. The application answer required three root calls and one search, with no sidecar delegation. Its pass does not show that recursion helped. Maintenance over the full-history fixture was deliberately limited to the first two occurrences. The remaining history was ingested without organization.

The controlled application run also exposed a semantic maintenance error: the smaller model published an `orion-service → boreal-service` relationship labeled `contradicts`, although the fixture explicitly describes different services. Structural validation preserved its source support and provenance but did not establish that the relation was true. The three answer checks still passed. This is a concrete reason to evaluate link precision and entity/scope confusion separately from successful application execution.

The final Python RLM verification, after cleanup and schema-budget fixes, passed with seven model calls, three child calls, five Python executions and one recursive invocation. It reported 5,847 input and 361 output tokens over 23.46 runtime seconds. An earlier successful verification used 5,852 input and 365 output tokens over 21.29 seconds. These are one-task observations. Prior runs included one malformed response and two completed protocols returning an incorrect missing-context answer. Completion status is therefore distinct from semantic success.

Docker server version was 29.7.2. The trusted image was `python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea`, inspected locally as `sha256:ec7d6c95cd3692a2e2d228a8b1ca74e4025b54121fcc4c5da6f09cfa473315ad`. An initial isolation test assumed only the loopback interface existed. Docker's network-none environment also exposed inactive tunnel interfaces. The corrected test checks no routes and a guest `ENETUNREACH` result, alongside Docker's network-none configuration, read-only filesystem, non-root user and lack of host mounts.

The documented `examples/recursive_memory.py` also ran through the real settings factory with default budgets and isolated local storage. It published one automatic link, answered the region question correctly with source citations, and used seven inference calls (two sidecar calls) plus one maintenance call. Its artifacts are under `runs/hosted-example/8036274d252546ccbb66f5d63215a75b/`. A final Docker check found no remaining containers with the runtime's `llgm-repl-` prefix.

The official evaluator was a clean checkout of [LongMemEval](https://github.com/xiaowu0162/LongMemEval/tree/9e0b455f4ef0e2ab8f2e582289761153549043fc), revision `9e0b455f4ef0e2ab8f2e582289761153549043fc`, evaluator SHA256 `ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251`. Alias `gpt-4o` resolved to `gpt-4o-2024-08-06`. Its retries and complete token/cost accounting are not exported, so those fields remain unknown. Neither one-case score estimates release-wide accuracy.

## Frozen B/D/H pilot

The [pilot protocol](../../experiments/NON_COLBERT_PILOT.md) and [run configuration](../../experiments/e03_non_colbert_pilot.json) were written before dispatch. The planned cohort contained five then-reserved LongMemEval cases: 15 original-query retrieval diagnostics and 45 iterative answering attempts across B/D/H × S/U/A. The pinned tokenizer supplied identical passage/accounting boundaries. ColBERT weights, model encoding and PLAID indexes were excluded. Full-matrix readiness remained false.

The completed study exposed these five cases. Later reuse is diagnostic
replication, and the remaining cases are not independent held-out histories
under the declared overlap rule. The original configuration's `stage="held-out"`
records the study's initial designation, not a guarantee of present non-exposure.

Retrieval completed all 15 case/backend attempts without failure:

| Backend | Source recall at 40 | Answer-turn recall at 40 | All required source/turn coverage | Mean cold indexing | Mean query | Embedding requests |
|---|---|---|---|---|---|---|
| B: BM25 | 1.00 | 1.00 | 5/5 | 0.073 s | 0.0033 s | 0 |
| D: dense | 1.00 | 1.00 | 5/5 | 22.99 s | 0.292 s | 98 |
| H: hybrid | 1.00 | 1.00 | 5/5 | 22.40 s | 0.358 s | 98 |

D and H each reported 782,736 embedding input tokens. Together they made 196 physical embedding requests. All five cases are answerable. Indexing is cold per case/backend, and D/H query times include hosted query embedding. These timings describe this machine/network/run, not general performance distributions. Partial overlap with an annotated answer turn is not proof that an exact answer span was retrieved. At this cutoff the sample cannot distinguish the retrievers' labeled coverage.

All 45 answering attempts completed, using 129 generation calls and 625 embedding requests. The official judge accepted 43 of 45 hypotheses across repeated evaluations of the same five questions. This is not an estimate over 45 independent questions.

| Arm | Official correct | Initial turn recall | Final retrieval turn recall | Bundle turn recall |
|---|---|---|---|---|
| B-S | 5/5 | 1.00 | 1.00 | 1.00 |
| B-U | 5/5 | 0.80 | 1.00 | 1.00 |
| B-A | 4/5 | 0.80 | 0.90 | 0.90 |
| D-S | 5/5 | 1.00 | 1.00 | 1.00 |
| D-U | 4/5 | 0.90 | 0.90 | 0.90 |
| D-A | 5/5 | 0.90 | 1.00 | 1.00 |
| H-S | 5/5 | 1.00 | 1.00 | 1.00 |
| H-U | 5/5 | 1.00 | 1.00 | 1.00 |
| H-A | 5/5 | 1.00 | 1.00 | 1.00 |

| Arm | Generation calls | Searches | Embedding requests | Mean total case time |
|---|---|---|---|---|
| B-S | 10 | 5 | 0 | 8.42 s |
| B-U | 15 | 20 | 0 | 6.77 s |
| B-A | 18 | 9 | 0 | 8.12 s |
| D-S | 10 | 5 | 98 | 29.56 s |
| D-U | 15 | 20 | 113 | 29.83 s |
| D-A | 18 | 9 | 102 | 31.06 s |
| H-S | 10 | 5 | 98 | 26.75 s |
| H-U | 15 | 20 | 113 | 31.19 s |
| H-A | 18 | 8 | 101 | 30.78 s |

The two rejected answers were B-A and D-U on the same case, `3fdac837` (case identities are retained in the judge summary). Single-query B-S, D-S and H-S each scored 5/5. More searches did not improve this sample's already perfect single-query scores. Differences of one answer in five cannot establish a preferred policy. Exact string match was zero for all arms. The separately invoked semantic evaluator is the relevant answer judgment here.

The 129 generation calls split into 45 root calls and 84 sidecar calls. Answering reported 234,491 generation input tokens, 9,698 generation output tokens and 4,696,945 embedding input tokens. Combined with retrieval diagnostics, the pilot made 821 embedding requests. The answering runner completed in 1012.65 seconds. The retrieval runner completed in 234.89 seconds. Judge time is separate. Every recorded generation/embedding attempt completed with reported usage. Currency cost remains unpriced and official-judge usage remains unknown.

The iterative pilot does not test the new application or Python RLM controller against each other. S starts with 40 original-query hits. U/A start with ten, so their first-retrieval coverage uses a different cutoff. Final-bundle coverage measures cited passages and turns, not whether every fact survives the summary. Answering case elapsed time includes preparation, indexing and cleanup. Its nested usage time measures inference only. Retrieval index timing includes index persistence, whereas answering indexes are not persisted, so those build timers should not be compared directly.

## Artifact map

Local raw artifacts live under ignored `runs/`. They are not packaged as benchmark evidence. Preserve them when reproducing or auditing this report. The pilot archives all 34 Python source modules under `runs/e03-non-colbert-20260910/source/`, verified against the startup manifest hashes. The running pilot retained its original imported code while final RLM cleanup/schema guards were completed separately. Those later guards do not affect the iterative pilot, which uses no native output schema. Frozen configuration, protocol, dependency versions and the pinned judge driver are retained beside the results. Text capture is enabled only for these public/synthetic diagnostics and can include source excerpts and model-generated code. No provider credentials are intentionally persisted.

| Evidence | Local artifact root under `runs/integration/` |
|---|---|
| Provider and embedding checks | `361ebc2a03094bf2a9ca00eb11d38338/` |
| Recursive malformed attempts | `a55a45a829954700afc1796648623f90/`, `378c26e9cd5a4b69b5616bf5cc281d13/` |
| Passing recursive protocol | `e17c7e8e06cf4b2f9405c17945b1e1f9/` |
| Recursive official judgment | `1f256ddea5544418ba1c37c387e4b552/` |
| Python RLM earlier failures | `794a112135854b4ea3b44dadf91700d5/`, `0575885c9a3543a899282ab4eefe68f2/`, `5192ae5e2cb146dba820122f6dd226ae/` |
| Passing Python RLM | `ec6a3cef5a684747a30c8f082e274a5a/`. Final verification `aa0fa6d40b974a7193398c49bfa2e525/` |
| Both memory application checks | `488aaa17e1cb41a9ad60147567877831/` |
| Application official judgment | `8895d289d4fb4b6cbb924cd07d5b9adc/` |
| Separate local dataset coverage run | `45220bc971ed4feeb2324f4929ffb4be/` |

## Interpretation and next research decisions

The core paths now execute with real hosted models, real storage and actual isolated Python. The remaining question is whether the graph, recursive delegation and journal interpretation improve outcomes enough to justify their cost. A fair next experiment needs matched budgets and independent examples, with direct-context, flat retrieval, iterative retrieval, structured recursion and Python RLM controls. Freeze the task cohort and scoring before comparing model allocations or changing prompts.

Broader link quality, semantic forgetting, S3 service behavior, distributed metadata, large-scale index updates, repeated model/provider cohorts, pricing admission and publication remain separate work. Anthropic and arbitrary compatible endpoints were not live-tested in this session. ColBERT/PLAID was deferred. The [implementation status](../../docs/reference/implementation-status.md) records current limitations. The [testing guide](../../development/testing.md) gives validation prerequisites and commands.

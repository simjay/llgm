# Remaining tasks

Agent backlog, reviewed September 11, 2026. Select work from the current user
request. This list does not authorize paid runs, provisioning, or publication.
Task IDs are stable references, not a required execution order.
Research plans and measurements are internal. Experiment instructions belong in
`experiments/` and must not become a public research section in `docs/`.
`docs/` serves library end users only. Contributor setup, test methodology and
publishing instructions belong in repository-only `development/`.
The [single working roadmap](../research/ROADMAP.md) owns the straightforward
benchmark goal, concrete solution suggestions and work order. The earlier
[assessment](../research/reports/remaining-work-2026-09-11.md) remains background.
Task IDs below index supporting work. They do not require completing every item
before the first benchmark comparison.

The user has now authorized execution, prioritizing removal of redundant test
suites and one trustworthy LongMemEval evaluation, with up to $500 API spend.
The canonical runner, shared call accounting, simple controls and focused trust
tests are implemented. Nine retired experiment-only suites were deleted. The
first paid ten-question plumbing run retained LLGM root-synthesis failures.
Root, maintenance-schema and reproduced Docker-removal fixes now pass the final
diagnostics. The user then requested five questions first. The full run was
stopped with all records preserved, and a five-question, three-arm pilot completed
under a separate $5 generation and $0.10 judge cap. It scored LLGM 2/5, BM25 4/5
and full context 3/5, costing $3.139091 including judging with no unknown usage.
LLGM missed facts within admitted nodes and combined dates incorrectly. It made
no recursive child calls. Review these concrete failures before expansion.
The completed five-question repair is recorded below. The
[node-search improvement order](../research/ROADMAP.md#node-search-improvement-order)
prioritizes verifying those changes, locating fact loss, then replaying candidate
fusion and testing useful follow-up discovery. Do not resume the full run automatically. Exact executed source is isolated from
subsequent checkout changes. The
[execution report](../research/reports/longmemeval-2026-09-11.md) owns preparation
measurements and proposed focused repairs. Mem0/Graphiti integration is paused
pending this pilot review and remains required before the named-framework goal
can be evaluated. LoCoMo and
other benchmark suites are deferred.

The user subsequently requested iteration until LLGM gets all five pilot
questions correct, with full test suites deferred until then. Candidate v2
removes the first-passage runnable prompt example and immediate-finish bias,
asks delegates for supported local facts and the root for evidence-checked
comparison. Its frozen LLGM-only run at `five-question-v2/` completed 1/5 for
$1.761923, regressing because models omitted print or invented batch wrappers.
Candidate v3 restores an explicit printed single-reference loop and supplies the
existing bounded turn-metadata page to all seeds. It completed 3/5 at
`five-question-v3/`, costing $1.631284. Reading and temporal arithmetic improved,
but the root chose an older value despite receiving the update. The clothing
count remains sensitive to groups versus individual objects. Candidate v4
clarifies those two root rules but again completed 3/5, for $1.639703 including
judging. Root-only replays show that presentation and interpretation matter after
successful reading. V5 presents attributed exact quotes before summaries and
uses a GPT-5.4 final root with GPT-4.1 delegates and maintenance, reaching 4/5.
V6 then reached 5/5 with explicit medium reasoning and the same prompts at
`runs/longmemeval-20260911/five-question-v6/`. The unchanged official evaluator
scored one fresh attempt per question. The default CLI and Make protocol select
v6. The full deterministic suite passed afterward. Preserve every earlier
failure and replay. The clothing count remains ambiguous between two pickups
and three individual pieces, which the final answer labels explicitly. A third
outstanding errand is not supported by the completed exchange. Previous GPT-4.1
control scores are unmatched to the new root model.

## Next research work

### T05: Measure link and journal interpretation quality

Structural link contracts and conservative scope/time/correction handling exist.
Broader semantic quality is unmeasured. A retained hosted scenario proposed an
Orion → Boreal `contradicts` link even though the fixture described separate
services. Passing answers did not validate that relation.

Build a frozen labeled cohort covering valid/spurious links, entity and scope
collisions, suggestions versus commitments, temporal changes, unsupported
replacements, and disagreement. Measure proposal precision/recall, evidence
support, interpretation accuracy, abstention, and maintenance cost. Retain actual
model failures and keep structural validity separate from semantic truth.

**Completion:** Declared scoring rules and frozen cases produce retained predictions
and failure analysis. Start in `src/llgm/memory/evidence.py`, `src/llgm/llgm.py`,
and `tests/integration/test_live_application.py`. General forgetting remains T10.

### T09: Run architecture ablations and model allocation studies

A frozen three-arm comparison and runner exist. The five controlled questions
compare one-search, recursive-search, and recursive-graph approaches. Historical
v1 readiness made zero model calls. The current
runner requires the independent-edge v3 protocol, which has no hosted measurements.
It remains a structured-operation ablation, not a test of the full Python seed pipeline.
Curated links hold maintenance fixed to isolate navigation. They do not validate
automatic maintenance.

The [comparison instructions](../experiments/comparison.md) retain the 15-trial
structured-operation protocol. The node-pipeline budget and branch-return
hardening below is verified for its declared diagnostic scope. Future runs must
preserve evaluator isolation, retained failures, and matched source/journal access.

Later comparisons should include direct context, flat retrieval, iterative
sidecars, ordinary RLM, and full LLGM under declared comparable budgets. The
allocation study calls for three roots × three sidecars across at least two
providers, plus same-root controls. Anthropic and other model/provider contracts
need live checks when introduced. See [experiment design](../research/design/experiments.md).

A fixed 15-case synthetic cohort and full Python node-pipeline versus one-search
root runner are prepared. See [node-pipeline instructions](../experiments/node-pipeline.md).
The user explicitly authorized API-key testing and supplied a temporary credential
on September 11. Both configured models passed account-access checks. The fixed
trials completed in two retained runs, separating lexical diagnostics, citation
validity and semantic judgment. The [live diagnostic](../research/reports/node-pipeline-live-2026-09-11.md)
records the initial failures and a complete repeat after prompt clarification.
Real nested children ran. The subsequent
[runtime hardening diagnostic](../research/reports/node-runtime-hardening-2026-09-11.md)
addresses the observed failures with branch finalization reservations, local
contribution prompts, attributed required gaps, relationship descriptors, and
recoverable schema/range observations. The original cohort was repeated under
its original budget. A separate focused cohort checks scope/time, negation,
independent contributions, conflict and abstention. Prescribed live diagnostics
verify nested delivery, completed-child preservation after injected parent
exhaustion, error recovery and an irrelevant-first edge choice. Frozen failures,
semantic reviews, canonical citations and accounting audits are retained.

The diagnostic gates are complete. Broad autonomous reliability, superiority,
and cost advantage remain unestablished. Occasional duplicate retrieval and
inefficient recovery are still model behaviors, even in successful runs. Use
independent histories and declared comparable budgets for the next study rather
than claiming generalization from these exposed cases. Empty-text abstention and
several hard failure boundaries have deterministic checks but were not all
individually forced in hosted trials.

The later [seed-answer comparison](../research/reports/seed-answers-2026-09-11.md)
found that a cheaper model pair on larger histories did not preserve the earlier
diagnostic's evidence delivery. A full GPT-4.1 delegate improved valid reads and
citations, but the root still ignored relevant returned facts or omitted their
citations. Models, budgets and history size changed between the earlier and later
studies, so the cause is not isolated to model size. Qualify reading, branch
selection, root use and final claim support separately before broad comparisons.
Two follow-up attempts failed during Docker cleanup confirmation despite eventual
container removal. A later controlled overlap reproduced Docker's removal-already-
in-progress error. Bounded exact-container absence verification now handles that
specific error and passed the repeated reproduction. Genuine daemon/permission
failures remain errors. The historical failures' exact cause remains unproved
because their removal stderr was not retained. The current
[execution report](../research/reports/longmemeval-2026-09-11.md) owns this repair.

**Completion:** Each declared cell has reproducible artifacts separating build,
maintenance, inference, and judge work. Report negative results and uncertainty. Shared histories are not independent experimental groups.

### T08: Expand independent development and evaluation data

Deferred under the current five-question LongMemEval scope. The work below is a
later generalization proposal, not the next execution task.

The B/D/H pilot and subsequent node-search diagnostic are complete. The latter
adds 24 LongMemEval-S questions with corrected opaque IDs and eight controlled
histories with 96 sources each. All 500 LongMemEval-S cases share one connected
history component under the current isolation rule. These additional questions
do not establish an independent held-out split. The controlled histories expose
crowding, paraphrase and scope failures, but use shared templates and deliberately
repetitive distractors. The [node-search report](../research/reports/node-search-2026-09-11.md)
owns these measurements and limitations.

**Next step:** Build independent development histories with enough distractors,
then freeze cohort selection, model/prompt/tokenizer pins, scoring, and measurement
conditions before new reserved evaluation. Declare cold build versus warm query
costs. The current E03 runner rebuilds per case/arm.

**Completion:** Artifacts retain all attempts and report retrieval cutoffs, admitted
and final evidence coverage, novel discoveries, duplicate retrieval, answer quality,
failures, usage, and latency. Start with [experiment instructions](../experiments/README.md).
The [completed pilot report](../research/reports/live-validation-2026-09-10.md)
owns earlier results. Reusing those IDs is diagnostic replication.

## Implementation and deferred choices

T14 and T15 from the [architecture update plan](../research/design/update-plan.md)
are implemented and locally verified. The
[delivery report](../research/reports/node-delegates-2026-09-11.md) records the
regression, dataset, Docker, documentation and installed-distribution checks.
The subsequent [code cleanup](../research/reports/code-cleanup-2026-09-11.md)
removes unused factories/settings and consolidates shared validation and resource
handling without changing the node/edge/journal design.

The delivered default is retrieval-first parallel Python node delegates followed
by one final root. Independent primary edges, eager operational journals,
canonical effective reads, conservative compaction and explicit schema-2 copying
are implemented. No source materialization or history deletion policy was added.
Hosted semantic evaluation and distributed storage/scaling remain separate work.

### T06: Add price accounting and budget admission

Call/token/time limits exist. Product-wide dollar admission does not. A non-null experiment
`currency_cap` is rejected. Add a dated price basis, estimated admission bounds,
and reconciliation for generation, embeddings, maintenance, indexing, and judging.
Keep missing usage/cost unknown and define cancellation/in-flight semantics.
Shared `evaluation/costs.py` now reserves priced calls for the primary LongMemEval
runner, including preparation, answering and judging. It retains unknown
liabilities after timeouts/cancellation and reconciles returned usage. Historical
seed-answer tools have separate local controls. External-framework physical call
instrumentation and product-wide admission remain incomplete.

**Completion:** Observable contracts cover exhausted budgets, missing prices,
unknown usage, and partial failures. Reports label measured versus estimated
costs without promising provider-side billing guarantees. Start in
`src/llgm/inference/budget.py`, `src/llgm/memory/maintenance.py`, and
`src/llgm/evaluation/`.

### T07: Actual ColBERTv2/PLAID workflow

**Completed for the initial live diagnostic, September 11.** The authenticated
Modal A10 workflow has passed full-checkpoint build/search, exact passage mapping,
separate-container persisted reopen and local library retrieval through real SDK
calls. The three-history BM25 comparison also passed. Both arms saturated labeled
coverage at top 40. The [dated report](../research/reports/colbert-modal-2026-09-11.md)
owns measurements, pins, failures and limitations. Commands remain in the
[ColBERT workflow](../experiments/colbert.md).

The user approved the scoped upload, persistent Volume and initial testing within
a $20 paid-spend limit. The earlier automatic-review upload rejection was resolved.
GPU jobs stop after completion. Checkpoint, indexes and artifacts remain on the
Volume. Actual billed charges are not available in the retained run records.

The subsequent four-arm node-search experiment is also complete. Both retrievers
ran through the actual application seed path at 12 and 40 passages with a fixed
three-node cap. The original 35-history audit found label-correlated session IDs
in benchmark passage headers. The corrected 24-question repeat used opaque IDs
and retained the original run and eight already-opaque controlled probes.
The [node-search report](../research/reports/node-search-2026-09-11.md) owns results.

The user authorized the next selector experiment with complementary BM25 and
ColBERT pools. The experimental helpers, frozen protocol and runner are implemented.
The hosted run completed with all planned repetitions and no failed calls. It
compared each top-40 pool and fused top 40 with matched first-owner controls at
three nodes. The [selection report](../research/reports/node-selection-2026-09-11.md)
owns measurements and semantic review. The selector improved controlled cases
but reduced complete annotated-source coverage on LongMemEval. Gold source sets
can contain redundant or older evidence, so these are not answer-accuracy scores.

The authorized final-answer comparison is complete with all 192 planned answer
attempts and a separate eight-case stronger-delegate qualification. The
[answer report](../research/reports/seed-answers-2026-09-11.md) retains official-prompt
accuracy, blinded citation support, all failures, latency and priced usage.
Evidence-delivery failures left no dependable selector winner. Product defaults
and both retriever controls remain unchanged.

**Development gate complete:** The final five-question candidate uses printed
local reads, source-first root evidence and GPT-5.4 medium reasoning to reach
5/5. Preserve the limits of this exposed result and the unmatched older controls.
A later comparison should use the same final reader and reasoning effort before
attributing gains to retrieval or graph architecture. Then replay balanced or
node-level BM25/ColBERT fusion under fixed evidence limits before a matched answer
test. The product's lexical/dense hybrid is not turnkey BM25/Modal-ColBERT
composition. Do not infer selector equivalence from failed evidence delivery or
tune only for complete source-label recall. Independent data under T08 remains
deferred. Keep benchmark handles opaque and the full run stopped.

Declare separate protocols for query length 32 versus 128 and exact ColBERT
scoring versus PLAID. Cold-start optimization is deferred by the user during
research. Before claiming interactive cold performance, reduce native-extension
compilation at container startup, then measure warm remote requests
and concurrency. The private deployed `connect()` example and manual GitHub CI
are prepared but were not executed. The verified local client used the ephemeral
test app's real RPC handles. These follow-ups are not additional spending authority.

### T10: Decide what forgetting means

The user withdrew “merge on write.” Immutable history and append-only journals
are implemented. An effective forgetting policy is not established.

**Completion:** A decision record chooses a forgetting objective and defines alternatives,
measurable tradeoffs, and falsification cases before implementation. Candidate
objectives include suppressing stale evidence, reducing query cost, compacting
derived state, or physically deleting information. Preserve scope/time provenance as the objective requires.

### T11: Validate storage and scaling contracts

T13 implemented immutable unique nodes, current journal reads, and ordinary
incremental FTS5. Historical-ranking/snapshot requirements are retired.

Incremental local indexing, restart reuse, and rollback/cancellation contracts
exist. History grows on disk, and individual reads can be large. Live S3 validation, distributed metadata and a complete restore contract remain
open. Conservative operational compaction and explicit schema-2 local conversion
are implemented. Physical history retention and source materialization remain open.

**Next step:** Run the existing S3 contract only with a dedicated configured prefix. Prioritize other work from measured bottlenecks. Measure the bounded local concurrency and Docker overhead before claiming
parallel scaling or distributed performance.

**Completion:** New adapters preserve references, the declared read semantics, idempotency, recovery,
and atomic publication under failure/concurrency. Measure growing histories,
distractors, updates, memory, and throughput. S3 blobs alone do not make SQLite
distributed. See the [architecture report](../research/reports/architecture-2026-09-10.md).

### T12: Validate release delivery and publish when authorized

The documentation, Makefile, distribution build, CI and hosting configurations
exist. `.github/workflows/docs.yml` builds and uploads the audited user site on
`main`. Its deployment job requires `DOCS_PUBLISH_ENABLED=true`. The repository
is private. Automatic approval review rejected enabling public Pages without
explicit approval for `https://simjay.github.io/llgm/`. Push and build are
authorized. Publication awaits that approval. The package remains unpublished.

**Next step:** Validate CI on the declared Python versions. After explicit
publication approval, enable Actions as the repository's Pages source, set the
publishing variable, run the workflow, and verify the public site. Connecting
the repository to Read the Docs is an alternative hosting route and requires
Business hosting while the repository is private. Before a PyPI release,
verify package ownership, metadata, README asset URLs, and reproducibility from an
installed distribution. Follow [publishing](../development/publishing.md).

**Completion:** Released artifacts reproduce supported workflows, and public
capability claims match completed verification. Research stays internal.
Publication requires its own authorization.

## Completed foundation

T01 (initial OpenAI/recursive checks), T02 (Docker and model-driven Python RLM),
T03 (targeted deterministic contracts), and T04 (integrated memory baseline) are
complete for their recorded scope. The small T08 pilot is also complete. T13
(unified LLGM API, organized packages, immutable nodes, explicit journal
overwrites, and node-targeted queries) is implemented with local verification. See the [refactor report](../research/reports/simplification-2026-09-10.md).
T14 (retrieval-first Python delegates and final synthesis) and T15 (independent
primary edges and operational journals) are complete for their locally verified
scope. See the [delivery report](../research/reports/node-delegates-2026-09-11.md).
Earlier hosted runs do not validate the changed runtime or reference format.
[Research reports](../research/reports/README.md) own their evidence and dates. Future changes need relevant regression checks, not repeated completion narratives.

When work closes, update its status and prerequisite here and put its measurements
in the appropriate dated report. Do not mark a gated dependency validated because
its test was skipped, or let a missing external prerequisite imply missing code.

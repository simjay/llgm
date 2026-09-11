# Remaining work and limits after runtime hardening

Assessment date: September 11, 2026. This is a repository review and proposed
work order. It records no new model runs, performance measurements or production
changes. Priorities below are recommendations, not commitments to every research
experiment or infrastructure feature.

LLGM now has a working retrieval-first node runtime, independent primary edges,
eager operational journals, canonical effective reads, recursive child returns,
and bounded final synthesis. The remaining work concerns semantic reliability,
evaluation quality, predictable resource use and long-lived storage. Another
large architecture reorganization is not supported by the current findings.

## What the latest results establish

The [runtime hardening report](node-runtime-hardening-2026-09-11.md) records 318
hosted generation calls across 89 retained trials. The final original cohort
passes 15/15 LLGM answers, the focused autonomous cohort passes 12/12, and five
controlled mechanism cases pass separately. All 135 returned citations match
their frozen source spans. The deterministic suite passes 776 tests and 230
subtests, and strict documentation checks pass.

Those results establish the observed cases and mechanisms. They do not establish
general accuracy, optimal graph traversal, learned graph quality or an economic
advantage. The cases are small development inputs. The coding agent reviewed
their semantics after execution, without blinding or an independent judge. The
five controlled cases prescribe traversal or inject faults. Two packaging tests
were skipped in the editable environment and integration tests were excluded
from the deterministic command, with hosted execution established separately.

The [ColBERT report](colbert-modal-2026-09-11.md) separately establishes actual GPU
indexing, persisted reopen in a fresh container and authenticated local retrieval.
That is completed work. It does not establish final LLGM answer quality with
ColBERT or superiority over BM25.

## Priorities

P1 means the next development or evaluation work. P2 means a gate for a specified
usage goal, such as large sources, long-lived workspaces or external release.
P3 means a later expansion whose value needs evidence first.

| Item | Priority | Kind | Backlog connection |
| --- | --- | --- | --- |
| Semantic graph and amendment proposal quality | P1 | Known historical failure, current quality unmeasured | T05 |
| Preservation of meaning in branch messages | P1 | Observed intermediate error, broader reliability unmeasured | T09 / E06 |
| Independent data and credible judgments | P1 | Evaluation gap | T08 |
| Fair architecture comparisons | P1, after independent data | Evaluation and runner work | T09 |
| Cost accounting and spend admission | P1 before larger paid sweeps | Missing implementation | T06 |
| Runtime efficiency and stressed execution | P1 measurement, P2 targeted changes | Residual model behavior and validation gap | T09, T11 |
| Journal growth and a working representation | P2 for long-lived workspaces | Storage limit and design decision | T10, T11 |
| Large-source host memory and I/O | P2 for large-node use | Current implementation limit | T11 |
| ColBERT end-to-end integration and cold performance | P2 when this backend is required | Integration and performance gap | T07, T09 |
| Backup/restore and optional S3 operation | P2 for durable service use | Recovery contract and live validation gap | T11 |
| Other providers and additional hosted failure cases | P2 when supported or deployed | Validation gap | T09 |
| Fresh installation, CI and contributor onboarding | P2 before external release | Delivery and usability verification | T12 |
| Multi-provider allocation grid and distributed metadata | P3 | Research or product expansion | T09, T11 |

## 1. Semantic graph and amendment proposal quality

**Current state.** Optional ingestion maintenance can propose and publish
primary edges. Its default `validated` mode checks structural support and an
allowed relation list. It does not prove that the relation is true.
`propose` and `disabled` modes already exist. Ordinary maintenance publishes
primary edges, not automatic semantic journal corrections.

The [earlier live report](live-validation-2026-09-10.md) retained an example in
which the model linked Orion and Boreal with `contradicts`, although they were
different services. Answer checks still passed. This is a historical semantic
failure, not a newly reproduced failure of the final runtime revision. It
remains relevant because the latest inference runs used curated edges and
patches and did not measure maintenance quality.

**Next work.** Create labeled positive and negative relationship cases covering
entity identity, scope, dates, dependency, support, contradiction and unrelated
records. Separately evaluate any model that proposes a journal overwrite.
Patch interpretation can be correct even when a proposed patch is unjustified.
Begin with the existing proposal interface and inspect failures before adding
another judging model or complex acceptance mechanism.

**Completion evidence.** Report relation precision/recall, unsupported publication
rate, false overwrite rate, abstention, downstream answer effects and maintenance
cost on frozen independent data. Declare acceptance criteria before evaluation.

**Owners.** `memory/evidence.py`, `memory/maintenance.py`, `llgm.py`, and the
maintenance/application tests.

## 2. Preservation of meaning in branch messages

**Current state.** Typed references and citation admission prove which text was
available. They cannot prove that a delegate interpreted it correctly. In the
focused Bluebell run, one local note called the destination code a recipient
code. The final root assembled the correct answer from the separate sources.
Other children expressed local uncertainty that a parent could resolve using
the relationship context.

**Why it matters.** A condensed child message can omit a negation, scope,
qualification or distinction between a source claim and a deduction. Valid
citations do not automatically reveal that loss. Passing a final short-code
answer does not measure fidelity across longer recursive chains.

**Next work.** Evaluate multi-fact messages, negation, conflicting authority,
numeric precision, temporal qualifiers and evidence needed from more than one
child. Compare exact excerpts with current selected findings while holding
retrieval and models fixed. Inspect intermediate messages as well as the root.

**Completion evidence.** Measure supported-claim rate, qualifier loss, wrong
attribution, child-to-parent evidence retention, final correctness and cost.
Add targeted regressions for demonstrated losses. Avoid adding a new message
format or model escalation rule without an observed benefit.

**Owners.** `inference/nodes.py`, `inference/results.py`, evaluation cases and
the E06 message/allocation protocol.

## 3. Independent data and credible judgments

**Current state.** The synthetic cases are exposed development fixtures. The
original cohort was reused to evaluate fixes. The final semantic review was
performed by the same coding agent after seeing the results. This is useful
debugging evidence, not an independent evaluation.

Under the repository's chosen session-ID/text-overlap rule, all 500
LongMemEval-S questions form one connected history component. Holding out other
question IDs does not establish independent held-out histories. Earlier top-40
retrieval diagnostics also saturated source/turn coverage, leaving little room
to distinguish retrievers. Turn coverage is not answer-span coverage.

**Next work.** Build genuinely separate development and evaluation histories
with realistic distractors, ambiguous entities, corrections and questions needing
multiple records. Freeze histories, prompts, models, scoring, repeats and budgets
before reserved evaluation. Use independent semantic assessment and review judge
disagreements instead of relying only on keyword matches.

**Completion evidence.** Publish internal split identities and overlap checks,
retain all failures, report uncertainty and distinguish source coverage,
answer support, semantic correctness and operational completion. If related
histories are used, acknowledge their dependence in the analysis.

**Owners.** T08, `evaluation/prepare.py`, `evaluation/node_pipeline.py`,
`evaluation/scoring.py`, and frozen experiment inputs.

## 4. Fair architecture comparisons

**Current state.** The final original LLGM arm used 83 model calls versus 15
for the one-search flat arm. LLGM reached 15/15 correct answers and flat reached
12/15, with the flat misses concentrated in deliberately graph-only credentials.
This does not establish better quality at comparable cost.

The structured-operation comparison v3 is prepared but has no hosted results.
It is a different executor protocol from the full Python node pipeline. The
broader model-allocation grid and full architecture controls are planned, not
implemented as a complete multi-pair experiment runner.

**Next work.** Start with one model pair and a small declared comparison:
direct context where it fits, static retrieval, iterative retrieval, an ordinary
RLM reference, and full LLGM. Hold source and journal access comparable. Then
disable graph traversal, recursive delegation or journal interpretation one at
a time to identify their contributions. Keep automatic maintenance separate
from fixed-graph inference comparisons.

Run several budget points and compare accuracy, cost, latency and failures.
An ordinary RLM control must declare its reference semantics. LLGM's
RLM-inspired protocol is not automatically an exact reproduction of that
reference implementation.

**Completion evidence.** Retained paired results and quality/cost curves show
where each mechanism helps, fails or adds overhead. The proposed three-root by
three-sidecar grid across providers is a later robustness study, not a
prerequisite for the first useful comparison.

**Owners.** T09, `evaluation/comparison.py`, `evaluation/node_pipeline.py`,
and `research/design/experiments.md`.

## 5. Cost accounting and spend admission

**Current state.** The runtime enforces call, context, evidence, operation, depth
and time allowances. It records reported usage. It has no dollar admission
mechanism, and retained currency costs remain unknown. Inference and maintenance
have separate budgets. A query call cap does not bound lifetime ingestion cost
or all remote work.

The default admission counter uses UTF-8 bytes as conservative accounting units.
It is explicitly not the provider tokenizer. Serialized provider framing is not
fully modeled, and observed provider tokens remain a separate measurement.

**Next work.** Add a dated price basis and estimates for generation, embeddings,
maintenance and optional GPU/index work. Reconcile actual reported usage with
estimates, including caching and any unreported usage. Define how concurrent or
already-dispatched work affects admission. Preserve unknowns instead of treating
them as zero. Explain admission units consistently in the user-facing settings.

**Completion evidence.** Known prices produce reproducible estimates, missing
prices/usage stay explicit, planned work is denied when it would exceed the
declared estimated allowance, and in-flight overshoot limitations are documented.
This still cannot promise a provider-side invoice ceiling.

**Owners.** T06, `inference/budget.py`, `memory/maintenance.py`, and evaluation
usage aggregation.

## 6. Runtime efficiency and stressed execution

**Current state.** Finalization reservations address observed starvation, but
models can still search twice, reread evidence or use too many repair steps.
The two controlled recovery cases consumed eight and seven total calls.
An autonomous two-hop case used direct remote reads rather than child delegation.
Both mechanisms are currently allowed.

The tests exercise small graphs and shallow chains. Large fanout, many overlapping
seed branches, tight budgets and realistic load have not been evaluated broadly.
The edge API returns a complete applicable descriptor list without pagination.
Large results must fit the existing callback allowance or fail. Whole-node reads
and replacement spans can likewise exceed admission limits. Model-call reserves
do not independently reserve every callback operation or guarantee cleanup fits
the deadline.

**Next work.** Measure repeated queries, duplicate evidence, useful findings per
call, time to first finding, end-to-end latency, Docker startup and concurrency.
Stress graph depth, degree, overlap, cancellation and resource exhaustion.
Optimize the measured cause. If high-degree nodes cause failures, add a small
paging or selection interface. If repeated work is material, evaluate limited
query-local reuse before designing a global cache or coordinator.

**Completion evidence.** Declared stress cases retain useful partial results,
respect observed resource contracts, leave no owned containers running and show
measured efficiency changes without losing answer support.

**Owners.** `inference/nodes.py`, `inference/repl.py`, `memory/evidence.py`,
`memory/query.py`, and scaling/evaluation tools.

## 7. Journal growth and the working representation

**Current state.** Operational compaction removes redundant superseded entries
while preserving raw history and canonical references. Different scopes, validity
intervals and correction dependencies can legitimately remain. The default
complete-journal allowance is 65,536 bytes. An irreducible journal exceeding its
allowance fails explicitly. Raw history still grows on disk.

There is no materialized working source copy and no physical history-retention
policy. Thus, the journal-sidecar strategy works for the tested bounded cases,
but it is not yet a demonstrated indefinite-growth strategy.

**Next work.** First measure repeated overwrites, many distinct patches, scoped
variants and corrections over long histories. If the active journal becomes
too large, design the smallest derived working representation that folds applied
patches while retaining a mapping to original canonical spans. The read result
must remain explainable after compaction and restart. Decide separately whether
old history remains online, moves to archival storage or can ever be deleted.

Forgetting also needs a concrete objective. Suppressing stale evidence in an
answer, reducing active state and physically deleting evidence are different
contracts. The earlier retraction of merge-on-write remains in effect.

**Completion evidence.** Growth tests distinguish active bytes, retained-history
bytes and read cost. A working representation, if adopted, preserves effective
answers, scope/time semantics and old references. Any deletion policy requires
its own explicit decision.

**Owners.** T10/T11, `memory/workspace.py`, `memory/query.py`, and journal
compaction tests.

## 8. Large-source host memory and I/O

**Current state.** The model and interpreter see selected spans and paginated
metadata. The host still fetches and JSON-decodes the entire owning source blob.
Even `source_info()` loads the source on the host before returning a small page.
Lazy model exposure therefore does not imply bounded host memory or small
physical reads for a very large source.

**Next work.** Measure resident memory, decoded bytes and latency for large nodes,
not only many small nodes. If the target workload needs larger records, use an
indexed/chunked physical layout that resolves stable turn/span references without
loading the full blob. Keep public immutable node identities and reference
semantics unchanged.

**Completion evidence.** A bounded span/metadata request has a documented bounded
I/O and memory cost as source size grows. Unicode offsets, original resolution,
patches and restart remain correct.

**Owners.** T11, `memory/workspace.py`, `storage/blobs.py`, and
`memory/query.py`.

## 9. ColBERT integration and performance

**Current state.** Official ColBERTv2/PLAID genuinely ran on a GPU. Persisted
reopen and authenticated local calls passed. Existing `evidence_factory` and
`Evidence.open(..., retriever=...)` support attaching a source retriever to
LLGM, with local journal retrieval retained. This is an existing integration
boundary, not a missing abstract interface.

The completed retrieval study did not run the complete ColBERT-to-LLGM answering
pipeline. The persistent deployment and its `connect()` lookup path were not
executed. The three-case top-40 comparison found no coverage gain over BM25.
Its approximately 96-second local client call included cold startup, while
server-only warm searches took roughly 17–19 ms. These are different timing
scopes. Compilation and loading need attention before claiming interactive cold
performance. The protocol used query length 128 and did not compare 32 or exact
ColBERT scoring against PLAID.

**Next work.** When ColBERT is required, run the complete injected-retriever
answer path, including journal amendments and canonical citations. Measure warm
remote latency separately from cold deployment/opening, reduce measured startup
cost and test the deployed lookup path. Use harder queries and lower cutoffs.
Define corpus/index refresh behavior when new sources are ingested, since a
persisted remote index does not automatically gain new source passages.

**Completion evidence.** End-to-end answers retain correct source mapping and
journal semantics, index freshness is explicit, and the selected retrieval
configuration has measured quality/latency/cost tradeoffs on independent data.

**Owners.** T07/T09, `retrieval/modal.py`, `retrieval/colbert.py`,
`examples/modal_retrieval.py`, and ColBERT experiment tools.

## 10. Persistence, restoration and optional S3

**Current state.** Immutable blobs, SQLite metadata, restart and publication
failure contracts exist. A complete supported backup/restore procedure remains
open. After restoring metadata, the current local guidance is to close users
and discard the old derived index before reuse. Arbitrary restore with a stale
index is not an established recovery contract.

An S3 blob adapter and gated service test exist, but live S3 validation is
pending. Metadata remains SQLite. Remote blobs do not make the application a
distributed database. Physical writes already in flight can complete after the
calling wait is cancelled.

**Next work.** Define what a consistent backup contains, how it is restored,
how integrity is checked and when indexes must be rebuilt. Exercise restoration
and interrupted operations. Run S3 tests only when that backend is needed and a
dedicated test prefix is configured. Decide on distributed metadata only from a
concrete multi-host or multiwriter requirement.

**Completion evidence.** A fresh process can restore a captured workspace,
resolve all expected references, rebuild derived state and answer the declared
queries. Optional cloud checks establish actual adapter operation separately.

**Owners.** T11, `memory/workspace.py`, `storage/lexical.py`, `storage/blobs.py`,
and storage integration tests.

## 11. Provider coverage and remaining hosted failure checks

**Current state.** The latest full pipeline uses one OpenAI root/sidecar pair.
Anthropic and compatible-endpoint adapters exist, but their current full-pipeline
behavior is not established by these runs. A compatible API does not guarantee
the same native schema, refusal, phase, usage or truncation behavior.

Deterministic tests cover empty-text abstention, hard context limits, cancellation,
missing evidence and other boundaries. The latest hosted faults specifically
exercise invalid ranges, invalid callbacks and exhaustion after child delivery.
They are not a live test of every failure mode.

**Next work.** For each provider/model added to supported use, run its adapter
contracts and a small complete pipeline cohort. Add selected real-boundary
checks for timeout, truncation and cancellation where they can reveal integration
behavior that scripted tests cannot. Retain deterministic fault injection for
precise invariants. Do not force every storage failure through a paid model.

**Completion evidence.** Supported provider/model combinations have declared
capabilities, accounted calls, correct partial/failure behavior and cleanup.
Additional model allocation studies remain separate from adapter bring-up.

**Owners.** `models/hosted.py`, provider integration tests, `inference/nodes.py`
and `inference/repl.py`.

## 12. Release, documentation and onboarding

**Current state.** Public architecture, concepts, walkthrough, configuration and
test-layer documentation exist. Strict rendering, API links, navigation and the
boundary excluding internal research pass. Local package builds were performed
at their recorded checkpoints. The configured Python 3.11–3.14 CI matrix and
hosted Read the Docs build have not been established by the latest local run.
The package remains unpublished.

Current `git status` also shows that much of this checkout is untracked. Before
external release, the intended source, tests and documentation need a coherent
reviewed source-control checkpoint. This report does not stage or publish them.

**Next work.** Build and test the final wheel/source archive in fresh environments,
run the declared CI matrix, and have a new developer follow only the public
instructions. Their walkthrough should cover installation, ingestion, primary
edges, journal patches, one recursive answer and interpretation of a partial
result. Verify those steps without hidden local artifacts. Resolve distribution
name/metadata and package-index README URLs before publication. Validate the
hosted documentation when hosting is requested.

**Completion evidence.** Fresh installation and public instructions reproduce
supported behavior, package tests run outside the checkout, the declared CI
matrix passes, and public claims match verified capabilities. A successful
Sphinx build alone is not evidence of onboarding clarity.

**Owners.** T12, `pyproject.toml`, CI, `docs/guide/`, `development/`,
examples and package tests.

## Deliberate boundaries that need no automatic expansion

| Boundary | Current contract and implication |
| --- | --- |
| Graphical-model motivation | Local node computation and bounded evidence messages justify the design motivation. Probabilistic factorization, minimality and convergence are not implemented claims. Evaluate locality and message value empirically. |
| Runtime graph | Discovered persistent relationships and recursive invocation traces develop during a query. There is no required upfront materialized component graph or permanent node daemon. |
| Final root | One synthesis call combines returned branches. It has no further tool phase. A root-driven retry/research phase would be a separate design change. |
| Current reads | Later operations can see later appends. `as_of_ms` selects applicability among available records, not an old database snapshot. Cross-node atomic snapshots and public source versions remain outside the chosen design. |
| IDs and time | Nodes have opaque immutable identities and optional Unix-ms source event times. No source import-time field or meaningful-name identity is required. |
| Journal semantics | Primary edges connect the graph. Journals assist effective reads and retain amendment provenance. They remain small eager sidecars, not another large RLM document to explore. |
| Historical formats | Schema-2 to schema-3 conversion is an explicit copy requiring pointer classification. Schema-1 and automatic in-place conversion are unsupported. Add them only for actual data migration needs. |
| Specialized executors | Separate experimental protocols remain useful for comparisons. Their existence does not require merging them into a generic executor framework. |

If current-read consistency causes a concrete application failure, first document
the required behavior and reproduce it. The existence of concurrent appends alone
does not justify restoring the discarded source-version/commit architecture.

## Recommended sequence and decision gates

1. Freeze the current tested checkpoint and build independent development cases
   for graph semantics and multi-hop message fidelity. This directly addresses
   the largest remaining correctness uncertainty.
2. Add usable cost estimates and measure runtime overhead. Freeze a small,
   comparable-budget baseline study on independent data. Run it before expanding
   the planned model grid.
3. Stress long journals, large sources and overlapping/high-degree graphs.
   Implement the smallest storage or runtime change required by measured limits.
4. Validate whichever product integrations are actually required: complete
   ColBERT answering, another provider, S3, and a restore procedure. Keep their
   results distinct from core inference quality.
5. Complete fresh-install, CI and onboarding gates before an external release.
   Publication is a separate action from validation.

For a research claim, independent data, semantic maintenance/message assessment
and fair ablations are the main gates. For large persistent production use, add
cost predictability, journal/source growth and recovery guarantees. A useful
local experimental library does not require building every P3 expansion first.

The internal remaining-tasks document maintains the canonical backlog.
Current public contracts remain in [implementation status](../../docs/reference/implementation-status.md)
and [architecture](../../docs/guide/architecture.md). Historical measurements stay
in their dated reports rather than being strengthened by this assessment.

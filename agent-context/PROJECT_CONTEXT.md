# Project context

Settled direction and constraints for agents. Last reviewed September 11, 2026.
Use [local state](LOCAL_STATE.md) for observed prerequisites and artifact locations,
and [remaining tasks](REMAINING_TASKS.md) for unfinished work.

`research/` is internal. Do not publish it through Sphinx, add a public research
section, or use it as contributor onboarding. Repository experiment protocols
and commands live in [experiments](../experiments/README.md). Public library and
end-user guidance lives in `docs/`. Contributor and maintainer guidance lives in
repository-only `development/`, linked from `CONTRIBUTING.md`.

The user explicitly requires the published documentation to serve library end
users only. Keep repository setup, test methodology, CI, publishing, branding
explorations and documentation maintenance outside `docs/`. User-facing guides
may describe runtime behavior, practical limitations and extension interfaces,
but must not contain internal benchmark iteration reports or maintenance plans.
The site navigation contains the User guide and Reference only. The documentation
audit rejects maintainer links and stale published maintainer artifacts.
User documentation must also read and build without research or agent context.
Concepts and architecture are tutorials with a concrete running example.
Prose, including rendered API docstrings, uses no em dashes or semicolons.
The default documentation check only validates public documentation. The optional
`make docs-links` command audits repository links separately.

## Product and research direction

- Deliver a usable Python library with documentation, then evaluate it rigorously and write a paper. The package remains unpublished. Benchmark superiority is unproven.
- The user approved [the roadmap](../research/ROADMAP.md), emphasized removing useless tests and focusing on one trustworthy LongMemEval evaluation, and authorized up to $500 API spend. The latest instruction is to improve LLGM until all five pilot questions are correct, using focused checks and postponing full test suites until then. Preserve every candidate and score, and never insert benchmark answers into generation. The full run was stopped and must not resume automatically. The default protocol is the five-question v6 LLGM-only development candidate. LongMemEval-S is the sole primary benchmark. Other benchmark expansion is deferred. The usefulness target remains outperforming Mem0 OSS and Graphiti by five percentage points at no higher total cost per question. Initial LLGM/BM25/full-context controls do not establish that framework win. Keep all failed attempts, opaque model-visible source IDs, separate gold, preparation cost and missing-usage liabilities. Current execution uses `evaluation/memory_benchmark.py` and `experiments/longmemeval.md`.
- The persistent evidence graph retains original text and stable passage references. Primary node-to-node edges must define its connectivity independently of journals. Per-node journals are a secondary mechanism for corrections, interpretation changes, and possible forgetting support, and may point to other nodes. Schema-3 workspace metadata stores independent primary edges with provenance, deduplicated publication and explicit withdrawal. Ordinary maintenance publishes edges without journal assertions.
- The complete operational journal must be eagerly loaded as small local read metadata. It is not a large RLM query target. Explicit patches guide lazy reads of large source content. A replacement pointer can resolve directly to a source span, with recursion reserved for further investigation of that source. Deterministic reconciliation removes superseded exact-applicability entries from the operational journal while retaining the raw history and canonical references. Bounded operational state does not imply bounded historical storage. Derived working-base layout and total-history retention remain open, with original references preserved and no deletion authorized.
- Edges represent evidence relationships. Local node computations exchange bounded findings and evidence. This is the graphical-model motivation. Runtime invocation/parent IDs describe the call tree, not a separately maintained factor graph. Do not claim mathematically minimal messages, probabilistic factorization, or convergence.
- A root model delegates context work to smaller sidecars. Recursive calls are an inference mechanism, not a collection of agent personas. Keep generation offloadable to hosted providers.
- The user approved running all admitted seeds with bounded concurrency. The cap queues excess branches. A subset requires an explicit selection/budget policy, and skipped or failed branches remain visible in the outcome. The node runtime uses shared budgets, one citation registry, bounded seed/model-call concurrency and isolated interpreters. Recursive waits do not hold model-call permits. Final synthesis has reserved call, operation, context and time capacity.
- The intended default query pipeline retrieves seed nodes, dispatches a smaller-model delegate at each seed, recursively collects and condenses branch evidence, then gives the findings to the root for final synthesis. The root retains its existing overall reasoning role. Each delegate acts as a local root using the same recursive mechanism. "Node delegate" is the terminology for that role, not another controller type or a permanent service. Runtime evidence graphs can develop as branches follow links. Upfront component partitioning and whole-graph materialization are not required. Default seed retrieval and branch dispatch/collection are implemented in `LLGM` and `inference/nodes.py`. Explicit `node_id` bypasses retrieval and supplies the sole seed.
- Validate mechanisms before broad benchmark sweeps. Scripted clients establish contracts. Live mechanism checks and semantic evaluation establish different things.
- Use the released ColBERTv2 and official PLAID for the reference retrieval arm. A substitute retriever is a different arm. The user authorized the Modal workflow, scoped source upload and persistent Volume with a $20 paid-spend limit. Real GPU build/search, separate-container reopen, local authenticated retrieval and the three-history BM25 comparison passed. These exposed diagnostics establish operation, not superiority or scalability. T07 records follow-up work and the dated report.
- Forgetting remains an open research decision. The user retracted “merge on write”. Do not treat ingestion-time reconciliation as settled.
- The user's Iceberg analogy is limited to an append-only per-node journal sidecar in the spirit of a WAL. Do not infer requirements for source editions, public commits, or historical table snapshots. The revised target uses unique immutable nodes. Journals point to other nodes containing superseding evidence. See the [simplification direction](../research/design/simplification.md).

## Implementation decisions to preserve

The [architecture update plan](../research/design/update-plan.md) records
the approved changes and acceptance checks. Local integration and validation are
complete for the gates recorded in the
[delivery report](../research/reports/node-delegates-2026-09-11.md).
The user authorized execution of the plan after the Python REPL walkthrough.
Implementation proceeds with Python node delegates, an explicitly stated
assumption based on that discussion. The complete pipeline has deterministic,
real Docker and installed-wheel validation.
Do not claim the separate historical Python RLM checks validate the new bridge.
The [full node-pipeline live diagnostic](../research/reports/node-pipeline-live-2026-09-11.md)
now records hosted execution of that bridge. It establishes actual nested
children while exposing autonomous planning and shared-budget limitations.
Completed child-selected findings admitted to a parent survive its later budget
exhaustion within the existing branch/root-context bounds. Unselected local reads
and undelivered child payloads are not promoted into findings.
The subsequent [runtime hardening diagnostic](../research/reports/node-runtime-hardening-2026-09-11.md)
verified autonomous nested returns, separate seed contributions, and recovery
through controlled live callback faults. A prescribed parent-exhaustion test
verified preservation after actual child generation and delivery. Keep controlled
mechanism evidence separate from autonomous answer quality.

The [ordinary LongMemEval pilot](../research/reports/longmemeval-2026-09-11.md)
located losses in within-node reading and cross-node synthesis, despite admission
of every annotated source for the misses. Iterations now correct the printed
single-reference Python interface and give every seed a bounded role/coordinate
page. Root synthesis sees exact attributed selected evidence before fallible
branch summaries, with explicit source dates and canonical metadata preserved.
The original 2/5 pilot and failed follow-ups remain immutable. V5 reached 4/5. The final v6
candidate reached 5/5 in one fresh run with a GPT-5.4 medium-reasoning root and
GPT-4.1 delegates and maintenance. The default benchmark command selects v6.
The requested full deterministic suite passed afterward.
Prior controls used GPT-4.1 readers and are not matched to this model allocation.
Root-only replays diagnose synthesis but do not replace full answer trials.
Counting-unit ambiguity must remain visible. Do not invent pending obligations
to match a scalar benchmark answer. The
[node-search improvement order](../research/ROADMAP.md#node-search-improvement-order)
owns later retrieval comparisons and useful follow-up discovery.

Node callbacks expose applicable edge relationship and provenance descriptors
alongside deduplicated target references. Delegates inspect their own contribution
first and may batch a lazy source read with edge discovery. A bounded initial
source-metadata page supplies span coordinates and roles for every seed and child.
It does not load their text into model context. Each frame reserves a final model
call, queued seeds reserve inspection/finalization calls, and child admission
preserves those reservations within the shared budget. Local model omissions may
be resolved by siblings. Host-owned `required_gaps` retain attributed operational
and evidence-interpretation failures. Callback schema/range mistakes are
recoverable observations, while unrecovered failures remain visible. An empty
answer with an explicit unresolved reason is a valid partial abstention. A finish
before any source inspection gets a corrective observation within existing limits.

`LLGM` is the single product entry point, implemented in `src/llgm/llgm.py`.
Shared records/configuration/errors are in `core/`. Source/journal/query/maintenance
operations are in `memory/`. Budgets/results/executors are in `inference/`.
The iterative and Python RLM experiments remain under `inference/`, with no
compatibility aliases or competing top-level application names.
Only `LLGM.from_settings()` owns configured application construction. Specialized
executors take caller-owned clients directly. Application `Settings` has no
experimental `search_policy` field. `make format` owns Python layout/imports,
and `make lint` checks both alongside unused-code rules.
The [cleanup report](../research/reports/code-cleanup-2026-09-11.md) records the
subsequent review and local validation of these simplifications.

Local environment files require explicit `load_env_file(path)` before settings
and client construction, or `--env-file` on `llgm ask` and the hosted example.
The dependency-free helper validates the entire file before filling missing
process variables. Existing variables, including empty values, win. It performs
no discovery, substitution, escape expansion, or hot reload. Settings precedence
remains explicit overrides, TOML, environment, then defaults. Environment-file
values have environment provenance. Credentials stay outside `Settings` and
belong in exported variables or an ignored local file. The
[configuration guide](../docs/guide/configuration.md) owns the public contract.
Frozen experiment JSON remains independent of application settings.

`answer(..., node_id=...)` starts at a node. The model's `query_node` operation
creates an isolated child that can inspect local evidence, discover links, and
query another node. Its selected findings/references return to the parent.
Child message history stays local. Children may search globally. Locality is the
computation model, not an authorization boundary. No per-node daemon is required.

Sources are immutable unique nodes. References have no source-version dimension.
New evidence gets a new node. Public commit/read-basis/snapshot APIs and historical
ranking machinery are removed. A private cursor refreshes the reusable FTS5 index.
Reads use current records and may observe appends during an answer. Workspace schemas 1 and 2
are rejected without automatic mutation. `copy_schema2_workspace` explicitly
copies a schema-2 local workspace into a new schema-3 directory after pointer
classification. Source/journal payload identity remains schema 2. No source is
silently upgraded or reclassified.

Journals are append-only. Explicit `record_kind="overwrite"` selects the latest
applicable append sequence for the same exact subject, relation, and declared
scope. Ordinary assertions remain multi-valued. Preserve stable spans, source
history, atomic publication, idempotency, provenance, and independent resource
ownership. Raw journals expose history. Interpreted views identify active entries. Effective reads apply only explicit
`replace` or `current_evidence` patches, preserving canonical provenance on every
original/replacement segment. Finite operational journal limits fail explicitly
when necessary distinct entries remain too large. Source-copy materialization
and deletion of cold history are not implemented or authorized.

Source event `timestamp_ms` is optional Unix milliseconds. No source import-time
field is required. Journal `recorded_at_ms` is audit time, not overwrite priority.
Use `as_of_ms` and `valid_from_ms`/`valid_until_ms` for machine selectors.
`query_date` is original text and never implies an instant. `parse_instant_ms`
requires a timezone or an explicit date-only timezone policy. Preserve missing
source time without inventing precision.

Gold answers and evaluator labels must stay outside model-visible evidence.
Benchmark source identifiers require the same treatment: LongMemEval support
session IDs can begin with `answer_`. The node-search audit found this perfectly
correlated with support labels in its 24-question cohort. Use `anonymize_case`
for the corrected experiment: opaque node IDs, date-only source metadata, and
original session aliases confined to scoring. Preserve original frozen runs as
historical diagnostics instead of rewriting their corpora or strengthening claims.
A structurally valid link or citation does not prove semantic truth. Unknown
usage/cost stays unknown. Call/time limits do not imply dollar-budget enforcement.
Live results under older schemas remain historical evidence, not validation of
this refactor. The [refactor report](../research/reports/simplification-2026-09-10.md)
records earlier local verification. Current node/edge delivery checks are in
the [September 11 report](../research/reports/node-delegates-2026-09-11.md).

Use small, replaceable interfaces and keep provider SDKs in adapters. Avoid a
registry or general framework without demonstrated variation. See
[code boundaries](ARCHITECTURE.md), [public architecture](../docs/guide/architecture.md),
and [code standards](../development/standards.md) before changing a contract.

## Evaluation constraints

- The user accepts reporting cold-start overhead separately and defers optimizing it during research. Prioritize measured retrieval/selection behavior now. Warm latency does not establish interactive cold performance.
- The completed [node-search diagnostic](../research/reports/node-search-2026-09-11.md) separates candidate-source coverage from initial seed selection through the actual application path. The completed [selection experiment](../research/reports/node-selection-2026-09-11.md) compared the two top-40 pools and reciprocal-rank fusion against matched first-owner controls at a three-node cap. Controlled gains did not transfer to complete LongMemEval source coverage, and fusion discarded one source available in the full union. The subsequent [final-answer comparison](../research/reports/seed-answers-2026-09-11.md) exposed evidence-delivery failures that prevent a useful selector conclusion. Later root-schema and cleanup repairs passed their diagnostics. The ordinary v1 pilot lost available facts semantically. Subsequent reading and synthesis changes reached 5/5 on the exposed development questions in v6, using a GPT-5.4 medium-reasoning root with GPT-4.1 delegates and maintenance. Older reader controls used GPT-4.1, so this is not a matched architecture comparison. V6 used no recursive children and establishes no graph-traversal benefit. Keep the first-owner policy and both retrievers. Next compare complementary source retention with replayed rankings before another hosted selector sweep. The stopped full run must not resume automatically. Annotated source sets may include redundant or older evidence, so their size is not a semantic minimum and their complete recall is not answer accuracy.
- Successful hosted diagnostics qualify their recorded models, prompts, budgets and source complexity only. API compatibility does not establish Python interface or evidence-use reliability. Shared evaluation accounting now provides priced admission and retained unknown-usage reservations for the primary LongMemEval runner. Historical seed-answer tools have their own controls. Neither is product-wide dollar enforcement or a provider billing guarantee.
- E03 isolates retrieval: B/D/H/C × S/U/A with one fixed model pair and graph/recursive policies disabled. Explicit `--required-backends B D H` limits readiness and dispatch to nine arms and records C as excluded. The default requires all twelve.
- Passage tokenizer identity is separate from ColBERT encoder/code identity. Tokenizer-only preparation does not run ColBERT or PLAID.
- All 500 LongMemEval-S cases share one connected history component under the chosen session-ID/text isolation rule. The five pilot questions do not leave 495 independent held-out histories. Repeated session IDs must retain distinct source occurrences and evaluator aliases.
- The pilot's five cases are exposed. Reuse is diagnostic replication. All three single-query retrievers saturated labeled coverage at top 40. This pilot cannot establish a benefit from added retrieval.
- The frozen architecture comparison uses curated links to isolate navigation. It does not measure learned maintenance. Readiness artifacts record zero model calls, not completed trials.
- Prepared E03 inputs remain immutable-node schema 2. The current structured-operation comparison is v3 with independent primary edges and numeric time. V1/v2 are preserved byte-identically and rejected by the revised runner. V3 does not measure the full Python seed pipeline. Use [comparison instructions](../experiments/comparison.md) and preserve older artifacts.

Internal proposals and dated results are indexed in [research](../research/README.md).
Current capabilities and limits belong in [implementation status](../docs/reference/implementation-status.md).
Do not copy changing test counts or report tables into this file.

## Working constraints

Inspect existing edits before modifying files and preserve untracked work. Do not
reset or clean the checkout to establish a baseline. Install only dependencies
required for the task. Documentation work does not require GPU packages or
environment replacement. Never persist credentials in code, docs, prompts or
artifacts. This handoff does not authorize paid infrastructure, publication or
a deferred experiment.

The user selected the **Return** logo. Preserve the selected geometry and stroke
width of 5 in `development/brand-explorations/04-return.svg`. Asset usage is owned
by [branding](../development/branding.md).

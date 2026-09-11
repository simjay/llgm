# LLGM architecture update plan

Implemented and locally verified, September 11, 2026. This document records the
approved requirements and implementation order. The
[delivery report](../reports/node-delegates-2026-09-11.md) records completed checks
and limits. This work does not authorize benchmark runs or publication. The plan is internal
and must remain outside public documentation and package archives.

Later retrieval and answer diagnostics have also completed. Use the
[current architecture](overview.md) and [roadmap](../ROADMAP.md) for present
behavior and next work. The implementation order below retains the approved plan.

## Intended result

LLGM stores original evidence in immutable nodes. Primary edges define the graph
independently of per-node journals. Journals record amendments to local evidence
and can point to other nodes containing replacement information.

An ordinary question retrieves starting nodes. A smaller-model delegate at each
selected node recursively investigates useful primary neighbors and applicable
journal amendments. Delegates return condensed findings with supporting evidence.
The same overall root combines those branch findings into the final answer.

```text
Question → retrieve seed nodes → node delegates → recursive local collection
                                                     ↓
Answer   ← root synthesis      ← cited branch findings
```

Branches form query-specific evidence graphs as they explore. A delegate acts as
the local root of its branch using the same recursive mechanism as its children.
The agreed scheduling policy is to run all admitted seeds with bounded
concurrency. The node runtime implements this policy. A concurrency limit queues branches
instead of silently dropping them. Whole-graph materialization, component
partitioning, persistent node daemons and distributed worker infrastructure are
not requirements of this update.

## Baseline to preserve

These earlier changes are already implemented. They need regression coverage,
not another redesign:

- One product entry point, `LLGM`, in `src/llgm/llgm.py`.
- Shared records, configuration and errors in `core/`, evidence operations in
  `memory/`, and execution, results and limits in `inference/`.
- Opaque generated node IDs, immutable original text and stable source spans.
- First-class optional source event `timestamp_ms`, with no source import-time
  requirement.
- Removal of source versions, public commits and workspace snapshot APIs.
- Append-only journals with exact subject/relation/scope overwrite precedence
  determined by eligible append order. Original evidence remains addressable.
- Node-targeted recursive calls, isolated child histories, selected citation
  returns, caller/resource ownership and shared-budget primitives.

The name LLGM reflects graph-organized local computation and bounded messages.
The design does not currently define probabilistic factors or prove minimal
messages or convergence. Public explanations should lead with what it does.

## Selected execution direction

The user authorized executing the plan after discussing Python REPL node
delegates. Implementation proceeds with that option, stated explicitly as an
assumption based on the discussion. Phase 3 and its checks include the graph
evidence bridge and actual isolated code execution. Implementation is in progress.

Current structured recursion exposes `read`, `search`, `neighbors`, `journal`
and `query_node`. The separate Python `RLMRuntime` exposes external context in
Docker and recursive `llm_query(prompt)` strings, without a graph-evidence bridge.
Do not describe those two existing mechanisms as already integrated.

If Python REPL execution is selected, connect local evidence and recursive node
queries to that environment and verify generated-code inspection at multiple
nodes. If structured operations are selected, reuse the existing interpreter and
describe its external-context recursion precisely. Neither option requires a new
controller for every graph, a second public application name, or a plugin registry.

## Implementation order

Each phase includes the corresponding public contract and example updates when
its behavior lands. Final documentation work connects those pieces into one
complete contributor explanation.

### 1. Primary edge records and maintenance

Add a small primary-edge record with source node, target node, relationship,
supporting provenance and declared applicability when relevant. Choose record
identity and explicit withdrawal semantics before exposing edge edits. Existing
source and journal references remain unchanged. An additional `EdgeRef` is not
needed for the initial traversal contract.

Store edges in the existing SQLite workspace with indexed outgoing lookup. Add
narrow publish/inspect operations with endpoint validation, atomic writes,
idempotency and duplicate prevention. Preserve directed relationships.

Change ordinary link acceptance and `LLGM.organize()` to publish primary edges.
Their deduplication and publication must not depend on unrelated journal append
sequences. `neighbors()` reads primary edges. Journal pointers no longer become
ordinary adjacency implicitly. Keep topology metadata available for explaining
why a neighbor was selected, without treating connectivity as proof of a claim.

Use the existing maintenance candidate search and model adapters. Source ingestion
must still survive maintenance failures. Leave lexical source/journal indexing
intact unless a concrete search requirement calls for indexing edge records.

**Owners:** `core/types.py`, `memory/workspace.py`, `memory/evidence.py`,
`memory/query.py`, `memory/maintenance.py`, `llgm.py`.

**Acceptance:** A graph with empty journals supports directed traversal and
recursive node queries. Accepted ordinary links create edge records without
semantic journal writes. Invalid endpoints, retries, concurrent duplicate writes
and reopen preserve the declared contract.

### 2. Fully loaded journals and amended source reads

Load the complete operational journal when opening a node. It is a small local
sidecar used by the reader, never another large RLM query target. The original
source and large replacement sources remain accessible through lazy reads. The
reader applies explicit journal patches before presenting selected evidence to
the model. The model must not need to discover the journal through optional
search or generated Python merely to obtain current evidence.

Keep raw source resolution and original offsets unchanged. Effective reads return
original or replacement segments with their own canonical references and amendment
provenance. The current ID/status-only journal view is insufficient. Replacement
text must not be attributed to the old source's offsets.

Preserve the existing exact overwrite key and append-order rules. Distinguish
amendment precedence from whether an amendment affects a selected source span.
A production-passage amendment must leave staging evidence and unrelated primary
edges intact. Suggestions and inapplicable amendments must not suppress committed
evidence.

An exact replacement reference can be resolved by the reader without a model
call. Recursive node queries remain available when the referenced source needs
further investigation. The recursive target is that source, not the journal.
Unspecified or conflicting amendments remain explicit unresolved conditions.
Do not silently use overwritten evidence when a replacement is unavailable.
Account for the full operational journal and exposed evidence in resource limits.
Notes without explicit patch semantics remain fully available local guidance.
The reader does not infer a replacement operation from arbitrary prose.

Keep journal application separate from ordinary neighbor discovery. A journal
reference need not be duplicated as a primary edge. Explicit relationship edits
must identify the affected relationship rather than infer its deletion from a
source amendment. General forgetting policy remains a separate research choice.

Add reconciliation for journal growth. Deterministically collapse superseded
updates with the same exact subject, relation and scope while preserving relevant
validity distinctions, scheduled future changes, unresolved records and provenance.
Do not use an LLM summary as the authoritative compacted state. Retain old
`JournalRef` targets outside the operational sidecar where needed for existing
citations and correction semantics. Historical records need not enter each node
delegate's context.

This reduces repeated-update overhead but cannot bound arbitrarily many distinct
live patches. At a declared size threshold, one proposed option is to write a
separate on-disk read copy with patches already applied and retain only subsequent
changes in the operational log. This is materialization, distinct from merely
compacting redundant journal records. It adds storage and original/replacement
reference mappings. It is not yet an agreed implementation choice. Settle its
scope/time reconstruction rules before implementation. The original immutable
sources remain authoritative evidence. This is local storage maintenance, not a
public source-version or workspace snapshot API.

Compaction publishes operational membership atomically and retains concurrent
appends. It does not materialize a working source copy. Bound the operational sidecar without
silently truncating it. Archived history can still grow on disk. Total-storage
retention remains an explicit open policy and does not authorize deletion.

**Owners:** `memory/workspace.py`, `memory/evidence.py`, `memory/query.py`,
selected inference executor.

**Acceptance:** A primary-edge visit to an old node finds an applicable correction
outside the retrieval seeds. The corrected answer has supporting references.
Missing targets, amendment chains, conflicts, scope mismatches and cycles produce
bounded, explicit outcomes. Unaffected passages retain their original meaning.
Reads apply the fully loaded journal without model-directed journal inspection.
Repeated corrections compact, distinct live patches trigger the declared growth
policy, and compaction preserves references, validity, retries and concurrent
appends across interruption and reopen.

### 3. One local delegate mechanism

Use one internal delegate result containing findings, selected evidence and
unresolved needs. Keep branch identity and outcome around that result for the
final root. Child context remains local. Carry the focused question and applicable
scope/time into each child instead of copying ancestor conversations.

The approved execution proceeds with Python delegates, as stated when work began.
Reuse the Docker transport and model adapters, expose journal-corrected lazy reads,
primary-edge discovery and recursive node callbacks, and retain canonical citations
across the bridge. The complete journal is supplied before the first local model
call. A bounded source outline lets a delegate choose exact source spans before
loading a large node. Text can remain in interpreter variables until selected
output is observed. Decoding an immutable JSON blob on the host is still a storage
cost, and this work does not claim streaming source storage.

Use active-request cycle detection plus total limits. Avoid caching by node ID:
the same node can receive different questions, scopes or temporal selectors.
Completed-result caching is unnecessary for the initial implementation.

**Owners:** `inference/recursive.py`, or `inference/rlm.py` and
`inference/repl.py` if selected, plus `inference/results.py` and `memory/query.py`.

**Acceptance:** A delegate follows A → B → C, resolves local amendments when
needed, and returns selected descendant evidence through the actual parent chain.
Siblings do not inherit each other's histories or gain citation visibility
automatically. Python selection additionally requires real code inspection and
callback integration checks, not merely scripted JSON operations.

### 4. Default seed dispatch and final root synthesis

Make `LLGM.answer(question)` run the configured retriever before local model
inference. Aggregate passage hits by node, preserving ranking, matched spans and
metadata. Use a bounded, configurable seed count. Multiple passages from one node
produce one initial branch. Retrieval is candidate selection, not exhaustive
relevance. Preserve `answer(node_id=...)` as an explicit single-seed entry.

The agreed default dispatches all admitted seeds with bounded concurrency.
Three seeds with a concurrency limit of two means two active branches and one
queued branch. A subset requires an explicit seed-selection or budget policy,
with failures and skipped work recorded. Do not silently stop after the first
useful answer. Reserve shared limits before concurrent work. Do not hold an
active-model-call permit while waiting for a recursive child.

Use one evidence registry, run ledger, query scope, temporal selector and execution
trace. Do not call the public runtime answer method independently per seed:
that resets budgets and citation IDs. Keep branch findings attributed until the
final root combines them. Preserve the public `AnswerResult` unless a demonstrated
contract requires extending it.

Define explicit outcomes for empty retrieval, skipped seeds, failed branches and
insufficient synthesis capacity. Retain successful branch evidence when another
branch fails operationally, and label the answer partial if evidence remains
unresolved. Cancellation and unexpected programming faults still propagate.
Operational failure must not become fabricated findings.

Reserve final synthesis capacity in the shared budget and bound aggregate root
input. Current root reservation preserves only one model call, not sufficient
context or wall time. Choose admission/default seed limits together with model,
context and deadline limits, and report exhaustion accurately. Individually valid
branch messages can still be too large when combined. Overlapping branches do
not establish independent corroboration of the same source.

Runtime evidence graphs develop through exploration. Trace the selected seeds,
traversal reasons, node requests, amendments, returns and final synthesis without
introducing a separate persistent runtime graph store. Use the existing retrieval
adapter boundary. ColBERT was deferred when this plan was prepared. The later
[Modal diagnostic](../reports/colbert-modal-2026-09-11.md) records its actual execution.

**Owners:** `llgm.py`, selected inference executor, `inference/budget.py`,
`inference/results.py`, `core/config.py`, `cli.py`.

**Acceptance:** A question with multiple useful seeds produces multiple local
branches and one overall synthesis using their attributed evidence. One branch's
failure preserves the other's findings. A trace proves upfront retrieval and
dispatch without scripted root instructions to perform those steps. Limits apply
once across the entire run, including initial retrieval and root synthesis.

### 5. Finish the timestamp contract

Source event time is already a first-class optional Unix-millisecond argument.
Complete the remaining consistency work for machine-readable query and journal
validity instants, which still use ISO strings. Proposed direction: one numeric
instant representation internally and explicit parsing at input boundaries.

Preserve unknown time and original human date text. Make date-only conversion and
timezone assumptions explicit. Keep half-open validity intervals and append-order
overwrite precedence. Journal recording time is audit information and does not
decide which interpretation wins. Do not reintroduce a source import-time field
or workspace snapshot semantics. Finalize public argument names and date-only
handling before changing signatures, examples and prepared inputs together.

**Owners:** `core/types.py`, `memory/evidence.py`, `memory/query.py`, `llgm.py`,
retrieval metadata and evaluation input adapters.

**Acceptance:** Equivalent declared instants select the same amendments. Unknown
or ambiguous dates do not acquire invented precision. Production/staging and
validity-boundary cases preserve their intended interpretation.

### 6. Developer documentation and evaluation fixtures

Write one end-to-end offline example covering primary edges, multiple retrieved
seeds, recursive descendant findings, a scoped passage correction and final root
synthesis. Label scripted choices clearly. Add a model-driven counterpart only
after the selected protocol has been exercised with the required environment.

Update concepts, architecture, quickstart, walkthrough, configuration, API,
implementation status and testing guides with each implemented contract. Explain
node, edge, journal amendment, delegate, runtime evidence graph and execution
trace with one consistent example. Show what each model sees and what is returned.
Give new developers a package map and pointers to owning implementation and tests.
Lead the graphical-model explanation with locality and message passing.

Replace claims that ordinary edges live exclusively in journals and descriptions
of root-optional initial retrieval when those behaviors change. Distinguish
proposed, implemented and verified behavior. Keep the current package grouping,
adding an owning module only for a concrete responsibility that no longer fits.

Update current experiment preparation and tests for explicit edges. Freeze a new
comparison protocol for changed topology or seed orchestration, and preserve old
protocols and measurements unchanged. Scripted execution, provider compatibility,
automatic link quality and answer quality remain separate forms of evidence.

Public prose stays in `docs/`, benchmark runbooks in `experiments/`, and this plan
and dated reports in internal `research/`. Research and agent context must remain
absent from published links, includes, downloads, search indexes and archives.
Align the hosted documentation build with the same rendered-output audit as CI.

**Acceptance:** A new developer can install, run the example, inspect its trace,
and locate implementation and tests using public guides alone. Examples run from
an installed wheel outside the checkout. Public claims match completed checks.

## Storage transition and delivery

Adding independently authoritative edges changes workspace semantics. Use an
explicit new format marker and a narrow conversion path to a new workspace copy.
Preserve source blobs, node IDs, journal IDs, sequences, payloads and retry records.
Keep the original workspace intact. Existing incompatible formats must not be
silently interpreted under new adjacency rules.

Old journal references mix navigation and amendments. Do not promote all pointers
or guess their role from a relation name. Inventory them and apply an explicit
mapping to primary edge, retained amendment or unresolved classification. Producer
provenance helps review but does not prove semantic correctness. Keep old journal
references resolvable even when a reviewed relationship becomes a primary edge.
This is a specific transition, not a general migration framework or runtime
compatibility layer. Fresh test workspaces can use explicit records directly.

Land phases as reviewable changes, with focused tests beside each behavior. Edge
storage and time cleanup can be developed independently after their contracts are
settled. Delegate execution uses isolated Python with the narrow evidence callbacks. The first full
retrieval-to-answer integration follows edges, amendment handling and delegates.

Delivery validation gates, with actual results in the linked report:

1. Run affected storage, evidence, application and recursive contract tests.
2. Cover journal-free adjacency, stale-seed correction, untouched passages,
   multiple seeds, overlapping descendants, partial failure, empty retrieval,
   context limits, amendment cycles, cancellation and explicit conversion. Include
   concurrent seed scheduling, nested-call deadlock prevention, full journal
   loading, compaction equivalence, growth thresholds and concurrent append.
3. Run `make check` and `make test-data` with the pinned local dataset available.
4. Run `make docs`, inspect affected rendered guides and audit the publication
   boundary from a fresh output directory.
5. Build distributions, check metadata and execute offline examples from an
   installed wheel outside the checkout. Verify archive exclusions.
6. Exercise the selected hosted/Docker protocol only with the required configured
   environment and authorization. Record new results separately from historical
   runs. A skipped integration test does not verify the new behavior.

The delivered default flow, independent edges and applicable amendments pass
their local checks. Source-copy materialization and physical history retention
remain explicit open choices. New hosted quality measurements remain separate.

# Design decisions

These are current design constraints, with links to their owning implementation
or guide. Revisit a decision when evidence warrants it, and update the relevant
contract and tests together. Open choices belong in [remaining tasks](REMAINING_TASKS.md).

## D01: Preserve original evidence

Source turns are immutable. A passage reference identifies a node, turn, and
character span. New related information appends to the existing topic node. Explicit journal patches
can change an effective read without changing the original source.

This keeps citations inspectable after corrections. Derived search chunks are
not a second source of truth. See [evidence records](../src/llgm/core/types.py),
[workspace storage](../src/llgm/memory/workspace.py), and the
[correction walkthrough](../docs/guide/walkthrough.md#follow-an-update-and-a-correction).

## D02: Separate links from amendments

Primary edges record generic connections between nodes. RLM readers interpret
their relationship for the current question. Edges have no relationship type field. Journals describe assertions
and amendments local to a node. Automatic maintenance proposes and validates
edges. It does not automatically decide which source text should be overwritten.

Keeping these operations separate lets navigation evolve without turning every
relationship into a correction. Exact subject, relation, declared scope, and
append order govern applicable overwrites. See
[maintenance](../src/llgm/memory/maintenance.py) and
[effective reads](../src/llgm/memory/evidence.py).

## D03: Start with retrieval, then use local model readers

The application selects distinct node owners from ranked passages before
delegate generation. It dispatches all admitted seeds with bounded concurrency.
Each delegate may read, search, inspect links, and start recursive children.
Only selected findings and evidence return to the parent or final main.

This keeps full source text and child conversation history out of a shared
prompt. Concurrency queues admitted work rather than silently dropping it.
The integrated path ends with one main synthesis call, with no further main tool
phase. See [node execution](../src/llgm/inference/nodes.py) and
[node search](../docs/guide/node-search.md).

## D04: Keep failures and resource limits visible

Branches share budgets and a citation registry. Finalization reserves capacity.
Skipped seeds, unrecovered operation failures, and required interpretation gaps
remain in the result. Cleanup runs on failure and cancellation. A valid citation proves
source identity, not that an answer is correct.

An explicit answer budget replaces the whole application budget. It is not a
partial settings merge. Resource ownership and failure behavior belong in
[LLGM](../src/llgm/llgm.py), [budgets](../src/llgm/inference/budget.py), and
[result handling](../docs/guide/quickstart.md#understand-the-result).

## D05: Make providers and retrieval replaceable

Hosted model SDKs live behind request/response interfaces. Search adapters return
references into canonical workspace evidence. Supplying a different model or
retriever should not require replacing storage or the node executor.

Keep adapter-specific settings in the adapter until a shared contract justifies
promoting them. Distinguish interface compatibility from measured model quality.
See [model contracts](../src/llgm/models/base.py),
[retrieval contracts](../src/llgm/retrieval/base.py), and
[custom search](../docs/guide/configuration.md#use-your-own-search-backend).

## D06: Keep current reads and retained history distinct

The workspace retains source and journal history. Its operational journal
reconciles exact applicable overwrites for current reads. This bounds some working
metadata, not total storage. Reads can observe appends made during an answer.

No general history deletion or snapshot contract follows from this design.
Schema migration is explicit and preserves the original workspace. See
[workspace storage](../src/llgm/memory/workspace.py) and
[migration](../src/llgm/memory/migration.py).

## D07: Separate correctness checks from quality measurements

Deterministic tests protect mechanics. Real boundary checks qualify their
configured services and models. Frozen evaluations measure answer quality under
declared inputs and budgets. Keep gold labels outside generation, retain failed
attempts, and preserve unknown usage rather than reporting it as zero.

No one layer substitutes for the others. See [testing](TESTING.md) and the
[LongMemEval runbook](../experiments/longmemeval.md).

## D08: Start with plain application inputs

`LLGM.from_settings()` reads the environment at context entry when settings are
omitted. Explicit settings keep their original behavior. `LLGM.ingest()` accepts
plain text as one user turn and chat turn sequences as one conversation. It
normalizes them to the existing `Conversation` record before writing anything.
Callers use that record directly for source IDs, metadata, and timestamps.

This removes setup wrappers from a first program while keeping one async
resource factory and one storage contract. Exact text, role attribution,
idempotency, maintenance behavior, and resource cleanup remain unchanged. See
[application entry-point tests](../tests/test_application_entrypoint.py) and the
[quickstart](../docs/guide/quickstart.md).

## D09: Conversation first with conservative topic growth

`answer()` persists incoming turns and nonempty replies. A stable conversation
ID resumes its active topic. The routing model prefers continuation, can return
to a retrieved topic, and creates a node only on a clear topic change. Size,
new sessions and subtopics do not independently justify a split. No automatic
size cap or within-import-batch segmentation is implemented.

Schema 5 stores appended turns as separate immutable blobs, preserving spans
without rewriting history. Coordinate pages avoid loading appended text.
A span loads one turn, while full-source reads remain explicitly materializing.
Schema 3 requires an explicit copying migration. Calls sharing a Workspace
serialize conversation mutation. Independent handles need caller coordination.

`ingest()` is the catch-up API. Explicit source IDs retain exact imports.
`answer(..., remember=False)` retains read-only evaluation. The current
LongMemEval runner routes supplied sessions as batches and can coalesce them.
Retired benchmarks and protocols were removed from the working tree. Earlier
measurements do not establish this construction policy or huge-node reasoning.
See [conversation tests](../tests/test_conversations.py) and
[the current runbook](../experiments/longmemeval.md).

## D10: Name model roles Main, Reader, and Graph

Settings configure main, reader, and graph clients independently. Topic
routing and generic connection proposals use the graph client. Readers
read recursively and interpret relationships at query time. Both conversational
answers and catch-up imports share topic routing. Imports route a batch as one
unit. Usage and admission limits distinguish graph calls from reader calls. Primary
edges, proposals, and traversal APIs contain no relationship type. Schema 5
removes that storage field and rejects schema 4 without modifying it.

Public constructor fields, settings prefixes, budget counters, and model trace
roles use main, reader, and graph consistently. There are no legacy role aliases.
The specialized iterative gatherer is EvidenceReader. The standalone RLM uses
main_model and reader_model, while child still denotes a recursive invocation.
MaintenancePolicy, MaintenanceResult, and maintenance outcomes name the process.
Benchmark protocol version 2 uses the same model and pricing keys. See
[configuration tests](../tests/test_config.py), [runtime tests](../tests/test_runtime.py),
and [benchmark tests](../tests/test_memory_benchmark.py).

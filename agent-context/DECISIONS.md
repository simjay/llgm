# Design decisions

These are current design constraints, with links to their owning implementation
or guide. Revisit a decision when evidence warrants it, and update the relevant
contract and tests together. Open choices belong in [remaining tasks](REMAINING_TASKS.md).

## D01: Preserve original evidence

Source nodes are immutable. A passage reference identifies a node, turn, and
character span. New information gets a new source node. Explicit journal patches
can change an effective read without changing the original source.

This keeps citations inspectable after corrections. Derived search chunks are
not a second source of truth. See [evidence records](../src/llgm/core/types.py),
[workspace storage](../src/llgm/memory/workspace.py), and the
[correction walkthrough](../docs/guide/walkthrough.md#follow-an-update-and-a-correction).

## D02: Separate links from amendments

Primary edges describe relationships between nodes. Journals describe assertions
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
Only selected findings and evidence return to the parent or final root.

This keeps full source text and child conversation history out of a shared
prompt. Concurrency queues admitted work rather than silently dropping it.
The integrated path ends with one root synthesis call, with no further root tool
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

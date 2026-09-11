# Design decision: immutable nodes and their journals

Implemented September 10, 2026. This decision replaces the source-version and
mandatory snapshot model in the [September 9 technical baseline](technical-spec.md).
The [architecture guide](../../docs/guide/architecture.md) describes the current
implementation. The [walkthrough](../../docs/guide/walkthrough.md) demonstrates
node recursion and scoped updates. Dated reports retain their original schemas
and measurement conditions.

## Local reasoning and bounded messages

LLGM uses a persistent evidence graph to organize original text and relationships.
Its graphical-model motivation is locality: a model works with selected evidence
at one node, asks focused questions of related nodes, and receives bounded findings
with references. This makes selective communication between local reasoning steps
an explicit part of the design.

`LLGM` in `llgm/llgm.py` is the integrated facade for ingestion, maintenance, and
answers. Shared contracts live in `core`, evidence and journals in `memory`, and
execution and budgets in `inference`. The recursive executor's
`query_node(node_id, question)` starts a child frame focused on that node. The
child can inspect evidence and ask another node a focused question. It returns
findings, citations, and unresolved issues to its parent. Children execute
sequentially under shared budgets and bounded depth. They receive local context,
not the parent's complete conversation.

The persistent evidence graph and transient invocation tree have distinct roles.
Traces record parent/child calls and target nodes. No additional runtime-graph
object is required. Locality and bounded communication are implemented mechanisms,
not a proof that messages are minimally sufficient. The system does not claim
probabilistic factorization, belief-propagation convergence, or measured superiority
over flat retrieval. Whether persistent relationships improve answer quality or
reduce repeated interpretation and total cost remains an empirical question.

## Immutable evidence, append-only interpretation

Each source occurrence is an immutable node with a unique opaque ID. Omitted IDs
default to `uuid.uuid4().hex`. An update creates another node. It does not replace
the original text or introduce a source-version dimension. Exact retries remain
idempotent, while reusing a node ID for different content is rejected.

The reference family is deliberately small:

- `NodeRef(node_id)` addresses a whole node.
- `SourceSpan(node_id, turn_id, start, end)` addresses a half-open range of Python
  Unicode code points in an original turn.
- `JournalRef(node_id, entry_id)` addresses a journal record. Paired `start` and
  `end` offsets can address its inline text value.

A node's journal records interpretations and links with provenance. Ordinary
`record_kind="assertion"` entries append independent values, so multiple supporting
or related nodes can coexist. An explicit `record_kind="overwrite"` replaces
earlier applicable interpretations with the **same exact subject, relation, and
declared scope**. For a passage subject, its node, turn, and both offsets must
match. A different passage, relation, or scope remains independent.

```text
Node A: "Atlas production uses PostgreSQL. Staging uses SQLite."
  journal sequence 1: assertion
    subject: A's production statement
    relation: current_evidence, scope: production, value: A's production span
  journal sequence 2: overwrite
    same subject, relation, and scope, value: B's production span

Node B: "Atlas production has migrated to MySQL."
```

This is a conceptual view of the executable update in the walkthrough. The latest
applicable overwrite wins by its owning node's `journal_sequence`, regardless of
input order or recording timestamps. Proposed, future, expired, unresolved, or
out-of-scope overwrites cannot suppress an applicable predecessor. An ordinary
assertion appended after an overwrite remains active until another applicable
overwrite replaces that same slot. Full journal history remains available.

The original A stays readable, including its unaffected staging statement. A
`record_kind="correction"` can instead target an earlier journal assertion through
a `JournalRef`, including retracting an overwrite. Source evidence, interpretations,
and corrections retain separate identities. Sequence determines precedence under
this policy. It does not prove an interpretation true. Search can discover older
evidence that needs a link, but mandatory reconciliation during ingestion is not
part of this decision.

## Time and current reads

Optional source `timestamp_ms` is an integer Unix timestamp in milliseconds for
the represented event. Unknown source time remains `None`. There is no source
import-time field. Journal `recorded_at_ms` records append time for audit. It does
not determine overwrite precedence. Validity bounds and `as_of` currently accept
ISO date/time strings. These select applicable interpretations rather than freeze
database visibility.

Reads observe currently published evidence. An answer may encounter a journal
entry appended after its initial search, so it does not promise one fixed instant
across all nodes. Original sources and journal history support questions about
past events. Exact replay of past workspace visibility or ranking is a separate
requirement. Citations retain the exact source spans and journal entries used.

The storage analogy is an append-only journal sidecar in the spirit of a WAL.
SQLite transactions provide atomic publication. The journal need not occupy a
separate file per node. Public `commit_id`, `ReadBasis`, source retirement, and
historical ranking machinery have been removed. A private append cursor remains
for incremental index refresh. Stable references, provenance, append preconditions,
idempotency, and recovery directly support the evidence model and remain in place.

## Validation and limits

Deterministic regressions cover immutable node identity, exact spans, safe retries,
journal ordering and corrections, scoped overwrite precedence, retained history,
current reads, restart and index reuse, and bounded recursive calls. The
[storage tests](../../tests/test_storage.py),
[interpretation tests](../../tests/test_evidence.py), and
[recursive tests](../../tests/test_recursive.py) own these contracts. The walkthrough's
scripts run without hosted models. Schema-1 workspaces are explicitly rejected
without rewriting their data. Automatic migration is not provided.

These checks establish mechanics. They do not measure automatic link quality,
message sufficiency, or comparative cost and accuracy. Those require the
[experiment design](experiments.md), including missing and incorrect links,
maintenance cost, and matched controls. Earlier reports remain evidence for their
original implementations and conditions.

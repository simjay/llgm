# Project context

LLGM stands for Large Language Graphical Model. It is a Python library for
answering questions from stored conversations with traceable source references.
It combines persistent evidence, passage retrieval, and recursive model reading.
The package is an experimental alpha, with its first PyPI release in progress.
The [quickstart](../docs/guide/quickstart.md) distinguishes package installation
from the repository installation available before that release.
Check [package metadata](../pyproject.toml) and
[capabilities and limits](../docs/reference/implementation-status.md) for current
requirements and supported behavior.

## Product model

A topic node contains append-only conversation turns. Related sessions can
share a topic, and new turns preserve earlier span identities. Search passages
point into those turns. Generic primary edges connect related nodes, and readers
interpret their meaning. Per-node journals record explicit amendments while
retaining original evidence and append history.
Edges and journals have separate responsibilities.

By default, an answer starts at its conversation's current topic, including in
read-only mode. Search can fill remaining seed slots with distinct node owners.
The current pointer is per conversation and survives restarts. Explicit read-only
node selection overrides it, and a read-only question without a pointer uses
retrieval alone. Reader-model delegates inspect the admitted nodes and can query
related nodes recursively. Their selected findings and evidence go to one final
main-model call. The persistent graph represents evidence relationships. It does not define
probabilistic factors or guarantee convergence.

Read [conversations](../docs/guide/conversations.md) for application inputs,
[concepts](../docs/guide/concepts.md) for terminology, and
[architecture](../docs/guide/architecture.md) for the complete answer lifecycle.
The [agent code map](ARCHITECTURE.md) points to the implementation.

## Current boundaries

- `llgm view` and `llgm.viewer.GraphViewer` provide local graph inspection.
  The viewer shows the selected conversation's current topic, pages original
  turns and journal records, and reuses evidence search. It does not edit memory.

- `LLGM` is the product entry point. `from_settings()` constructs and owns its
  configured resources, reading environment settings at context entry when no
  `Settings` object is passed. Direct construction accepts caller-owned resources.
- `answer()` is the primary conversational API. It saves new user turns and
  nonempty assistant replies, persists the active topic per conversation ID,
  and conservatively routes clear topic changes. `remember=False` queries
  without writing. Routing and inference share an answer allowance.
- `ingest()` catches up on earlier batches. Related sessions can reuse a topic.
  It routes each batch as one unit. An explicit Conversation.node_id retains
  the exact import path. Retry keys protect imported batches.
- Schema 5 stores appended turns separately and pages their coordinates.
  Span reads load one appended turn. Explicit imports and legacy base blobs
  still load whole base sources. Schema 3 requires an explicit copy.
- Primary connections have no relationship type field. Journals retain
  explicit amendment semantics. Model topic and connection quality is unmeasured.
- SQLite remains the default metadata store. Configured application and CLI
  viewer search defaults to BM25 plus ColBERT on Modal. Source generations refresh
  lazily, with exact ColBERT below 64 passages and PLAID from 64 onward.
  Explicit `sqlite_fts5` retains local BM25. Quickstart deliberately selects it
  for a first program. See [node search](../docs/guide/node-search.md) for costs
  and the remaining full-snapshot reindexing limitation.
- Model generation can use native OpenAI, Anthropic, or configured compatible
  endpoints. SDKs stay in adapters. Node readers use DSPy RLM and the default Deno/Pyodide sandbox. Model
  calls and evidence access run on the host through LLGM admission. The main
  remains one ordinary synthesis call. The optional rlm extra installs DSPy
  and Deno. NodeRuntime is the sole answer controller, also available for direct
  seed-based queries. The alternative query runtimes have been removed.
- Model roles are Main, Reader, and Graph. Constructors and settings use
  main_model, reader_model, and graph_model. Reader and graph call budgets and
  trace roles stay separate. Graph maintenance remains the process name.
- Settings accept environment variables, TOML, and Python overrides. Environment
  files must be loaded explicitly. Provider credentials are not settings values.
- Current reads can observe new appends. They do not provide a workspace snapshot.
  Query scope selects edge and journal applicability, not source access control.
- Call, context, and time limits are implemented. Ordinary application answers
  have no dollar spending cap. Unknown provider usage stays unknown.
- General forgetting, history deletion, distributed metadata, and broad
  benchmark superiority are not established product capabilities.

Configuration details belong in the
[configuration guide](../docs/guide/configuration.md). Stable rationale belongs
in [decisions](DECISIONS.md), and unfinished work in
[remaining tasks](REMAINING_TASKS.md).

## Working conventions

Use small replaceable interfaces, docstrings on every Python definition, and
comments that explain a constraint or non-obvious choice. Avoid duplicate public
entry points and abstractions without a demonstrated contract or ownership boundary. Follow
[code standards](../docs/contributing/standards.md) and select meaningful checks through
[testing](TESTING.md).

The user site publishes the guides and API reference for library users.
Contributor operations belong in `docs/contributing/`, benchmark runbooks in
`experiments/`, and shared coding-agent
knowledge here. Private research and machine state are not required inputs to
the library, its tests, or its documentation build.

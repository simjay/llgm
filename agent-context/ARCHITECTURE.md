# Architecture and code map

Use this map to locate the implementation and its contract tests. The
[user architecture tutorial](../docs/guide/architecture.md) explains the answer
lifecycle, and the [API reference](../docs/reference/api.md) owns signatures.
Read the owning code before changing a boundary. This page is a navigation aid,
not a second specification of every method.

## Where to change behavior

| Change | Owning code | Relevant tests |
| --- | --- | --- |
| Application construction, ingestion and seed selection | [llgm.py](../src/llgm/llgm.py) | [Application](../tests/test_application.py) |
| Records, references, settings and time selectors | [core](../src/llgm/core) | [Storage](../tests/test_storage.py), [settings](../tests/test_config.py), [environment files](../tests/test_env_file.py) |
| Source publication, edges and operational journals | [workspace.py](../src/llgm/memory/workspace.py) | [Storage](../tests/test_storage.py), [edges](../tests/test_edges.py), [journal compaction](../tests/test_journal_compaction.py) |
| Raw evidence, amendments and query scope | [evidence.py](../src/llgm/memory/evidence.py), [query.py](../src/llgm/memory/query.py) | [Evidence](../tests/test_evidence.py), [effective reads](../tests/test_effective_reads.py) |
| Relationship proposals and maintenance policy | [maintenance.py](../src/llgm/memory/maintenance.py) | [Application](../tests/test_application.py) |
| Blob storage, local indexing and explicit migration | [storage](../src/llgm/storage), [migration.py](../src/llgm/memory/migration.py) | [Storage](../tests/test_storage.py), [index](../tests/test_lexical_index.py), [migration](../tests/test_storage_migration.py) |
| Seed delegates, child queries and final synthesis | [nodes.py](../src/llgm/inference/nodes.py) | [Node runtime](../tests/test_nodes.py) |
| Shared budgets, result records and Docker execution | [inference](../src/llgm/inference) | [Runtime](../tests/test_runtime.py), [REPL](../tests/test_repl.py), [Docker](../tests/test_repl_docker.py) |
| Provider requests, response handling and embeddings | [models](../src/llgm/models) | [Model contracts](../tests/test_models.py), [live providers](../tests/integration/test_live_providers.py) |
| Passage ranking and remote retrieval | [retrieval](../src/llgm/retrieval) | [Retrieval](../tests/test_retrieval.py), [Modal adapter](../tests/test_modal_retriever.py) |
| Remote ColBERT jobs and persisted index records | [colbert_modal.py](../tools/colbert_modal.py), [colbert_worker.py](../tools/colbert_worker.py) | [Job contracts](../tests/test_colbert_modal.py), [worker](../tests/test_colbert_worker.py) |
| Evaluation preparation, execution and accounting | [evaluation](../src/llgm/evaluation) | [Memory benchmark](../tests/test_memory_benchmark.py), [costs](../tests/test_costs.py), [judging](../tests/test_answer_judging.py) |
| CLI behavior | [cli.py](../src/llgm/cli.py) | [CLI](../tests/test_cli.py) |
| Documentation publication and distribution contents | [check_docs.py](../tools/check_docs.py), [pyproject.toml](../pyproject.toml) | [Documentation](../tests/test_docs.py), [packaging](../tests/test_packaging.py) |

## Preserve the evidence boundary

- Source text and published journal entries retain their original identities.
  Retrieval passages are derived records, not durable evidence identities.
- Publish source bytes before making their metadata visible. A maintenance
  failure must not undo successful source ingestion.
- Primary directed edges establish graph connectivity independently of journals.
  A journal pointer does not create an edge. Ordinary organization proposes
  relationships, while explicit amendments control effective reads.
- `Evidence.read()` resolves original text. `QueryEvidence` applies amendments
  by exact subject, relation, scope, validity and append order. Preserve the
  references of replacement segments and report unresolved replacements.
- Reads use current stored records, not a workspace snapshot. `as_of_ms`
  selects declared validity periods. It does not reconstruct past storage.
- External retrieval results must resolve to canonical workspace evidence.
  Preserve local journal retrieval when injecting a source retriever.

## Preserve the execution boundary

- `LLGM` is the application entry point. Initial retrieval chooses distinct seed
  owners before delegate generation. An explicit `node_id` supplies the sole seed.
- Schedule every admitted seed with bounded concurrency. Children use isolated
  interpreters and working contexts, while sharing budgets and citation tracking.
  Global search remains possible, so a node delegate is not an access boundary.
- Only delivered, selected evidence reaches the final root call. The root has
  no further query phase. Keep skipped seeds and required failure reports visible
  in the result even when the model omits them from its answer text.
- Reserve finalization capacity within the shared budget. Preserve available
  findings on supported failure paths and complete cleanup under cancellation.
- Keep provider response phases distinct. Provider commentary must not become
  executable operation JSON or replace the final structured response.
- `from_settings()` owns resources it creates. Directly supplied workspaces,
  model clients and retrievers remain caller-owned. Close per-operation evidence
  handles on success, failure and cancellation.

## Keep extensions narrow

The default application uses `NodeRuntime`. `RecursiveRuntime`, `IterativeRuntime`
and `RLMRuntime` expose different protocols for direct callers. Share validated
behavior where the contracts agree, without forcing these executors into one
interface or adding application settings for hypothetical modes.

Keep optional provider and GPU imports lazy. The Modal transport validates a
pinned remote index against local corpus identity. Native GPU dependencies belong
in its worker environment. The separate local ColBERT adapter retains its own
explicit prerequisites.

Follow the [code standards](../development/standards.md) for small interfaces,
meaningful docstrings and comments that explain constraints. Follow the
[testing guide](../development/testing.md) to choose checks. Update this map when
ownership changes, and update the owning user guide when behavior changes.

# Code boundaries for agents

Use this map to locate implementation and contract tests. The
[public architecture guide](../docs/guide/architecture.md) owns behavior and resource
ownership. The [API reference](../docs/reference/api.md) owns supported signatures.
Paths in the implementation column are relative to `src/llgm/`.

| Boundary | Implementation | Start with these tests |
| --- | --- | --- |
| Evidence records and references | `core/types.py`, `core/errors.py` | `tests/test_storage.py` |
| Persistence and blob publication | `memory/workspace.py`, `storage/blobs.py` | `tests/test_storage.py` |
| Current evidence access and journal interpretation | `memory/evidence.py`, `memory/query.py`, `core/time.py` | `tests/test_evidence.py`, `tests/test_effective_reads.py` |
| Primary edges, compact journals and explicit conversion | `memory/workspace.py`, `memory/migration.py` | `tests/test_edges.py`, `tests/test_journal_compaction.py`, `tests/test_storage_migration.py` |
| Incremental local index | `storage/lexical.py` | `tests/test_lexical_index.py` |
| Benchmark retrieval | `retrieval/` | `tests/test_retrieval.py` |
| Official ColBERTv2/PLAID | `retrieval/colbert.py`, `retrieval/tokenizers.py` | `tests/test_retrieval.py`, `tests/integration/test_live_colbert.py` |
| Modal retrieval transport | `retrieval/modal.py`, repository-root `tools/colbert_modal.py`, `tools/colbert_worker.py`, `tools/colbert_client_check.py` | `tests/test_modal_retriever.py`, `tests/test_colbert_modal.py`, `tests/test_colbert_worker.py`, `tests/test_colbert_client_check.py` |
| Hosted models and embeddings | `models/` | `tests/test_models.py`, `tests/integration/test_live_providers.py` |
| Iterative retrieval and shared accounting | `inference/iterative.py`, `inference/budget.py`, `inference/results.py` | `tests/test_runtime.py` |
| Executor JSON boundary | `inference/_json.py` | `tests/test_runtime.py`, `tests/test_recursive.py`, `tests/test_rlm.py`, `tests/test_repl.py` |
| Retrieval-first Python node delegates | `inference/nodes.py` | `tests/test_nodes.py`, node Docker integration |
| Structured evidence recursion | `inference/recursive.py` | `tests/test_recursive.py`, `tests/integration/test_live_recursive.py` |
| Python RLM and Docker callbacks | `inference/rlm.py`, `inference/repl.py` | `tests/test_rlm.py`, `tests/test_repl.py`, `tests/test_repl_docker.py` |
| Product API and bounded maintenance | `llgm.py`, `memory/maintenance.py` | `tests/test_application.py`, `tests/integration/test_live_application.py` |
| Query scope and journal-aware navigation | `memory/query.py` | `tests/test_application.py`, `tests/test_evidence.py` |
| Settings, local environment files and CLI | `core/config.py`, `core/environment.py`, `cli.py`, `__init__.py` | `tests/test_config.py`, `tests/test_env_file.py`, `tests/test_cli.py` |
| Experiment preparation and E03 execution | `evaluation/prepare.py`, `evaluation/matrix.py`, `evaluation/runner.py`, `evaluation/retrieval_diagnostic.py` | `tests/test_evaluation.py`, `tests/test_retrieval_diagnostic.py` |
| Controlled comparison and scaling | `evaluation/comparison.py`, `evaluation/scaling.py` | `tests/test_comparison.py`, `tests/test_scaling.py` |
| Primary LongMemEval comparison | `evaluation/memory_benchmark.py`, `evaluation/baselines.py`, `evaluation/benchmark_preflight.py` | `tests/test_memory_benchmark.py`, `tests/test_baselines.py`, `tests/test_benchmark_preflight.py` |
| Physical call and price accounting | `evaluation/costs.py`, `evaluation/framework_calls.py` | `tests/test_costs.py`, `tests/test_framework_calls.py` |
| Historical Python node-pipeline diagnostic | `evaluation/node_pipeline.py` | Product contracts in `tests/test_nodes.py`, retained protocol in `experiments/node-pipeline.md` |
| Passage retrieval and seed-selection diagnostic | `evaluation/node_search.py`, repository-root `tools/node_search_worker.py`, `tools/node_search_summary.py`, `tools/node_search_cases.py` | `tests/test_node_search.py` |
| Experimental model-based node selection | `evaluation/node_selection.py`, repository-root `tools/node_selection_experiment.py`, `tools/node_selection_summary.py` | `tests/test_node_selection.py`, `tests/test_node_selection_experiment.py` |
| Historical frozen-seed answers and current judging | `evaluation/seed_answers.py`, `evaluation/answer_judging.py`, repository-root `tools/seed_answer_experiment.py`, `tools/seed_answer_summary.py`, `tools/seed_answer_qualification.py` | `tests/test_seed_answers.py`, `tests/test_answer_judging.py`, shared cost tests |
| Documentation and packaging | `docs/`, `tools/check_docs.py`, `pyproject.toml` at repository root | `tests/test_docs.py`, `tests/test_packaging.py` |

## Guardrails for changes

The primary benchmark entry point is `evaluation/memory_benchmark.py`. Historical
protocol, reporting and qualification suites were retired where they duplicated
its trust contracts. Preserve their frozen artifacts without restoring redundant
tests. The [node-search guide](../docs/guide/node-search.md) owns current selection
behavior, while the roadmap owns proposed improvements.

- `LLGM` in `llgm.py` is the product entry point. Do not restore application aliases or obsolete top-level modules. Budgets/results belong in `inference/`, persistent operations in `memory/`, and shared records/settings/errors in `core/`.
- Default `LLGM.answer` retrieves unique seed owners before any model call and runs all admitted seeds with bounded concurrency. `answer(node_id=...)` and `query_node` start local reasoning with a node handle. Evidence remains external until read. Child histories are isolated, but global search is permitted. This is not an access-control boundary.
- Preserve immutable source/journal references. Current reads do not provide a workspace snapshot. Exact journal subject/relation/scope and append order determine replacements. Derived retrieval chunks are not durable evidence identities.
- Blob publication precedes metadata visibility. Source ingestion survives maintenance failures, including partial edge publication.
- Retain speaker roles, dates, scope, negation, and attribution through evidence transport. Metadata counts toward admission limits.
- Keep direct caller ownership separate from resources created by `from_settings`. Per-answer evidence handles must close on failure and cancellation.
- Recursive frames share limits, isolate sibling context, and return findings to the parent. Repeated cancellation must still drain cleanup.
- Preserve final-answer phase selection and replay in OpenAI adapters. Commentary must not become executable operation JSON.
- Modal retrieval binds a pinned remote index to a verified local corpus. Remote hits contain IDs/scores, while canonical text and references come from that corpus. Keep the optional SDK lazy and GPU dependencies remote. The worker owns build/open records and native search, the Modal job owns execution and Volume publication, and the local client owns response validation.
- Environment loading reserves `LLGM_TEST_` and `LLGM_REPL_` for their consumers while rejecting unknown application settings.
- Failed/interrupted evaluation attempts remain in artifacts and denominators. Cleanup precedes cancellation propagation and partial-summary finalization.
- Prepared inputs and source/journal payloads declare immutable-node schema 2. Workspace metadata is schema3. The structured-operation comparison requires v3 primary edges. Reject older source-version formats without rewriting retained evidence or frozen protocols.
- Sphinx publishes end-user library documentation only. Contributor and maintainer guidance lives in repository-only `development/`. `research/` is internal, and `experiments/` owns benchmark operator instructions. Do not publish those directories, agent context, or brand explorations through pages, downloads, assets or search indexes.

Keep the integrated node runtime and specialized experimental executors distinct
where their actual protocols differ. They share strict JSON validation and run
accounting. Benchmark `SearchPassage` contains source spans. Runtime
`EvidencePassage` may also refer to journals. Introduce a common interface only
when a demonstrated shared contract justifies it.

Follow the [testing guide](../development/testing.md) for validation commands
and layer boundaries. Read implementation and tests before trusting a handoff
summary. Update this map when ownership changes.

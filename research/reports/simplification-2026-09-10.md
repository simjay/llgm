# Immutable-node and node-query refactor

September 10, 2026. Local implementation and validation record. No hosted model,
Docker, GPU, or live S3 run was performed for this refactor. Earlier reports retain
their own code, schemas, inputs, and measurement scope.

## Implemented change

- Unified the public facade as `LLGM` in `src/llgm/llgm.py`. Configuration and
  shared records/errors live in `core/`, memory behavior in `memory/`, and
  execution/accounting in `inference/`. Experimental controllers are not exported
  as competing package-root application entry points.
- Removed source versions and public commit/read-basis/snapshot contracts. Each
  source occurrence has an immutable identity. Schema 1 workspaces are rejected
  before mutation. No automatic migration or deletion is performed.
- Replaced source-retirement and historical BM25 machinery with ordinary reusable
  SQLite FTS5 search, refreshed through a private change cursor.
- Added explicit same-subject/relation/scope journal overwrites, ordered by local
  append sequence after applicability checks. Ordinary supporting assertions remain
  multi-valued. Original text and older journal entries remain readable.
- Added optional source event `timestamp_ms` and numeric journal `recorded_at_ms`.
  Source ingestion does not require a separate import-time field. Journal audit
  time does not determine overwrite priority.
- Added node-targeted recursive queries and starting-node selection. Tests follow
  real stored links, derive turn references from returned metadata, read a precise
  descendant span, and return it without the ancestor's uncited text or message
  history. Exact node-ID bytes are preserved, including whitespace.

## Validation

| Check | Result |
| --- | --- |
| `make check` | Ruff passed. 1,293/1,293 definitions have docstrings. 481 deterministic tests and 172 subtests passed. Two installed-distribution tests skip in the editable environment. 19 integration tests are deselected. |
| `pytest tests/integration/test_longmemeval_local.py -q` | Five passed against the already available pinned dataset. No download. |
| `make docs` | Strict Sphinx build, source links, public/agent boundary, rendered navigation and API checks passed. |
| Both Python walkthroughs and `examples/offline.py` | Executed successfully with deterministic clients and real SQLite storage. |
| Build sdist and wheel without dependency isolation | Both built successfully. |
| Installed wheel probes in a clean temporary environment | Both tests passed: no eager optional dependencies or import side effects. Modules, metadata, typing marker and CLI entry point present. |
| Strict Twine metadata checks | Both wheel and sdist passed. |

The source package fingerprint below hashes sorted repository-relative Python
paths and file bytes, each separated by a NUL byte:

`8841c3c38259c15245088bb9bdbbafd66e0e63d61b44087b773114c0ddd7b0b0`

The package check initially caught `memory/` being excluded by an unanchored
runtime-data ignore rule. Anchoring that rule to the repository root restored
`llgm.memory` in the sdist and wheel. Installed checks passed after rebuilding.

The comparison v2 protocol preserves v1's cases, model settings, budgets, seed,
and order. Its SHA256 is
`6b406e2172762a5721e2660d314b5a22f857114780461eb61d65e7d92c256971`.

The checks establish storage, reference, interpretation, recursion, packaging,
and documentation contracts. They do not establish model correctness, useful
message compression, graph benefit over retrieval, or the semantic accuracy of
learned journal links.

## Changed experiment boundary

Prepared references and the architecture comparison have a new format/protocol
identity. Existing frozen protocols and reports remain preserved. New runs must
use the revised preparation and protocol. Old scores are not silently relabeled.

## Operational limits

An answer uses current reads and can encounter appends after its initial search.
The journal reducer picks the latest applicable explicit overwrite for an exact
subject. It does not freeze a database-wide instant. Raw journal reads retain
history, and active traversal follows the interpreted view.

A node query is an isolated model computation with local starting evidence.
Children can search globally. Messages are selected and bounded, without a proof
of mathematical minimality or inference convergence. The runtime records an
invocation tree rather than maintaining a separate factor-graph object.

# Code cleanup after the node-runtime update

September 11, 2026. This pass reviewed all library packages and their tests,
application examples, public guides, development tools, and CI configuration
against the simplified node/edge/journal design. It preserved the existing
checkout, including untracked work. The before-copy and validation artifacts
are in ignored `runs/code-cleanup-20260911/`.

## Changes

- Removed the unused iterative settings factory, its resource-ownership state,
  the inert `EvidenceSidecar.mode` argument, the ineffective application
  `search_policy` field, and an unused ColBERT class alias. Configured application
  construction belongs to `LLGM.from_settings()`.
- Separated question answering and experiment dispatch in the CLI. Application
  commands no longer import evaluation runners merely to dispatch a question.
- Moved shared strict JSON parsing into `inference/_json.py`, removing private
  helper dependencies between the product and experimental runtimes. Duplicate
  object keys fail explicitly. `RunLedger.count()` validates accounting results
  across executors, and model events carry attribution at creation.
- Reused scope/time classification for primary edges and journal interpretation.
  Edge filtering no longer creates synthetic journal entries. Timestamp records
  reuse the same scalar validator.
- Centralized SQLite transaction boundaries while preserving blob-before-metadata
  publication, rollback, early idempotent returns, and reference validation.
  Traversal and link acceptance avoid redundant source loads. Effective reads
  reuse unchanged source fragments with their original Unicode offsets.
- Included applicability in automatic link retry identities so distinct scoped
  proposals cannot share a retry key. The model proposal response schema is
  unchanged; the regression supplies scoped `LinkProposal` records directly.
- Fixed normalization of extreme finite embedding vectors and strict embedding
  response-index validation. Hybrid searches drain both components after failure
  or cancellation. Evaluation runners share cancellation-safe native-work draining.
- Replaced stale snapshot, source-version, journal-as-graph and implementation-history
  commentary with actual contracts. Public guides explain the remaining executors
  and client ownership. Ruff formatting and import ordering now have matching
  local and CI checks, with a pinned formatter version.

The four executors remain because they implement different supported protocols:
integrated Python node inference, structured node operations, iterative retrieval
policies, and standalone recursive Python context inspection. Data records,
provider protocols, and resource-owning adapters remain separate where their
contracts differ. No persistent controller, generic registry, new graph layer,
source-version API, or public commit system was introduced.

## Validation

Validation used Python 3.11.13 and Ruff 0.16.6. The source-tree digest, individual
source hashes, frozen protocol hashes, and distribution hashes are retained in
`runs/code-cleanup-20260911/validation.json`.

| Check | Result |
| --- | --- |
| `make check` | 572 passed, 230 subtests passed, two editable-install packaging skips, 20 integration tests deselected |
| Formatting, imports and lint | Passed across 92 files |
| Docstring presence | 1,529/1,529 definitions across 90 code files; wording also reviewed manually |
| `make test-data` | Five pinned local LongMemEval checks passed |
| Real Docker integration | Five tests passed against the existing image `sha256:ec7d6c95cd3692a2e2d228a8b1ca74e4025b54121fcc4c5da6f09cfa473315ad` |
| Strict Sphinx and documentation audit | Passed, including source links, publication boundaries and rendered API/navigation |
| Wheel and source distribution | Built from the source archive, passed strict metadata checks, and matched checkout source bytes |
| Isolated wheel installation | Two packaging tests passed in a fresh environment outside the checkout; ask/migrate CLI help passed |
| Installed offline example | Passed with real Docker: two seeds, one recursive descendant, scoped replacement, and one final root call |

Independent cross-reviews covered changed scope/time, transaction, reference,
JSON, accounting, cancellation and CLI boundaries. Frozen comparison protocols
v1, v2 and v3 retain their recorded SHA-256 digests. Logs and fresh HTML live in
the run directory. Internal research and agent context remain excluded from
distribution archives.

No hosted generation, paid embeddings, ColBERT execution, S3 calls, remote CI,
or publication was performed. Frozen comparison protocols and historical research
measurements remain unchanged. This cleanup does not rerun their experiments.

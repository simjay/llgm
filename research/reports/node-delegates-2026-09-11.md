# Primary edges, operational journals and Python node delegates

Implementation record, September 11, 2026. This updates LLGM's application flow
under the approved architecture plan. It is local mechanism validation, not a
hosted quality measurement or a benchmark result.

## Delivered behavior

`LLGM.answer` retrieves source owners before model execution, applies the declared
seed limit, and runs every admitted seed with bounded concurrency. An explicit
node bypasses retrieval. Each node uses the same local Python delegate mechanism.
Children receive isolated context, and return selected findings and canonical
citations. One final root model combines attributed branch returns.

The shared ledger accounts for retrieval, model calls, evidence, context,
operations, recursion and elapsed time. Collection reserves the final model call
and operation, allocates serialized root context across seed returns, and leaves
20% of remaining query time, capped at 30 seconds, for synthesis. Cleanup can use
that time and a provider can still fail. Skipped seeds and operational failures
remain explicit. Source handles can be paged without sending source text to the
model. The JSON blob implementation still decodes the owning source on the host.

Schema-3 metadata stores primary edges independently of journals. Publication,
withdrawal, duplicate prevention and maintenance preserve directed provenance.
Journal pointers never implicitly become adjacency. Complete operational journals
are loaded before local reasoning. Exact patches resolve directly into canonical
original/replacement segments. Overlaps, missing replacement evidence, unknown
scope/time and amendment cycles remain unresolved without stale-text fallback.

Deterministic compaction removes dominated exact-applicability updates from the
operational sidecar while preserving historical JournalRefs, correction semantics
and retry records. Necessary distinct live patches can overflow its declared
byte limit. No source materialization, LLM-authoritative log summary, public source
version, workspace snapshot, graph framework, distributed service or history
retention/deletion policy was added.

The explicit local schema-2 converter copies into a new workspace after every
pointer has been classified. Original sources, journal identities and retry
records remain intact. It neither mutates the original workspace nor infers a
pointer's role from its relation name. Schema-1 conversion remains unsupported.

## Verification scope

The saved validation artifacts are under `runs/architecture-update-20260911/`.
The final validation manifest records source/distribution hashes, commands and
outcomes. The local interpreter is Python 3.11.13. Docker Desktop 29.7.2 uses an
existing Python image, with no image pulls and no hosted model calls.

| Gate | Result |
| --- | --- |
| `make check` | Ruff clean, 1,509/1,509 docstrings, 555 passed, 2 skipped, 20 integration tests deselected, 191 subtests |
| `make test-data` | 5 passed against the pinned local dataset |
| Actual Docker integration | 5 passed, including source-info paging and recursive node callbacks |
| Fresh `make docs` | Strict build plus source/rendered/publication-boundary audit passed |
| Browser inspection | Architecture Mermaid rendered correctly. Sidebar, component table, walkthrough code and example links inspected |
| Distribution checks | Wheel built from sdist, strict Twine checks passed, internal archive exclusions verified |
| Installed-wheel contracts | 2 passed outside the checkout, side-effect-free imports and CLI help verified |
| Installed-wheel example | Two seeds, one descendant, exact patch, original-source resolution and one final root passed |

The editable-install packaging skips in `make check` are covered separately by
the actual installed-wheel checks. Optional hosted/S3/ColBERT gates were not run.
The delivered wheel's Python files were compared byte-for-byte with current
source. The sdist includes the final public guide and example, with no internal
research or agent context.

The regression suite covers primary-edge independence, source retention on
maintenance failure, scoped canonical patches, complete journals, compaction and
concurrent append, exact timestamp parsing, seed selection/queuing, overlapping
branches, recursive permit safety, failed callback citation visibility, escaped
context admission, final synthesis reservation, failure/cancellation cleanup and
explicit workspace conversion. The converter checks include a live committed WAL
and repeated cancellation while its backup worker is active.

The Docker check executes real Python callbacks and recursive child interpreters
with scripted model responses. The standalone installed-wheel example retrieves
two seeds, follows a primary edge to a third node, applies an exact amendment,
returns selected canonical citations and invokes the root once. These scripted
choices do not measure planning quality.

The public documentation audit checks authored and rendered links, navigation,
API anchors and exclusion of internal content. HTML inspection checks tables,
code blocks, diagrams and example downloads. It does not certify all prose or
remote hosting. Archives exclude research, agent context and agent instructions.

## Experiment and delivery boundaries

`architecture_comparison_v3.json` separates authored primary edges from journal
amendments and uses Unix-millisecond validity. Its SHA-256 is
`5d6baaf49e1bace643591c86e40eca603b160f6136b8c99097de1b0eac278272`.
V1 and v2 remain byte-identical. V3 is a structured-operation navigation ablation,
not a comparison of the full Python seed pipeline. No hosted v3 trial was run.

Earlier hosted results remain historical. No new provider-quality study,
ColBERT execution, S3 integration, remote CI run, documentation deployment or
package publication is included. Total retained history can grow on disk.
Optional working-source materialization and general forgetting remain separate
choices.

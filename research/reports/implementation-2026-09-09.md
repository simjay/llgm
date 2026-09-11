# Initial implementation and experiment preparation

Historical checkpoint: later application/RLM implementation, live checks and the non-ColBERT pilot are recorded in [Live validation](live-validation-2026-09-10.md). The results and next execution gate below describe the September 9 state. Use [Implementation status](../../docs/reference/implementation-status.md) for current capabilities and limitations.

Date: 2026-09-09. Version: `0.1.0a1`, unpublished. This records completed local work, not benchmark findings.

## Delivered

The repository contains a typed, dependency-free Python core with an installable CLI, wheel and source distribution. Immutable conversation versions preserve exact text and stable Unicode source references. SQLite metadata and local blobs support read cutoffs, append-only node journals, provenance, idempotency, and explicit conflict handling. An optional S3 adapter has fake-service coverage. Postgres and distributed operation remain pending.

Hosted model adapters support native OpenAI, native Anthropic, and compatible chat endpoints. Providers and root/sidecar models are independently configurable through typed environment/file settings or Python injection. Tests cover request shapes, refusal/incomplete responses, error paths, ownership, and usage accounting. Actual paid provider integration has not run.

The runtime provides single-query, upfront multi-query, and adaptive iterative evidence gathering, with a root answer model and a smaller-model sidecar. Search, model-call, evidence, bundle, context, and time limits are explicit. Results retain source references, unresolved questions, traces, and known versus unavailable usage. Recursive RLM execution, automatic PEG traversal, semantic reconciliation, and forgetting remain later milestones.

See the [implementation status](../../docs/reference/implementation-status.md), [quickstart](../../docs/guide/quickstart.md), and [configuration guide](../../docs/guide/configuration.md) for the supported API surface.

## E03 preparation

The [matrix](../../experiments/e03_matrix.json) requires twelve configurations: BM25, dense cosine, hybrid rank fusion, and official ColBERTv2/PLAID, each with single, upfront, and adaptive search. The harness prepares isolated case corpora, executes selected configurations, records failures and physical calls, evaluates source/turn coverage, and exports predictions for official scoring. The full preflight checks all twelve configurations. It does not replace a missing ColBERT backend.

The official cleaned LongMemEval-S release was downloaded and hashed. Its 500 questions form one connected group under the declared policy that shared session IDs or identical session text remain in the same split. Therefore this release cannot provide the planned history-disjoint development/evaluation split under that policy. All benchmark IDs remain reserved. A five-case subset was materialized for preparation checks. Separate controlled histories supply 50 development cases, including five smoke cases, with ten additional controlled histories reserved. These authored templates require expansion before supporting research conclusions.

The actual released ColBERT tokenizer was downloaded and pinned, without model weights. Shared passage preparation produced 84 controlled-development passages and 5,706 passages for the reserved five-case preview, all within the 180-token document limit. Original questions reach 69 ColBERT tokens and 61 exceed 32. `query_maxlen=128` preserves those inputs. This choice precedes answer-quality tuning. The model-repository tokenizer revision is distinct from the official ColBERT implementation revision still needed for execution.

Public source pins are in [source_pins.json](../../experiments/source_pins.json).
Dataset preparation used source content and structural labels but did not run
answer-quality comparisons. The five-case preview described here was subsequently
consumed by the [September 10 pilot](live-validation-2026-09-10.md#frozen-bdh-pilot).
Its original reserved status does not apply to later reuse.

## Preparation artifacts

The following ignored directories identify the original preparation outputs.
They are dated records, not prerequisites assumed to exist in a new checkout.

| Path | September 9 contents |
| --- | --- |
| `data/longmemeval/split-audit/` | Strict history-isolation audit with no development queries emitted |
| `data/controlled/prepared-colbert/` | 50 controlled development cases, including five smoke cases, with 84 shared passages |
| `data/longmemeval/evaluation-colbert/` | Five benchmark cases with 5,706 passages, subsequently consumed by the pilot |
| `data/models/colbertv2-tokenizer/` | Released tokenizer/configuration files without model weights |
| `runs/verified-offline-smoke/` | Five deterministic B-S fixture checks and their traces |
| `runs/e03-preflight.json` | Twelve-arm readiness report with missing requirements |

The [runbook](../../experiments/retrieval.md) owns current preparation commands.
This report does not identify a complete source/test archive for its validation
checkpoint. The artifact hashes below identify the original distributions, but
a new build from current code does not reproduce that historical checkpoint.

## Validation completed locally

| Check | Result |
| --- | --- |
| Python 3.11 development environment, with native provider SDKs installed | 95 tests passed, 2 installed-distribution checks skipped, 15 subtests passed |
| Clean Python 3.11 wheel installation, outside the checkout, without provider SDKs | 95 tests passed, 2 optional SDK checks skipped, 15 subtests passed |
| Clean Python 3.14 source-distribution installation, outside the checkout, without provider SDKs | 95 tests passed, 2 optional SDK checks skipped, 15 subtests passed |
| Deterministic offline B-S smoke | Five fixture checks passed. No provider calls. Scores are plumbing diagnostics |
| Ruff | Passed |
| Sphinx/MyST HTML, warnings treated as errors | Passed |
| Wheel/source build and strict Twine metadata checks | Passed |
| Distribution contents and import behavior | No bundled datasets, runs, environment files, or build caches. Optional SDKs/models are not imported eagerly. No import-time network/storage writes |
| Historical research brief | Byte-identical to the original downloaded file |

CI is configured for Python 3.11–3.14, installed artifacts, mocked provider contracts, linting, and documentation. Remote CI has not run. Test coverage and clean installations do not establish production scale or live provider/ColBERT integration performance.

Built artifacts:

- `dist/llgm-0.1.0a1-py3-none-any.whl`, SHA-256 `14d32e0563478ac130d7dd9a17abbe0ccd76caa475ed8fafbecf34bcdfb80759`.
- `dist/llgm-0.1.0a1.tar.gz`, SHA-256 `25de4b922f15bac6da892598ce850d1ee5bf142e634eb99da77bed335d46fa4e`.

## Next execution gate

At this checkpoint, `runs/e03-preflight.json` reported `ready=false` and zero model calls. A live run still needed explicit root/sidecar model IDs, provider credentials, and the full ColBERT checkpoint, checksum, official code revision, optional runtime dependencies, and compatible hardware. No genuine ColBERT index/search, hosted embedding comparison, paid answer run, or official judge run had occurred.

The next action is to complete those pins in a run-specific copy of the matrix, then rerun preflight on `data/controlled/prepared-colbert`. Start with the five controlled smoke cases across all twelve configurations. Inspect failure rates, source recall, and actual usage before widening development. Broaden the controlled workload and set decision thresholds before freezing the configuration for reserved LongMemEval evaluation.

The package is not published. The PyPI name remains provisional. No benchmark superiority, scalability result, or novelty claim has been established.

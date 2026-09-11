# Component tests and verification

Historical checkpoint: later application/RLM implementation, live checks and the non-ColBERT pilot are recorded in [Live validation](live-validation-2026-09-10.md). The results below describe their original measurement. Use [Implementation status](../../docs/reference/implementation-status.md) for current capabilities and [Coverage](coverage-2026-09-10.md) for the subsequent deterministic measurement.

Verified on September 10, 2026, using Python 3.11.13. This report describes software and integration checks, not benchmark performance.

## Implemented coverage

| Mechanism | Tests and implementation |
|---|---|
| Structured recursive evidence access | `tests/test_recursive.py`: root requests and continuation, actual nested parent/child returns, sibling isolation, branch combination, externally addressed sources, source/journal citations, missing evidence, malformed operations, shared budgets, depth/step limits, repeated-request cycles, deadlines and cancellation |
| Evidence and graph operations | `tests/test_evidence.py`: real SQLite/source reads, exact Unicode spans, rechunking, snapshots, restart, inline journal search, pointer resolution, directed navigation, cycles/diamonds, bounded traversal, trusted metadata preservation, stale/missing backend references |
| Journal interpretation | Explicit scope, event-time applicability, corrections, retraction of corrections, historical reads, suggestions and unresolved conflicting replacements. No physical deletion or general forgetting policy |
| Automatic link proposals | Candidate discovery on disconnected nodes, validated endpoint spans, proposal/publication separation, provenance, version checks, idempotency and concurrent journal conflicts. Scripted protocol tests do not establish semantic model accuracy |
| External-context Python REPL | `tests/test_repl.py`: persistent variables, framing, actual trusted-worker callback roundtrips, output limits, invalid frames, host callback budgets, cancellation and cleanup. `tests/test_repl_docker.py` separately checks the real container boundary when enabled |
| Existing library contracts | Source publication failures, retries, configuration, provider payloads/refusals/truncation, embeddings, BM25/dense/hybrid arithmetic, ColBERT asset/provenance validation, corpus isolation and scorer export remain covered |
| Established-dataset sanity | `tests/integration/test_longmemeval_local.py`: complete pinned LongMemEval histories through actual SQLite and BM25, every generated span resolved, unchanged retrieval identity after restart, frozen historical source visibility |
| Optional live services | Native OpenAI/Anthropic schema calls, embeddings, official ColBERT build/search/reopen, two-level recursive hosted inference, official semantic judging, real S3 and Docker tests have explicit gates |

This checkpoint added `SnapshotEvidence`, `RecursiveRuntime`, a conservative journal reducer/link proposal interface, and `DockerREPL`. The structured JSON interpreter and the Python execution adapter have separate contracts. Their presence does not constitute reproduction of the original RLM paper or completion of every proposed LLGM feature.

Two bugs found during testing/review were fixed: dates and speaker metadata were lost at the runtime evidence boundary, and an already-expired tool deadline could leave an unawaited coroutine. Metadata now survives source reads, search, child returns and final evidence bundles, and counts against exposure/context limits.

## Observed results

| Check | Result |
|---|---|
| Full checkout suite | 208 passed, 13 skipped, 109 subtests passed |
| Fresh installed wheel, isolated Python, working directory outside checkout | 208 passed, 13 skipped, 109 subtests passed |
| Local LongMemEval integration | All five tests passed. These are included in the totals above |
| Ruff | Passed |
| Strict Sphinx build | Passed |
| Source distribution and wheel built from that distribution | Passed |
| Twine artifact checks | Passed |
| Enabled live gates with required configuration missing | Fail before provider work. No successful fallback/skip |

The checkout skips comprise seven gated hosted/ColBERT/S3 checks, four Docker checks and two installed-distribution-specific checks. In the fresh wheel run, the two distribution checks pass. Two optional SDK signature checks skip because that clean environment installs only the core library and pytest. No skipped test counts as a passed capability.

## Real dataset scope

The tests verify the complete 277,383,467-byte cleaned LongMemEval-S file against SHA256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` and release revision `98d7416c24c778c2fee6e6f3006e7a073259d48f` before selecting four fixed cases: `001be529`, `01493427`, `00ca467f`, `031748ae_abs`. The selected histories contain 808–843 passages under the diagnostic word tokenizer. This is not the ColBERT tokenizer configuration.

The tests preserve full per-case histories, keep evaluator labels out of stored/model-visible sources, and report source/turn coverage only after retrieval. They do not require a tuned minimum retrieval score. The fifth test performs a controlled source replacement on a real history to check snapshot behavior.

The hosted recursive integration test, when enabled, supplies evaluator-selected whole source handles and explicitly requires two delegation levels. It measures the execution protocol. A separate official judge consumes its predictions and applies the declared small-sample acceptance threshold. Oracle handles, forced delegation, and tiny sample sizes must remain disclosed. None establishes autonomous retrieval quality or a benchmark advantage.

## Unexecuted checks at this checkpoint

No hosted model/embedding or official judge requests were made at this checkpoint.
The hosted, ColBERT, S3 and Docker integrations lacked their configured
prerequisites or were not enabled. Their implementations had no live result in
this measurement. Later live outcomes are recorded in the
[validation report](live-validation-2026-09-10.md).

Dollar-denominated admission, a complete paper-compatible RLM controller, automatic link quality measurements, production reconciliation, incremental index scaling and distributed metadata remain separate work. Unit tests verify the implemented contracts rather than claiming those research questions are solved.

See [Testing](../../development/testing.md) for exact commands, environment flags, model/checkpoint/evaluator pins and artifact locations. A manual GitHub Actions workflow in `.github/workflows/benchmark-sanity.yml` can download the pinned public dataset and rerun the local corpus checks without model API credentials. That workflow was authored but not dispatched.

Those instructions describe the current checkout. This report does not identify
a complete retained source/test checkpoint for its 208-test measurement, so
current test commands do not reproduce its historical denominator.

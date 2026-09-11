# Research reports

Dated records of completed work and observed results. Each report retains its
original measurement scope. A later code change does not rerun an earlier study.
Current capabilities and limitations are maintained in the
[implementation status](../../docs/reference/implementation-status.md).

| Date | Report | Scope |
|---|---|---|
| September 11, 2026 | [LongMemEval five-question pilot](longmemeval-2026-09-11.md) | LLGM 2/5, BM25 4/5, full context 3/5, concrete evidence-delivery failures and costs, with the earlier full run stopped |
| September 11, 2026 | [Final answers from frozen seed choices](seed-answers-2026-09-11.md) | Matched first-owner and selector answers, official-prompt judging, evidence-delivery failures, and a separate delegate qualification |
| September 11, 2026 | [Passage retrieval and node selection](node-search-2026-09-11.md) | Actual BM25/PLAID cutoff comparison, source-versus-seed coverage, controlled crowding/scope probes, and benchmark-identifier correction |
| September 11, 2026 | [Small-model node selection](node-selection-2026-09-11.md) | Hosted selection over BM25, ColBERT and fused candidate pools, paired controls, repeated trials and priced usage |
| September 11, 2026 | [Remaining work and limits](remaining-work-2026-09-11.md) | Prioritized semantic, evaluation, cost, storage, integration and release gaps after runtime hardening |
| September 11, 2026 | [Node runtime hardening](node-runtime-hardening-2026-09-11.md) | Branch finalization, local gaps, edge descriptors and recovery fixes, with original/focused hosted cohorts and controlled delivery checks |
| September 11, 2026 | [ColBERTv2 and PLAID on Modal](colbert-modal-2026-09-11.md) | Real GPU indexing, fresh-container reopen, canonical passage verification, local client and three-case BM25 comparison |
| September 11, 2026 | [Full node-pipeline live diagnostic](node-pipeline-live-2026-09-11.md) | Two complete hosted cohorts, independent semantic/citation review, actual recursion and budget-return failure analysis |
| September 11, 2026 | [Code cleanup](code-cleanup-2026-09-11.md) | Removed unused construction paths, consolidated shared rules, corrected comments and verified behavior |
| September 11, 2026 | [Primary edges and Python node delegates](node-delegates-2026-09-11.md) | Retrieval-first orchestration, effective amendments, operational compaction and local delivery checks |
| September 10, 2026 | [Immutable nodes and node queries](simplification-2026-09-10.md) | Unified API, organized modules, journal overwrites, and local refactor validation |
| September 10, 2026 | [Coverage](coverage-2026-09-10.md) | Deterministic coverage, denominators and earlier measurements |
| September 10, 2026 | [Architecture improvements](architecture-2026-09-10.md) | Incremental indexing, application contracts and local scaling |
| September 10, 2026 | [Live validation](live-validation-2026-09-10.md) | Hosted models, Docker execution, integrated memory and the B/D/H pilot |
| September 10, 2026 | [Component verification](components-2026-09-10.md) | Earlier component and local dataset checks before live validation |
| September 9, 2026 | [Initial implementation](implementation-2026-09-09.md) | Package delivery and retrieval experiment preparation |

Generated datasets, traces and build artifacts are ignored by Git. Reports name
their run directories and record the evidence needed to interpret their results.
See the [research index](../README.md) for design proposals and source material.

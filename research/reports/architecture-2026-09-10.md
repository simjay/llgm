# Architecture improvements

September 10, 2026. This records an implementation-hardening checkpoint.
It does not replace the earlier live results or claim better answer quality.
API names and storage behavior below describe that checkpoint. See
[implementation status](../../docs/reference/implementation-status.md) for the
current package after subsequent refactoring.

## What changed

| Before | Now |
|---|---|
| Editing metadata on a search result could change later reads from the same snapshot. | Returned metadata is independent. Original dates, scope and provenance remain intact. |
| The iterative reader could lose separately stored attribution and scope. | Text and metadata travel together and both count toward limits, including fallback evidence. |
| Different answer paths treated missing evidence and errors differently. | Unresolved answers are partial. Expected operational errors return failed/budget-exhausted results with attempted usage and evidence. |
| Repeated cancellation could detach direct REPL cleanup. | Startup, execution and close finish cleanup while retaining ownership, even under repeated cancellation. |
| Each application answer reread the whole corpus and rebuilt an index. | A saved local index processes changed commits. Source text is read on demand. |
| The integrated application could not accept another retriever. | One explicit evidence factory serves both maintenance and answers, with snapshot and ownership checks. |

`MemoryApplication` is the recommended application entry point. The three
execution mechanisms remain separate because they test different behaviors.
Shared admission checks now live in `RunLedger`. No generic runtime framework,
plugin registry or additional factor-graph layer was added.

The intentional alpha API change is in `LLGM.answer`: expected operational
`LLGMError`s now return failure results, matching structured recursion. Invalid
caller input, unexpected errors and cancellation still raise. Direct
`EvidenceSidecar.gather` remains a lower-level interface that may raise operational
errors. Corrected metadata accounting can change future admission decisions.
Previous frozen experiment results are historical observations of their original code.

## Local performance measurement

Two actual local runs used 100, 1,000 and 10,000 records, first with 256-character
records and then with 4,096-character records. The table below uses the longer
records: about 41 million source characters at the largest size.

| Records | First index build | Reuse unchanged index | Process one replacement | Reopen after restart | Median warm search |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.080 s | 0.246 ms | 1.391 ms | 1.011 ms | 0.292 ms |
| 1,000 | 1.504 s | 0.333 ms | 1.931 ms | 1.084 ms | 2.823 ms |
| 10,000 | 15.544 s | 0.304 ms | 1.482 ms | 1.163 ms | 25.912 ms |

At each size, preparation read all source blobs on the first build, zero on an
unchanged reopen, exactly one after a replacement, and zero after restart.
Historical results and scores were unchanged after the replacement. Restarted
results matched the latest view. These are preparation and local-search times,
not LLM answer latency. No model calls were made.

The histories are generated local storage workloads, not conversation-quality
benchmarks. The search probe uses a unique term and three repetitions per phase.
The table reports the median of the unchanged-index warm phase. Timings
come from this machine without isolation from concurrent work. Python allocation
measurements exclude SQLite/native memory and are not process RSS. The current
scorer still scans visible passage metadata for corpus statistics: the table
does not establish constant-time search or distributed scale.

The workload can be rerun with a new output directory:

```bash
python tools/probe_index_scaling.py --output runs/scaling-new \
  --sizes 100 1000 10000 --source-chars 4096
```

Recorded artifacts are `runs/scaling-20260910/` and
`runs/scaling-long-records-20260910/`, including manifests, source-code identity,
phase timings, byte/read counts, index descriptors and historical checks.
These generated directories are ignored by Git. Their manifests retain source
hashes, but this report does not identify a complete source archive for the
measured checkpoint. The recorded Git commit alone is insufficient because the
implementation was untracked when measured. Running the current tool creates a
new measurement with its current storage contracts. It does not reconstruct the
historical implementation or establish identical timings. Exact historical
reproduction requires recovering the matching source and environment first.

## Tests and packaging

- 422 deterministic tests and 159 subtests passed. Two installed-package checks
  skipped in the editable environment.
- Five separate tests using checksum-pinned LongMemEval histories passed.
- Coverage: 89.36% of lines and 79.55% of branches across the full package.
- Ruff, all 1,228 checked docstrings, strict Sphinx and rendered API/navigation
  checks passed.
- Wheel and source distribution builds passed metadata validation. Both
  installed-package probes passed in a clean core-only environment outside the
  checkout, without downloading dependencies.

The new index tests cover updates, empty replacements, concurrent refresh,
restart, rollback, cancellation, metadata isolation and historical ranking.
Its visible-corpus BM25 scores were also checked against SQLite FTS5 on an
independent corpus. The index descriptor identifies `sqlite-visible-bm25`.
E03's separate SQLite FTS5 retriever and tokenized passages remain distinct.

## Ready experiment and remaining work

The [frozen comparison](../../experiments/architecture_comparison_v1.json) and
[run instructions](../../experiments/comparison.md)
cover five independent development questions and three approaches: one search,
recursive search, and recursive search with graph navigation. They share source
and journal evidence, model pair and limits. Scoring uses exact answers and
reference coverage. Failures and unknown usage remain visible.

Readiness recorded **zero model calls** at this checkpoint. This
comparison has been prepared and contract-tested, but has no live quality result.
Its curated links isolate navigation. Measuring automatically generated link
quality is still a separate task. ColBERT remains deferred.

Other remaining limits include growing index history on disk, large individual
source/journal reads, distributed storage, compaction and metadata backup/restore.
After restoring metadata, close the workspace and discard its old derived
`.indexes` directory before reopening. The library does not yet implement an
automatic restore protocol or general semantic forgetting.

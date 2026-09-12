# Remaining tasks

This backlog records open work, not an execution order. The current request
sets priority. [Capabilities and limits](../docs/reference/implementation-status.md)
owns supported behavior, and [experiment runbooks](../experiments/README.md)
own evaluation commands. Close tasks with reproducible evidence of their outcome.

## T05: Measure relationship and amendment quality

Structural validation exists. Broader semantic quality remains to be measured.
Evaluate valid and spurious links, entity collisions, scope, time and unsupported
updates. Start with [maintenance](../src/llgm/memory/maintenance.py) and
[evidence operations](../src/llgm/memory/evidence.py).
**Done when:** Frozen labeled cases yield retained predictions, support checks,
precision, recall and failure analysis distinct from structural validation.

## T06: Define application cost admission

Application limits bound calls and time, while evaluation has separate priced
admission. Extend the application contract while preserving unknown usage. Reuse relevant
[evaluation accounting](../src/llgm/evaluation/costs.py) contracts without conflating them.
**Done when:** Price identity, admission, reconciliation, cancellation and missing
usage have observable tests. Estimates are clearly separate from provider bills.

## T07: Improve complementary retrieval

Both retriever adapters exist. Combined retrieval is not an application default.
Compare BM25, ColBERTv2/PLAID and combined candidate pools under fixed seed and
evidence limits. The [LongMemEval runbook](../experiments/longmemeval.md) locates tooling.
**Done when:** Retained rankings and matched answer trials separate candidate
coverage, seed choice, evidence delivery and answer quality, including failures.

## T08: Establish independent evaluation histories

Freeze development and evaluation membership with history overlap checks,
distractors and reproducible source pins in the [evaluation tools](../src/llgm/evaluation/).
**Done when:** Isolation rules and overlap audits support the claimed split, and
all attempts retain the source, prompt, model and scoring identities needed to reproduce them.

## T09: Compare architectures and model allocation

LLGM, BM25 and full-context reader arms exist. Other framework integrations remain
incomplete. Add matched controls and complete integrations around the
[answer runner](../src/llgm/evaluation/memory_benchmark.py). Transport tests alone are insufficient.
**Done when:** Each declared arm runs end to end with comparable source access,
models and limits, reporting preparation, maintenance, inference and judging separately.

## T10: Define forgetting before implementing it

Choose between suppressing stale evidence, compacting derived state and deleting
history. Existing operational compaction preserves original sources and raw journals.
**Done when:** A defined policy has counterexamples and measurable tradeoffs for
retrieval, provenance, query cost and retention, with visibility distinct from deletion.

## T11: Validate storage growth and recovery

Measure growing histories and complete restore and service-boundary contracts.
Start with [storage guidance](../docs/guide/configuration.md#storage-choices) and [scaling probes](../src/llgm/evaluation/scaling.py).
**Done when:** Recovery preserves references and retry semantics under failures,
and measurements separate indexing, source reads, Docker overhead and concurrency.

## T12: Complete release delivery

The PyPI workflow validates release tags, calls the existing CI workflow and
publishes its tested distribution artifact through Trusted Publishing. The
GitHub `pypi` environment and PyPI publisher registration are external setup
requirements. Adding the workflow does not establish successful publication.

Validate installable artifacts, package metadata, README assets and versioned
documentation through the [publishing guide](../development/publishing.md).
**Done when:** A released distribution reproduces documented workflows outside
the checkout, and release documentation states the capabilities actually verified.

## T13: Measure topic boundaries and gigantic-node reasoning

Conversation routing and segmented turn storage are implemented. Deterministic
checks protect node reuse, split decisions, restart, citations and bounded span
I/O. Hosted routing quality and huge-node answer quality remain unmeasured.
Imported batches route as one unit. Splitting within a mixed-topic batch remains
open. Compare archive size and node size independently with a direct-reader
control. **Done when:** Retained measurements cover false splits, missed splits,
topic returns, large turns, per-node RLM work, graph growth and complete costs.

# Remaining tasks

This backlog records open work, not an execution order. The current request
sets priority. [Capabilities and limits](../docs/reference/implementation-status.md)
owns supported behavior, and [experiment runbooks](../experiments/README.md)
own evaluation commands. Close tasks with reproducible evidence of their outcome.

## T15: Future plan for a simpler, incrementally maintained LLGM

This is the coordinated follow-up plan for product direction, application
simplification, memory maintenance and evaluation. The work is planned. It does
not establish a new implemented contract or authorize a paid benchmark run.
Keep detailed research designs and dated measurements in local research.
Update [design decisions](DECISIONS.md) only when a replacement contract is settled.

Start with the [conversation pipeline next steps](../docs/contributing/next-steps.md)
for the bounded execution-contract review and implementation gap list. It records
the proposed six-stage flow and acceptance scenarios before the broader work below.

### 1. Define the graph's computational purpose

- Write one governing product contract and a falsifiable advantage over a strong
  filesystem-backed RLM with comparable retrieval, caching and model access.
- Evaluate sparse dependency tracking, reusable local findings and selective
  recomputation as the initial structural hypothesis. Specify what nodes,
  dependencies and returned messages mean. Generic navigation links do not
  establish computational dependencies or conditional independence.
- If pursuing probabilistic graphical inference, first define the variables,
  factors, message operations and assumptions under which inference is correct.
  Measure extraction errors, correlated evidence, cycles and approximation
  error separately. Natural-language summaries do not inherit sum-product
  exactness or convergence guarantees.
- Start with a small controlled domain and repeated questions between updates.
  Compare full recomputation with incremental results, including missing
  dependencies, new relevant evidence and changes that affect the whole graph.

**Done when:** A frozen experiment can establish or reject the proposed graph
benefit through supported answer quality, update freshness and complete cost.
Keep only mechanisms justified by the result. Coordinate with T05 and T09.

### 2. Simplify the public API and its implementation

- Specify complete first-use examples before changing signatures: open memory,
  continue a named conversation, import recorded turns, ask without writing,
  inspect evidence and close resources. Use those examples to define the API.
- Separate conversational writes from read-only queries. Bind persistent chat
  identity to a conversation handle instead of repeating routing controls on
  every call. Give common inputs one obvious shape and keep exact source imports
  in the advanced storage interface.
- Provide simple model and storage defaults. Keep provider setup, reader tuning,
  maintenance policy, time and scope selectors, budgets and interpreter factories
  out of ordinary calls. Expose advanced options only at their owning boundary.
  Replacing many keywords with an equally confusing options object is insufficient.
- Evaluate a synchronous default and an explicit asynchronous client with matching
  behavior. Define resource lifetime and event-loop ownership once. Avoid separate
  inference implementations or a new event loop for each call.
- Reduce [llgm.py](../src/llgm/llgm.py) to application coordination. Move connection
  proposal and publication policy into the existing maintenance boundary, compose
  resources in one place, and centralize update handling. Preserve one answer
  controller. Remove duplicate state, unnecessary wrappers and unused public exports.
- Audit every public symbol for an actual use case, an owning guide, examples,
  side effects and failure behavior. Update the contributor code map and remove
  obsolete API guidance as interfaces change.

**Done when:** The ordinary workflows run from concise documented examples and
the facade no longer implements storage policy, graph algorithms or duplicated
runtime accounting. Contract tests protect persistence, retries, concurrency,
cancellation, provenance and ownership through the new interface.

### 3. Unify live updates and historical replay

- Define one memory-update contract for incoming live turns and imported history.
  Resolve the current difference where ingestion invokes organization after each
  batch but conversational answering does so only for newly created topic nodes.
- Drive affected-evidence discovery from appended turns and their dependencies.
  Current organization searches from a bounded node prefix, which can miss the
  significance of later appends to a large topic.
- Define maintenance timing, completion and failure visibility. Preserve committed
  source turns when organization fails and make deferred work observable.
- Replay original user and assistant exchanges without generating replacement
  replies. Compare whole-session imports with incremental exchange replay and
  specify topic handling within mixed-topic imports.
- Preserve the distinction between navigation, dependency invalidation and exact
  journal amendments. Explicit corrections are implemented. Automatic semantic
  amendment generation requires its own policy and quality evidence.

**Done when:** Equivalent recorded updates exercise the same declared maintenance
rules through live and replay paths. Tests include existing-topic appends, late
cross-topic dependencies, corrections, duplicate delivery and interrupted work.
Coordinate with T05, T10 and T13.

### 4. Validate the conversational lifecycle

- Exercise topic continuation, changes and returns, interleaved conversations,
  active-topic read-only queries, restart and growing histories.
- Check intermediate memory states and answers after updates, including stale
  claims, contradictory sources, scoped changes and historical questions.
- Measure topic placement and large-node reading separately from archive size.
  Report memory growth, indexing work, maintenance work and latency per update.
- Exercise configured retrieval separately from the benchmark's local BM25 path.
  Distinguish controlled-model tests, real sandbox checks and hosted quality.

**Done when:** Retained lifecycle evaluations reveal failures that a single
question after history import cannot, with freshness and resource measurements
for each declared configuration. Coordinate with T07, T11, T13 and T14.

### 5. Freeze an evaluation that can distinguish the mechanisms

- Audit supplied session order against timestamps. Declare release-order replay
  and chronological replay explicitly and freeze a new protocol for any change.
- Keep the final question, gold labels and identifying annotations out of memory
  construction. Preserve original roles, text, dates and session attribution.
- Compare a retrieval reader, filesystem-backed RLM, LLGM without graph use and
  the proposed graph mechanism. Match models, retrieval, evidence access, caching
  opportunities and budgets, or declare a separate cost-quality comparison.
- Include repeated-query and update workloads alongside LongMemEval. Report useful
  dependency traversal, finding reuse and invalidation accuracy, not edge counts
  alone. Include disconnected and misleading graphs and correlated source repeats.
- Freeze development and evaluation membership with history-overlap checks.
  Retain all attempts and separate construction, updates, answering and judging
  costs. Measure current topic construction and DSPy execution under their own
  identities. Historical results cannot validate a later implementation.

**Done when:** Complete denominators and retained artifacts support an independent
comparison and a clear decision about which graph features deserve to remain.
Coordinate with T08, T09 and T14. Keep frozen protocols and commands in
[experiments](../experiments/README.md).

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

Combined retrieval is now the configured application and CLI viewer default.
Live workspace transport and generation reuse have deterministic contract tests.
The new tiny-corpus native ColBERT integration test remains an explicit live gate.
Full source snapshot uploads and PLAID rebuilds on source changes remain a scale limitation.
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
and measurements separate indexing, source reads, sandbox overhead and concurrency.

## T12: Complete release delivery

The PyPI workflow validates release tags, calls the existing CI workflow and
publishes its tested distribution artifact through Trusted Publishing. The
GitHub `pypi` environment and PyPI publisher registration are external setup
requirements. Adding the workflow does not establish successful publication.

Validate installable artifacts, package metadata, README assets and versioned
documentation through the [publishing guide](../docs/contributing/publishing.md).
**Done when:** A released distribution reproduces documented workflows outside
the checkout, and release documentation states the capabilities actually verified.

## T13: Measure topic boundaries and gigantic-node reasoning

Conversation routing and segmented turn storage are implemented. Deterministic
checks protect node reuse, split decisions, restart, citations and bounded span
I/O. Hosted routing quality and huge-node answer quality remain unmeasured. The
[conversation guide](../docs/guide/conversations.md) owns the current user contract.
Imported batches route as one unit. Splitting within a mixed-topic batch remains
open. Compare archive size and node size independently with a direct-reader
control. **Done when:** Retained measurements cover false splits, missed splits,
topic returns, large turns, per-node RLM work, graph growth and complete costs.

## T14: Measure DSPy reader quality

The node readers use DSPy's loop with the
Deno/Pyodide sandbox. Deterministic contracts and real sandbox checks protect
execution, citation admission and cancellation. Hosted quality has not been
measured for this controller. Use the new DSPy protocol variants in the
[runbook](../experiments/longmemeval.md), preserving historical Docker runs.
**Done when:** Approved hosted trials retain complete predictions, failures,
source and runtime identities, costs and comparisons to the frozen controls.

# LLGM roadmap: prove that the architecture is useful

Updated September 11, 2026. This is the single working roadmap. The benchmark
choices and numerical target below guide the implementation phase. The user has
authorized execution, with LongMemEval as the single primary benchmark and a
$500 API-spend ceiling for the comparison. The latest instruction is to improve the same
five development questions until all are correct, then run full test suites. A full-benchmark result is not yet established.

## Current execution

The active work consolidates testing around ordinary LongMemEval ingestion,
retrieval and answering. Nine historical protocol/report/qualification suites
have been removed. Product storage, journal, runtime, isolation and citation
regressions remain. Shared priced-call accounting and one comparison runner are
implemented. Paid plumbing runs exposed two root-synthesis failures and a Docker
cleanup failure. Root citation admission now constrains the actual returned
evidence, and a separately reproduced Docker removal race has a verified fix.
The historical smoke failure's exact cleanup cause remains unproved. A repeated
53-session construction pilot completed all maintenance outcomes after repairing
native proposal-schema forwarding, versus 11 failed outcomes before that fix.
Both runs are retained. This is readiness evidence, not benchmark superiority.

The full run was stopped at the user's request, with its records preserved.
The completed pilot selected five questions spanning recall, updates, time and
abstention. LLGM scored 2/5, BM25 4/5 and full context 3/5, for $3.139091 including
judging. LLGM failed to extract available facts from admitted nodes and combined
dates incorrectly. It made no recursive child calls. The
[pilot report](reports/longmemeval-2026-09-11.md) explains those failures and the
repair to local fact collection and final synthesis. Follow-up v2 regressed to 1/5 because reads were not printed. V3 fixed that
interface and reached 3/5. V4 retained 3/5, with remaining losses in synthesis.
V5 presents exact dated evidence before summaries and reached 4/5 with a GPT-5.4
root and GPT-4.1 delegates. V6 reached 5/5 in one fresh run by enabling medium
reasoning in that final root call, with unchanged prompts. Full deterministic
checks passed after the five-case gate. The default benchmark command selects
this v6 development protocol. This model
allocation differs from the initial controls, so their scores are unmatched.
The pilot questions are exposed development data and do not establish general
accuracy. The [node-search improvement order](#node-search-improvement-order) below
separates this work from changes to candidate retrieval. Mem0 OSS and Graphiti integration is paused and
still requires instrumented preparation, embedding and reranking calls before
entering a trustworthy cost comparison. These control arms do not satisfy
the named-framework win. LoCoMo and other experiment expansion are deferred until
the LongMemEval evaluation is established. Operator commands and exact limits
belong in [the LongMemEval workflow](../experiments/longmemeval.md).

## 1. The goal

**We want LLGM to answer more long-term memory questions correctly than Mem0
and Graphiti on LongMemEval-S, without increasing total cost per question.**

That is the usefulness claim we are trying to establish. Passing our own tests
gets the implementation ready to compete. Beating credible alternatives under
the same evaluation conditions demonstrates why someone should use LLGM.

The proposed initial success target is **at least five percentage points higher
answer accuracy than the strongest reproduced baseline at no higher mean total
cost per question**. Five points is a suggested engineering target worth pursuing,
not an official benchmark threshold. Freeze it with the experiment protocol
before evaluating the final candidate.

For example, if the strongest comparison gets 425 of 500 questions right, our
target is at least 450, with no increase in mean total cost. Those numbers are
illustrative. We have not measured those framework scores in this repository.
If accuracy improves only by spending substantially more, report that tradeoff
and continue improving. The stated goal has not yet been met.

### What we will compete on

| Role | Choice | Reason |
| --- | --- | --- |
| First public benchmark | The pinned cleaned LongMemEval-S release, all 500 questions | It exercises long-term conversational recall, multi-session reasoning, time, updates and abstention. The repository already has its data loader and evaluator integration. |
| Deferred independent confirmation | LoCoMo question answering | A possible later workload after the LongMemEval evaluation is established. No current expansion or execution is planned. |
| Direct memory-framework comparisons | Mem0 open-source library and Graphiti | They are concrete alternative memory systems with inspectable implementations. |
| Simple controls | BM25 plus a reader, full-context reading where it fits, and iterative retrieval | They reveal whether LLGM adds value beyond finding text and giving the reader more opportunities. |
| Recursive control | The authors' RLM implementation, configured with isolated execution | It tests whether persistent graph organization and journals add value beyond recursive context inspection. |

Benchmark definitions and implementations are available from the
[LongMemEval authors](https://github.com/xiaowu0162/LongMemEval),
[LoCoMo authors](https://github.com/snap-research/locomo),
[Mem0](https://github.com/mem0ai/mem0),
[Graphiti](https://github.com/getzep/graphiti), and
[RLM authors](https://github.com/alexzhang13/rlm).

Mem0 OSS is a specific comparison. It is not a substitute for testing Mem0 Cloud.
Graphiti is likewise distinct from the managed Zep product. Label the exact
implementation and configuration. Published vendor numbers can appear in a
separate reference table, but our primary comparison reruns systems under one
declared protocol. A higher number under different models or question subsets
does not establish a win.

LongMemEval-V2 also exists and addresses memory over agent trajectories. It is a
later target if we extend the claim to that workload. Start with the conversational
memory comparison already supported by our data pipeline. See its
[official repository](https://github.com/xiaowu0162/LongMemEval-V2).

## 2. What LLGM is supposed to do better

Consider this history:

| Record | Source text |
| --- | --- |
| Original decision | Atlas production uses PostgreSQL. Deployment details are in the registry record. |
| Another environment | Atlas staging uses SQLite. |
| Later correction | Atlas production now uses MySQL, replacing the earlier PostgreSQL decision. |
| Registry | Atlas production is deployed in eu-west-1. |

The question is: **What database and region does Atlas production currently use?**

LLGM should retrieve an entry node, apply the relevant journal correction while
reading it, follow the primary edge to the registry when needed, and return
**MySQL, eu-west-1** with the supporting references. The staging fact stays
separate. The root combines the selected findings.

This example has three possible sources of value:

1. Edges help find a related fact that initial retrieval missed.
2. Journal interpretation prevents an old fact from being presented as current.
3. Node delegates combine facts while sending only relevant findings upward.

We need to measure all three. If a simple reader gets the same answer more
cheaply, this example provides no advantage for LLGM. Harder examples must
demonstrate the benefit without being constructed from evaluator answers.

The execution machinery already works on small cases. Earlier focused runs
passed 15 original LLGM cases, 12 focused autonomous cases and five controlled
mechanism checks. Those runs used curated relationships and patches. They do
not yet establish the performance of automatically constructed memory on a
public benchmark. The exact results remain in the
[runtime hardening report](reports/node-runtime-hardening-2026-09-11.md).

## 3. First deliverable: one trustworthy comparison runner

**Problem in plain language:** We can now run ordinary LLGM, BM25 and full context
through one pinned LongMemEval comparison. We cannot yet obtain a fair
LLGM-versus-Mem0-versus-Graphiti answer table. The remaining wrappers need to
account for each physical extraction, embedding and reranking request, including
failed requests and library fallbacks.

**Implementation:** The canonical runner reuses the LongMemEval parser and
artifact writer and calls the actual `LLGM.ingest()` and `LLGM.answer()` path.
Add experiment-only wrappers around the external systems next.
Each wrapper needs to prepare a history, answer a question, report usage, and
close its resources. A handful of functions and records is enough.

The flow should be understandable as:

```text
for each benchmark history:
    create an isolated workspace for each comparison system
    ingest the supplied history and measure construction cost
    for each question belonging to that history:
        ask every system using its frozen configuration
        save the answer, status, usage, latency and available evidence
    close each workspace

score the saved answers using the same frozen evaluator
write one comparison table, including failures
```

Memory preparation receives history only. It cannot inspect the question's gold
answer, evidence labels or answer-bearing source annotations. For systems with
query-time organization, the question becomes available only at query time and
all extra work is charged. Evaluation questions and generated answers are not
added back into persistent memory before later questions.

**Concrete files:** `src/llgm/evaluation/memory_benchmark.py` owns the loop.
`evaluation/baselines.py` owns the common BM25/full-context readers.
`evaluation/costs.py` records and prices every attempted provider request.
`experiments/longmemeval_v1.json` freezes the current control comparison settings.
External framework adapters remain unimplemented. Keep their optional dependencies
isolated from the core library.

**Definition of done:** Ten separately authored smoke questions go through each
wrapper, every attempt produces an answer or explicit failure record, and the
same evaluator scores all records. This gate checks the runner, not whether
every model answers every smoke question correctly.

Bring up LLGM, BM25 and full-context reading first. Add Mem0 and Graphiti next.
Add iterative retrieval and the authors' RLM control before drawing a conclusion
about the value of LLGM's graph and recursion.

### The rules that make the comparison fair

| Question | Concrete rule |
| --- | --- |
| Which model answers? | Use the same final reader model. Start from the tested root/sidecar pair where supported, and pin exact settings after smoke compatibility checks. Keep configurable extraction/delegate models from the same declared pool. Label unmatched configurations separately. |
| What history is available? | The same source turns, roles, timestamps and ordering. Preserve distinct occurrences of repeated session IDs. Each system constructs its own memory from that history. |
| Can LLGM receive manually supplied edges or patches? | Only in separately labeled diagnostics. The scored end-to-end arm must construct its own relationships and amendments or report that a mechanism was unused. |
| How is correctness measured? | Export LongMemEval predictions as `question_id` and `hypothesis`, then use a pinned official evaluator and judge for every arm. Include abstention questions and operational failures in the answer denominator. |
| What is fixed? | Dataset checksum, framework revisions, model settings, prompts, retrieval parameters, budgets, evaluator and repetition count. Give baselines a recorded development-tuning opportunity. |
| What if a run fails? | Retain and score the declared attempt. Report provider interruptions separately. Do not silently retry until a passing answer appears. |
| What about citations? | Audit LLGM's reference claims separately. Missing native citations in another framework do not automatically make an otherwise correct benchmark answer wrong. |

The LongMemEval output/evaluator contract is documented in the
[official instructions](https://github.com/xiaowu0162/LongMemEval#testing-your-system).

## 4. Second deliverable: know what each answer really costs

**Problem in plain language:** A system can look cheap at query time because
it already spent many model calls building its memory. Comparing only the final
answer call would reward hidden work.

**Current implementation:** Shared priced-call accounting records preparation,
answering and judging requests, including unsuccessful and unknown-usage calls.
The runner stores the price basis, reservations and measured usage. Extend this
boundary to each physical extraction, embedding and reranking call when external
frameworks are added. Local compute, storage and a provider invoice are still
separate from these API estimates.

For each history, calculate:

```text
total cost per question =
    (memory construction + all query costs)
    / actual number of evaluated questions for that history
```

If a history is used for one question, charge its whole construction cost to
that question. If it is used for twenty, divide construction across those twenty.
Do not assume hundreds of future questions to make the system appear cheaper.
Also show query-only cost so the construction tradeoff is visible.

Include paid compute and report local CPU/GPU time and storage. Price local
compute under an explicit common assumption if it is included in a total-dollar
claim. Missing prices or usage stay unknown. Judge costs belong to evaluation
expenditure and are reported separately from the system's answer cost.

After the smoke run, estimate the whole comparison cost and freeze its envelope.
Use existing call/time limits plus estimated cost admission before dispatch.
Already-dispatched work can still incur charges, so estimated admission is not
a provider invoice guarantee.

**Concrete files:** Implemented `evaluation/costs.py` and the comparison runner,
with dated prices in the frozen experiment configuration. The existing
`inference/budget.py` remains responsible for runtime admission.

**Definition of done:** For every comparison row we can explain construction
cost, query cost, mean total cost, median/p95 answer latency and unknown fields.
This is required before claiming the equal-cost part of the goal.

## 5. Third deliverable: fix the failures that actually lose questions

Use the completed five-question pilot to diagnose the current candidate first.
The earlier suggestion of sixty independent histories is deferred alongside
additional benchmark suites. Keep answers and labels outside model inputs.
LongMemEval-S
has shared histories under our overlap rule, and some cases have already been
used during development. We can still report a frozen public-benchmark result,
but cannot describe the whole corpus as untouched independent data. Record that
exposure and freeze the candidate. A separate confirmation workload remains a
later decision, not a prerequisite for this focused repair.

For each missed development question, assign the earliest failure stage and use
the corresponding solution below. Change one cause at a time and rerun its
regression and the same exposed pilot. A later frozen evaluation is needed to
measure generalization.

### Node-search improvement order

This is the current proposed sequence. It does not start another paid run or
resume the stopped full comparison. Keep the existing BM25 default, optional
ColBERT backend and first-three-distinct-owners selection while testing changes.
Use the existing runner and retained traces rather than adding another benchmark
framework or a separate fixed-case acceptance suite.

**1. Fact-collection development gate completed.** In all three original LLGM
misses, every annotated source node was admitted. Iterations repaired local
reading and root presentation, then qualified GPT-5.4 with medium reasoning for
synthesis. One fresh v6 run answered all five correctly with supported citations.
The original attempts, intermediate regressions and counting-unit ambiguity are
retained in the dated report. The final root differs from the older control
readers. This establishes recovery on exposed questions, not a general accuracy
estimate or an isolated architecture gain. A later matched comparison must hold
the final reader and reasoning effort fixed and retain construction variability.

**2. Record where each needed fact disappears.** Extend analysis of existing
traces with one row per question and evidence need:

```text
candidate source -> admitted node -> useful span read
-> supported fact returned -> fact correctly used -> claim cited
```

Classify the earliest observed loss. Track role, entity, scope and event date
beside each fact, retaining the canonical quote. Model-generated explanations
do not establish these transitions by themselves. Gold and manual labels belong
only in offline analysis. If a needed span lacks annotation, mark it unknown or
retain a separately identified human review instead of silently extending gold.
This makes source recall, useful reading and synthesis measurable separately.

**3. Preserve complementary sources when combining BM25 and ColBERT.** The
[selection study](reports/node-selection-2026-09-11.md) found all annotated sources
in the full union on 22/22 cap-feasible benchmark cases. Passage-level RRF cut to
forty retained them on 21/22. It discarded a uniquely useful BM25 source for
`06f04340`. This is a specific fusion loss, not evidence that either backend
should be removed.

First replay the saved real rankings without new model or GPU calls:

| Alternative | Concrete comparison |
| --- | --- |
| Existing control | Passage-level RRF, forty passages, first three distinct owners |
| Balanced merge | Alternate unique passages from the two rankings until forty, preserving each backend's order |
| Node-level ranking | Fuse each owner's best rank from each backend, select three owners, then expose their highest-ranked distinct spans under the same byte allowance |

Freeze tie-breaking, exact-span deduplication, maximum visible passages and bytes
before replay. Retain full-union availability as a diagnostic upper bound, not
as a budget-matched deployable arm. Compare source coverage, relevant-turn/span
coverage where labels permit, repeated/overlapping content, owner concentration,
selected evidence volume and paired regressions. Backend quotas are an option
to test, not a guarantee of keeping the useful source. A long node must not gain
unlimited weight from many near-duplicate passages.

The node-level alternative changes fusion and owner selection together. Label
it as a combined policy intervention, and do not attribute its result to fusion
alone. A separate matched owner-selection control is needed to isolate that cause.

Carry only a promising policy into a matched answer comparison once fact
delivery works. Count both backend searches and their latency/cost. Product
composition also needs a shared canonical corpus-identity contract. The existing
lexical/dense `HybridRetriever` is not turnkey BM25/Modal-ColBERT wiring.

**4. Recover a missing fact with bounded additional discovery.** After local
reading is reliable, compare existing fixed seeds with a bounded follow-up policy
using an explicit unresolved fact, such as the date of the second event. Try
local turn inspection first, then a useful edge or global search when the source
does not establish it. Measure new facts actually delivered and used, duplicate
reads, extra calls and answer improvement under the same total budget. Merely
creating a child or increasing the seed count is not a success metric. A root
requery phase would be a separate architecture intervention because the current
root has only one final synthesis call.

**5. Check whether graph preparation earns its cost.** The pilot published eighty
edges but made no recursive child calls or cross-node reads. Automatic maintenance
cost $1.359618, about 90% of LLGM's API cost in that pilot. After fact delivery is
reliable, compare maintenance enabled and disabled, charging construction to the
actual questions asked. Preserve graph-enabled controls and measure whether
edges cause useful discoveries. The present result neither proves edges useless
nor justifies making costly preparation unconditional for every workload.

Defer query rewriting, learned selectors, seed-cap sweeps, exact-ColBERT versus
PLAID tuning and cold-start optimization until the earlier stages identify a
need. Source labels are not necessarily a minimal sufficient evidence set.
Do not optimize complete label recall at the expense of supported answers.

| What went wrong? | Example | Concrete suggested solution | How we know it helped |
| --- | --- | --- | --- |
| Initial retrieval missed the right entry | The question says “billing currency” and the record says “invoices settled in CAD.” | Compare current BM25 with a hybrid or injected ColBERT retriever using the existing `evidence_factory`. Hold the reader fixed. | Better entry-node recall and final answers under the cost envelope. |
| The needed edge was never built | The entry mentions a registry but no relationship leads to it. | Run existing maintenance automatically during preparation. Inspect its candidate retrieval. Widen candidates or improve relation instructions only for demonstrated misses. | The required source is discovered without evaluator-provided links. |
| An incorrect edge was built | Orion and Boreal get a `contradicts` edge despite being different services. | Ask proposals to identify the entity, attribute, scope and supporting excerpts for both endpoints. Reject incomplete evidence. Contradiction needs the same entity/attribute and overlapping applicability. Keep uncertain proposals in the existing proposal path. | Fewer unsupported links and fewer downstream mistakes on labeled positive and negative cases. Semantic judgments still need evaluation. |
| An old fact wins over a correction | Production still answers PostgreSQL after an explicit MySQL replacement. | First determine whether the amendment was absent or misapplied. If absent, add a bounded amendment-proposal function that cites the old exact span and new source, then validates them before append. Scope ambiguity produces an unresolved proposal. | Current and historical/scope-specific questions return the appropriate evidence. No manual benchmark patches are needed. |
| The child message changes the meaning | “Not approved” becomes “approved,” or destination becomes recipient. | Return the exact supporting excerpt beside the finding and preserve its qualifiers. Compare that small message change with the current format before adding another model. | Lower attribution/qualification error through the parent chain and better final answers. |
| Repeated work spends the answer budget | Two seeds independently reread the same node for the same question. | Measure duplicates first. If material, reuse only a completed result within the same query when node, question, selectors and observed local evidence basis match. If current appends cannot be checked safely, skip reuse. | Fewer charged calls without stale answers or lost branch attribution. |
| A large callback fails | A high-degree node returns more edges than the response allowance permits. | Add a bounded page/cursor to edge discovery, modeled on `source_info`, with explicit remaining-page information. | High-degree cases can inspect the useful relationship without a whole-response failure. |
| A branch stops before finishing | Inspection consumes time or operations even though a model finish call was reserved. | Use the retained trace to distinguish model, callback, time and context exhaustion. Adjust the relevant admission rule or unnecessary operation sequence. Preserve required gaps and completed returns. | The reproducing case finishes or returns an honest useful partial result within its declared limits. |

These are solution candidates, not a requirement to implement every row now.
For example, automatic journal proposal generation is currently a gap, but we
should first establish how often it would change an incorrect answer. Ordinary
maintenance currently creates primary edges, not semantic journal overwrites.

**Concrete owners:** Retrieval and link proposals in `memory/evidence.py`,
maintenance policy in `memory/maintenance.py`, amendment application in
`memory/query.py`, and delegate messages/admission in `inference/nodes.py`.

**Definition of done:** Every adopted change fixes a recorded failure, has a
regression, and improves the declared development result or removes a concrete
correctness defect. We keep unsuccessful experiments in the record.

## 6. Fourth deliverable: demonstrate the win and its cause

Freeze the selected configuration and run all 500 LongMemEval-S questions for
the declared comparison systems. Report overall accuracy and question-type
results, total and query-only cost, latency, failures and coverage of usage.
Choose the repetition count with the execution budget before dispatch. A
recommended confirmation is three runs of the final comparisons, retaining all
of them. Shared histories mean question outcomes are not independent samples.

Then run three development comparisons that remove one LLGM mechanism at a time:

| Comparison | What changes | Question it answers |
| --- | --- | --- |
| No graph navigation | Prevent edge discovery/navigation while retaining source search and reads. | Do primary relationships improve evidence discovery? |
| No child delegation | Keep the source/edge access available but require the current delegate to inspect it directly. | Does recursive distribution improve the quality/cost tradeoff? |
| No journal interpretation | Read the original sources, including later correction sources, without applying journal patches. | Does the amendment mechanism improve update reasoning beyond reading the same evidence? |

Keep the source inventory and budgets comparable. These controlled comparisons
use fixed preparation to isolate each mechanism. Charge and evaluate automatic
preparation separately in the main end-to-end result. If disabling a mechanism
has no benefit or loss, we have not demonstrated its contribution on that workload.

Confirm the frozen approach on LoCoMo. Use its official per-category QA scoring,
including the adversarial category, and report category counts. The released
evaluator uses F1-based scoring for ordinary questions and a distinct rule for
adversarial abstention. If we add a common LLM judge, label that score separately.
Do not mix it with a published percentage from another metric or filtered subset.
See the [official evaluator](https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py).

**Definition of done:** One results table says whether the numerical goal was
met, how much it cost, which abilities improved and which mechanisms caused the
gain. A miss remains a miss. We do not revise the target after inspecting results.

## 7. Remaining engineering work, with concrete triggers

The following items are important for a reusable library. They enter the active
work when they block the benchmark or the intended deployment. They do not all
need to precede the first comparison.

| Item | What the limit means | Suggested implementation and concrete acceptance check |
| --- | --- | --- |
| Journal growth | Distinct active patches can exceed the default 64 KiB allowance. History also grows on disk. | Generate histories with 100, 1,000 and 10,000 amendments. If redundant compaction is insufficient, prototype a derived working representation made of canonical source segments plus active patches. Verify identical effective reads, scope/time handling, restart and old references. Keep historical deletion a separate decision. |
| Large-node memory | A small span read still loads the full JSON source on the host. | Measure metadata and 1 KiB reads against 1, 10 and 100 MiB nodes. If memory/I/O scales with the full source beyond acceptable use, store indexed turns or chunks behind the same public spans. Test Unicode boundaries and cross-chunk patches. These sizes are proposed probes. |
| ColBERT operational use | GPU retrieval works, but full answering, deployed lookup and refresh need validation. | Run the injected retriever through `LLGM.answer`, verify journal corrections and canonical references, then test the deployed `connect()` path. Prepare an explicit new index when the corpus changes. Measure cold opening separately from warm remote search. |
| Recovery | Restart is tested, but complete backup/restore is not a supported workflow yet. | First support a quiescent backup: close writers, capture metadata and referenced blobs, restore into an empty directory, discard/rebuild derived indexes, verify references and answer a fixed test set. Add online backup only if required. |
| Other providers | Adapter code exists without the latest full-pipeline proof for every provider. | Add one provider at a time. Run native-output contracts and a small full-pipeline cohort including refusal, truncation and timeout handling. |
| Additional live faults | Several precise boundaries currently have deterministic tests only. | Add real Docker/provider-boundary checks for cancellation, timeout and cleanup where external behavior matters. Keep exact schema/storage fault cases deterministic when a paid model adds no evidence. |
| S3 and distributed use | S3 blobs are available as an adapter, while metadata stays local SQLite. | Run the existing S3 test against a dedicated prefix if that backend is needed. Define a concrete multi-host workload before choosing distributed metadata. |
| Release and onboarding | Local checks do not prove clean installation or new-developer usability. | Build the final wheel/source archive, test outside the checkout, run the configured Python CI matrix, and follow the public quickstart in a clean environment through a corrected recursive answer. Establish a reviewed source-control checkpoint before release. |

## 8. What to do next, in order

1. The five-question repair is complete. Before claiming an architecture gain,
   compare controls using the same final reader and reasoning effort, with every
   construction and query call priced. Preserve all earlier candidates.
2. Follow the node-search order above: locate evidence loss, replay complementary
   fusion options, and test useful follow-up discovery under matched limits.
3. Once the candidate is reliable, resume the proposed Mem0/Graphiti integration
   with complete priced-call accounting. Add iterative and RLM controls when
   assessing the contribution of graph navigation and recursion.
4. Freeze any broader comparison separately. Keep the stopped full run stopped
   until its scope is explicitly resumed. Report benchmark exposure and isolate
   the contribution of edges, journals and recursive delegates.
5. Address storage, integration and release limits when the measured workload
   or deployment requires them.

The completed development repair is **use the relevant evidence already found**. The
project milestone is **beat the named alternatives on answer accuracy at the
same total cost**. This keeps each proposed feature connected to a measurable
reason for building it.

This roadmap preserves the adopted design: one `LLGM` entry point, opaque
immutable nodes, primary edges, small eager journals, current reads and bounded
node delegates. No source-version/commit system or permanent node service is
required by this plan. Existing measurements remain in their dated reports.

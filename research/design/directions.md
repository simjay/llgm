# LLGM research directions and open choices

Open research choices, reviewed September 11, 2026. These alternatives develop the
[current research architecture](overview.md) within the adopted
[simplification decision](simplification.md). Unique immutable nodes and current
journal reads are the baseline, not alternatives to source editions or mandatory
workspace snapshots. For implemented behavior use the
[architecture guide](../../docs/guide/architecture.md). For completed measurements
use [research reports](../reports/README.md).

The [experiment design](experiments.md) gives these options objective IDs, controls, measurements and decision gates. The [September 9 technical baseline](technical-spec.md) is retained historical context. Supported interfaces and contributor requirements belong in the public documentation.

The working architecture has a **frontier root reasoner**, **smaller-model node delegates**, and a **Persistent Evidence Graph (PEG)**. The host retrieves and selects initial seeds deterministically. Each delegate inspects its source, follows primary edges or searches when needed, and returns selected findings through a bounded recursive call chain. The root receives branch findings for one final synthesis call with no further tool phase. Each PEG node preserves complete source text and owns one **node journal** for interpretation notes and explicit amendments. Independent primary edges define graph connectivity. Journal values can be inline text or stable evidence references without also creating an edge. Search passages, edges, journal records, and returned evidence share source-addressing infrastructure.

Standalone query nodes and a materialized Runtime Inference Graph are no longer required. Calls and dependencies may be recorded in an execution trace. The PGM connection is an inspiration for locality, not a claim that the runtime performs probabilistic factor-graph inference.

Forgetting semantics and reconciliation scheduling remain research choices. The journal structure does not select when semantic maintenance must run.

The [roadmap](../ROADMAP.md#node-search-improvement-order) owns the current node-search priorities. The active gate is review of the completed five-question LongMemEval pilot and its evidence-use failures. Broader retrieval/model sweeps and additional benchmarks, including LoCoMo, are deferred. The alternatives below remain a research map, not an instruction to restart those experiments.

## 1. Canonical nodes and stable evidence references

**Question:** How can nodes preserve conversational context while supporting precise, durable references?

| Option | Explanation and tradeoff |
| --- | --- |
| Fixed contiguous source windows | Simple closure rules and reproducible ingestion. Related turns can straddle boundaries. |
| Conversation or checkpoint boundaries | Natural full discussions. Node lengths vary considerably. |
| Small-model boundary detection | Potentially coherent segments, with inference cost and segmentation errors. |
| Simple canonical boundaries with flexible search passages | Separates source identity from retrieval optimization. Needs reliable source-range mappings. |

The common requirement is immutable addressability. The source reference contains a unique node ID, turn ID and half-open Python Unicode-code-point range. A list of ranges can identify evidence spread across turns. `NodeRef` addresses a complete immutable node. `JournalRef` identifies an immutable journal entry and can select an inline-text range. Journal corrections target those entry references rather than modifying original source text.

Search chunk IDs must not become the only durable evidence IDs. Rechunking, overlap, a changed embedding model, or a rebuilt index should preserve existing journal references. Canonical text can be stored once while indexes keep ranges and derived representations. A citation identifies text. It does not certify that the text is true.

**Compare:** Vary node and search-passage boundaries independently. Measure boundary-spanning recall, index overhead, needed context expansion, and reference resolution after rechunking, reindexing, and journal corrections. Include Unicode, quoted text, and identical passages in different nodes.

## 2. Primary relationships and node interpretation journals

**Question:** How should relationship and amendment records share evidence references while preserving their separate meanings?

| Option | Explanation and tradeoff |
| --- | --- |
| Whole-node primary relationship with supporting references | Simple navigation. Does not by itself specify which source statement should change. |
| Precise journal subject with an inline-or-reference value | Addresses an interpretation or amendment. Requires clear scope and precedence. |
| Rich applicability and provenance in either record | Makes decisions inspectable, with annotation overhead and extraction errors. |
| Minimal authoritative records plus derived annotations | Keeps persistent contracts small while allowing experiments. Derived views need policy identifiers. |

Implemented primary edges have their own identity, endpoints, relationship, provenance, applicability, and withdrawal. Ordinary maintenance publishes these edges independently of journals. Journal records identify a subject, relation, inline or referenced value, provenance, and append order. A subject can identify one statement inside a larger Q&A or an earlier journal entry. Inline values remain addressable rather than becoming anonymous summaries.

Primary-edge records are authoritative for graph connectivity, with indexed outgoing lookup. Journal views are authoritative about applicable interpretation records. A replacement pointer need not also become a navigation relationship, and an ordinary edge does not amend its source. A derived incoming-edge index remains a possible access optimization. Correcting a mistaken journal entry appends a record identifying the earlier entry. Automatic semantic amendment discovery and broader forgetting policy remain open.

The journal is authoritative about which assertions were recorded, including user and model assertions. It does not make every assertion true or applicable. A source decision being superseded is distinct from correcting a mistaken journal assertion. Original sources and journals are durable inputs. Views, indexes, and caches are rebuildable outputs.

**Compare:** Test whole-node versus span-level subjects, inline versus referenced values, and simple versus rich envelopes. Measure annotation cost, incorrect scope changes, traceability, and reconstruction of the same view from identical journal entries under the same interpretation policy.

## 3. Automatic semantic links and affected-evidence discovery

**Question:** Which earlier evidence should be examined when a new discussion arrives or a reader discovers a connection?

| Option | Explanation and tradeoff |
| --- | --- |
| Explicit references and chronology | Cheap, grounded links. Misses implicit dependencies. |
| Semantic-neighbor candidates | Broad discovery through retrieval, with many merely similar passages. |
| Entity and scope filters followed by retrieval | Reduces comparisons but can miss aliases, implicit references, and cross-scope effects. |
| Hybrid candidates plus selective model classification | Combines routes and richer relation judgments, with maintenance cost. |

Candidate discovery can serve both ordinary navigation links and potential edits. The decision rules cannot be identical: `related_to` requires less evidence than `superseded_by`. A model-proposed link should retain its supporting source references and provenance. A candidate that has not been published as a primary edge can remain an ephemeral search result. Publishing an edge does not amend evidence or settle an assertion's truth.

PEG need not be connected. Global retrieval remains available when useful evidence lies outside existing neighborhoods. Reverse indexes should make an update in C discoverable from an older source A without requiring two independently authored relationship records.

**Compare:** Measure affected-source recall separately from relation classification. Track missed updates, spurious replacements, useful traversal discoveries, fan-out, and comparisons per new passage. Exhaustive comparison on small histories supplies a diagnostic reference. Bounded search must demonstrate how much semantic coverage it sacrifices as history grows.

## 4. How to forget: current applicability and historical evidence

**Status: active research. Policy not settled.** The journal gives this topic a concrete representation, but it does not itself decide which statement governs an answer.

| Direction | Meaning and tradeoff |
| --- | --- |
| Interpret sources and journal entries during each query | Flexible and question-sensitive. May repeat expensive work or produce inconsistent decisions. |
| Derive explicit applicability relationships | Makes supported revisions reusable. Incorrect scope or precedence can suppress useful evidence. |
| Present unresolved alternatives together | Preserves uncertainty. Increases reader burden and can leave answers unresolved. |
| Adjust retrieval priority using journal state | Can reduce interference. Stale ranking or aggressive suppression may hide historical evidence. |

A useful behavioral target is that older evidence stops governing a current-state answer when a supported change applies, while remaining accessible for historical questions. This is distinct from physical deletion, archival, recency decay, or model-parameter unlearning. No automatic retention or destruction policy follows from accepting the journal architecture.

Start with concrete histories:

| History variation | Behavior to specify and evaluate |
| --- | --- |
| Explicit replacement of a production database | Current and historical questions identify different decisions. |
| Suggestion to consider another database | A proposal does not silently become the adopted decision. |
| Production changes, staging stays unchanged | Only the affected scope changes. |
| Later report about an earlier event | Event applicability and recording order are distinguished. |
| Two incompatible reports without a resolution | Preserve the disagreement instead of selecting the newest text by default. |
| An incorrect model-authored journal edit is corrected | Recover the supported interpretation while retaining the correction trail. |

**Compare:** Measure stale answers, false supersessions, historical accuracy, qualification retention, and recovery after mistaken edits. Temporal knowledge graphs and conflict-aware memory are relevant prior work, including [Zep](https://arxiv.org/html/2501.13956v1) and the August 2026 [LatticeMind preprint](https://arxiv.org/abs/2608.08236). The journal idea alone does not establish novelty.

## 5. Reconciliation timing and current views

**Question:** When should the system interpret journal records and make their consequences available to readers?

| Option | Explanation and tradeoff |
| --- | --- |
| During ingestion | Pays interpretation cost before future queries. Higher ingestion latency and work on rarely queried material. |
| During reads | Defers work until an actual information need exists. Higher and less predictable query latency. |
| Background or periodic reconciliation | Amortizes maintenance. Introduces freshness lag and a tail requiring attention. |
| Hybrid by evidence type or workload | May adapt to demand, with additional scheduling and correctness complexity. |

Appending an observation, deciding its semantic relation, and materializing a current view are separate actions. One logical node journal does not require a single physical table or a single timing policy.

A current view depends on question scope, relevant time and the journal entries read. It is not simply a global active/inactive bit on every statement. Immutable source identity and the per-node sequence used for journal ordering serve different purposes.

The adopted storage decision uses an append-only journal sidecar. Database recovery logs remain an implementation detail. Neither that analogy nor historical questions require a public commit model, source editions or a mandatory workspace-wide snapshot.

The current reader eagerly loads the complete compact operational journal as local metadata and applies explicit scoped source patches during lazy reads. Deterministic compaction retains original history and canonical references. The timing options here concern broader semantic maintenance and derived views, not a proposal to make the small operational journal another recursive source-reading task.

Current reads may observe journal entries published after the initial search. Record the exact entries used and preserve immutable source references. A background design needs a declared treatment for evidence newer than its derived view. A future workload requiring fixed cross-node visibility must justify and test that additional guarantee separately.

**Compare:** Hold the semantic policy fixed while changing ingestion/read/background timing. Sweep query-to-update ratios and update burstiness. Measure total cost, answer freshness, ingestion latency, p95 query latency, and replay reproducibility. Storage scalability does not prove semantic comparison scalability.

## 6. Retrieval backend and passage-to-node aggregation

**Question:** Which search method finds useful evidence at acceptable index, query, and update cost?

| Option | Explanation and tradeoff |
| --- | --- |
| Lexical retrieval | Strong for exact names, identifiers, and quoted wording. Weak paraphrase matching. |
| Single-vector dense retrieval | Compact semantic indexing. Can blur fine distinctions. |
| Lexical/dense hybrid | Complementary coverage, with fusion and reranking choices. |
| ColBERTv2 with PLAID | Candidate late-interaction backend. Requires evaluation of index footprint, updates, and conversational data. |
| BM25 plus ColBERTv2/PLAID | Combines lexical and token-level matching. Fusion can discard complementary evidence when the union is truncated. |

ColBERTv2 supplies contextual token representations and late interaction. PLAID is an efficient retrieval engine for this class of representation. Neither is an LLM strategy for deciding what to search next. [ColBERTv2](https://aclanthology.org/2022.naacl-main.272/), [PLAID](https://arxiv.org/abs/2205.09707)

Search should return passage hits with stable evidence references, parent node identity, and index-version information. Node aggregation can use best-hit scoring, capped combinations of distinct hits, or coverage of separate information needs. Overlap must not let a long node crowd out complementary sources.

The current application selects the first distinct owners in passage rank order, retaining all distinct matched references for each admitted owner. Its defaults are 12 passage hits and three seed nodes. The real ColBERT/PLAID [deployment diagnostic](../reports/colbert-modal-2026-09-11.md), [node-search comparison](../reports/node-search-2026-09-11.md), and [selector comparison](../reports/node-selection-2026-09-11.md) are complete. BM25 plus ColBERT fusion and model selection remain experimental policies. The [answer comparison](../reports/seed-answers-2026-09-11.md) did not establish a selector winner because reading and synthesis failed. Candidate-source coverage, selected-source coverage, and correct supported answers remain separate measurements. The original twelve-arm E03 experiment is incomplete.

**Compare:** Hold the sidecar fixed and measure passage recall, all-required-source recall, duplicate crowding, latency, memory/storage footprint, and incremental indexing cost. Search relevance is only a candidate signal. Journal-aware applicability and evidential sufficiency require later inspection.

## 7. Searching original evidence and journal-derived views

**Question:** How should search expose useful current interpretations without hiding original or historical evidence?

| Option | Explanation and tradeoff |
| --- | --- |
| Search raw passages, then inspect relevant journal entries | Faithful access path. Extra work when many old decisions compete. |
| Search raw text and journal text together | Can find concise clarifications. Model-generated text may dominate unless provenance is preserved. |
| Separate source and interpretation indexes | Gives explicit control over access. Requires merging and version coordination. |
| Search a scoped view, with source and historical fallback | Potentially efficient current-state retrieval. Incorrect view construction can hide needed evidence. |

Indexing journal text should preserve its entry anchor and whether it came from a user, source, or model interpretation. Searching a model summary twice does not create two independent sources. Query intent may concern current state, a past state, or the history of a decision. A single latest-only search view cannot serve all three.

**Compare:** Use the same journals and semantic policy while changing access paths. Measure current and historical evidence recall, missed corrections, provenance preservation, and raw-source fallback cost. Include updates whose new wording is less similar to the question than the obsolete wording.

## 8. Sidecar ownership of search and follow-up

**Question:** How much adaptive search should node delegates perform, and would an additional root follow-up phase justify its cost?

| Option | Explanation and tradeoff |
| --- | --- |
| Root chooses every next operation | Simple diagnostic baseline, but repetitive frontier work may dominate cost. |
| Sidecar executes an upfront plan | Predictable delegation. Can miss dependencies discovered later. |
| Sidecar owns an adaptive information request | Searches, reads, and refines until a budget or completion condition. Can drift or repeat work. |
| Adaptive sidecar with root escalation | Keeps routine loops inexpensive while exposing consequential ambiguities. Escalation needs a measurable rule. |

The current implementation uses adaptive smaller-model node delegates after deterministic initial seed selection. Requests carry the question, scope, source handles, output contract, and shared budget. Delegates can create children and return findings. The root is responsible for the final answer but cannot request another investigation after inspecting branch returns. A root follow-up or escalation phase is a proposed intervention requiring a declared trigger, reserved budget, and comparison against the existing single synthesis call. No permanent process per source is required.

Preserve original wording alongside rewrites. Delegates can follow primary edges or search globally. Journal replacement pointers resolve as effective reads and can motivate further source investigation. Iterative retrieval has precedents in [Baleen](https://arxiv.org/abs/2101.00436) and [IRCoT](https://aclanthology.org/2023.acl-long.557/).

**Compare:** Isolate query decomposition, adaptive follow-up, and PEG traversal. Track frontier invocations, small-model work, query drift, recovered dependencies, and answer quality at matched total cost.

## 9. Direct reading, RLM access, and compositional relays

**Question:** Which source-access operations justify recursive computation?

| Option | Explanation and tradeoff |
| --- | --- |
| Retrieved passages plus neighboring turns | Cheap direct reader. Sensitive to candidate omissions. |
| Full-node reading when it fits | Preserves context, with larger prompts. |
| External source inspection without recursive model calls | Isolates the benefit of selective access and text tools. |
| Recursive inspection and subrequests | Supports decomposition and local composition, with coordination and transformation errors. |

A relay remains possible without standalone query nodes: while reading B, the sidecar discovers a need in C, investigates C, and combines that evidence with B before returning. Compare this with pure forwarding and with direct access to all relevant sources. Calls belong to execution. Primary edges belong to persistent evidence organization, while journals govern explicit interpretations and amendments.

RLM already uses external context, recursive calls, different model tiers, and deeper recursion in its revised experiments. Small-model subcalls and chaining alone are not LLGM novelty. [RLM, version 3](https://arxiv.org/html/2512.24601v3)

**Compare:** First provide correct source nodes, requiring readers to find evidence within them. Vary distractors, source lengths, chain depth, and missing links. Compare against ordinary RLM over the same unpartitioned evidence. Then restore realistic retrieval. Measure qualification loss and whether the intermediate interpretation adds value beyond extra calls.

## 10. Bounded evidence messages and escalation

**Question:** What should the sidecar return, and when does a request need a stronger model?

| Option | Explanation and tradeoff |
| --- | --- |
| Exact excerpts | Preserve wording. Consume context and leave interpretation to the receiver. |
| Query-conditioned findings with source references | Compact and relevant. Can omit qualifications or overinterpret text. |
| Structured findings, unresolved needs, and expandable handles | Supports inspection and progressive disclosure, with interface overhead. |
| Adaptive stronger-model review | Can recover difficult interpretation, but checks and retries may erase savings. |

Messages should distinguish original claims, journal interpretations and the sidecar's own deductions. References resolve to immutable text, with consulted journal entry IDs and interpretation policy recorded. A pointer to “whatever C currently believes” must not silently replace a reference to C's original decision. Unavailable evidence, not-found results and contradictory evidence need different statuses.

Small models can perform both PEG maintenance and runtime work. Stronger-model escalation is a candidate for unsupported conclusions, scope ambiguity, or repeated failure. Self-reported confidence alone needs calibration. Deterministic resolution, indexing mechanics, accounting, and exact deduplication should not require LLM calls.

Hosted execution of every generative role is a library requirement, independent of the model-allocation experiment. Providers can differ by role under a capability-aware model interface. Native APIs, compatible endpoints and optional structured-output helpers are integration choices. Embedding/reranking interfaces remain separate. The [configuration guide](../../docs/guide/configuration.md) owns supported environment and Python settings. [E08](experiments.md) separates backend scaling measurements from model-quality comparisons.

**Compare:** Vary message format and model allocation separately. Attribute errors to retrieval, semantic maintenance, reading, compression, or final synthesis. Report successful escalations, unnecessary escalations, and total retry cost.

## 11. Budgets, cycles, concurrency, and execution traces

**Question:** How can recursive work remain bounded as history and branching grow?

| Option | Explanation and tradeoff |
| --- | --- |
| Fixed calls, rounds, text, and time limits | Reproducible and simple. Can terminate useful work early. |
| Adaptive budget allocation | Prioritizes consequential gaps. Allocation itself can be wrong. |
| Sequential execution | Easier accounting and dependency handling, with higher latency. |
| Bounded concurrency with shared run budget | Reduces independent-read latency, with coordination and duplicate-work risks. |

Each root and sidecar invocation must satisfy:

```text
instructions + question + retained state + selected evidence
    + output reserve <= invocation context limit
```

Depth limits do not bound a branching call tree. All concurrent subcalls spend from a shared ledger. Evidence and intermediate material remain external. A growing sidecar session must not recreate the full-context problem. Request identity, source lineage, and pending-dependency checks should prevent cyclic pointers from causing endless calls or false corroboration.

An execution trace can record requests, versions, references inspected, dependencies, results, costs, and stop reasons. It need not be a materialized factor graph or mandatory second graph database. Trace retention is an implementation choice distinct from source semantics.

**Compare:** Sweep call and token budgets, concurrency, and stopping rules. Include cyclic references and repeated evidence. Report quality-cost curves, p95 latency, repeated work, and unresolved requests.

## 12. Growing journals, indexes, and caches

**Question:** How does the system avoid reading every old journal entry or rebuilding every index for each operation?

| Option | Explanation and tradeoff |
| --- | --- |
| Replay journal entries on demand | Simple authority model. Growing read amplification. |
| Derived checkpoints plus journal tail | Faster reconstruction, with checkpoint maintenance and versioning. |
| Incremental indexes with a recent-evidence overlay | Faster foreground updates. Readers must account for unindexed material. |
| Cached findings with dependency tracking | Reuses expensive reads, with invalidation complexity. |

A checkpoint is a derived representation, not permission to discard original evidence or authoritative journal history. Index IDs and cache keys should remain separate from stable source references. Cache dependencies can include immutable source and journal entry IDs, scope, request, model/prompt and interpretation policy. New relevant evidence can invalidate an answer even when previously read sources are unchanged.

Current local search already refreshes an incremental SQLite index, and operational journal compaction removes redundant entries without deleting history. Irreducible complete-journal overflow fails explicitly. Checkpoints, overlays, and finding caches in this section are further alternatives rather than prerequisites still missing from basic source access.

**Compare:** Replay streams with queries immediately after updates. Measure index/view freshness, journal entries inspected per read, maintenance work, storage amplification, stale-cache answers, and recovery after rebuilding derived state. Test realistic skew, where some nodes receive many edits and others remain almost unused.

## 13. General inference and the graphical-model connection

**Question:** Can the architecture improve reasoning beyond finding and reconciling history?

| Direction | Explanation and tradeoff |
| --- | --- |
| One synthesizer with evidence sidecar | Clear initial memory claim. Does not establish a general ensemble improvement. |
| Alternative-hypothesis or constraint-specific subcalls | Broadens reasoning, with correlated errors and decomposition overhead. |
| Proposal followed by evidence verification | Separates generation and checking. The verifier can also fail. |
| Task-specific probabilistic formalization | Defines variables and factors for a restricted task. Adds a separate modeling burden. |

The LLGM name currently describes a research direction inspired by graph locality. PEG sources are not automatically random variables, and calls do not become factors because they connect evidence. A factor-graph extension would require explicit variable domains, local functions, and an inference objective. [Factor Graphs and the Sum-Product Algorithm](https://www.isiweb.ee.ethz.ch/papers/arch/aloe-2001-1.pdf)

**Compare:** Match resources against stronger direct reasoning, repeated sampling with synthesis, and verification baselines. Include tasks requiring broad aggregation or dense dependencies, where selective locality may perform poorly. No benchmark about conversational recall alone can establish a general “Jarvis” capability.

## 14. Benchmark portfolio

**Question:** Which public tasks test the separate mechanisms rather than only an overall answer score?

LongMemEval-S is the sole current primary benchmark. The active scope is review of five already exposed questions, not a full run. The remaining benchmarks below are deferred candidates. Dataset and harness versions must be pinned before any future expansion, which must follow the current roadmap rather than this portfolio alone.

| Benchmark | What it tests for LLGM | Recommended use and limitation |
| --- | --- | --- |
| [LongMemEval](https://github.com/xiaowu0162/LongMemEval) | Conversational extraction, multi-session reasoning, knowledge updates, temporal reasoning, and abstention. | Sole current primary benchmark. Review the completed five-question pilot before expanding. Correct-source variants are diagnostics, not deployable retrieval results. |
| [MemoryAgentBench: FactConsolidation](https://github.com/HUST-AI-HYZ/MemoryAgentBench) | Conflict resolution under incremental input. | Focused semantic-update evaluation. Include a simple version-aware baseline and follow the task's actual scoring protocol. |
| [BEAM](https://github.com/mohammadtavakoli78/BEAM) | Long conversational memory at released 128K, 500K, 1M, and 10M scales. | History-length scaling. Document the selected buckets and synthetic conversation construction when interpreting generalization. |
| [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) | Difficult iterative search and reasoning against a fixed document corpus. | Tests follow-up retrieval beyond chat memory. Hold the corpus and retrieval access fixed. It does not test personal-history updates by itself. |
| [OOLONG](https://github.com/abertsch72/oolong) | Long-context reasoning and aggregation. | Stress broad evidence coverage, where reading only a few relevant-looking passages may fail. Pin the chosen variant. |
| [LoCoMo](https://github.com/snap-research/locomo) | Long conversational memory tasks. | Supplementary comparison with existing memory systems. The small set of source conversations limits independent-history coverage. |
| [LongMemEval-V2](https://github.com/xiaowu0162/LongMemEval-V2) | Memory over multimodal web and enterprise trajectories, evaluating answer accuracy and query latency. | Later extension toward experiential memory. Requires a broader source model and careful adherence to the fixed downstream reader and context interface. |

The official LongMemEval repository distinguishes cleaned data from its earlier release. MemoryAgentBench contains several source tasks and metrics. Avoid treating every reported accuracy as an identical measure. LongMemEval-V2 is a separate protocol, not a drop-in replacement for conversational LongMemEval. These distinctions matter when comparing published scores.

**Compare:** Start with independent controlled development histories and the already exposed LongMemEval pilot only for diagnostic replication. The current release's 500 cases form one connected history component, so the pilot does not leave 495 independent held-out histories. Freeze new evaluation eligibility before extending to scale and corpus reasoning. Benchmark selection should follow the claim under test rather than whichever headline score is easiest to improve.

## 15. Baselines, metrics, and experimental controls

**Question:** Which comparisons isolate the value of the sidecar, graph, and journal?

Use a baseline ladder with matched root and smaller models, source evidence, and accounting:

| System | Mechanism isolated |
| --- | --- |
| Strong flat retrieval plus root answer | Passage access without adaptive inspection. |
| Iterative evidence sidecar over flat sources | Benefit of delegated follow-up search and reading. |
| Ordinary RLM over the same external evidence | Existing recursive access without LLGM's persistent organization. |
| Sidecar with PEG traversal | Incremental value of graph access. Use navigation relations without update interpretation. |
| Same sidecar and graph with semantic journal interpretation | Incremental value of clarification, revision, and historical applicability. |

PEG navigation comes from independent primary edges. The last ablation adds journal interpretation while keeping navigation available. It does not change edge authority. The current LongMemEval runner supplies LLGM, BM25 and full-context controls. Mem0 OSS and Graphiti are the named framework targets, with their integrations and complete cost instrumentation still pending. Iterative and ordinary-RLM controls are needed before claiming a benefit specific to recursion or graph access. Match common interfaces while allowing each method a competent configuration. Do not intentionally weaken ordinary RLM or retrieval.

| Measurement | Required reporting |
| --- | --- |
| Answer quality | Official scores and per-ability results. Supported claims, stale answers, historical accuracy, and abstention. |
| Evidence access | Passage and jointly required-source recall, scope errors, dependency discovery, and source lineage. |
| Semantic maintenance | Affected-evidence recall, false replacements, missed updates, correction recovery, and view freshness. |
| Cost | All root/sidecar calls, retries, ingestion, embeddings, indexing, journal interpretation, maintenance, and storage. |
| Latency | Median and p95 query latency, ingestion delay, update-to-visible delay, and throughput. |
| Scaling | History length, update rate, edits per node, queries per update, branching, and necessary evidence volume. |

Report both query cost and lifecycle cost. A useful accounting convention is:

```text
lifecycle cost = ingestion + indexing + semantic maintenance
    + all query execution + storage over the measured period

amortized cost per query = lifecycle cost / evaluated query count
```

Specify the period and workload. Show resource/quality curves rather than one arbitrary budget, and distinguish measured dollars from raw token/call counts. A cheaper sidecar call may lead to more retries. An expensive ingest step may pay off only with many later queries.

Ingest only source evidence available by the query time defined by the protocol. A retrospective question asked now can use later reports about earlier events. A query replayed at an earlier time cannot use future source events. Keep held-out questions out of memory preconstruction, while allowing the current question to guide online search and reconciliation within the measured budget. Exclude gold answers and evaluator-only metadata throughout. Isolate query-conditioned journal changes, caches, and answers across otherwise independent evaluation cases. Pin prompts, models, data, interpretation policies, index versions, judges, and seeds where supported. Separate development from held-out cases, use paired comparisons, and report uncertainty with clustering by shared history when appropriate.

## 16. Research claims and next experiments

**Question:** What result would justify a paper contribution?

| Candidate claim | Necessary evidence |
| --- | --- |
| Delegating evidence loops reduces frontier expenditure | Quality survives matched total-cost comparisons with root-driven search and ordinary RLM. |
| Persistent relationships improve dependency discovery | Gains survive equal retrieval access, passage boundaries, models, and budgets. |
| Shared references and journals improve evolving-memory reliability | Current and historical answers improve without excessive false edits or maintenance cost. |
| Incremental reconciliation scales with growing histories | Measured affected-evidence recall and lifecycle cost remain favorable as histories and updates grow. |
| Local computation improves broader reasoning | Gains extend beyond recall to independent reasoning tasks under fair compute comparisons. |

Storage versioning, recursive model calls, and temporal memory all have prior work. A narrower possible contribution is reliable, bounded evidence access through a unified passage-addressed journal while accounting for maintenance and inference together. That remains a hypothesis until ablations support it.

The current sequence is owned by the [node-search improvement order](../ROADMAP.md#node-search-improvement-order). The broader research dependencies remain:

1. Preserve the implemented source, primary-edge, journal, reference, and runtime contracts while identifying the earliest stage that loses evidence in the five-question pilot.
2. Compare direct reading, iterative small-model inspection and ordinary RLM when the active evaluation warrants those controls. Known-source diagnostics remain separate from ordinary retrieval.
3. Test passage retrieval, owner selection, follow-up requests, and primary-edge traversal as separate interventions after local reading and synthesis are reliable enough to expose a retrieval effect.
4. Revisit semantic-edit and reconciliation alternatives only when recorded update failures justify them. FactConsolidation remains a deferred extension.
5. Freeze a broader memory evaluation only after reviewing the pilot. Scaling, additional benchmarks and multimodal trajectory memory remain later claim-specific work.

Each stage should permit retaining a simpler mechanism when the additional structure provides no measurable benefit. Implementation choices and eventual paper claims should follow these results.

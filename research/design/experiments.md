# LLGM experiment design

Evaluation plan, reviewed September 11, 2026. The objective map describes future
comparisons under the [current research architecture](overview.md). Section 6.1
retains the original E03 baseline with its subsequent data and execution status
stated explicitly. The [live validation report](../reports/live-validation-2026-09-10.md)
records the completed scoped B/D/H pilot under its original schema.
[Implementation status](../../docs/reference/implementation-status.md) owns current
capabilities and validation limits. Real ColBERTv2/PLAID deployment and node-search
diagnostics are complete. The original full twelve-arm E03 comparison remains
incomplete. These are different experiments, with separately frozen inputs.

LLGM should develop through controlled research decisions. Each objective below names competing options, the comparison that distinguishes them, and the evidence needed to adopt or defer a mechanism. The intended deliverable is a reusable Python library. The [research architecture](overview.md) defines the design, and [research directions](directions.md) provides alternatives within it. The [September 9 technical baseline](technical-spec.md) is historical context rather than the current implementation contract.

The working architecture has a frontier root reasoner, smaller-model node delegates, unique immutable full-text nodes, independent primary edges, stable source and journal references, and one interpretation journal per node. The host selects initial seeds deterministically, delegates inspect and recursively investigate evidence, and the root makes one final synthesis call without further tools. Primary edges define connectivity independently of journal amendments. Current reads can observe later appends during inference. Record the exact evidence used instead of requiring a public workspace snapshot. Broader reconciliation and forgetting policy remain open.

The [roadmap](../ROADMAP.md#node-search-improvement-order) owns current priorities.
The completed five-question LongMemEval pilot requires review of local fact
collection and final synthesis before a broader run. LongMemEval-S is the sole
primary benchmark. Broad backend/model sweeps and other benchmarks, including
LoCoMo, are deferred. The objective map below preserves alternatives and research
dependencies rather than authorizing their execution.

## 1. Decision process and scope

Use the following cycle for each objective:

1. Define a behavioral example or a measurable hypothesis and its competing options.
2. Identify what must remain fixed and what differs between variants.
3. Select a development workload, resource envelope, metric, and practically meaningful effect.
4. Run the smallest informative comparison and inspect failures, including unsuccessful runs.
5. Record an adopted, revised, rejected, or unresolved decision with supporting artifacts.
6. Confirm the selected configuration on untouched evaluation data before making a general claim.

A decision record must distinguish a correctness requirement from an empirical preference. References resolving to the wrong text is a correctness defect. Whether overlapping retrieval chunks improve answers is an empirical choice. Distributed infrastructure is deferred until the workload demonstrates a need.

**Implementation status:** The integrated LLGM baseline performs initial retrieval, bounded node delegation, recursive smaller-model requests, source/journal access, automatic primary-edge proposals, and explicit journal interpretation. Deterministic and hosted mechanism checks are complete for their recorded configurations. The current LongMemEval runner compares ordinary LLGM, BM25 and full-context answering with priced call accounting. The pilot exposed fact-extraction and synthesis failures, so functional integration is not reliable benchmark performance. Automatic semantic-amendment generation, Mem0/Graphiti integration, and the broader architecture controls remain separate work. A root-controlled follow-up phase is a proposed alternative, not current runtime behavior.

Do not invent a universal success threshold. Before a confirmatory comparison, record the smallest quality improvement worth the added cost, or the maximum quality loss acceptable for a saving. Choose these tolerances from the intended use and development variability, then freeze them. A wide uncertainty interval is an unresolved decision, not evidence that two methods are equivalent.

| Stage | Purpose | Allowed adaptation | Exit condition |
| --- | --- | --- | --- |
| Behavioral fixtures | Specify reference, journal, and time semantics | Revise contracts explicitly | Required deterministic behaviors pass. Ambiguous cases have documented outcomes |
| Mechanism diagnostics | Determine whether a capability helps with correct inputs | Hand-authored links, known relevant nodes, detailed inspection | Benefit or failure mechanism is understood well enough to justify automation |
| Development | Tune a realistic implementation | Prompts, models, budgets, indexes, policies | Configuration and decision thresholds are frozen for evaluation |
| Held-out evaluation | Test the selected hypothesis | No tuning on evaluated cases | Report paired effects, uncertainty, costs, and all failures |
| Scaling and transfer | Test workload limits and broader applicability | Separately declared configurations and development data | Claims are restricted to measured scales and task families |

Hand-authored annotations and known-source results are diagnostics, not deployable performance. Keep development histories and held-out histories separate where questions share a conversation. When a public benchmark lacks an official split, publish the deterministic split procedure and report it as a project-defined split, not an official leaderboard result.

## 2. Objective map

| ID | Objective | Main decision |
| --- | --- | --- |
| E00 | Reference and journal correctness | Can every artifact resolve its immutable evidence and identify the journal entries actually inspected? |
| E01 | Baseline inference and delegation | Does a smaller sidecar justify its coordination cost? |
| E02 | Source segmentation and passage construction | Which organization preserves context while enabling effective search? |
| E03 | Retrieval and adaptive search | Which backend and follow-up policy find necessary evidence efficiently? |
| E04 | PEG navigation and link discovery | Do independent primary relationships add value beyond competent search? |
| E05 | Semantic edits and historical interpretation | Which interpretation policies handle change without erasing needed evidence? |
| E06 | Messages, model allocation, and relays | How should evidence move between bounded computations? |
| E07 | Reconciliation, projections, and caches | When should semantic work occur and how should derived state stay usable? |
| E08 | Lifecycle cost and scaling | Where are the quality, cost, freshness, and throughput limits? |
| E09 | Broader compound inference | Do benefits extend beyond conversational memory? |

## 3. E00: Reference and journal behavioral correctness

**Objective:** Establish a trustworthy measurement substrate before evaluating model quality.

**Contract:** Use unique immutable node IDs, turn-relative Unicode spans and journal-entry references. Named passages may provide a derived lookup over those references. Source editions and public read-basis tokens are not alternative identity schemes for the current design. For journal views, compare replay with checkpoints plus a journal tail while preserving the exact input entry IDs and policy identity.

**Fixtures:** Repeated identical text in different nodes. Unicode and boundary offsets. Sources split into different retrieval windows. Journal inline text cited by another entry. A correction to a mistaken annotation. References to unavailable targets. Concurrent appends. A source containing several unrelated decisions. And query replay before versus after a journal correction. Include directed links and their derived incoming lookup without independently authored reverse records.

**Controls and measurements:** Use deterministic sources and explicit expected resolutions. Verify exact text, unique node identity, provenance, journal-entry identity, per-node append ordering and the evidence inventory captured during each query. Rebuild projections and indexes from declared records, then compare results with the same inputs. Rechunking must not alter old citations. Missing references must return explicit failures rather than similar substitute text. Crash or interrupted-write fixtures must expose either a published record or its absence, never partial authoritative state. Include journal reads after an intervening append and corrections that target earlier `JournalRef` records.

**Decision gate:** Required invariants pass before model evaluation. Annotation disagreement is allowed as data. Corrupted references, inaccessible published entries and silent source replacement are not. Observing a later journal append is valid under current-read semantics, provided the consumed entry is recorded. Evaluation replay must still exclude evidence that its protocol has not yet made available. This objective establishes correctness, not a benchmark advantage for one offset encoding.

## 4. E01: Baseline inference and sidecar delegation

**Hypothesis:** Delegating several evidence operations to a smaller model can reduce frontier expenditure while preserving useful answer quality at comparable total resources.

| Option | Role in the comparison |
| --- | --- |
| Strong flat retrieval plus root | Establishes passage access without adaptive inspection |
| Root-directed iterative retrieval | Measures the cost of frontier control over successive evidence operations |
| Smaller adaptive sidecar over flat sources | Isolates delegated search and reading without PEG traversal |
| Ordinary RLM over the same external evidence | Tests against recursive context access without LLGM-specific organization |
| Direct long-context or recent-context reader | Supplementary baseline where the chosen context regime is feasible |

**Protocol:** Start with independent controlled dependency histories. Reuse exposed LongMemEval pilot cases only as diagnostic replication. For a reader diagnostic, provide relevant source nodes rather than answer-bearing spans. The reader must still locate and interpret evidence. In end-to-end runs, remove this assistance. Give each baseline a competent documented configuration and equal development tuning opportunity.

Match the frontier and smaller model versions where roles exist, total evidence availability, timestamp and speaker metadata, answer contract, and resource envelope. Ordinary RLM receives the same evidence in externally addressable form. Its representation and access tools must be documented. An extension with common search tools should be labeled separately from a reproduction of the paper. A method need not use every allocated resource, and forcing identical call counts can handicap a method. Compare several budget points instead.

Distinguish iterative tool use, a structured recursive interpreter, and an isolated code-executing REPL adapter in the manifest. A structured sidecar must not be labeled a reproduction of ordinary RLM solely because it can make nested model calls. The baseline's external-context inspection, recursive calls, and execution environment must match its declared protocol.

**Metrics:** Official answer score and ability-level results. Required-source recall. Root versus sidecar calls and tokens. Unresolved requests. Total cost. Median and p95 latency. Include failures and truncations.

**Decision gate:** Retain delegated control if its quality-cost frontier supports the predeclared tradeoff. Lower root token use alone is insufficient when sidecar retries or maintenance make total cost worse. Freeze comparison configurations before evaluating primary-edge navigation and semantic interpretation. Smaller recursive models already appear in [RLM v3](https://arxiv.org/html/2512.24601v3). Their use alone is not a novelty claim.

## 5. E02: Source segmentation and passage construction

**Hypothesis:** Separating complete source nodes from adjustable search chunks provides useful context expansion without binding semantic edits to a particular retrieval window.

**Options:** Node boundaries at conversations, sessions, fixed contiguous groups of turns, or topic-sensitive contiguous boundaries. Search can use turn windows, token windows with overlap, or structure-aware windows. Candidate results can rank chunks directly, aggregate scores by parent, or reserve diversity across parent nodes.

**Protocol:** Run two separate sweeps. First keep canonical sources and the reader fixed while changing retrieval chunks. Then vary source grouping while preserving exact input text, chronology, and immutable reference mapping. Measure node organization without secretly supplying extracted facts unavailable to other variants. Use short and long nodes, interleaved topics, repeated boilerplate, and evidence straddling boundaries.

**Metrics:** Passage recall where gold spans exist. Jointly required-source recall. Duplicate candidate crowding. Context-expansion bytes. Answer quality. Ingestion and index-build cost. Storage amplification. And reference stability. Report recall denominators and annotation coverage. When a benchmark supplies source IDs but no exact spans, label source recall honestly rather than inventing passage-level ground truth.

**Decision gate:** Select chunking and aggregation on development quality-cost measurements. Keep stable addressing regardless of the winning chunk size. If a grouping change helps only through additional metadata or a larger inspected-text budget, report that interaction rather than attributing the gain solely to nodes.

## 6. E03: Retrieval backend and adaptive search

**Hypothesis:** Focused follow-up search recovers missed dependencies, while the best retrieval backend depends on evidence distribution and available resources.

| Axis | Options |
| --- | --- |
| Backend | Lexical. Dense. Lexical/dense hybrid. ColBERTv2 with PLAID. BM25 plus ColBERT fusion as a separately declared arm |
| Search policy | Original question only. Upfront decomposed queries. Adaptive follow-ups |
| Candidate access | Raw source chunks. Sources plus journal text. Separate source and interpretation results |
| Candidate selection | Score-only. Parent diversity. Estimated coverage of information needs |

**Protocol:** Compare backends within each fixed search policy and policies within each fixed backend. Preserve original question text beside rewrites. Include exact identifiers, paraphrases, less-similar replacement statements, multi-source dependencies, and cases requiring global search beyond initial neighbors. Journal-aware search is a separate condition because it exposes derived semantic information.

Current seed selection takes the first distinct source owners in passage order,
retaining every distinct matched reference for each selected owner. Defaults are
12 passage hits and three seeds. The completed [node-search](../reports/node-search-2026-09-11.md)
and [selection](../reports/node-selection-2026-09-11.md) diagnostics used their
own 12/40-hit and fused-pool conditions. The model selector and BM25 plus ColBERT
fusion are experimental, not application defaults. The subsequent
[answer comparison](../reports/seed-answers-2026-09-11.md) exposed evidence-use
failures rather than a dependable selector winner. Later runtime fixes do not
change those frozen outcomes.

For the next diagnostic, distinguish evidence absent from the candidate pool,
owners excluded by the cap, relevant spans left unread, facts omitted from
branch returns, returned facts ignored in synthesis, and missing final citations.
The [current improvement order](../ROADMAP.md#node-search-improvement-order)
prioritizes the observed pilot failures before another retrieval sweep.
Complete annotated-source recall is not answer accuracy, and annotated sessions
may contain redundant or older evidence rather than a semantic minimum.

Hold model, sidecar prompt, candidate/reading budgets, source eligibility, and indexes' source cutoff fixed. Provide lexical lookup and metadata filters consistently when they are shared tools. If access differs, treat that as the intervention. Tune each backend within a declared development budget, including its hardware requirements.

Keep deployment separate from retrieval method. Hosted generation can power the reader and search policy in every condition. Dense embedding APIs and ColBERT's token-level representations are different backend choices. Selecting a hosted chat model does not supply a compatible ColBERT encoder. Record whether encoding, indexing, and retrieval run locally or through a service, with their latency and cost. Compare local/remote deployment of the same algorithm separately from comparisons between retrieval algorithms.

**Scientific reference versus deployment:** The required C arm remains the released Stanford ColBERTv2 checkpoint with the official PLAID index/search implementation. Calling that exact stack remotely is acceptable. Substituting Jina-ColBERT, AnswerAI-ColBERT, Weaviate, Qdrant, or a candidate reranker is a separately named intervention. Hosted alternatives may be useful library integrations, but they do not replace the scientific reference arm. General library use does not require a user-operated GPU. The selected Modal workflow has executed the actual reference stack on project corpora, including persisted reopening and authenticated local retrieval. This is our managed deployment, not evidence that an arbitrary hosted search product implements the same algorithm. The [deployment report](../reports/colbert-modal-2026-09-11.md) owns its pins, measurements and limits.

The [official ColBERT server](https://github.com/stanford-futuredata/ColBERT/blob/main/server.py) exposes a Searcher over HTTP, and [DSPy's ColBERTv2 client](https://dspy.ai/api/tools/ColBERTv2/) calls a retrieval endpoint. These are examples of remote deployment, not proof that an arbitrary commercial search service implements PLAID. [RAGatouille's PLAID wrapper](https://github.com/AnswerDotAI/RAGatouille/blob/main/ragatouille/models/index.py) executes the ColBERT stack but can modify indexing defaults and algorithms, so wrapping a library also requires recording actual configuration.

A remote reference deployment must expose or export the checkpoint hash, code revision, tokenizer settings, passage identities, corpus/index snapshot, compression, pruning and search parameters. Include encoder and index build cost, query-side compute, transport overhead, cache policy, and failures. Preserve per-case haystack isolation. A public Wikipedia retrieval endpoint cannot substitute for indexing the benchmark's own evidence. Engine latency and end-to-end service latency are separate measurements. Opaque hosting cannot support claims about unobserved hardware efficiency.

**Metrics:** First-retrieval and final evidence recall. All-required-evidence coverage. Useful discoveries per search. Query drift. Search latency and index memory. Offline and incremental indexing cost. End-to-end quality and total work.

**Decision gate:** Keep follow-up search when it recovers consequential evidence at acceptable total cost. Select backend by the measured operating point, not a presumed ranking. [ColBERTv2](https://aclanthology.org/2022.naacl-main.272/) provides contextual late-interaction retrieval and [PLAID](https://arxiv.org/abs/2205.09707) an efficient retrieval engine. Neither selects the sidecar's reasoning policy. [Baleen](https://arxiv.org/abs/2101.00436) and [IRCoT](https://aclanthology.org/2023.acl-long.557/) are relevant iterative-retrieval precedents.

### 6.1 Original E03 retrieval baseline and current status

The September 9 plan specified B/D/H/C × S/U/A. The completed September 10
pilot executed the nine B/D/H configurations under a separately frozen protocol,
with 15 retrieval diagnostics and 45 answering attempts over five questions.
Those measurements used the earlier implementation and do not validate the
current storage refactor. The full twelve-arm comparison remains incomplete.
ColBERTv2 with official PLAID remains the scientific reference for C. Its three
E03 answering cells were not completed in that pilot. Separate September 11
ColBERT deployment, retrieval and seed-selection diagnostics subsequently ran
successfully. They do not complete this historical matrix. This plan does not
authorize resuming or expanding it.

The matrix, representation and resource point below retain the original design.
Here H means BM25 plus dense retrieval, not the later BM25 plus ColBERT fusion.
The [frozen pilot protocol](../../experiments/NON_COLBERT_PILOT.md) and its
configuration own the completed run. New experiments must freeze a new manifest
for the current source schema rather than rewriting that protocol or its results.

**Data eligibility:** The pinned cleaned LongMemEval-S release has one connected
history component containing all 500 cases. Its reused sessions prevent the
original strict 50-case development/held-out split. The five pilot questions are
now exposed, and the remaining 495 questions are not independent held-out
histories. Use separate controlled histories for development. Reusing pilot cases
is diagnostic replication. Any further evaluation must state its exposure and
history overlap instead of claiming an untouched history-disjoint split.
Preparation emits a split audit and requires explicit cohort selection.
Dataset hashes and counts are recorded in the
[experiment guide](../../experiments/retrieval.md).

Use four implementations, with no retrieval-model fine-tuning in this experiment:

| ID | Retrieval implementation | Role |
| --- | --- | --- |
| B | SQLite FTS5 BM25, pinned tokenizer and query escaping | Lexical baseline |
| D | Hosted `text-embedding-3-large`, pinned dimensions, exact cosine search over stored vectors | Dense baseline without approximate-search error |
| H | B + D, equal-weight reciprocal rank fusion. Initial rank constant 60 | Hybrid baseline |
| C | Released ColBERTv2 checkpoint and the official ColBERT implementation's PLAID index/search path | Test the proposed late-interaction approach directly |

The dense model is an initial experimental choice, not a required library provider. Record its actual identifier, dimensions, and encoding date. For C, pin the downloaded checkpoint checksum, repository revision, compression, encoder limits, and search parameters before running. Use the released checkpoint linked by the official repository rather than silently substituting a newer model. The official repository includes ColBERTv2 and PLAID in its main implementation. [OpenAI embedding model](https://developers.openai.com/api/docs/models/text-embedding-3-large), [ColBERT implementation](https://github.com/stanford-futuredata/ColBERT)

Run each backend with three search policies, giving **12 configurations**:

| Policy | Concrete procedure | Candidate allowance |
| --- | --- | --- |
| S: single query | Search the original question once. Then read and compose evidence | Top 40 hits |
| U: upfront queries | Search the original question plus three smaller-model rewrites/decompositions produced before inspecting results | Top 10 per query, 40 hit slots total |
| A: adaptive queries | Search the original question. Inspect results. Optionally formulate the next query from missing evidence. Stop after at most four searches | Top 10 per query, at most 40 hit slots total |

U controls for extra search opportunities when comparing A with S. Both U and A retain the original question as their first query. The same smaller model and common reading/composition instructions serve all arms. Only the allowed search-control procedure differs. Use a fixed iterative reader for this retrieval diagnostic. Recursive execution and model allocation remain separate E01/E06 interventions.

For H, each logical search fuses the top 40 B and top 40 D candidates, then exposes only its policy's allowed hit count. Internal backend work is recorded and charged. Equal visible hit allowances do not imply equal compute. Candidate text/previews count as evidence exposure. Duplicates consume returned hit slots, and the common resolver deduplicates evidence by stable source spans. U/A retain findings across queries without pretending raw scores from different queries are comparable.

**Original data plan:** The proposed sequence used five smoke cases, then targeted
50 development questions stratified by ability, with seed 1729 and shared histories
kept together. The split audit invalidated that selection plan before a disjoint
split could be constructed. The completed pilot instead used its own five-case
frozen protocol. This paragraph preserves the planning history and is not a cohort
selection instruction. The benchmark provides session and answer-turn labels for
scoring. [LongMemEval data and protocol](https://github.com/xiaowu0162/LongMemEval)

For each question, use only its supplied haystack as the searchable corpus. Do not union histories across benchmark cases. Make one released session a source node and index its full original evidence through shared overlapping passages. Initial windows contain at most 180 tokens under the pinned ColBERT tokenizer including rendered metadata, with 32 source-token overlap where a full window permits it. Never cross session boundaries. All four backends index identical rendered passages with exact source mappings. Preflight encoder limits and reject silent truncation. Pin and check ColBERT's query limit as well as its document limit, including special tokens, for original questions and generated follow-ups. Record overflow as a failure unless a common preprocessing policy was declared before comparison. Source roles and dates remain available. Gold answers, `has_answer`, and evidence IDs remain evaluator-only. Source grouping, chunk-size tuning, automatic links, and semantic reconciliation are disabled as interventions in this matrix.

**Initial resource point:** Allow 8,000 cumulative newly exposed evidence tokens per case and a 4,000-token evidence bundle for the fixed root model. Use one pinned accounting tokenizer for these cross-backend limits. Separately record each provider's actual token usage. Repeated evidence in prompts, query-generation calls, intermediate outputs, and all retries still count toward total cost. All arms share a maximum of eight sidecar generation calls and one root answer call. They need not consume the allowance. Record exact hosted root/sidecar model IDs, prompts, sampling, deadlines, and an overall run budget before dispatch. No default model alias is allowed to vary between arms. These are proposed pilot limits, not measured optima or an authorization to spend.

**Readouts:** First run a separate retriever-only original-query top-40 diagnostic and compare passage cutoffs 5, 10, 20, and 40, mapping hits to their source sessions. Those evaluator-only results must not leak extra initial candidates to U/A. Record diagnostic work separately from online answer cost. Report session recall and fraction of cases with all required sessions represented. Separately measure labeled evidence-turn coverage where labels exist, final evidence actually inspected, evidence delivered to the root, and official answer quality. A matching session is not proof that the supporting turn was read. Abstention cases remain in answer evaluation but have no required-source recall denominator. Report cost, search/model calls, duplicate hits, latency, index-build time, bytes, hardware, and transferred data.

Produce one row per declared configuration with paired per-case outputs and error traces. The original decisions were whether C improves the quality-cost tradeoff against B/D/H and whether A improves over both S and U. The C/S, C/U and C/A answering cells remain incomplete in this E03 matrix. The five-case pilot saturated labeled retrieval coverage for single-query B/D/H at top 40, so it cannot establish a benefit from added retrieval. Use independent development histories with enough distractors for future operating-point selection, then freeze eligible evaluation data and paired uncertainty. Backend speed claims require separately stated hardware/network conditions, and PLAID scalability belongs in E08 on larger collections.

**Then test PEG's contribution:** Carry forward C and the best non-ColBERT backend selected on development data. For each, compare traversal off/on under the same fixed search policy and evidence limits. Both conditions can search/read identical source-grounded journal annotations. Linked reads count toward the same evidence allowance. Start with hand-authored links as E04 diagnostics, then test automatically built links without exposure to evaluation questions. Reuse graph-off results only when their inputs and settings are identical. This prevents a retrieval-model gain from being misreported as an LLGM graph gain.

That paragraph retains the original journal-derived navigation proposal. A new
schema-3 comparison must instead hold independent primary-edge descriptors and
journal annotations separately fixed, with equal information access in its
controls. It needs a new manifest and cannot silently reuse an earlier graph-off
result with changed records or runtime semantics.

## 7. E04: PEG navigation and automatic link discovery

**Hypothesis:** Persistent relationships help discover dependencies missed by equally available search, especially when a local source explains why another source matters.

**Options:** No traversal. Chronology/reference links. Hand-authored semantic links for diagnostics. Automatic semantic links. Or automatic links with selective review. Traverse one hop, bounded adaptive hops, or combine traversal with global search. Independent primary edges provide navigation. Journal amendments are held fixed and evaluated separately under E05. An edge's existence does not apply a semantic overwrite.

**Protocol:** First toggle navigation on/off over the same hand-authored, source-grounded relationships. This is a diagnostic of mechanism usefulness, not a formal upper bound. Then replace annotation with automatic discovery, keeping runtime access unchanged. Use chains, branches, disconnected evidence, distractor links, cycles, and missing links. Evaluate links that were built without seeing test questions.

Separate information from access: include a control that can search or read the same relation annotations but cannot traverse adjacency. An on/off traversal comparison that entirely withholds useful annotation text measures both metadata and navigation. For automatic linking, separately measure candidate discovery and relation classification. Link creation must be able to discover previously unconnected nodes.

**Metrics:** Affected-source/candidate recall. Relation precision by type. Recovered dependencies. Unnecessary traversals. Duplicate evidence lineage. Answer quality. Annotation cost. And net query savings. Inspect whether B's context is needed to interpret C, or whether the link only supplies another search hint.

**Decision gate:** Continue automation only if reliable annotations show potential value and automated quality retains enough of it. If search alone performs as well, retain passive source relationships for provenance without claiming a graph-inference gain.

## 8. E05: Semantic edits, historical interpretation, and “how to forget”

**Hypothesis:** Explicit, scoped journal interpretations can improve answers about changing evidence while preserving historical recovery and avoiding false replacements.

**Options:** Raw evidence only. A simple scope-aware chronological/version baseline. Explicit source-declared changes only. Model-proposed relations with unresolved outcomes. Or richer semantic reconciliation. Compare interpretation enabled/disabled while holding navigation access and journal annotation availability fixed. A secondary control presents journal text to the ordinary reader without an interpretation-view operation.

**Fixtures:** Actual replacements versus suggestions. Production versus test environments. Partial replacements within a longer Q&A. Late reports about earlier events. Retractions. Contradictory sources without authority to resolve them. Mistaken model annotations and later corrections. Cyclic replacement claims. And multiple supporting passages derived from one original source.

Ask three distinct question families: what applies now, what applied at a past event time, and what was known at a past observation time. A retrospective question asked now may use a later report about earlier events. A query replayed before that report cannot. Source arrival time, stated event time, journal commit time, and evaluation cutoff must not collapse into one timestamp.

Use hand-authored journal records to validate behavioral semantics, then automated candidate discovery and classification when the active roadmap warrants it. LongMemEval update/temporal abilities remain within the primary benchmark. [MemoryAgentBench's FactConsolidation task](https://github.com/HUST-AI-HYZ/MemoryAgentBench) is a deferred extension requiring its own released protocol.

**Metrics:** Current-answer accuracy. Historical accuracy. False replacement rate. Missed applicable updates. Affected-evidence recall. Correction recovery. Unresolved-conflict handling. Stale answers. And semantic work per update/query. Label gold relations explicitly when manually annotated and measure annotator disagreement rather than treating interpretation as self-evident.

**Decision gate:** An improvement in current answers does not justify unacceptable historical or scope errors. Prefer conservative unresolved outcomes when evidence does not justify replacement. This objective studies changes in answer applicability. Archival and physical deletion are separate, currently unspecified policies. No deletion or universal forgetting mechanism follows from a journal.

## 9. E06: Messages, model allocation, and compositional relays

**Hypothesis:** Bounded, query-conditioned evidence messages preserve enough qualifications for accurate synthesis while allowing routine work to use a cheaper model.

**Options:** Exact excerpts. Generic summaries. Query-conditioned findings with references. Or structured findings with unresolved needs and expandable handles. Model policies include fixed smaller sidecar, stronger sidecar, root review on specified conditions, and selective escalation. Reader policies include direct reading, external inspection without recursive calls, recursive subrequests, pure forwarding, and local composition during return.

**Protocol:** Change message format with models fixed, then model allocation with format fixed. Begin with known-source diagnostics. End-to-end runs then reveal retrieval interactions. Vary chain depth, branch count, contradictory evidence, necessary qualifications, and required output precision. Use the same evidence opportunity and count every transformation. An exact-excerpt condition may need a different quality-cost point, not artificially truncated evidence selected to make summaries win.

**Metrics:** Support and reference correctness. Qualification loss. Distinctions between quoted claims and deductions. Answer quality. Evidence duplication. Subcall/retry count. Total cost. Escalation precision and recovery. And maximum active context. Separate unavailable source, no finding, unresolved conflict, and budget exhaustion.

**Decision gate:** Select the simplest message contract preserving essential evidence at the target budget. Escalation rules must outperform their extra cost on held-out cases. Self-reported confidence is not a calibrated trigger by assumption. Relays are useful only if they improve dependency discovery or interpretation beyond ordinary recursion or direct access.

### 9.1 Required model allocation comparisons

This section preserves the broader model-robustness plan. The nine-pair sweep is
deferred while the five-question LongMemEval review qualifies the current local
reader and root synthesis. API compatibility alone does not qualify a model for
the Python evidence interface or establish that it uses returned facts correctly.

One root/sidecar pair is sufficient to bring up the implementation, but not the planned evidence for robustness across models. The planned model comparison is three root models by three smaller sidecar models, including at least two providers overall, plus a same-root-model sidecar control for each root. The initially selected GPT-6 Astra / GPT-5.6 Terra pair is one cell. Every additional model ID, reasoning setting, prompt adaptation, endpoint, and price table must be pinned before evaluation. This section does not imply those credentials or runners are ready.

| Comparison | Planned configurations | What it isolates |
| --- | --- | --- |
| Retrieval and search policy | Existing B/D/H/C by S/U/A, with one fixed model pair | Retrieval backend and adaptive search |
| Root and smaller-sidecar allocation | Three roots by three smaller sidecars: nine pairs, with C fixed | Sensitivity to root/sidecar capability and cost |
| Same-model allocation | Each root also performs sidecar/recursive work using that same model | Whether gains require delegation to a cheaper model |
| Architecture controls | Within each declared model pair: ordinary RLM, iterative flat-evidence sidecar, and full LLGM | Added value beyond recursion or adaptive retrieval alone |
| Core ablations | Full LLGM with recursion, graph traversal, or journal interpretation individually disabled | Contributions of the proposed mechanisms |
| Retriever transfer | Repeat a predeclared subset of model pairs and key comparisons with hybrid retrieval | Whether conclusions depend on C |

For every root, also run direct full-context inference where the evidence fits, ordinary static RAG, and root-directed iterative retrieval. These root-only baselines are not duplicated as independent observations for every sidecar column. Long-context overflow must be reported separately from a declared truncation or compaction baseline. Ordinary RLM must use the pinned reference execution semantics. An RLM variant augmented with our retrieval tools is an additional, explicitly labeled control. Compare both the reference configuration and matched root/submodel allocations where supported.

Keep the source corpus, supplied journal entry inventory, maintenance model, interpretation policy and evidence opportunities fixed during inference-model comparisons. Isolated workspaces can enforce those experimental inputs without a public snapshot API. Otherwise changing the sidecar could also change graph construction and confound the result. Evaluate maintenance-model variation separately and count its cost. Report controlled fixed-annotation experiments and complete automatic-pipeline experiments distinctly. The latter are required to establish deployable behavior.

Use multiple declared budget points and report quality-cost curves, source coverage, latency, failures, and paired uncertainty. Equal call counts are not equivalent compute when models, reasoning effort, caching, or tools differ. Give each baseline a documented development tuning budget. Do not change models after inspecting reserved benchmark outcomes. Replicate stochastic runs on a predeclared cohort and describe shared-history dependence instead of treating every reused history as independent.

This is a staged experimental design rather than an immediate Cartesian product of every backend, model, ablation, and dataset. Functional live integration has been exercised, but the current evidence-use failures must be resolved before broader model/architecture comparisons. If the nine-pair sweep is later selected, retain all declared pairs in its report, including failures. Do not report only the winning pair. Run sizes and repetition counts must be frozen with a feasible execution budget before dispatch. The current runner uses one configured pair. The complete multi-pair architecture comparison remains unimplemented.

The [RLM paper, v3](https://arxiv.org/html/2512.24601v3#S3.SS2), compares multiple base models and direct, retrieval/code, compaction, and recursive variants. Its GPT-5 configuration uses GPT-5-mini subcalls, and its results report API costs from OpenAI, Fireworks, and Anthropic. This supports using hosted inference with explicit limitations. It does not establish a balanced root-by-sidecar factorial sweep or eliminate the need for our own controls.

## 10. E07: Reconciliation timing, projections, and caches

**Hypothesis:** Reusing semantic work can reduce lifecycle cost, but the best schedule depends on update/query frequency and freshness requirements.

**Options:** Ingestion-time, query-time, periodic/background, or hybrid reconciliation. Journal replay versus checkpoints plus tail. Raw index plus recent-evidence overlay versus synchronous index updates. Uncached versus dependency-aware cached findings.

**Protocol:** First compare timing while holding semantic candidate rules, models, prompts and information available to reconciliation fixed. This isolates schedule and reuse. Query-conditioned reconciliation is a distinct experiment because it has extra query information. Replay a common stream with different queries-per-update ratios, skewed nodes, bursts and queries immediately after an update. Record each query's consumed source and journal entry inventory, publication events during the query, and pending reconciliation/index work. Current reads may observe later appends within the permitted stream. When the view is stale, evaluate raw-source/tail fallback or explicit unresolved behavior rather than silently assuming the update was processed.

Cache experiments include relevant new evidence outside previously inspected nodes. Unchanged dependencies are not enough to prove an old answer remains valid when the search universe has grown. Benchmark result caches must never cross independent cases unless cache reuse is the declared workload.

**Metrics:** End-to-end answer quality. Update-to-visible delay. Read amplification. Reconciliation calls. Foreground/background work. Cache hit and stale-answer rates. Rebuild time. And query plus amortized lifecycle cost.

**Decision gate:** Select schedules for declared workload regimes. Charge background work to the workload that induced it and report unfinished backlog at termination. A low query price achieved by unreported precomputation or deferred maintenance is not a saving.

## 11. E08: Lifecycle cost, budgets, and scaling

**Hypothesis:** Selective access and incremental maintenance can make some growing-history workloads practical, without implying constant work for arbitrary inference.

**Options:** Sequential versus bounded concurrent execution. Fixed versus adaptive request budgets. Full versus incremental derived-index rebuilding. And replay versus checkpointed journal reading. Evaluate these only after functional comparisons identify useful mechanisms.

Sweep history length, irrelevant-history volume, necessary evidence volume, updates per unit time, queries per update, journal entries per node, graph degree, branching, and concurrent queries. Separate fixed necessary evidence with growing distractors from tasks whose required evidence grows. Use released [BEAM](https://github.com/mohammadtavakoli78/BEAM) scales where feasible plus controlled streams. Report actual tested sizes and hardware.

The library's deployment requirements add an explicit backend matrix:

| Profile | Source bytes | Transactional metadata | Purpose |
| --- | --- | --- | --- |
| Local | Local filesystem | Local SQLite | Low-overhead correctness and single-host baseline |
| Mixed | S3 | Local SQLite | Isolate remote source-fetch overhead. SQLite still has local concurrency limits |
| Remote | S3 | Postgres | Evaluate shared persistence, concurrent clients, publication, and recovery |

First hold source data, interpretation policy, retrieval implementation/ranking, and model responses fixed using a recorded replay or deterministic client. This isolates infrastructure behavior. If switching metadata databases also changes the lexical index, use a shared retrieval service for this isolation experiment or report the ranking change as a separate intervention. Then confirm representative operating points using live hosted models. Compare model allocations/providers in E06 rather than attributing their quality differences to storage.

Sweep object size, source reads per request, cache warmth, connection-pool size, I/O concurrency, ingest/query ratios, and index lag. Measure database contention, object requests and transferred bytes, network/storage cost, p95 latency, throughput, and update visibility. Include failed uploads, database failure before/after publication, retries, concurrent journal appends, and index recovery. Verify no visible reference points to unpublished source data. Test equivalent environment/file/Python configuration and record resolved capabilities. No backend may silently weaken a requested visibility guarantee.

Each invocation must satisfy a context bound and every child must spend from its parent's shared run ledger. Reserve budget before concurrent calls. Record actual usage and any estimation error. Depth alone does not bound branching. Cancellation, provider retries, partial responses, timeouts, and budget exhaustion remain measured outcomes. Price caps need conservative estimates and explicit provider limitations. Raw token and tool limits are also reported.

A strict currency-cap mode must reject calls without a defensible upper bound. When a provider cannot supply or bound usage, an estimate is not a guarantee. Mark the run's accounting as incomplete and do not silently treat missing usage as zero. Distinguish enforceable admission limits from provider-reported final usage.

```text
lifecycle cost = ingestion + indexing + semantic maintenance
    + all query execution + storage and network over the measured workload

amortized cost per query = lifecycle cost / evaluated query count
```

Report query-only and lifecycle quality-cost curves, median/p95 latency, throughput at stated concurrency, update lag, peak context, memory, bytes stored, and evidence/journal entries inspected. Keep currency estimates separate from observed token/call counts and measured compute. Pin price tables and state assumptions. Do not add embedding tokens to generation tokens without identifying the categories.

**Decision gate:** Claim scalability only across measured workloads with acceptable quality and freshness. Include cold build, warm steady-state, and rebuild/recovery costs. Distributed infrastructure is justified by a measured bottleneck, not by the graph metaphor.

## 12. E09: Broader compound inference

This objective is deferred until the LongMemEval evaluation supports a useful
memory result. The tasks below remain candidates for a later, broader claim.

**Hypothesis:** Persistent organization and bounded local composition may benefit difficult multi-source inference beyond recalling a personal history.

**Options:** Root plus evidence sidecar. Hypothesis-specific subcalls. Proposal followed by verification. Repeated independent sampling with synthesis. And ordinary RLM. Include direct larger-context reasoning where feasible.

Use [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) for fixed-corpus search and reasoning and a pinned [OOLONG](https://github.com/abertsch72/oolong) variant for broad evidence aggregation. Follow task-specific corpus and scoring restrictions. Do not import conversation-specific semantic replacement rules into a static corpus without a meaningful interpretation.

Hold accessible corpus, models, tools, answer format, and total resources fixed. Measure official quality, evidence coverage, support, cost, and latency. Include negative cases where selective access misses dense interactions or where correlated submodel errors survive synthesis.

**Decision gate:** A memory gain supports a memory claim. General inference requires benefits on additional task families over competent equally resourced inference baselines. No result here by itself establishes a general assistant or a probabilistic graphical-model algorithm.

## 13. Run protocol and reproducibility

Every run uses an isolated workspace. Shared immutable source bytes and declared immutable index snapshots may be reused. Authoritative query-conditioned journal writes, temporary findings, and answer caches must not leak between independent cases. For an explicitly longitudinal workload, preserve state in event order and record that this is a different evaluation unit. Do not ingest evaluation answers or previous benchmark predictions as fresh evidence.

A machine-readable manifest must record:

- Run ID, objective ID, variant, stage, hypothesis, primary metric, tolerances, and stopping rule.
- Code commit plus dirty patch/hash, package and dependency versions, execution platform, hardware, and provider endpoints.
- Redacted resolved configuration and field provenance, blob/database/retrieval adapters and capabilities, pool/concurrency limits, and role-to-provider assignments.
- Dataset release/commit, content hashes, case IDs, split method, eligibility cutoffs, and any manual annotations.
- Immutable source IDs and hashes, journal entry inventories, index/configuration identity, current-read policy, reconciliation schedule, and cache warm/cold policy.
- Exact prompt/template hashes, resolved model identities, sampling parameters, supported seeds, tool permissions, and tokenizer assumptions.
- Per-invocation and shared budgets, concurrency, retry/cancellation rules, price-table version, and evaluator configuration.

Do not store API keys in manifests. Record enough provider metadata to explain reproducibility limits when model behavior or service deployment cannot be pinned.

Proposed run artifacts:

```text
runs/<run_id>/
    manifest.json
    cases.jsonl
    traces.jsonl
    usage.jsonl
    predictions.jsonl
    judgments.jsonl
    metrics.json
    decision.md
```

`cases.jsonl` declares case IDs, eligible input records and source-event cutoffs. Traces identify the exact source and journal references consumed, operation IDs, parent/dependency IDs, tool arguments/results or retained-artifact references, timing, stop reasons and errors. Usage records include all model and maintenance invocations with estimated and actual usage. Predictions retain failed and abstaining cases. Judgments retain rubric version and evaluator output. Metrics specify denominators, aggregation, missing values and uncertainty. These are target artifact requirements, not a claim that every current runner exports each field. Artifact access and redaction must preserve auditability of permitted evidence without storing private model reasoning or unrestricted sensitive payloads.

Separate questions and gold/evaluator data from the ingestion pipeline. Gold answers, annotated relevant sources, and evaluator-only metadata are accessible only to scoring, except in explicitly labeled oracle diagnostics. The current question may guide online retrieval or measured reconciliation, but must not shape advance construction of held-out memory.

Use official scorers when available. Pin judge models/rubrics, blind variant names, and assess a manually reviewed sample of disagreements and unsupported answers. For repeated stochastic runs, use paired cases and seeds where supported. Report paired uncertainty with clustering by shared history or workload. Small diagnostic sets are not significance tests, and repeated tuning on the evaluation set invalidates a held-out claim.

## 14. Interaction tests, error analysis, and initial backlog

After individual ablations, test plausible interactions: retrieval backend × graph traversal, chunking × message budget, automatic annotation quality × reconciliation timing, and update frequency × cache policy. A small planned factorial comparison is more interpretable than comparing only an all-features system against a weak baseline. Avoid an exhaustive parameter grid before individual mechanisms show value.

Classify inspected failures by source eligibility, retrieval, link discovery, semantic classification, stale projection/index/cache, reference resolution, reading, message transformation, root synthesis, and resource termination. Multiple causes may apply. Record whether adequate evidence was available, retrieved, read, communicated, and correctly used. This makes the next implementation change concrete.

The sequence below records research dependencies. It is not a current task list
or a claim that every foundation still needs implementation:

| Order | Deliverable | Objectives and decision |
| --- | --- | --- |
| 1 | Source/reference contracts, minimal journals, fixture runner, and run artifact writer | E00: reliable evidence addressing and replay |
| 2 | Model/tool adapters, shared budgets, retrieval baseline, iterative sidecar, and ordinary-RLM adapter | E01: establish reproducible quality-cost baselines |
| 3 | B/D/H/C retrieval adapters and the 12-configuration S/U/A matrix in section 6.1. Then chunking sweeps | E03: test ColBERT and adaptive search explicitly. E02: evaluate representation changes separately |
| 4 | Independent primary-edge traversal with hand-authored then automatic links | E04: test navigation benefit before broad automation |
| 5 | Scoped interpretation fixtures and policy adapters | E05–E06: validate semantic changes and evidence communication |
| 6 | Schedule, projection, and cache adapters with stream replay | E07–E08: measure lifecycle tradeoffs |
| 7 | Frozen held-out memory evaluation, then corpus/aggregation adapters | E08–E09: support limited, auditable claims |

These deliverables describe experimental dependencies rather than one-to-one
software releases. Retrieval diagnostics need a common iterative reader and the
declared retrievers. They do not depend on ordinary RLM. A delegation claim does
need that reference comparison. Current execution scope, including incomplete E03 cells,
must be declared in each run instead of inferred from the historical milestone
names in the September 9 technical baseline.

The functional baseline now runs real model calls with inspectable predictions, evidence, usage, and failures. The current comparison runner and completed five-question pilot supply the next failure review, described in the [roadmap](../ROADMAP.md#node-search-improvement-order). Broader controlled comparisons remain conditional on that review and a newly frozen execution scope. The package and public documentation exist, with publication pending. Their existence is not evidence that LLGM outperforms a baseline.

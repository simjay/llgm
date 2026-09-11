# Experiments

Repository protocols, operator instructions and source pins used by LLGM's
evaluation tools. These pages are not published in the library documentation.
Run commands from the repository root. Datasets and generated outputs belong
under the ignored `data/` and `runs/` directories.

## Primary evaluation

[LongMemEval](longmemeval.md) is the single current answer benchmark. Its runner
uses ordinary memory construction and retrieval, preserves every scheduled
attempt, and judges saved predictions separately. Start with the
[five-question LLGM-only pilot](longmemeval_pilot_v6.json), which is the CLI and
Make default. It uses a `gpt-5.4-2026-03-05` root with medium reasoning and no
sampling temperature, plus `gpt-4.1-2025-04-14` delegates and maintenance at
temperature zero. Each run freshly ingests all 237 session occurrences for the
same five exposed development questions. This is a development diagnostic, not
a benchmark accuracy estimate.

The [initial three-arm pilot](longmemeval_pilot_v1.json) remains an explicit
comparison option with GPT-4.1 in every generation role. Its BM25 and full-context
readers differ from the v6 root, so comparing their results cannot isolate an
architecture advantage. The [full protocol](longmemeval_v1.json) remains a
separate later choice, not the default. See the
[operator guide](longmemeval.md#commands) for explicit protocol commands.
The [ten-question smoke protocol](longmemeval_smoke_v3.json) validates integration
without claiming benchmark accuracy.

## Historical diagnostics and optional infrastructure

The files below preserve earlier experiments and reproduction instructions.
They are not competing acceptance suites. Their fixed-case report, protocol and
qualification tests have been retired where they duplicated the primary
evaluation contracts. Keep original measured runs and frozen configurations.
The ColBERT workflow remains an optional integration check.

| File | Purpose |
|---|---|
| [Full node-pipeline diagnostic](node-pipeline.md) | Fifteen fixed synthetic cases comparing LLGM Python delegates with one-search root synthesis |
| [Node-pipeline cases](node_pipeline_cases_v1.json) | Curated sources, primary edges, exact amendments and evaluator-only expectations |
| [Node-pipeline configuration](node_pipeline_v1.json) | Model, Docker, budget and arm-order settings for the fixed diagnostic |
| [Retrieval protocol](retrieval.md) | E03 data preparation, isolation, preflight, execution and official scoring |
| [ColBERTv2 and PLAID on Modal](colbert.md) | Remote GPU preparation, integration checks and BM25 comparison |
| [Node-search diagnostic](node-search.md) | Frozen BM25 and ColBERT/PLAID comparison at passage cutoffs 12 and 40 through actual node admission |
| [Node-selection diagnostic](node-selection.md) | Hosted selection over frozen BM25, ColBERT/PLAID and fused top-40 pools with matched three-node controls |
| [Node-selection configuration](node_selection_v1.json) | Pinned retrieval artifacts, model snapshot, repeated trials and local execution limits |
| [Delegate qualification configuration](seed_answer_qualification_v1.json) | Eight-case follow-up changing the per-answer delegate allocation after the frozen answer comparison |
| [Final answers from frozen seeds](seed-answers.md) | Full Python node-pipeline comparison of retained first-owner and hosted-selector choices with separate accuracy and citation-support judgments |
| [Frozen-seed answer configuration](seed_answers_v1.json) | Prior selector repetition, six-arm cohort, model snapshots, Docker image, shared budgets and dated accounting |
| [Node-search configuration with opaque IDs](node_search_opaque_v1.json) | Corrected four-arm comparison on the same 24 expanded histories with benchmark session IDs excluded from retrieval |
| [Original node-search configuration](node_search_v1.json) | Retained original protocol with an answer-correlated session-ID caveat for its LongMemEval arm and a separately valid controlled corpus |
| [Modal ColBERT configuration](colbert_modal.json) | Official code, checkpoint, environment and retrieval pins |
| [Non-ColBERT pilot](NON_COLBERT_PILOT.md) | Frozen five-case B/D/H diagnostic protocol and reproduction commands |
| [Architecture comparison instructions](comparison.md) | Controlled comparison design, version boundary and execution command |
| [E03 matrix](e03_matrix.json) | Twelve-arm template. Unresolved pins must be supplied before execution |
| [Pilot configuration](e03_non_colbert_pilot.json) | Frozen B/D/H × S/U/A configuration for the September 10 pilot |
| [Architecture comparison v3](architecture_comparison_v3.json) | Current structured-operation ablation using independent primary edges and compact journals |
| [Architecture comparison v2](architecture_comparison_v2.json) | Historical immutable-node protocol retained unchanged |
| [Architecture comparison v1](architecture_comparison_v1.json) | Historical protocol retained unchanged. Incompatible with the current runner |
| [Focused node cases](node_pipeline_cases_focused_v1.json) | Twelve autonomous cases and five controlled mechanism checks |
| [Focused node configuration](node_pipeline_focused_v1.json) | Explicit models and execution limits for the focused checks |
| [Source pins](source_pins.json) | Dataset and tokenizer revisions, checksums and preparation provenance |

Current preparation writes schema 2 with immutable node references. Regenerate
older prepared corpora in a new directory. Preflight rejects their schema instead
of changing stored references. The v3 architecture comparison preserves earlier model/budget pins and case
content while separating primary edges from amendments. It has no hosted
measurements and does not evaluate the Python seed-delegate pipeline.

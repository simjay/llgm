# Local state

Observed September 11, 2026 in `/Users/jaehyunsim/Documents/projects/llgm`.
This is machine-specific working context. Recheck before relying on it. Ignored
artifacts and installed services will not exist in a fresh clone.

## Environment and prerequisites

The documentation site contains only end-user guides and API reference.
Contributor and maintainer guides live in `development/`, and logo explorations
in `development/brand-explorations/`. Concepts, architecture, search and usage now
teach through the Atlas example. Strict Sphinx and the public documentation audit
passed, including a build from a temporary tree without research, agent context,
maintainer guides or root Markdown. Default validation reads only public sources.
`make docs-links` separately audits repository links. All 67 documentation tests
passed. Rendered prose checks include API docstrings and browser titles.

The downloadable Docker example had retained the old root payload format.
`examples/offline.py` now reads branch `findings` and the shared `evidence` list.
Its complete Docker run passed with two seeds, a recursive child, source
replacement and retained original text. The standalone correction tutorial also
ran successfully. All nine public Python snippets parsed. Hosted examples were
reviewed against current interfaces but no hosted calls were made for this audit.
The earlier package check confirmed that source archives exclude internal notes
and design explorations. No public site has been deployed.

The project was pushed to private `simjay/llgm` at `902ec7b`. GitHub Actions
[built and audited the documentation](https://github.com/simjay/llgm/actions/runs/34658822776)
and uploaded the HTML artifact. Deployment was skipped. The first CI run exposed
two test dependencies on the checkout import path and working directory. Both
tests now resolve repository tooling and fixtures from their own file location.
A clean installed-wheel run outside the checkout passed 1,022 tests and 245
subtests, with 56 optional checks skipped. GitHub validation of these repairs is
pending. Read the Docs is not connected. Its Community service needs a public
repository, while the current private repository requires Business hosting.

| Resource | Last observed state |
| --- | --- |
| Python | `.venv/bin/python`, Python 3.11.13 |
| uv | `/Users/jaehyunsim/.local/bin/uv` |
| Optional packages | OpenAI 2.54.0, tiktoken 0.11.0, Anthropic, tokenizers, Modal SDK, Coverage.py 7.16.0 |
| ColBERT | `colbert` and `torch` absent locally. Full pinned checkpoint and GPU dependencies verified on Modal |
| Docker | Docker Desktop server 29.7.2 was running during live validation |
| Managed GPU | Modal A10 real integration, separate-container reopen, local SDK client and three-history BM25 comparison passed September 11. User authorized scoped upload and persistent Volume within $20 paid spend. No permanent app deployment or GitHub dispatch was performed |
| Package | `llgm==0.1.0a1`, unpublished, no mandatory runtime dependencies |
| Hosted docs / CI | Private repository `simjay/llgm`, default branch `main`, administrator access verified. The Pages workflow builds and uploads user HTML. Deployment requires `DOCS_PUBLISH_ENABLED=true` and Pages setup. Automatic approval review blocked enabling public Pages pending explicit approval for `https://simjay.github.io/llgm/`. Read the Docs configuration is also prepared |

Credential availability is not established by this file. Check the required
environment variables without printing their values when a live task requires
them. Model, Docker image and evaluator identities for completed runs belong in
the internal [live report](../research/reports/live-validation-2026-09-10.md).
Use run-specific pins. The runtime does not pull Docker images. Canonical live
gates and prerequisites are in [testing](../development/testing.md).

`make setup` installs development/docs dependencies additively. `uv sync` can
remove extras not selected in that invocation. `make setup-colbert` now installs
the pinned optional `modal` extra additively. GPU dependencies belong to the
remote image, not this local virtual environment.

Modal assets are retained in `llgm-colbert-assets`. The successful runs are
`runs/colbert-modal/prepare-20260911-b/`, `test-20260911-a/` and
`benchmark-20260911-a/`. Their result files own status and pins. The initial
HTTP package-download build was stopped and replaced with HTTPS mirrors.
`runs/colbert-modal/local-validation-b/` retains the final local test/coverage
and package checks. See the [ColBERT report](../research/reports/colbert-modal-2026-09-11.md)
for measured conditions and limitations. Fresh serving-container compilation
dominated the first remote query. No actual invoice amount was retrieved.

The authorized node-search experiment completed both its original 35-history run
at `runs/colbert-modal/node-search-20260911-a/` and corrected 24-question run at
`runs/colbert-modal/node-search-opaque-20260911-a/`. The audit found label-correlated
benchmark IDs in original rendered headers. Use the corrected opaque-ID results
for the benchmark comparison. The eight original controlled histories already
use opaque IDs and remain a separate cohort. Both runs retain matching executed
sources in `executed-source/`. Both apps stopped and the final container inventory
was empty. Final local validation and stopped-container evidence are retained in
`runs/node-search-20260911/opaque-validation/`. Actual billed cost remains unknown.
The [node-search report](../research/reports/node-search-2026-09-11.md) owns results,
limitations and the proposed next selection experiment.

The user subsequently authorized the hosted selector comparison. Its preflight
and independent input audit passed under `runs/node-selection-20260911/preflight-a/`.
The run at `runs/node-selection-20260911/live-a/` completed all 288 calls with
pinned `gpt-4.1-mini-2025-04-14` and no failed attempts. It replayed retained
retrieval without a new GPU job. Token-based estimated cost was $0.9514792 against
the approximately $4.01 preflight reservation. Independent input/output/cost audits
and a targeted semantic review are retained with the run. The local process exited
successfully. Validation artifacts are in `runs/node-selection-20260911/validation/`.
Actual invoices were not retrieved. Scoring labels and credentials stay outside
model-visible requests. The
[selection report](../research/reports/node-selection-2026-09-11.md) owns results.

The subsequent final-answer comparison completed all 192 planned attempts under
`runs/seed-answers-20260911/live-a/`, using the saved selector's repetition zero
and the same retained retrieval. All 2,552 provider call records are present.
Two timed-out delegate calls have unknown usage, and one of 192 accuracy
judgments is unknown. A separate eight-case qualification with full GPT-4.1
delegates completed under `runs/seed-answers-20260911/qualification-a/`, retaining
71 calls, two canceled calls with unknown usage and one unknown accuracy judgment.
It improved interface execution but exposed ignored findings and incomplete
citations at the root. Both processes exited successfully and no GPU jobs ran.
The [answer report](../research/reports/seed-answers-2026-09-11.md) owns denominators,
model/budget pins, frozen-source identities, semantic findings and limitations.

Combined known API cost is $3.0807217. Four unknown-usage calls retain $0.0789584
of reservations, for $3.1596801 of local admission liability under the $20 ceiling.
These are dated token-price calculations, not invoices or a complete actual-cost
estimate. Historical selector cost is separate. Independent final input,
request, citation, accounting and summary audits are saved with both runs.
Validation logs, coverage and distribution checks are in
`runs/seed-answers-20260911/validation/`.

The qualification had two Docker cleanup-confirmation failures that aborted
their answers. Docker events confirmed all 24 containers were eventually removed.
A further 12-session real lifecycle diagnostic had no failures and left no
containers, with no hosted calls. It is retained under
`qualification-a/lifecycle-diagnostic/`. An automatic-removal race is plausible
but unproven because the original trace omitted removal stderr. That diagnostic
made no production cleanup change or replacement qualification answer attempts.
The pinned image was
`sha256:ec7d6c95cd3692a2e2d228a8b1ca74e4025b54121fcc4c5da6f09cfa473315ad`.

Subsequent cleanup verification is retained under
`runs/longmemeval-20260911/cleanup/`, using the same pinned image and no model
calls. `before.json` records 16 successful ordinary sessions. The controlled
same-container overlap in `before-overlap.json` records eight cleanup failures
with actual Docker stderr: `removal of container ... is already in progress`.
All eight containers were then confirmed absent. Production cleanup now handles
only that exact-name conflict by inspecting until the container is absent,
within the existing cleanup deadline. It keeps `--rm`, issues no second removal,
and retains failure for visible containers or daemon errors. Repeated
cancellation still retains cleanup lock ownership. `after-overlap.json` records
eight successful controlled sessions, and `after.json` records eight successful
ordinary sessions. Every final exact-container inspection confirmed absence.
Focused REPL and node checks passed (75 tests and 90 subtests). This establishes
the controlled failure mechanism and its handling, not the exact cause of the
earlier qualification or smoke failures whose removal stderr was not retained.

The root citation-schema change and branch-local absence clarification were
exercised by the ten-question synthetic development run
`runs/longmemeval-20260911/smoke-v3/`. All nine admitted root calls returned
citations within their branch-selected ID enum. The remaining LLGM attempt
failed during interpreter cleanup before root synthesis. Earlier smoke attempts
remain unchanged. This tiny authored cohort is not a LongMemEval result or
evidence of reliable full-history reasoning. The later cleanup diagnostic did
not replace or re-score its failed answer.

## Data and artifacts

The user authorized up to $500 API spend, then limited testing to five questions
first. The full run at `runs/longmemeval-20260911/full-v1/` was stopped with SIGINT.
Its Python process and process-scoped `caffeinate` both exited. It retains 74
attempted trials, including four interrupted trials, $14.692706 known estimated
API cost and one unknown call reservation of $0.035528. No judging ran.
`stop.json` records the stop reason. Do not restart this full run automatically.

The five-question pilot completed at
`runs/longmemeval-20260911/five-question-v1/`, with its log beside it. It uses the
immutable `full-candidate-v1/llgm` package through `PYTHONPATH` and copies those
imported sources into the artifact. It schedules 15 attempts across three arms,
two concurrent cases, $5 generation and $0.10 judging allowances, and a shared
400,000-token generation admission window. Later checkout edits do not change
the executed candidate. All 15 answers and judgments are retained. Results are
LLGM 2/5, BM25 4/5 and full context 3/5, with $3.139091 known estimated API cost
including judging, no unknown usage and no runtime/provider failures. All 237
LLGM maintenance outcomes completed and published 80 edges, but no recursive
child ran. The Python process exited successfully. No recurring automation was
created. `independent-audit.json` verifies all 291 calls, 24 final cited spans and
60 executed Python files. All annotated source nodes were selected for the three
LLGM misses, which arose in local reading and reasoning. The default CLI and
Make protocol now select this five-question cohort.

The user then requested iteration until LLGM gets all five correct, deferring
full test suites until that point. The LLGM-only v2 run completed 1/5 at
`runs/longmemeval-20260911/five-question-v2/`, costing $1.761923 including judging
with no unknown usage. Its prompt edit led to bare reads without printed results
and invalid batch wrappers. V3 corrects that interface example, compacts prompt
text, and supplies turn metadata to all seed types. It completed 3/5 at
`runs/longmemeval-20260911/five-question-v3/`, costing $1.631284 including judging
with no unknown usage. The remaining update failure was at the root, which
received both values. Candidate v4 again completed 3/5, costing $1.639703 including judging.
All 272 physical calls have usage and no query callback failed. Root-only replays
are retained in `root-presentation-v5/`, `root-presentation-v5b/`, `root-model-v5/`,
`root-model-v5-neutral/`, `root-model-v5-count/` and `root-model-v5-calculation/`.
They isolate final synthesis on the two failed v4 inputs and are not replacements
for complete trials. V5 completed 4/5 for $1.6602105 including judging. The final v6 run at
`runs/longmemeval-20260911/five-question-v6/` completed 5/5 with an explicit
GPT-5.4 medium-reasoning root and GPT-4.1 delegates/maintenance. Its frozen
`five-candidate-v6/llgm` package was imported via PYTHONPATH. Generation cost
$1.6534215 and judging cost $0.0030275, with no unknown usage. The preceding
`root-reasoning-v6/` replay cost $0.0480175 and did not replace final trials.
The final answers come from one fresh five-question candidate.

The default CLI and Make protocol now select `longmemeval_pilot_v6.json`.
The frozen runtime is unchanged by this subsequent command-default edit and a
recorder docstring punctuation edit. Full `make check` passed after the five-case
gate: 1,039 tests, 245 subtests, two skips, 21 opt-in integration cases deselected,
plus lint and docstrings. Output is in `check-after-five-v6.log`. Strict docs
and rendered-page checks passed in `docs-after-five-v6.log`. The updated default
benchmark preparation validated exactly five questions with zero calls, retained
in `default-v6-preflight/`. V6's independent audit passed 555 checks and verified
all ten final citations and 61 frozen modules. V6 completed all 237 maintenance
outcomes, accepted 67 edges, made 53 delegate read callbacks and no recursive
child query, edge inspection or journal amendment. All five runtime results are
partial from 76 explicitly skipped seed candidates across the five questions.
No full500 or paid model run is active. `development-costs.json` reconciles this
LongMemEval sequence at $28.3853245 known estimated API cost plus $0.039778 in
older unknown reservations. This includes preparation and the stopped full run,
not only the final five questions. The authorized ceiling remains $500.

Preparation has $2.0055855 known estimated API cost plus a $0.00425 unknown
reservation, reconciled directly from physical call files in
`preparation-costs.json`. Do not infer zero cost from absent fields in an older
summary. `final-audit.json` and `full-reader-preflight.json` passed before the
stopped full run. The [LongMemEval report](../research/reports/longmemeval-2026-09-11.md)
owns diagnostic results and limitations. Actual invoices remain unknown.

The user supplied a temporary OpenAI credential for the September 11 node-pipeline
task. It is stored in the explicitly requested local `.env`, ignored by Git and
restricted to owner read/write permissions. The runner loads it only through
`--env-file .env`. Never copy it into reports or frozen artifacts. Read-only access
checks passed for both configured models. Two complete fixed live comparisons
are retained under `runs/node-pipeline-20260911/live-v1/` and `live-v2/`. The
[live diagnostic](../research/reports/node-pipeline-live-2026-09-11.md) records
their distinct outcomes and interpretation limits.
The subsequent hardening task retained two original-cohort runs and a separate
focused autonomous/controlled sweep under `runs/node-hardening-20260911/`.
The [hardening report](../research/reports/node-runtime-hardening-2026-09-11.md)
owns outcomes and distinctions. All three runs finished, and `full-v2` and
`focused-v1` match the final package source hashes. The directory also contains
semantic reviews, canonical-reference/accounting audits, mechanism assertions,
and final local checks. No deferred hosted work remains from this task.

Paths below are relative to the checkout and ignored by Git. Reconstruction
instructions and source identities are indexed in [experiments](../experiments/README.md).

| Path | Contents and limitations |
| --- | --- |
| `data/longmemeval/longmemeval_s_cleaned.json` | Verified official 500-case cleaned release, about 277 MB |
| `data/models/colbertv2-tokenizer/` | Pinned tokenizer/config only. No usable encoder checkpoint |
| `data/controlled/prepared-colbert/` | 50 development cases, 84 passages. Insufficient distractors to establish ranking quality |
| `data/longmemeval/evaluation-colbert/` | Five cases, 5,706 passages, consumed by the September 10 pilot despite the earlier reservation manifest |
| `data/evaluators/LongMemEval/` | Pinned official evaluator checkout with separate dependencies |
| `runs/verified-offline-smoke/` | Synthetic deterministic fixtures, not live scores |
| `runs/e03-preflight.json` | Prior readiness report. Regenerate after configuration changes |
| `runs/integration/` | Local and live diagnostic artifacts with pins and outcomes |
| `runs/coverage/` | Separate unit, local-data, and targeted orchestration measurements. Do not combine unlike denominators |
| `runs/e03-non-colbert-20260910/` | Frozen source/configuration, 15 retrieval and 45 answering attempts, and official judgments |
| `runs/build-validation/`, `runs/review-validation/`, `runs/architecture-hardening-build/` | Historical distributions and their installed-package checks |
| `runs/scaling-20260910/`, `runs/scaling-long-records-20260910/` | Local indexing diagnostics at 100/1,000/10,000 sources. No model latency measurements |
| `runs/architecture-comparison-readiness/` | Frozen comparison preflight with missing credentials and zero model calls |
| `runs/documentation-site-validation/`, `runs/documentation-reorganization/`, `runs/docs-update/` | Earlier documentation and source-archive validation outputs. These do not describe the current public site boundary |
| `runs/architecture-update-20260911/` | Node/edge delivery manifest, check logs, rendered docs and distributions before subsequent code cleanup |
| `runs/code-cleanup-20260911/` | Cleanup before-copy, focused and integrated check logs, rendered docs and distribution validation |
| `runs/node-pipeline-20260911/` | Synthetic 15-case preparations, two complete hosted comparisons, frozen sources, semantic/citation reviews and recursion/accounting audits |
| `runs/node-hardening-20260911/` | Retained hardening failure and final repeat, focused hosted mechanisms, exact frozen inputs, post-run semantic reviews and trace/accounting audits |
| `dist/`, `docs/_build/` | Generated builds. Edit source files instead |

Exposed pilot IDs: `3fdac837`, `86b68151`, `e61a7584`, `720133ac`, and
`gpt4_4ef30696`. Local integration separately uses four pinned IDs across five
tests. Its diagnostic word-tokenizer counts differ from the E03 tokenized corpus.
The historical architecture readiness used
`experiments/architecture_comparison_v1.json`, SHA-256
`148549682d7ed456fca86c18c7a1848c33057e57475228abeaa05cff1c5c9000`.
The current runner requires `experiments/architecture_comparison_v3.json`.
Use the [comparison instructions](../experiments/comparison.md)
for its hash and execution command. Old prepared corpora and workspace databases
need explicit schema handling. Preserve them and prepare new inputs separately.

Traces retain queries and may include source text. Inspect their contents before
sharing them. Summary records and sanitized adapter events are different outputs.

## Reading validation history

[Research reports](../research/reports/README.md) own dated outcomes and measurement
counts. Later fixes do not rerun historical live experiments. The live report,
architecture report, and coverage report refer to different checkpoints.
Installed-package results also belong to the exact distribution that was checked.
Research records remain internal. Old generated research pages under
`docs/_build/` or earlier validation directories are stale build artifacts, not
part of the intended public documentation.

After restoring a metadata backup, close all workspace users and discard the old
derived `.indexes` directory before reuse. Arbitrary restore with a stale index
is not a supported recovery contract. Source and journal history is authoritative.

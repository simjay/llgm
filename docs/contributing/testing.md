# Testing

Use deterministic tests to protect implementation contracts, optional integration
checks to verify real service boundaries, and the LongMemEval runner to measure
answer quality. A passing unit test or a valid citation does not establish answer
accuracy.

## Choose a test layer

| What you need to know | Check | What actually runs |
| --- | --- | --- |
| Did a code change break storage, routing or evidence handling? | `make check` | Local storage and runtime code with controlled model responses and interpreter fixtures |
| Can the real reader sandbox execute and shut down safely? | DSPy sandbox checks below | Real Deno/Pyodide, with fixed model outputs and no hosted model calls |
| Can LLGM process the pinned real histories? | `make test-data` | Complete local LongMemEval histories through storage and retrieval, without answer models |
| Does an adapter work against its service? | Opted-in provider, S3 or ColBERT tests | The selected real service with explicit credentials or assets |
| Are answers useful and what do they cost? | LongMemEval runner | Frozen questions, generated answers, separate judging and recorded costs |

Keep these layers because they answer different questions. A deterministic
interpreter fixture cannot prove isolation, and a successful hosted request
cannot prove answer accuracy. The unused alternative answer controllers and
their dedicated test suites have been removed. `NodeRuntime` is the single
answer controller to test. Shared accounting still needs its own budget tests.

## Product checks

Run from the repository root:

```bash
make setup
make check
```

`make check` runs Ruff, the docstring audit, and deterministic tests without
hosted calls. During development, run the affected test files first. The main
contracts are:

| Area | What the tests protect |
| --- | --- |
| Storage and journals | Exact Unicode spans, append-only topic nodes, restart, idempotency, concurrent publication, scoped and timed amendments, canonical replacements, and explicit missing-target failures. |
| Retrieval and graph | Reusable indexes, current-write visibility, independent primary edges, withdrawal, canonical retrieval references, and bounded source metadata. |
| Node inference | All admitted seeds are scheduled. Children have isolated context. Only delivered, selected evidence can be cited. Budgets reserve finalization, and failures or cancellation retain admitted findings and close interpreters. |
| Application and adapters | Resource ownership, maintenance accounting, configuration precedence, provider schemas, refusals, truncation, and known versus unknown usage. |
| Evaluation | Gold and label-bearing source IDs stay outside generation inputs. Frozen sources and scheduled membership are checked. Failures remain in denominators, and judging and cost accounting stay separate. |

The fixtures in `tests/node_support.py` and `tests/test_repl.py` provide finite
interpreter responses while DSPy's loop, LLGM's scheduler, evidence access and
budget ledger run. They do not execute arbitrary generated Python. Real sandbox
checks cover execution and isolation separately.

Hybrid search tests use controlled RPC and encoder fixtures to check generation
refresh, failure handling and canonical source references. Despite its filename,
`test_live_index_service.py` is a deterministic test of the live-index service,
not a real GPU run. `test_workspace_retrieval.py` also makes no Modal calls.

Prefer observable behavior and known failure cases over assertions about private
helpers or exact implementation recipes. Keep tests for recovered reads, local
versus global gaps, journal conflicts, and valid abstention. Do not retry a model
run until it passes or treat a larger test count as stronger semantic evidence.

`make test-data` additionally exercises checksum-pinned local LongMemEval histories
through storage, BM25 retrieval, and restart. Its fixture is
`tests/fixtures/longmemeval-s.json`. It expects
`data/longmemeval/longmemeval_s_cleaned.json`. Set
`LLGM_TEST_LONGMEMEVAL_PATH` to use another local file. Missing optional data skips
these checks. An explicitly configured missing or mismatched file fails. Tests
never download the dataset. These checks do not run an answer model.

### Additional local checks

For other changes:

```bash
make coverage  # line and branch reports under runs/coverage/unit/
make docs      # strict Sphinx build plus rendered API and navigation checks
make build     # wheel/sdist build and metadata validation
```

Coverage identifies unexecuted code. It is not an accuracy score and has no global
percentage gate. Packaging changes also need the installed-distribution checks
outside the checkout, with `VERIFY_LLGM_WHEEL=1` after installing the build.

The PyPI release workflow calls the same CI workflow at the release commit and
waits for every job to pass. CI retains its tested wheel and source archive as
`python-distributions`. Publishing consumes those files without rebuilding.
See [publishing](publishing.md#release-the-package-to-pypi) for setup and release
commands. When changing Actions workflows, run `actionlint` if installed, and
verify changed build and validation commands locally.

### Documentation checks

`make docs` checks public source links, navigation, rendered API symbols, prose
punctuation, and publication exclusions. It does not read internal documents.
`make docs-links` is a separate optional link audit for the whole repository.
Neither command judges clarity or verifies claims about runtime behavior.
`tests/test_documented_examples.py` separately executes the complete storage
tutorials and checks their documented outputs. Review other prose against the
implementation and read tutorials in order. These local checks
also do not establish remote website availability.

After moving or deleting a documentation page, use a fresh `DOCS_BUILD_DIR` or
remove only `docs/_build/` before rebuilding. The audit rejects obsolete HTML,
repository-only pages, and internal content in published output.

## Optional boundary checks

Install only the relevant optional dependencies. Gates must be exactly `0` or `1`.
A disabled gate skips its check. Enabling it requires its configured credentials,
assets, and services. A skip is not evidence that a boundary works. Hosted calls,
S3 writes, and remote GPU work can incur charges.

### DSPy sandbox

The development group includes DSPy and its managed Deno executable. Allow
runtime asset downloads on first startup, then run:

```bash
LLGM_TEST_SANDBOX=1 .venv/bin/python -m pytest tests/test_repl_sandbox.py -m sandbox -q
.venv/bin/python examples/offline.py
```

The tests execute real Pyodide Python with fixed model outputs. They check state
persistence, recovery after Python errors, async host callbacks, host filesystem
and environment isolation, denied network permissions, deadlines and process
cleanup. The example also exercises canonical citations, amendments and recursive
node readers through real storage. Neither command makes hosted model calls.
Deterministic unit tests run DSPy's actual loop with finite `CodeInterpreter`
fixtures. They do not evaluate generated Python on the host.

### Providers

Install the corresponding `openai` or `anthropic` extra and export
`OPENAI_API_KEY` or `ANTHROPIC_API_KEY` through your normal environment. Use exact
model identifiers:

```bash
LLGM_TEST_OPENAI=1 LLGM_TEST_OPENAI_MODEL=YOUR_MODEL \
  .venv/bin/python -m pytest tests/integration/test_live_providers.py -m openai -q

LLGM_TEST_ANTHROPIC=1 LLGM_TEST_ANTHROPIC_MODEL=YOUR_MODEL \
  .venv/bin/python -m pytest tests/integration/test_live_providers.py -m anthropic -q

LLGM_TEST_EMBEDDING=1 LLGM_TEST_EMBEDDING_MODEL=YOUR_EMBEDDING_MODEL \
  .venv/bin/python -m pytest tests/integration/test_live_providers.py -m embedding -q
```

These verify native response schemas, actual model identity and usage, or physical
embedding batches. They do not measure general reasoning quality.

### Hosted application diagnostic

`tests/integration/test_live_application.py` checks the whole reader pipeline
with real models and a pinned local history. Enable `LLGM_TEST_APPLICATION=1`
and configure `MAIN`, `READER` and `GRAPH` under the
`LLGM_TEST_APPLICATION_` prefix. Each role needs both `_PROVIDER` and `_MODEL`,
plus the selected provider's credential. For example, the main role uses
`LLGM_TEST_APPLICATION_MAIN_PROVIDER` and `LLGM_TEST_APPLICATION_MAIN_MODEL`.
It also needs the local dataset and `rlm` extra.

```sh
LLGM_TEST_APPLICATION=1 .venv/bin/python -m pytest tests/integration/test_live_application.py -q
```

This diagnostic imports fixed source identities and uses local BM25. It checks
hosted reading and maintenance over that history. It does not measure automatic
topic placement or configured Modal hybrid search. Use the frozen evaluation
runner for comparative answer scores.

### S3 and ColBERT

For S3, install the `s3` extra and configure boto3's normal credential chain:

```bash
LLGM_TEST_S3=1 LLGM_TEST_S3_URI=s3://YOUR_BUCKET/nonempty-test-prefix/ \
  .venv/bin/python -m pytest tests/integration/test_live_s3.py -q
```

The test creates a unique child prefix, checks real blob publication and restart,
and cleans up only its attempted objects. It requires the corresponding object
read, write, and deletion permissions.

For a local official ColBERT/PLAID installation, provide the pinned checkpoint,
checksum, code revision, and local dataset:

```bash
LLGM_TEST_COLBERT=1 \
LLGM_TEST_COLBERT_CHECKPOINT=/path/to/local/checkpoint \
LLGM_TEST_COLBERT_CHECKPOINT_SHA256=YOUR_CHECKPOINT_SHA256 \
LLGM_TEST_COLBERT_REVISION=YOUR_OFFICIAL_CODE_REVISION \
  .venv/bin/python -m pytest tests/integration/test_live_colbert.py -q
```

`LLGM_TEST_COLBERT_GPUS` optionally selects the GPU count. The tests build, search and reopen actual indexes. A small-corpus case exercises
exact ColBERT scoring and hybrid refresh before the PLAID size threshold. These
checks never substitute lexical retrieval for the semantic component.
Remote setup and frozen retrieval protocols remain manual workflows in the
repository's `experiments/` directory. Available tests do not imply that S3 or GPU
validation has been performed in a particular environment.

## LongMemEval evaluation

`llgm.evaluation.memory_benchmark` supports LLGM, BM25 and full-context reader
arms. Its default protocol runs LLGM on five exposed development questions from
LongMemEval-S. Other arms need an explicitly frozen comparison protocol. Preparation verifies pinned local inputs and records the
protocol and planned schedule without making model calls. Execution requires an
explicit flag and a new output directory. It copies and hashes the imported
package sources before dispatch. Generation and official-prompt judging are
separate phases. Repository setup, model choices,
frozen protocols, and run commands belong in the
[LongMemEval runbook](../../experiments/longmemeval.md).

Report scheduled, completed, failed, and unstarted trials together. Preserve
unknown usage and unavailable judgments, and keep retrieval coverage separate
from answer accuracy. A preparation record establishes readiness, not a completed
evaluation.

### Find the stage that failed

Use the same retained question and exact source spans to distinguish these stages:

| Stage | Question to check |
| --- | --- |
| Candidate retrieval | Did the retrieved passages include the needed evidence and its node? |
| Seed admission | Was that node among the admitted seeds, or explicitly skipped by the cap? |
| Local inspection | Did its delegate actually read the relevant turn and value? |
| Branch return | Did the branch select that evidence and return the supported fact? |
| Main synthesis | Did the main model use the returned facts correctly and cite all necessary evidence? |
| Execution | Did a model, budget, callback, or cleanup failure interrupt any of these stages? |

Source annotations can include redundant or older evidence. Complete annotated
node recall is a useful diagnostic, not a semantic minimum or an answer score.
Exact citation matching proves source identity, not that the cited text supports
the generated claim. A recursive mechanism claim requires an observed child call
and delivered findings. Merely constructing or inspecting an edge is insufficient.

Readiness and structural tests cannot qualify a model's local reading or final
reasoning. Small real-history diagnostics can reveal dropped updates, inaccurate
counts, and unsupported date comparisons even when retrieval found every
annotated source. Compare separately frozen candidates on the same development
questions, preserve earlier failures, and avoid treating repeated exposed
questions as an independent holdout. See [node search](../guide/node-search.md)
for the current selection contract.

### Check evaluation accounting

Shared accounting tests cover concurrent priced admission, token pacing,
pre-dispatch cancellation, missing usage, and durable request records. Queue
delay stays in measured latency. A request canceled before dispatch releases its
reservation, while missing usage after dispatch keeps cost unknown and retains
the reserved liability. These evaluation controls are separate from the ordinary
application's call and time limits.

The synchronous framework transport is tested at the physical HTTP boundary so
response usage survives a later SDK parsing failure. It supports bounded
complete-response requests, not streaming. Its caller must drain worker threads
before closing the client or accounting loop. These tests prepare for framework
comparisons. They do not establish a working or measured Mem0 or Graphiti arm.

## Local graph viewer

Run `tests/test_viewer.py` for metadata-only graph reads, paged records, canonical
search, and the HTTP request boundary. The optional loopback test requires no
provider or GPU service:

```sh
LLGM_TEST_VIEWER=1 .venv/bin/python -m pytest tests/test_viewer.py -q
```

After changing viewer assets, inspect graph selection, conversation switching,
search, record expansion, and file links in a browser at desktop and mobile widths.

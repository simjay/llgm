# Testing

Use deterministic tests to protect implementation contracts, optional integration
checks to verify real service boundaries, and the LongMemEval runner to measure
answer quality. A passing unit test or a valid citation does not establish answer
accuracy.

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
| Storage and journals | Exact Unicode spans, immutable nodes, restart, idempotency, concurrent publication, scoped and timed amendments, canonical replacements, and explicit missing-target failures. |
| Retrieval and graph | Reusable indexes, current-write visibility, independent primary edges, withdrawal, canonical retrieval references, and bounded source metadata. |
| Node inference | All admitted seeds are scheduled. Children have isolated context. Only delivered, selected evidence can be cited. Budgets reserve finalization, and failures or cancellation retain admitted findings and close interpreters. |
| Application and adapters | Resource ownership, maintenance accounting, configuration precedence, provider schemas, refusals, truncation, and known versus unknown usage. |
| Evaluation | Gold and label-bearing source IDs stay outside generation inputs. Frozen sources and scheduled membership are checked. Failures remain in denominators, and judging and cost accounting stay separate. |

Replay transports simulate known callbacks while the real scheduler, evidence
reader, and ledger run. They never evaluate generated Python. Real Docker checks
cover that boundary separately.

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

### Documentation checks

`make docs` checks public source links, navigation, rendered API symbols, prose
punctuation, and publication exclusions. It does not read internal documents.
`make docs-links` is a separate optional link audit for the whole repository.
Neither command judges clarity or verifies claims about runtime behavior. Review
prose against the implementation and read tutorials in order. These local checks
also do not establish remote website availability.

After moving or deleting a documentation page, use a fresh `DOCS_BUILD_DIR` or
remove only `docs/_build/` before rebuilding. The audit rejects obsolete HTML,
repository-only pages, and internal content in published output.

## Optional boundary checks

Install only the relevant optional dependencies. Gates must be exactly `0` or `1`.
A disabled gate skips its check. Enabling it requires its configured credentials,
assets, and services. A skip is not evidence that a boundary works. Hosted calls,
S3 writes, and remote GPU work can incur charges.

### Docker

Use an already available trusted Python image and a running daemon:

```bash
LLGM_TEST_DOCKER=1 LLGM_REPL_DOCKER_IMAGE=YOUR_LOCAL_PINNED_IMAGE \
  .venv/bin/python -m pytest tests/test_repl_docker.py \
  tests/test_seed_answers.py -m docker -q
```

These tests use real containers with deterministic model decisions. They check
persistent Python state, callbacks, output limits, source citations, isolation,
and cleanup without hosted model calls. Images are never pulled. A missing image
or daemon fails an enabled check.

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

`LLGM_TEST_COLBERT_GPUS` optionally selects the GPU count. The check builds,
searches, and reopens an actual index. It never substitutes lexical retrieval.
Remote setup and frozen retrieval protocols remain manual workflows in the
repository's `experiments/` directory. Available tests do not imply that S3 or GPU
validation has been performed in a particular environment.

The retained hosted application and specialized executor diagnostics use these
additional gates. Each configured role requires both a `PROVIDER` and a `MODEL`
variable under the listed prefix, plus that provider's credential:

| Test file under `tests/integration/` | Gates | Model setting prefix and roles | Additional inputs |
| --- | --- | --- | --- |
| `test_live_application.py` | `LLGM_TEST_APPLICATION=1`, `LLGM_TEST_DOCKER=1` | `LLGM_TEST_APPLICATION_`: `ROOT`, `SIDECAR`, `MAINTENANCE` | Local dataset and `LLGM_REPL_DOCKER_IMAGE` |
| `test_live_recursive.py` | `LLGM_TEST_RECURSIVE=1` | `LLGM_TEST_RECURSIVE_`: `ROOT`, `SIDECAR` | Local dataset and prescribed oracle source handles |
| `test_live_rlm.py` | `LLGM_TEST_RLM=1`, `LLGM_TEST_DOCKER=1` | `LLGM_TEST_RLM_`: `ROOT`, `CHILD` | `LLGM_REPL_DOCKER_IMAGE` and prescribed external-context protocol |

For example, the application root uses
`LLGM_TEST_APPLICATION_ROOT_PROVIDER` and `LLGM_TEST_APPLICATION_ROOT_MODEL`.
These are manual compatibility diagnostics. Their results do not replace the
LongMemEval evaluation below.

## LongMemEval evaluation

`llgm.evaluation.memory_benchmark` compares ordinary LLGM with BM25 and
full-context readers. Its default protocol selects five development questions
from LongMemEval-S. Preparation verifies pinned local inputs and records the
protocol and planned schedule without making model calls. Execution requires an
explicit flag and a new output directory. It copies and hashes the imported
package sources before dispatch. Generation and official-prompt judging are
separate phases. Repository setup, model choices,
frozen protocols, and run commands belong in the
[LongMemEval runbook](../experiments/longmemeval.md).

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
| Root synthesis | Did the root use the returned facts correctly and cite all necessary evidence? |
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
questions as an independent holdout. See [node search](../docs/guide/node-search.md)
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

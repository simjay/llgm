# ColBERTv2 and PLAID on Modal

This workflow runs the official ColBERT implementation with the released
ColBERTv2 checkpoint and PLAID search on a remote GPU. The local machine needs
Python and the Modal SDK. It does not need CUDA, PyTorch or FAISS.

The [configuration](colbert_modal.json) owns the upstream code and checkpoint
revisions, file hashes, runtime dependencies, hardware and retrieval parameters.
The [source pins](source_pins.json) identify the original LongMemEval release.
Do not substitute a hosted embedding model or a tokenizer-only checkpoint.

## Setup and commands

Run from the repository root:

```bash
make setup
make setup-colbert
.venv/bin/python -m modal setup
make colbert-prepare
make test-colbert
```

Authentication is a one-time local step. Existing Modal authentication can be
reused. `setup-colbert` adds the pinned SDK to the selected environment without
installing the local `colbert` extra or replacing other optional dependencies.

| Command | Action |
| --- | --- |
| `make colbert-prepare` | Build the remote environment and download or verify pinned weights and public data. |
| `make test-colbert` | Prepare assets, then check real build, search, reopen and local client retrieval. |
| `make benchmark-colbert` | Compare BM25 and ColBERTv2 + PLAID with the configured real-history inputs. |
| `make colbert-shell` | Open an interactive shell with the GPU test image and Volume mounts. Exit the shell when finished. |
| `make colbert-deploy` | Deploy the remote retrieval functions for authenticated SDK calls from local LLGM. |

The remote commands use the authenticated Modal workspace and can incur charges.
Set the workspace's spend limit before running them. SDK timeouts and concurrency
limits bound execution. They do not implement strict dollar admission. No OpenAI
or Anthropic credentials are used by these retrieval commands.

Each Make target wraps `tools/colbert_modal.py`. Direct invocation supports
`--action prepare`, `--action test` and `--action benchmark`, with optional
`--run-id` for an explicit artifact identity:

```bash
.venv/bin/python -m modal run tools/colbert_modal.py --action test --run-id colbert-check-001
```

Use a new run ID for a new attempt. Keep failed attempts alongside successful
runs. Local outputs are retained under `runs/colbert-modal/`.

## Persistent assets and development

The Modal Volume `llgm-colbert-assets` retains checkpoint files, the pinned
dataset, immutable indexes and run artifacts. Preparation verifies downloaded
content against the declared pins. Tests use those verified local container
paths and block implicit model downloads.

An index belongs to its exact source passages, tokenizer, checkpoint and index
configuration. Reuse a compatible index after code changes that do not change
those inputs. Changes to an indexed input require a new index identity. An
incomplete or incompatible index must fail clearly instead of being searched or
silently rebuilt by an open operation.

For local library changes, first run the affected deterministic tests, then
`make test-colbert`. The remote image receives the current checkout's library
and test code. Compiled GPU dependencies and downloaded assets remain remote.
The shell uses the same environment for investigating an upstream compilation
or search failure.

After a successful test and `make colbert-deploy`, query the saved index from
the local machine using that run's downloaded canonical passage records:

```bash
.venv/bin/python examples/modal_retrieval.py --run-dir runs/colbert-modal/YOUR_RUN_ID
```

The example checks corpus and checkpoint identity and reports whether the remote
ranking matches the saved result. Supply `--query` for another question. The
library client is `llgm.retrieval.modal.ModalColBERTRetriever`. It implements the
existing `Retriever` interface and can be supplied through an application's
`evidence_factory` with `Evidence.open(..., retriever=...)`. Application evidence
search also merges journal results. Use raw retrievers for the B/C comparison.

## Integration acceptance

The first gate uses a complete checksum-pinned LongMemEval-S history. It runs the
existing real ColBERT integration test and additional persisted-index checks:

1. Build a genuine PLAID index using the full checkpoint and actual tokenizer.
2. Retrieve top-40 passages and map every result to its original source span.
3. Open the saved index without rebuilding and compare passage IDs and scores.
4. Repeat the open and search in a fresh process, independently of the original
   Python objects, in a separate Modal function container. Require different
   container identities in the saved result.
5. Reject mismatched passages or settings, and keep histories isolated.
6. Call the index from the local `ModalColBERTRetriever` through authenticated
   SDK functions. Compare all 40 passage IDs, ranks, scores and canonical
   evidence records with the saved baseline. Retain `client.json` with timing.

The client check uses the running test app and does not require a permanent
deployment. The separate example uses `connect()` against the deployed app.

This gate checks operational correctness. It does not assert an invented recall
threshold or establish benchmark superiority. A fresh process establishes
process-independent reopen. A fresh container establishes the additional remote
storage boundary only when the retained run records that check.

## Retrieval comparison

The initial arms are **B: BM25** and **C: ColBERTv2 + PLAID**. They use identical
source histories, tokenizer-defined passage boundaries, queries and retrieval
cutoffs. Graph expansion, recursive model calls and answer generation are absent.
Evaluation labels are used after retrieval and never inserted into passages.

The diagnostic configuration uses `doc_maxlen=180` and `query_maxlen=128`,
including special and marker tokens. The adapter rejects text beyond those
limits before encoding. The upstream default query length is 32. Its
[query tokenizer](https://github.com/stanford-futuredata/ColBERT/blob/cc4f3dc91c0b45d2d08c251d9d95178285c65f1c/colbert/modeling/tokenization/query_tokenization.py)
pads shorter queries to the configured length with mask tokens. Their embeddings
participate in MaxSim scoring, so 128 can change ranking and compute even when a
query would fit within 32 tokens. Report this as the 128-token configuration.
A comparison with the 32-token setting requires a separate protocol and is not
part of this diagnostic run.

Record separately:

- Source- and turn-level evidence coverage at the declared retrieval cutoffs.
- Index build duration and persisted index bytes.
- First-query duration and repeated warm-query latency.
- Remote invocation elapsed time, separately from time spent searching.
- Code, image, dependency, checkpoint, dataset and configuration identities.
- Every failure and its affected case and arm, with missing costs left unknown.

The worker's `build_seconds` measures the complete `ColBERTRetriever.build` call,
including checkpoint verification, model loading, passage validation, native
index construction and searcher initialization. Its descriptor's `build_seconds`
measures the native `Indexer` construction and index call only. BM25 build time
measures `SQLiteBM25Retriever.from_passages` on a new SQLite file. Keep these
scopes distinct when comparing setup costs.

Warm server samples measure the public retriever search call, including query
validation and result conversion. They exclude the worker's preceding manifest
verification and index opening, as well as Modal dispatch. The remote response
reports `open_seconds` separately. Client elapsed time includes dispatch,
container startup, server preparation, search and response validation. Subtracting
server search time from client elapsed time does not isolate network latency.
The local retrieval example records client/server timing. The B/C worker
benchmark alone does not measure remote per-query invocation time.

Source coverage is a retrieval diagnostic. A partial passage overlap with a gold
turn does not prove the answer was present. Abstention questions have no positive
evidence recall denominator. Repeated diagnostic questions are exposed development
inputs, not independent held-out benchmark evidence. All 500 LongMemEval-S cases
share one connected history component under the current session-ID/text rule.

ColBERTv2 is the retrieval model and PLAID is its efficient search engine. This
B/C comparison evaluates their combination against BM25. Isolating PLAID itself
would require a separate comparison with exact or uncompressed ColBERT search
over the same encoded passages and scoring model. The initial workflow does not
claim that engine ablation.

## Manual CI

`.github/workflows/colbert.yml` runs only through `workflow_dispatch`. Configure
repository secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`, then dispatch the
workflow from the intended revision. The job installs the local SDK, runs
`make test-colbert`, and uploads `runs/colbert-modal/` even after a failure when
artifacts are available. Runs share a concurrency group and have a 100-minute
job timeout. Pushes and pull requests do not launch these paid GPU checks.

Ordinary CI still tests core contracts without requiring Modal credentials.
The remote workflow's existence is not evidence of a completed CI run.

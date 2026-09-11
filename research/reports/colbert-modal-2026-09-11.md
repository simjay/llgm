# ColBERTv2 and PLAID on Modal

Implementation and live execution record, September 11, 2026. The official
ColBERTv2/PLAID path built a real index on an A10, reopened it in a different
container, and returned matching results through the local remote-retrieval
client. A three-case comparison then found both BM25 and ColBERT covered all
labeled sources and turns at top 40. BM25 was faster and used smaller indexes on
these histories. These checks establish operational behavior on exposed
LongMemEval inputs. They do not establish a held-out quality improvement or
broad scalability.

Later September 11 audit: the [node-search study](node-search-2026-09-11.md)
found support-label cues in original LongMemEval session IDs rendered in passage
headers. This run used original IDs. Its measurements and operational checks
remain historical evidence; label-neutral retrieval quality requires the
separately recorded opaque-ID comparison.

## Delivered workflow

The repository now provides pinned remote image construction, verified checkpoint
and dataset preparation, persisted index records, separate build/reopen jobs,
a BM25 comparison job, a Modal implementation of the `Retriever` interface, Make
commands, and a manually triggered GitHub Actions workflow. The command contract
is maintained in [the experiment guide](../../experiments/colbert.md).

The remote payload contains scoped library, job and test files. It excludes
credentials, internal research, agent context and earlier local results. The
user authorized exporting that scoped source and running jobs within the
workspace's $20 paid-spend limit. No hosted generation API was used.

The first image build was aborted during HTTP Ubuntu package setup. Changing
the package mirrors to HTTPS allowed image construction and preparation to
complete. This was an environment-build interruption before the GPU tests, not
a failed encoder or PLAID run.

## Frozen execution identities

The protocol is [experiments/colbert_modal.json](../../experiments/colbert_modal.json).
Complete per-file checkpoint hashes and dependency pins remain in that protocol
and its dependency lock. The live environment records these identities:

| Item | Recorded value |
| --- | --- |
| Official ColBERT commit | `cc4f3dc91c0b45d2d08c251d9d95178285c65f1c`, verified through PEP 610 installation metadata |
| Checkpoint | `colbert-ir/colbertv2.0` at `c1e84128e85ef755c096a95bdb06b47793b13acf` |
| Checkpoint directory SHA256 | `482743edb1820fda2de676d3260032ec9f15970a110bd8e468c1c6506e3ab804` |
| Full model file | `model.safetensors`, 438,349,816 bytes, SHA256 `3f58890b1dfdfec066ef12ba431fa9d56992da9e30a53489242c4156e37e9017` |
| Dataset | `xiaowu0162/longmemeval-cleaned`, revision `98d7416c24c778c2fee6e6f3006e7a073259d48f` |
| Dataset file SHA256 | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`, 277,383,467 bytes |
| CUDA base image | `nvidia/cuda@sha256:da6791294b0b04d7e65d87b7451d6f2390b4d36225ab0701ee7dfec5769829f5`, CUDA 12.4.1 development image, Ubuntu 22.04 |
| Built Modal image | `im-uDGyOQ1G1mqwiWmccYHjrK` |
| Runtime | Python 3.11.5, PyTorch 2.5.1, Transformers 4.45.2, FAISS CPU 1.8.0.post1, Modal 1.5.5 |
| Observed GPU | NVIDIA A10, driver 580.95.05, 23,028 MiB reported memory |
| Requested allocation | One A10, 4 CPU cores, 16,384 MiB host memory |
| Dependency-lock SHA256 | `2b75036d9a89996e51c92a4e50224c3ec1b8ca3d8957b483fb20930751378030` |
| Test-run source SHA256 | `df3160d4c9d597ffe8bc0744e53c7b3322515b708a2601a7a86c5f72ab11f9a5` |
| Benchmark-run source SHA256 | `6ba88c7685cce546da69b4ba93c36ebc5433ced62197ecba87b190f4c4ad975e` |

The source digest covers the recorded 54 library and `tools/colbert*.py` source
files. It is not a Git commit or a digest of the entire checkout. Integration-test
files are outside that digest. The separate local-validation manifest records
the changed test and workflow file hashes. Unrelated node-pipeline files changed
locally after the cloud payload was captured, so these results must retain their
recorded source scope. The three files that differ between test and benchmark
payloads are `evaluation/node_pipeline.py`, `evaluation/node_protocol.py` and
`inference/nodes.py` under `src/llgm/`. They are outside this raw-retrieval path.
The recorded retrieval and worker code, frozen parameters, dependency lock and
Modal image agree between runs.

The reference settings use document length 180, query length 128, two-bit
residuals, two searched cells, centroid threshold 0.45, 1,024 candidate documents,
one rank/GPU, indexing batches of 64 and four k-means iterations. FAISS trains
centroids on CPU. ColBERT encoding and PLAID search use the GPU. This is the
official implementation with its real weights and native kernels.

The query-length setting also controls mask padding. Even a short query is
augmented to 128 positions, which can change MaxSim scores and work compared
with the upstream default of 32. The sanity query contains 16 tokens including
special and marker tokens before that padding. The experiment does not compare
the two query-length settings.

## Live integration and local client

Preparation artifacts are under `runs/colbert-modal/prepare-20260911-b/`.
The completed test artifacts are under `runs/colbert-modal/test-20260911-a/`.
They include `result.json`, `environment.json`, `junit.xml`, the integration
summary, build/reopen logs and results, saved canonical passages, baseline
ranking, and `client.json`.

| Check | Observed result |
| --- | --- |
| Pinned preparation | Complete checkpoint and dataset hashes matched |
| Genuine integration test | 1 passed, 0 skipped, 127.671 seconds in JUnit |
| Full history | Case `001be529`, 51 sources, 1,154 unique passages |
| Canonical span checks inside integration test | 80 passed across initial and reopened top-40 results |
| Durable worker build | New index, no reuse, top 40 distinct passages returned |
| Fresh-container reopen | Same ordered top-40 IDs, ranks and scores within relative/absolute tolerance `1e-6` |
| Local remote-retrieval client | Same ranking, scores and canonical evidence through real authenticated Modal RPC |
| Independent artifact audit | All 1,154 saved passage bodies matched their exact original source slices. The full local dataset hash and original query also matched |

The durable corpus fingerprint is
`ecdfe3b9342b2414fc4658468d0e9eca99451d8f7aa9fa9944b905972846053d`.
Its persisted index ID is
`25862d17164fdecfb0173673e427ad143521250dd606d744cff60c63796330dd`.
Builder container `ta-01M27N3A0DQQC9W3WSCE7J5P6R` and reopen container
`ta-01M27N8QVVMXBKXD8AG81PTW6R` differ. Reopen performs no index build.

| Durable-worker measurement | Value |
| --- | ---: |
| Adapter build | 30.256 seconds |
| Native index construction | 26.505 seconds |
| Index bytes | 6,045,581 |
| First query after build | 0.208 seconds |
| Fresh-container adapter open | 90.345 seconds |
| First search after fresh-container open | 1.521 seconds |
| Local client search elapsed | 95.982 seconds |
| Server search inside that client call | 1.432 seconds |
| Full test action, including client check | 403.903 seconds |

The roughly 96-second client call includes a cold container and its setup. It
does not measure warm serving latency. Cold opening includes model loading and
native extension preparation. Subtracting server search time does not isolate
network latency. The faster post-build query shares the builder's initialized
process and compiled extensions.

Top-40 source and labeled-turn recall are both 1/1 for this sanity question.
These are session/turn annotations. A matching passage can overlap an annotated
turn without containing its answer span, so this is not an answer-accuracy score.

The FAISS build log warns that 122,442 training points for 4,096 centroids fall
below its recommendation of 159,744 points. The build and searches completed.
This small-corpus warning limits interpretation of quantization quality and is
not evidence of a large-corpus speed or recall result.

## BM25 comparison

The three-case diagnostic comparison passed, with artifacts under
`runs/colbert-modal/benchmark-20260911-a/`. `benchmark.json` retains per-case
rankings, exact floating-point timing samples and configurations. `environment.json`
and `benchmark.log` record the actual process environment and native execution.
The complete action took 224.252 seconds, including 210.857 seconds in the worker
job. All three cases built new indexes and completed both arms without failure.

Both retrievers received the same original query and token-bounded passages
within each case and returned top 40. Source/turn coverage was identical:

| Case | Sources in history | Passages | Labeled sources covered, both arms | Labeled turns covered, both arms |
| --- | ---: | ---: | ---: | ---: |
| `001be529` | 51 | 1,154 | 1/1 | 1/1 |
| `01493427` | 47 | 1,142 | 2/2 | 2/2 |
| `00ca467f` | 47 | 1,141 | 3/3 | 2/2 |

Durations below are rounded from the retained raw samples. Build time measures
the complete adapter call for ColBERT and construction of a fresh SQLite index
for BM25. Index size is the worker's recorded persisted file footprint, not
an isolated compressed-vector size or total service storage.

| Case | Arm | Build seconds | Index bytes | First query ms | Warm median ms | Warm p95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `001be529` | BM25 | 0.183 | 2,461,696 | 8.478 | 6.990 | 7.584 |
| `001be529` | ColBERTv2 + PLAID | 112.101 | 6,045,576 | 255.419 | 18.832 | 21.125 |
| `01493427` | BM25 | 0.168 | 2,420,736 | 7.723 | 6.257 | 6.329 |
| `01493427` | ColBERTv2 + PLAID | 27.857 | 5,961,235 | 32.027 | 17.423 | 19.360 |
| `00ca467f` | BM25 | 0.168 | 2,469,888 | 7.700 | 6.299 | 6.394 |
| `00ca467f` | ColBERTv2 + PLAID | 27.491 | 5,990,208 | 24.509 | 18.166 | 20.242 |

Each warm summary has five retained repetitions. Nearest-rank p95 is therefore
the largest observed sample, not a stable tail-latency estimate. These are
server-side public retriever calls. They include validation and result conversion
but exclude preceding manifest verification, index opening and Modal dispatch.
Per-query remote latency was not measured by this worker comparison.

The first ColBERT build includes native extension compilation in a new container.
Later cases share that process and its compiled extensions. Native index-call
durations were 107.549, 25.373 and 25.026 seconds respectively, with the first
also including compilation. These measurements do not compare cold setup under
identical cache conditions across cases. First-query timings likewise refer to
the first query for each index, not a fresh service process for every case.

All three histories triggered FAISS's small-training-sample warning: 122,442,
120,022 and 121,010 points respectively, against its recommended 159,744 for
4,096 centroids. The official four-iteration builds completed. No checkpoint,
retrieval parameters or upstream implementation were changed to improve scores.

Both arms saturated the available labeled coverage on these exposed questions.
The measured warm median was about 6–7 ms for BM25 and 17–19 ms for ColBERTv2 +
PLAID. This run supplies no retrieval-quality gain from ColBERT at top 40 and
does not establish the performance ordering for harder queries or larger
collections. Lower cutoffs, additional independent data and exact-versus-PLAID
evaluation remain future experiments.

## Local validation and remaining boundaries

`runs/colbert-modal/local-validation-b/summary.json` records 776 passing tests,
230 passing subtests, two skips and 20 integration deselections. Ruff,
documentation publication-boundary checks, distribution construction, strict
Twine validation and source-archive content checks passed. This is a dated
local run, not a claim that every later parallel edit was tested by it.

Its deterministic coverage is 6,151/6,790 statements (90.59%) and 1,911/2,360
branches (80.97%), giving 88.11% combined coverage. Native ColBERT/CUDA execution
is outside that instrumentation. The real GPU test is established separately
by the retained cloud artifacts.

The real local client check supplied authenticated RPC functions from the
running Modal app to `ModalColBERTRetriever`. A persistent deployment and the
`ModalColBERTRetriever.connect` lookup path against that deployment were not run.
The manually triggered GitHub Actions workflow exists but has not been
dispatched. Credentials for that CI environment and an actual manual CI run
remain separate setup/validation steps.

The tested B/C design compares BM25 with ColBERTv2 plus PLAID. An exact or
uncompressed ColBERT reference arm is absent, so it does not isolate PLAID's
approximation or speedup. The diagnostic questions are exposed development
inputs. LongMemEval-S's shared-history structure does not supply 500 independent
held-out histories. Graph expansion, recursive retrieval, hosted generation,
final-answer accuracy and maintenance quality are outside this retrieval run.

Actual billed dollars were not measured. All retained cost fields remain null.
The workspace spend limit bounds authorized paid usage but is not a measured
cost or a per-job dollar admission mechanism. No statement in this report
estimates the actual invoice from model-call counts or wall time.

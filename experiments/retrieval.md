# E03 retrieval protocol

`e03_matrix.json` declares the required twelve B/D/H/C × S/U/A arms. B is SQLite FTS5 BM25. D is exact cosine over hosted `text-embedding-3-large` vectors. H fuses the top 40 B and D results with equal-weight RRF at constant 60. C is the official ColBERTv2/PLAID implementation. No fine-tuning, graph links, or semantic reconciliation is active.

S returns at most forty original-query hits. U uses the original question and
three queries generated before seeing evidence, with ten hits per query. A
generates follow-ups after inspection, with at most four searches and forty hit
slots. All arms use the same source representation, reader/model pair and
evidence limits. Each case searches only its own supplied haystack.

This guide covers data preparation and execution. Model IDs, local ColBERT assets, checkpoint checksum, and repository revision remain explicit unresolved pins in the full-matrix template. No dependency or checkpoint downloads happen on import. Optional adapters fail with actionable capability errors rather than changing retrieval methods. The completed B/D/H subset is documented in the [pilot protocol](NON_COLBERT_PILOT.md).

These commands use the installed checkout and produce a new measurement. They do
not reproduce the historical runner or migrate its prepared artifacts. For the
September 10 pilot, use its [archived-source procedure](NON_COLBERT_PILOT.md#replicating-the-historical-workload).

Current `prepare` writes `schema_version=2` and
`evidence_schema=immutable-node-v2`. References identify an immutable node and
turn-relative span. Preflight rejects old manifests and versioned source
references. Use a new preparation directory for the commands below if an earlier
output already exists. Keep historical corpora and results.

The scientific reference remains the actual released Stanford ColBERTv2 checkpoint and official PLAID implementation. Hosting that same implementation remotely is a deployment choice. Jina/AnswerAI models, other index engines, or hosted candidate rerankers are different experimental arms and cannot replace C under its existing name. A remote reference service must preserve corpus isolation and provide reproducible model/index configuration plus timing and usage records.

The separate [Modal workflow](colbert.md) implements authenticated remote retrieval
through `ModalColBERTRetriever`. E03's C arm still constructs a local
`ColBERTRetriever`. Connecting this matrix to the remote adapter requires
additional runner integration.

The twelve-arm matrix holds one model pair fixed to isolate retrieval. The current runner accepts one model pair. Broader model-allocation orchestration remains unimplemented. The [implementation status](../docs/reference/implementation-status.md) records the available mechanisms.

## Offline plumbing

```bash
llgm offline-smoke --output runs/offline-smoke
```

This creates six isolated synthetic histories, selects five development cases, and runs B-S with deterministic local readers. It produces manifests, predictions, traces, physical-call accounting, judgments, metrics, and a decision note. Its normalized exact-match score is a fixture check. It is not model accuracy or a benchmark result. The fake reader is restricted to these dedicated fixtures.

## Real dataset preparation and the isolation finding

The cleaned LongMemEval-S file verified during implementation contains 500 questions and all 500 belong to one connected history component under the declared rule: cases sharing any source session ID or identical session text stay together. Shared distractors also count. Therefore a 50-question development split with untouched benchmark history groups is unavailable under this rule. Preparation writes an audit and returns `blocked_history_isolation`. It does not silently select the entire benchmark for development.

The checked dataset is the official `xiaowu0162/longmemeval-cleaned` release at revision `98d7416c24c778c2fee6e6f3006e7a073259d48f`, file `longmemeval_s_cleaned.json`, SHA256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` (277,383,467 bytes). Data and generated runs are not bundled with the package. An explicit optional download command is available:

```bash
llgm experiment download --output data/longmemeval/longmemeval_s_cleaned.json \
  --dataset-revision 98d7416c24c778c2fee6e6f3006e7a073259d48f \
  --sha256 d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442

llgm experiment prepare --dataset data/longmemeval/longmemeval_s_cleaned.json \
  --dataset-revision 98d7416c24c778c2fee6e6f3006e7a073259d48f \
  --output data/prepared/longmemeval-split-audit
```

The default word tokenizer is explicitly diagnostic. Substantive shared passages
require the pinned local tokenizer prepared below. It loads with local-files-only
and records a content hash. Every rendered passage, including session/date/role
metadata plus encoder special tokens and marker, fits 180 tokens. Overlap is 32
source tokens when enough source content fits. Passages never cross a session or
turn boundary. Turn-relative Unicode offsets remain canonical when passage
windows change. Original questions and generated follow-ups are checked against
the pinned query encoder limit. Overflow fails instead of truncating silently.

Some released cases repeat a session ID, sometimes with identical text at different dates. Each occurrence is preserved in a distinct source node. `benchmark_session_id` records its original identity. Evaluator-only alias maps collapse session recall onto the original benchmark ID while turn labels retain exact occurrence/turn identity. No copies are silently dropped. Only role, content, dates, and ordinary source identity reach model inputs. Answers, `has_answer`, and required-evidence labels remain in `evaluator/gold.jsonl`.

### Prepare the pinned tokenizer

[source_pins.json](source_pins.json) owns the dataset and tokenizer source
identities. The following explicit download obtains only its five tokenizer and
configuration files. It verifies each length and SHA256 before writing, and
checks existing files instead of replacing them. It downloads no model weights.

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

pins = json.loads(Path("experiments/source_pins.json").read_text())
tokenizer = pins["colbert_tokenizer"]
destination = Path("data/models/colbertv2-tokenizer")
destination.mkdir(parents=True, exist_ok=True)
base = (
    "https://huggingface.co/" + tokenizer["repository"]
    + "/resolve/" + tokenizer["revision"] + "/"
)
for name, expected in tokenizer["files"].items():
    path = destination / name
    if path.exists():
        content = path.read_bytes()
    else:
        with urlopen(base + name, timeout=60) as response:
            content = response.read()
    if len(content) != expected["bytes"]:
        raise SystemExit(f"Unexpected file length: {name}")
    if hashlib.sha256(content).hexdigest() != expected["sha256"]:
        raise SystemExit(f"Checksum mismatch: {name}")
    if not path.exists():
        path.write_bytes(content)
print("Verified pinned tokenizer files:", destination)
PY
```

The preparation commands below use its model-repository revision
`c1e84128e85ef755c096a95bdb06b47793b13acf`. This is separate from the official
ColBERT implementation revision required for indexing and search. The original
query-length audit found 61 of 500 questions exceeding 32 ColBERT tokens, with a
maximum of 69. The template uses `query_maxlen=128` to preserve those inputs.
Generated queries must also fit that limit or fail visibly. This was an
input-compatibility choice made before answer-quality tuning.

## Separate development and evaluation

The experiment design's fallback is to tune on separate controlled histories, then freeze before LongMemEval evaluation. Generate a small template workload with lookup, updates, temporal questions, scoped suggestions, cross-session dependencies, and abstention:

```bash
llgm experiment controlled-fixtures --output data/controlled-development.json --count 60
llgm experiment prepare --dataset data/controlled-development.json \
  --dataset-kind controlled-development --dataset-revision controlled-v1 \
  --tokenizer-directory data/models/colbertv2-tokenizer \
  --tokenizer-revision c1e84128e85ef755c096a95bdb06b47793b13acf \
  --output data/prepared/controlled
```

With the defaults, the balanced seed 1729 selection yields 50 development questions, five smoke cases drawn from them, and ten reserved controlled histories. These simple templates establish behavior and plumbing. Broaden the workload before treating development results as research evidence.

Materialize an evaluation workload explicitly, limiting the number of cases if
needed. The five-case selection below was already exposed by the completed pilot.
Reusing it is diagnostic replication.

```bash
llgm experiment prepare --dataset data/longmemeval/longmemeval_s_cleaned.json \
  --dataset-revision 98d7416c24c778c2fee6e6f3006e7a073259d48f \
  --selection evaluation --case-limit 5 \
  --tokenizer-directory data/models/colbertv2-tokenizer \
  --tokenizer-revision c1e84128e85ef755c096a95bdb06b47793b13acf \
  --output data/prepared/longmemeval-evaluation
```

This writes an evaluation selection and creates no benchmark development set.
Omit `--case-limit` to prepare all 500 cases. The runner requires
`stage="held-out"`, `configuration_frozen_at`, and the separate
`development_dataset_sha256` for an evaluation run. These manifest fields record
the declared protocol. They do not establish that cases remain unseen. The
completed pilot exposed five cases, and the other 495 are not independent held-out
histories under the declared overlap rule. Record prior exposure in every later
study that uses this release.

## Preflight and live dispatch

Create a run-specific copy of the template with independent tokenizer provenance:

```bash
python - <<'PY'
import json
from pathlib import Path

matrix = json.loads(Path("experiments/e03_matrix.json").read_text())
pins = json.loads(Path("experiments/source_pins.json").read_text())
matrix["tokenizer"] = {
    "local_path": str(Path("data/models/colbertv2-tokenizer").resolve()),
    "revision": pins["colbert_tokenizer"]["revision"],
}
destination = Path("runs/e03-new/matrix.json")
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("x") as output:
    json.dump(matrix, output, indent=2)
    output.write("\n")
PY
```

Fill in `runs/e03-new/matrix.json` with exact root/sidecar model IDs, provider
endpoints if needed, the local ColBERT checkpoint path and directory-content
SHA256, the full official implementation repository revision, compression,
encoder limits, and hardware/search settings. Keep the standalone `tokenizer`
section. Do not put the implementation revision in its `revision` field. The
adapter verifies a clean official Git checkout or PEP610 VCS installation. The
directory checksum can be computed with
`llgm.retrieval.colbert.checkpoint_sha256(path)` after explicitly obtaining the
released checkpoint. The full C arm remains unverified. Complete its prerequisites
before dispatching a full matrix.

```bash
llgm experiment preflight --prepared data/prepared/controlled \
  --matrix runs/e03-new/matrix.json --output runs/e03-new/preflight.json

llgm experiment run --prepared data/prepared/controlled \
  --matrix runs/e03-new/matrix.json --split smoke \
  --output runs/e03-new/answering --execute
```

Full preflight checks all 12 arms, even when dispatching a subset using `--arms B-S`. C is required. No lexical fallback makes an incomplete matrix appear ready. `--diagnostic` is exclusively the local B-S plumbing path. `--execute` marks the live provider dispatch boundary. Preparation/preflight do not invoke models. The runner isolates every case and arm and uses cold indexes for each. Build costs and embedding calls are recorded, including the work behind H. Generation and embedding request limits, case-arm limits, context/evidence budgets, and deadlines are explicit. The runner does not claim to enforce a strict currency cap. `currency_cost` remains unavailable unless computed separately from recorded usage and a pinned price table. Local ColBERT indexing may finish an in-flight operation before the next deadline check.

Outputs live under `<run>/<arm>/`. Raw retrieval IDs/scores/references, original and generated queries, provider-reported usage, failures, and final cited references are auditable. Experiment traces explicitly retain query text. Ordinary retrieval adapter events retain only query hashes. Treat run artifacts as containing the source workload's potentially sensitive content.

Metrics distinguish first retrieval, final retrieval, admitted evidence, and the final evidence bundle. Report session recall, answer-turn hit recall, all-required-session coverage, and all-answer-turn-hit coverage with explicit non-abstention denominators. A hit anywhere in an annotated answer turn does not prove recall of its exact answer-bearing span. Failed cases remain in evaluation denominators. Unsupported claims and semantic correctness still require judging.

For the separate original-question top 40 retrieval diagnostic:

```bash
llgm experiment preflight --prepared data/prepared/controlled \
  --matrix runs/e03-new/matrix.json --retrieval-only
llgm experiment retrieve --prepared data/prepared/controlled \
  --matrix runs/e03-new/matrix.json --output runs/e03-new/retrieval --execute
```

This checks all four backends and issues no generation calls. D/H still invoke hosted embeddings. C still requires its real local encoder and PLAID stack. Predictions contain ranked source references, with source/turn recall, complete-label coverage, and index/query costs scored separately. It is not answer-quality evaluation. Missing C assets keep substantive retrieval preflight blocked.

## Official scoring

```bash
llgm experiment export-official --predictions runs/e03-new/answering/B-S/predictions.jsonl \
  --output runs/e03-new/answering/B-S/official-input.jsonl
```

This produces the released scorer's `question_id`/`hypothesis` schema, retaining failed cases with their recorded prediction. It never launches a paid judge. Invoke a separately checked-out, revision-pinned official `evaluate_qa.py` with the chosen judge model, this export, and the original dataset. Record judge/rubric pins and outputs before reporting official scores. Local normalized exact match is always labeled diagnostic.

References: [LongMemEval data and scorer](https://github.com/xiaowu0162/LongMemEval), [official ColBERT and released checkpoint](https://github.com/stanford-futuredata/ColBERT).

The C arm targets the official PLAID Indexer and Searcher. The complete twelve-arm
E03 matrix remains unverified. Separate Modal integration checks have passed real
index building, search, persisted reopening, and retrieval through the local client.

## Explicit non-ColBERT scope

The [five-case pilot](NON_COLBERT_PILOT.md) freezes the B/D/H × S/U/A subset in `e03_non_colbert_pilot.json`. Pass `--required-backends B D H` to preflight and execution. C is unchecked and cannot be dispatched outside the declared scope. The full matrix remains incomplete. A standalone `tokenizer` section holds the local passage tokenizer path/revision without using ColBERT encoder pins.

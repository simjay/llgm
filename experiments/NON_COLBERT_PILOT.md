# Non-ColBERT diagnostic pilot

The configuration was frozen at 2026-09-10T07:07:42Z in
`e03_non_colbert_pilot.json`, before the pilot's retrieval or answering calls.
All 15 retrieval diagnostics and 45 answering attempts completed. The
[dated validation report](../research/reports/live-validation-2026-09-10.md#frozen-bdh-pilot)
records the results. This was a five-case diagnostic, not a confirmatory experiment
or a benchmark superiority result.

The five LongMemEval-S cases were `3fdac837`, `86b68151`, `e61a7584`, `720133ac`,
and `gpt4_4ef30696`. Their prepared, isolated question histories contained 5,706
passages in total. These cases are now exposed. Reusing them is diagnostic
replication. The separate controlled development dataset verified pipeline
behavior. All 500 original LongMemEval cases belong to one connected
history-overlap component. The other 495 cases are not an independent held-out set.

The first run made one original-question top-40 retrieval per case with B (SQLite BM25), D (hosted `text-embedding-3-large`, 3,072 dimensions, exact cosine), and H (equal-weight reciprocal rank fusion, constant 60, top 40 from each component). It recorded source and answer-turn coverage, all-required coverage, indexing/query latency, and embedding requests/tokens. Gold labels were applied only after retrieval.

The second run evaluated the nine B/D/H × S/U/A combinations on the same five
cases. S used one original query. U allowed three upfront reformulations after
the original query. A allowed up to three adaptive follow-ups. Root generation
used `gpt-6-astra`, and the sidecar used `gpt-5.6-terra`. Each case-arm allowed at
most four searches, eight sidecar calls, nine total model calls, 8,000 exposed
evidence tokens, a 4,000-token bundle, a 16,000-token context and 1,024 output tokens
per call. This tested the iterative retrieval controller.

Both runs used cold per-case/per-arm indexes. Indexing work and costs were
included repeatedly. This did not measure a production embedding cache. Shared
passage windows were 180 tokens with 32 overlap, using the pinned tokenizer. No
ColBERT model, PLAID index, weights, GPU workload, or C arm was executed.
Standalone tokenizer pins were separate from the unconfigured ColBERT encoder pins.

Each runner had independent overall limits of 45 case-arm attempts, 405 generation calls, 2,000 embedding requests and 3,600 seconds. Each retrieval case allowed at most 100 embedding requests. These were call/time caps, not an enforced dollar limit. Provider failures and incomplete answers remained in their original denominators. There were no application retries or post-result threshold changes.

Answer hypotheses were exported separately for the unmodified official LongMemEval evaluator at commit `9e0b455f4ef0e2ab8f2e582289761153549043fc`, script SHA256 `ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251`, alias `gpt-4o` resolving to `gpt-4o-2024-08-06`. All judgments were reported without a minimum quality assertion or selecting a winner. The official script owns retries and does not report complete physical request/token/cost accounting.

## Replicating the historical workload

Historical reproduction requires the retained
`runs/e03-non-colbert-20260910/source/llgm/` archive and its startup manifest at
`runs/e03-non-colbert-20260910/answering/B-S/manifest.json`. These ignored artifacts
are not included in a fresh checkout. Obtain the retained archive before following
this procedure. The current package is being refactored and is not a substitute
for those historical modules or their prepared-data schema.

Use a separate Python 3.11 environment. The startup manifest records Python
3.11.13, OpenAI SDK 2.54.0 and Transformers 4.57.6. Its package list is not a
complete dependency lockfile. Record the replication environment and returned
provider model identities. Hosted outputs and timings need not repeat.

Run from the repository root. Verify the source archive before using it:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

archive = Path("runs/e03-non-colbert-20260910")
manifest = json.loads((archive / "answering/B-S/manifest.json").read_text())
for name, expected in manifest["code"]["source_sha256"].items():
    path = archive / "source/llgm" / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Source checksum mismatch: {name}")
print("Verified", len(manifest["code"]["source_sha256"]), "historical modules")
PY

historical_llgm() {
  PYTHONPATH="$PWD/runs/e03-non-colbert-20260910/source" python -m llgm.cli "$@"
}
```

Obtain the dataset using the [pinned download command](retrieval.md#real-dataset-preparation-and-the-isolation-finding)
and obtain the five tokenizer files using the [checksum-verified tokenizer procedure](retrieval.md#prepare-the-pinned-tokenizer).
Those downloads use the identities in [source_pins.json](source_pins.json).
Prepare new artifacts with the archived CLI, retaining the historical seed and
tokenizer revision. Use a new prepared-output directory and a new controlled
fixture file:

```bash
historical_llgm experiment controlled-fixtures \
  --output data/controlled-development-pilot-replication.json --count 60
historical_llgm experiment prepare \
  --dataset data/longmemeval/longmemeval_s_cleaned.json \
  --dataset-revision 98d7416c24c778c2fee6e6f3006e7a073259d48f \
  --selection evaluation --case-limit 5 --seed 1729 \
  --tokenizer-directory data/models/colbertv2-tokenizer \
  --tokenizer-revision c1e84128e85ef755c096a95bdb06b47793b13acf \
  --output data/prepared/e03-pilot-replication
```

Verify the prepared inputs and create a portable configuration copy. The original
frozen JSON contains the original workstation's tokenizer path. Change that path
only in the new copy. Its freeze timestamp continues to identify the original
protocol, while the new run manifest records the replication date.

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

pins = json.loads(Path("experiments/source_pins.json").read_text())
matrix = json.loads(Path("experiments/e03_non_colbert_pilot.json").read_text())
prepared = Path("data/prepared/e03-pilot-replication")
manifest = json.loads((prepared / "prepared.json").read_text())
development = Path("data/controlled-development-pilot-replication.json")
checks = {
    "dataset": manifest["dataset"]["sha256"] == pins["longmemeval_s"]["sha256"],
    "tokenizer": manifest["tokenizer"]["sha256"] == pins["tokenizer_digest"],
    "cases": set(manifest["selection"]["evaluation_ids"])
        == set(matrix["pilot_scope"]["evaluation_ids"]),
    "development": hashlib.sha256(development.read_bytes()).hexdigest()
        == matrix["development_dataset_sha256"],
}
if not all(checks.values()):
    raise SystemExit(f"Historical input mismatch: {checks}")
matrix["tokenizer"]["local_path"] = str(Path("data/models/colbertv2-tokenizer").resolve())
destination = Path("runs/e03-pilot-replication/matrix.json")
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("x") as output:
    json.dump(matrix, output, indent=2)
    output.write("\n")
print("Verified historical inputs and wrote", destination)
PY
```

The following dispatch commands make real hosted requests. Each output directory
must be new, and the named provider credential environment variables must be set.

```bash
historical_llgm experiment preflight --prepared data/prepared/e03-pilot-replication \
  --matrix runs/e03-pilot-replication/matrix.json --required-backends B D H
historical_llgm experiment retrieve --prepared data/prepared/e03-pilot-replication \
  --matrix runs/e03-pilot-replication/matrix.json --required-backends B D H \
  --split evaluation --output runs/e03-pilot-replication/retrieval --execute
historical_llgm experiment run --prepared data/prepared/e03-pilot-replication \
  --matrix runs/e03-pilot-replication/matrix.json --required-backends B D H \
  --split evaluation --output runs/e03-pilot-replication/answering --execute
```

The narrower scope is explicit. Default preflight still requires all four
backends. A successful replication never marks the full twelve-arm matrix ready
or complete. Export and judge the new hypotheses separately using the
[official scoring procedure](retrieval.md#official-scoring), with
`historical_llgm` and the new prediction/output paths. Preserve the original
results and disclose the exposed cohort in the replication report.

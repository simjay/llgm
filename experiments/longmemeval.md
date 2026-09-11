# LongMemEval: the primary answer evaluation

This is the current evaluation workflow. It answers a concrete question:
**Does LLGM answer long-term memory questions better than credible alternatives,
and what does that improvement cost?**

Start with the [five-question LLGM-only pilot](longmemeval_pilot_v6.json), covering
single-session recall, multi-session recall, knowledge updates, temporal reasoning
and abstention.
Selection takes the first released question in each category, without choosing by
generated answers. The pilot has a $5 generation allowance and $0.10 judge allowance.
It rebuilds all 237 source-session occurrences for these five exposed development
questions. They are not a benchmark accuracy estimate. The CLI and Make commands
below default to v6. Scaling to all 500 questions is a separate explicit decision.

The retained [v1 comparison](longmemeval_pilot_v1.json) runs ordinary LLGM,
BM25 plus a reader, and full-context reading when explicitly selected. Mem0 OSS
and Graphiti remain required competitors before claiming the roadmap's framework
win. They are not aliases for these
controls. Their extraction, embedding and reranking calls need complete accounting
before they can join the paid comparison.

## What runs

The default v6 protocol selects only the LLGM column. The v1 comparison selects
all three arms:

| Stage | LLGM | BM25 reader | Full-context reader |
| --- | --- | --- | --- |
| Preparation | Ingest every supplied session and attempt automatic primary-edge maintenance | Ingest the same complete history without model maintenance | Ingest the same complete history without model maintenance |
| Evidence access | Ordinary initial search, three admitted seeds at most, recursive Python delegates in Docker | One local BM25 search, up to 40 passages | Every source turn, or an explicit input-budget failure |
| Answer | One final root combines returned findings | One reader call | One reader call |
| Scoring | Same pinned LongMemEval task prompts and judge | Same | Same |

V6 uses `gpt-5.4-2026-03-05` with explicit `medium` reasoning for the root and
omits its sampling temperature. Delegates and maintenance use
`gpt-4.1-2025-04-14` at temperature zero. Judging uses
`gpt-4o-2024-08-06` through Chat Completions.

V1 uses `gpt-4.1-2025-04-14` for every generation role, including the final reader
in all three arms. Its reader matches across arms. Comparing those controls with
v6 changes the root model and reasoning settings, so the comparison is unmatched
and cannot isolate an architecture advantage.

Each case and arm gets a fresh workspace containing only its supplied history.
Shared distractors do not authorize merging histories across questions. Repeated
session IDs retain distinct occurrences. Every model sees opaque source handles,
original role/text, source date text and the question's date at query time.
Original benchmark IDs, answer flags and gold answers stay outside generation.
Preparation never receives the evaluation question or gold. No manually chosen
seeds, edges or journal patches enter this comparison. Automatic journal amendment
generation is not implemented, so its contribution is not measured here.

Sources remain in their supplied order. Each source's complete role-labeled
conversation is ingested once per case and arm. Automatic relationships can be
wrong, and their failures are recorded. Successful ingestion is not rolled back
when maintenance fails. Construction is charged to the one question actually
evaluated against that workspace. There is no assumed future-query amortization.

## Prerequisites

Install the package and benchmark dependencies in the existing environment:

```bash
uv pip install --python .venv/bin/python -e '.[benchmark]'
```

Obtain the cleaned dataset and evaluator revision named in
[the pilot protocol](longmemeval_pilot_v6.json). The runner verifies their exact SHA-256 hashes.
It never downloads the dataset or evaluator. Tokenizer initialization can fill
tiktoken's verified local asset cache before model dispatch. The dataset release is also pinned in
[source pins](source_pins.json). Use the
[official dataset instructions](https://github.com/xiaowu0162/LongMemEval) to
obtain the release. The required local paths are explicit in the protocol.

Start Docker with the already available trusted image named in the protocol.
The runtime does not pull images. Native model credentials belong in exported
environment variables or an explicitly supplied, ignored local environment file.
No credential values belong in a protocol or retained artifact.

## Commands

Validate the five-question plan without model calls:

```bash
make benchmark-prepare BENCHMARK_OUTPUT=runs/longmemeval-preflight
```

Run the five selected LongMemEval questions with the default LLGM-only protocol:

```bash
make benchmark-longmemeval BENCHMARK_OUTPUT=runs/longmemeval-pilot-v6 ENV_FILE=.env
```

Or select the same v6 protocol explicitly:

```bash
.venv/bin/python -m llgm.evaluation.memory_benchmark \
  --protocol experiments/longmemeval_pilot_v6.json \
  --output runs/longmemeval-pilot-v6 --env-file .env --execute
```

Every output directory must be new. Existing runs are never overwritten or
silently retried. Removing `--execute` validates and prepares a schedule only.
Use a new, explicitly labeled protocol for a changed model, cohort or limit.

Rebuild a report from retained records without making calls:

```bash
.venv/bin/python -m llgm.evaluation.memory_benchmark \
  --output runs/longmemeval-pilot-v6 --report
```

### Retained comparison and development variants

Run the initial three-arm pilot with its matched GPT-4.1 readers by selecting v1:

```bash
.venv/bin/python -m llgm.evaluation.memory_benchmark \
  --protocol experiments/longmemeval_pilot_v1.json \
  --output runs/longmemeval-comparison-v1 --env-file .env --execute
```

The [v5 variant](longmemeval_pilot_v5.json) retains the earlier GPT-5.4 root at
its default `none` reasoning and temperature zero. V1, v5 and v6 use exactly the
same five exposed questions. Keep their configurations and retained results
separate.

The [ten-question smoke protocol](longmemeval_smoke_v3.json) remains an optional
integration check with its own pinned models. Its independently authored
questions are not LongMemEval data, and it does not qualify a different model:

```bash
.venv/bin/python -m llgm.evaluation.memory_benchmark \
  --protocol experiments/longmemeval_smoke_v3.json \
  --output runs/longmemeval-smoke --env-file .env --execute
```

Root-only replays reuse retained branch evidence to diagnose synthesis behavior.
They do not repeat ingestion, retrieval, delegate inspection or evidence selection,
and cannot replace an end-to-end run or establish its accuracy or lifecycle cost.

### Final-reader reasoning

V6 sets the top-level field `"root_reasoning_effort": "medium"`.
It configures the native OpenAI final
reader in every selected arm, including the LLGM root, BM25 reader, and
full-context reader. Delegates and maintenance keep their own settings. This
changes the existing synthesis call and adds no agent, verifier, or extra call.

For an explicit effort other than `"none"`, the runner omits sampling temperature
from final-reader requests. Omitting the field preserves the provider's default
effort and the existing temperature-zero requests. The configured output limit
includes both reasoning and visible answer tokens. Model compatibility must be
qualified, and a changed effort must be retained in a new protocol and output
directory. V1 and v5 leave effort at the provider default.

## Reading the result

Start with `results.md`, then inspect `summary.json`. The report includes every
scheduled question in each arm's denominator. It distinguishes an incorrect
answer, an operational failure, an unstarted attempt and a missing judge result.
Missing judgments leave accuracy incomplete and produce lower/upper bounds.
Failed or empty answers receive an explicit operational-failure score of zero.
An explanatory abstention is a real answer and goes to the judge.

`predictions-ARM.jsonl` uses the official `question_id` and `hypothesis` fields.
Unstarted questions are exported with empty hypotheses. The schedule and trial
records retain their status, so this does not disguise them as model responses.
There is no cherry-picked successful subset or best-of answer selection.

Category summaries put abstention questions in their own group. Other questions
retain their released question types. On the full release, that means 30
abstention questions and 470 others. This grouping differs from a table that
leaves abstention questions inside those types, so compare category percentages
only when their denominators match. Accuracy uses every scheduled question per
arm, five in the pilot and 500 in the full protocol.

Judging uses the checksum-pinned official prompt function and its exact question,
gold-answer and hypothesis inputs. The configured Chat Completions request uses
temperature zero and ten output tokens. A strict yes/no parser treats malformed
responses as missing judgments. The upstream substring-based verdict is also
recorded for audit. This is an instrumented official-prompt evaluation, not an
unmodified invocation of the authors' script. Their script has internal retries
and does not expose complete request accounting. Scores using other judges,
models, subsets or parsers must be labeled separately.

Citation integrity is a distinct contract from semantic correctness. LLGM's
selected quotes are checked against exact immutable source spans. A structurally
valid quote can still be irrelevant or misinterpreted. Judge accuracy does not
certify evidence support or the truth of generated edges.

## Cost and failure accounting

The per-call recorder writes a request before dispatch and retains its response
or sanitized error type. It records actual token counts, cached input and the
dated published price basis. SDK retries are disabled. A failed call can still
cost money. Missing usage remains unknown and consumes its reserved allowance.
Local reservations include concurrent in-flight requests. They are conservative
admission estimates, not provider invoice guarantees.

The five-question pilot runs two cases concurrently under $5 generation and
$0.10 judging allowances. V6 runs LLGM only. In the v1 three-arm comparison,
arm order rotates between cases and remains serial within each case.
Every LLGM answer retains its own three-delegate concurrency
limit. Judging has a separate time window after generation, so a long construction
phase cannot consume the judge's entire deadline. The retained full protocol
specifies four concurrent cases and $485 generation plus $5 judging allowances.
It is a separate configuration and is not the default run.

Generation roles share a sliding 60-second admission window of 400,000 estimated
tokens. A provider qualification request reported a 450,000-token limit for the
selected GPT-4.1 snapshot. The local estimate tokenizes the actual messages and
schema with `cl100k_base`, adds 10 percent and 256 framing tokens, then reserves
the full output allowance. Judging has its own window. These estimates reduce
rate-limit collisions but do not guarantee provider admission. Other account
activity can consume capacity.

Queue delay remains in wall-clock latency and existing runtime deadlines.
Each request records that delay and whether it actually reached the provider.
Cancellation while waiting costs zero and releases its monetary reservation.
An already dispatched request with missing usage keeps its complete reservation.
The pilot stops an allowance after 16 unknown-usage calls, or earlier
when its cost or time admission limit is reached. No rate-limit response triggers
a retry. The summary distinguishes recorded requests from physical model calls.
Summed per-request queue delays can overlap and must not be subtracted from
query wall time to claim an unpaced latency.

Construction and query costs are separated. Their sum is divided by the actual
scheduled question count, with incomplete execution explicitly marked. Judge
cost is separate evaluation expenditure. Local process CPU time, workspace bytes
and wall time are measured. They are not priced. Process CPU time during a trial
includes concurrent cases and excludes Docker's separate processes, so it cannot
be attributed to that arm. The report therefore labels API cost separately
and leaves total cost unknown. Do not claim the equal-total-cost goal yet.

`protocol.json`, `schedule.json`, copied imported package sources and source hashes preserve
the attempted configuration. Each trial retains construction outcomes, automatic
edges, runtime traces, selected evidence, provider attempts and durations. Judge
records live separately from generation. Interruptions retain individual records.
The offline report can reconstruct incomplete results without resubmitting work.

LongMemEval-S has overlapping histories, and earlier diagnostics exposed some
questions. A full result is a frozen public-benchmark measurement. It is not an
untouched independent holdout, and 500 question outcomes are not 500 independent
history samples. Preserve negative results and keep development changes separate
from the frozen comparison.

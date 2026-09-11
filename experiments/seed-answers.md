# Final answers from frozen seed choices

This completed historical diagnostic retains its original protocol and run
commands. The primary evaluation now uses [LongMemEval](longmemeval.md).
Later runtime repairs and the five-question pilot are separate from these frozen
attempts.

This diagnostic asks whether different initial node choices improve actual
answers after recursive evidence gathering. It follows the
[node-selection experiment](node-selection.md), whose source-coverage scores
did not establish answer correctness. Product retrieval and selection defaults
remain unchanged. [seed_answers_v1.json](seed_answers_v1.json) freezes the input
hashes, cohort, model snapshots, runtime limits, order, judging, and price basis.

## Inputs and comparison

The protocol requires the prior selection run's exact `prepared.json` and
`node-selection.json`, its original protocol, and the checksum-pinned official
LongMemEval evaluator source. Selection preparation also verifies the original
LongMemEval dataset, retained real BM25/ColBERT results, and source manifests.
These ignored local data and run artifacts are prerequisites. They are not
packaged library assets. Restore their exact bytes before running preparation.
Changed artifacts require a separately frozen protocol.

| Initial candidate pool | Initial node choices |
| --- | --- |
| `bm25_40`: retained BM25 top 40 | First three distinct owners and frozen hosted selector |
| `colbert_40`: retained ColBERTv2 + PLAID top 40 | First three distinct owners and frozen hosted selector |
| `rrf_40`: retained reciprocal-rank fusion, truncated to 40 | First three distinct owners and frozen hosted selector |

Each model-selector arm uses **repetition 0** from the completed selection run,
chosen before answer outcomes are available. Selection does not run again.
Preparation reconstructs its candidate pool and reparses the original response
to verify the saved IDs. Both policies use all saved spans belonging to their
selected nodes.

The comparison includes all 24 prior LongMemEval questions and all eight
controlled histories: 32 histories × three pools × two policies × one answer
attempt = **192 trials**. The two benchmark questions with four annotated
sources remain included. Annotated source count does not determine the minimum
evidence needed for a correct answer. Benchmark and controlled scores remain
separate.

Six arm combinations rotate by case index. Two cases can run concurrently, and
each case runs its six arms sequentially. Identical seed sets still receive
independent answer attempts. No best-of choice, implicit answer retry, or
outcome-dependent prompt adjustment occurs. This version evaluates the existing
truncated fusion pool. It does not introduce a new union-preserving selector.

The root is `gpt-4.1-2025-04-14`, delegates are
`gpt-4.1-mini-2025-04-14`, and both judges use `gpt-4o-2024-08-06`.
All calls use temperature zero. Each answer has the same 13-call total limit,
12-sidecar-call limit, eight-search limit, 180-second deadline, recursion depth
three, eight steps per frame, 96 shared operations, and concurrency three.
The per-call output limit is 1,536 provider tokens. Evidence and context have
65,536 admission units each and the returned bundle has 16,000. The current
runtime's conservative admission counter counts UTF-8 bytes, not provider tokens.

## What executes

The experiment supplies frozen initial node choices to the ordinary
`LLGM.answer` pipeline. Every selected node receives all of its saved canonical
passage references, in retrieval-rank and reference order. Exact repeated spans
deduplicate. Overlapping spans remain separate. Each trial admits at most three
nodes. An empty selection remains empty and produces the runtime's ordinary
partial response without a hidden fallback search.

The helper replaces only initial seed admission. It charges one search against
the shared answer budget and records `source="frozen_replay"` in the trace.
Initial retrieval uses the retained results of actual BM25 and official
ColBERTv2/PLAID execution. It is not rerun or timed here. Selected IDs are visible
to the runtime. Pool identities, selector explanations, evaluator labels, and
unselected candidate records are excluded. This replay does not report discarded
candidate owners as failed or skipped branches. Consequently the initial trace
is an experimental admission record, not a byte-identical production search
trace.

After admission, smaller-model delegates run actual Python in separate Docker
interpreters. They inspect sources, perform follow-up searches, and recursively
query other nodes through the existing callbacks. The root model synthesizes
their returned findings. All arms use the ordinary local BM25 index for these
follow-up searches, including trials whose initial seeds came from ColBERT.
This holds the continuation policy fixed while varying initial choices. It is
not an end-to-end comparison of alternative recursive search backends.

Each trial sees its full case history, with opaque node IDs and original source
text, dates, roles, and timestamps. Gold answers and original benchmark aliases
remain outside the workspace. The helper validates an exact source-only
workspace with no journal edits or primary edges. It performs no learned graph
maintenance. Models can discover other sources through global search despite
the absence of prepared edges.

## Prepare, execute, and summarize

Use the repository environment with the OpenAI extra installed. Preparation
requires the exact local artifacts and the SDK for environment metadata. It
makes no hosted model calls and does not start Docker or GPU jobs. Run from the
repository root with a new output directory:

```bash
.venv/bin/python -m tools.seed_answer_experiment \
  --output runs/seed-answers/PREPARE_UNIQUE
```

Preparation saves `prepared.json`, evaluator-only `scoring.json`, `protocol.json`,
`environment.json`, and an `executed-source/` snapshot with source hashes.
These are local experiment artifacts. Source text and labels must remain
separate from the fields explicitly sent in model requests.

Execution requires `OPENAI_API_KEY`, a running local Docker daemon, and the
already-installed image pinned in the protocol:
`sha256:ec7d6c95cd3692a2e2d228a8b1ca74e4025b54121fcc4c5da6f09cfa473315ad`.
The experiment uses real isolated Python sessions and does not pull an image.
Supply the key through an exported variable or an ignored environment file.
Existing process values take precedence over file values.

```bash
.venv/bin/python -m tools.seed_answer_experiment \
  --env-file .env --execute \
  --output runs/seed-answers/EXECUTE_UNIQUE
```

Omit `--env-file` when using an exported key. Execution repeats preparation and
requires another new directory. It does not resume or overwrite a prior run.
The protocol's input hashes are verified before any provider call.

Each case imports its full history and prebuilds the common follow-up BM25 index
once, before its six answer attempts. Per-trial wall time includes validation
and application execution. Workspace setup is recorded separately. Runtime
usage also separates preparation, inference, and cleanup. Retained selector
latency is reported as historical overhead, not a fresh latency measurement.

`seed-answers.json` checkpoints trials, statuses, answers, citations, traces,
judge verdicts, and accounting. Each dispatched request and response is also
saved under `trials/<trial-id>/call-*.json`. Completed attempt records are written
as `result.json`. Sources and local indexes remain in `workspaces/`. Model
requests may contain sensitive source text, so these files are local experiment
artifacts rather than public documentation. Credentials are not stored in them.

```bash
.venv/bin/python -m tools.seed_answer_summary \
  runs/seed-answers/EXECUTE_UNIQUE/seed-answers.json \
  --output runs/seed-answers/EXECUTE_UNIQUE/seed-answers-summary.json
```

The summary preserves planned denominators and reports paired wins, regressions,
and ties within each pool, separate cohort results, citation support, source
discoveries beyond the initial seeds, latency distributions, and accounting.
One generated answer per arm does not estimate answer-generation variability.

For the deterministic contract checks and the separate actual-Docker test:

```bash
.venv/bin/python -m pytest tests/test_seed_answers.py -m 'not integration'
LLGM_TEST_DOCKER=1 \
  LLGM_REPL_DOCKER_IMAGE=sha256:ec7d6c95cd3692a2e2d228a8b1ca74e4025b54121fcc4c5da6f09cfa473315ad \
  .venv/bin/python -m pytest tests/test_seed_answers.py -m docker
```

The Docker test uses deterministic model decisions and actual Python transport.
It makes no hosted calls. Once opted in, unavailable Docker or image execution
fails the test rather than silently skipping it.

## Outcomes and evidence

Answer correctness and citation support are distinct measurements. The accuracy
judge uses the checksum-pinned LongMemEval `get_anscheck_prompt` function with the
question, reference answer, and generated answer. The loader extracts only that
trusted function. It does not execute the evaluator CLI. This is an
official-prompt-based evaluation with explicitly configured judge behavior, not
an unmodified run of the official evaluation script.

Controlled positive cases use the official multi-session accuracy template, and
their ability names are mapped for that purpose. The controlled abstention uses
the official abstention template. Report these adapted diagnostic results
separately from the LongMemEval results.

A separate support judge receives the question, question date, generated answer,
and only the evidence records actually cited by the final root. It sees exact
quoted text with canonical spans, speaker roles, and dates. Every quote is checked
against the original source before judgment. Uncited local reads and unselected
child findings are not silently promoted into supporting evidence. This judge
receives no reference answer, ability label, pool identity, selector reason, or
operational metadata. Its outcomes distinguish supported, unsupported,
insufficient, and not-applicable responses. Support does not establish answer
completeness. Matching a reference answer does not establish citation support.

Retain empty answers, operational failures, unresolved evidence, malformed judge
responses, and incomplete trials. Do not silently drop them from a denominator
or replace them with successful retries. Equal limits provide equal admission
budgets. They do not force equal actual model calls, tokens, searches, or latency.

## Cost admission

The protocol allows up to $16 in local generation reservations and $4 in judging
reservations. Before each request, the runner reserves uncached input cost from
serialized message/schema UTF-8 bytes plus a 2,048-token framing allowance and
the full output limit. Shared reservations include concurrent calls. Returned
usage settles the reservation against the frozen standard synchronous prices.
Cached input is counted once at its separate rate. Unknown usage retains the
reservation. A bound breach or 16 unknown-usage calls stops further admission
for that allowance. These controls do not enforce provider billing.

The native generation and support-judge adapters use Responses. The accuracy
judge uses Chat Completions with the official prompt and a strict yes/no parser.
SDK retries are disabled and each request has a 60-second timeout. The overall
four-hour limit stops admission of new generation and judging requests through
their shared allowances. Already-admitted work retains its own answer and request
deadlines. Trial failures and missing judges remain visible even when all planned
attempts have been dispatched.

## Interpretation limits

The benchmark questions and controlled histories are exposed development data.
The benchmark histories share the project's existing isolation component. The
controlled histories share authored templates. They do not constitute an
independent held-out evaluation. A frozen selector repetition avoids choosing a
favorable seed set after observing answers. It does not estimate variability over
all selector samples or answer generations.

Report inference, judging, prior selection, and prior retrieval separately.
Replaying saved retrieval does not measure current GPU startup or retrieval
latency. A combined pool requires both original retrieval lists even when later
selection admits the same three-node limit. Usage-based dollar estimates use the
frozen price basis and are not invoices or provider-enforced spend caps. Missing
usage must remain unknown.

## Delegate qualification follow-up

[seed_answer_qualification_v1.json](seed_answer_qualification_v1.json) defines a
separate eight-case diagnostic prompted by observed interface failures in the
main run. The original comparison remains unchanged. This follow-up uses the
first question from each of its six LongMemEval strata and controlled cases
`ns-01` and `ns-03`, with the exact retained BM25 first-owner seeds.

Only the per-answer delegate model changes, from GPT-4.1 mini to GPT-4.1. The
root, prompts, temperature, source histories, canonical handles, inference limits
and judging remain fixed. The follow-up executes cases sequentially while the
main run has two concurrent cases. It does not isolate latency or scheduling
effects. Report actual reads, branch evidence returns, final citations and answer
quality separately. It has no outcome-dependent success threshold and makes no
general reliability or selector-quality claim.

The runner refuses preparation until the main run completes with no requests
in flight. It reconciles retained known costs and unresolved reservations and
admits its full $2 generation plus $0.50 judging allowance only if the combined
local liability stays within $20. Missing provider usage remains unknown rather
than becoming an exact cost. Preparation hashes the completed parent result and
execution rechecks its exact bytes before dispatch. A new output directory
preserves the separate attempt:

```bash
.venv/bin/python -m tools.seed_answer_qualification \
  --output runs/seed-answers/QUALIFY_PREPARE_UNIQUE

.venv/bin/python -m tools.seed_answer_qualification \
  --execute --env-file .env \
  --output runs/seed-answers/QUALIFY_EXECUTE_UNIQUE
```

The output `seed-answer-qualification.json` retains all eight planned trials,
judgments, calls, references, traces and parent identity. Do not pass it to the
six-arm summary tool, whose planned denominator is specific to the main
comparison. Keep this post-hoc qualification distinct from the frozen benchmark
comparison, including any failed or unstarted attempts.

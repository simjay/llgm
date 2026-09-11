# Controlled architecture comparison

Repository instructions for the [frozen v3 protocol](architecture_comparison_v3.json).
The comparison asks five independent controlled questions using three approaches:

1. Search once, have the sidecar read the results, then ask the root to answer.
2. Let the models search, read and delegate follow-up questions.
3. Use the same recursive approach with graph navigation enabled.

The questions cover a direct lookup, a chain of aliases, a scoped update,
conflicting claims and an unknown answer. Each history includes forty distractor
sources. All approaches receive the same source and compact operational journal evidence, model pair
and per-run budget limits. Setup is shared and timed separately. Execution order
is seeded. Failures stay in the results. Exact answer matching and supporting
reference coverage are scored separately after inference.

The links are curated and explicitly labeled. The comparison measures the
navigation shortcut, not automatic link quality or the full value of graph
maintenance. Primary edges are curated independently from journal amendments. Source text
contains the alias relationships in every arm, and disabling navigation removes
only the primary-edge lookup operation. Python RLM needs a separate
comparison because its context and citation contracts differ. This small
development diagnostic is not a benchmark or paper result. The separate
[E03 protocol](retrieval.md) owns retrieval-backend and query-policy comparisons.

## Protocol version and evidence

Version 3 records independent primary edges, complete operational journals and
Unix-millisecond validity selectors. The two alias-chain relationships are
explicitly authored as edges. Sources, keys, distractors, budgets, model pair and
seeded arm order remain unchanged. This is a structured-operation ablation. It
does not measure LLGM's retrieval-first parallel Python node delegates.

The [v1 protocol](architecture_comparison_v1.json) and
[v2 protocol](architecture_comparison_v2.json) remain byte-identical historical
inputs. The current runner rejects them so changed semantics cannot reuse an old
protocol name. No hosted v3 trial has been run.

## Run the comparison

Run from the repository root with the OpenAI extra installed,
`OPENAI_API_KEY` configured and a new output directory:

```bash
python tools/compare_inference.py run \
  --protocol experiments/architecture_comparison_v3.json \
  --sha256 5d6baaf49e1bace643591c86e40eca603b160f6136b8c99097de1b0eac278272 \
  --output runs/architecture-comparison-v3-live --execute
```

The command makes real hosted calls. It checks the frozen file hash, records
model identities and source-code provenance, and writes predictions, judgments,
usage, traces and a summary. Missing credentials produce a `not_run` record with
zero model calls. They never trigger scripted replacement responses.

Freeze another file before changing models, budgets, inputs or scoring. Retain
the original file and its results. Keep measurements in the internal
[research reports](../research/reports/README.md). Neither these instructions nor
the internal research records are part of the published library documentation.

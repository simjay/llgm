# Test fixtures

This directory contains a manifest for optional real-data checks. It does not
contain the LongMemEval conversations themselves.

[longmemeval-s.json](longmemeval-s.json) pins the full dataset's revision, byte
size, checksum and four diagnostic case IDs. Each selected case uses its complete
history. The manifest is not a development split or a benchmark score.

## Use the local dataset

Place the pinned release at `data/longmemeval/longmemeval_s_cleaned.json`, relative
to the repository root, or set `LLGM_TEST_LONGMEMEVAL_PATH` to an existing copy.
Then run:

```sh
make test-data
```

The fixture checks the complete file's size and SHA256 before loading cases.
Missing optional data skips these tests. An explicitly configured missing file
or a mismatched checksum fails. Tests never download or substitute data.

## Find other test inputs

- Most unit tests create short histories inside the test that needs them.
- `tests/node_support.py` supplies controlled model and interpreter responses
  for runtime contracts. These responses do not establish hosted model quality.
- [Smoke cases](../../experiments/longmemeval_smoke_cases.json) contain independently
  authored histories and evaluator-only labels for the answer runner.

The [testing guide](../../docs/contributing/testing.md) explains what each layer
runs, which checks use real services, and how to interpret their results.

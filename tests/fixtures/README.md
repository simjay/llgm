# Test fixtures

`longmemeval-s.json` contains only the dataset identity, immutable revision, full-file checksum, and fixed diagnostic case IDs. Benchmark conversation excerpts, gold answers, and model credentials are not committed here.

Place the separately obtained pinned release at `data/longmemeval/longmemeval_s_cleaned.json`, or set `LLGM_TEST_LONGMEMEVAL_PATH` to that exact file. The integration fixture verifies its size and SHA256 before loading cases. It never downloads or substitutes data.

These cases support component and protocol checks, not a development split or a benchmark score. See [Testing](../../development/testing.md) for optional provider, recursion, ColBERT, and official-judge checks.

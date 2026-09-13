# Contributing to LLGM

The [contributor guide](docs/contributing/README.md) covers environment setup,
local commands, and package builds. From a checkout, start with:

```sh
make setup
make check
```

Keep changes focused on an observable behavior. Describe the problem, the
resulting behavior, and the checks you ran when submitting a change.

- Follow the [code standards](docs/contributing/standards.md) for Python and docstrings.
- Use the [testing guide](docs/contributing/testing.md) to choose the appropriate checks.
- Follow [documentation ownership](docs/contributing/documentation.md) when updating guides or moving pages. Run `make docs` for documentation changes.

For work with a coding agent, [AGENTS.md](AGENTS.md) is the entry point to the
repository instructions and shared project context.

Report reproducible bugs or propose changes through [GitHub issues](https://github.com/simjay/llgm/issues).

## Repository layout

| Directory | Contents |
| --- | --- |
| `src/llgm/` | Library, CLI, retrieval adapters and local graph viewer |
| `tests/` | Deterministic contract tests and separately enabled integration checks |
| `docs/guide/`, `docs/reference/` | Published user guides and API reference |
| `docs/contributing/` | Repository-only setup, testing, standards, releases, and branding |
| `examples/` | Runnable library examples |
| `experiments/` | Benchmark protocols, configurations, and runbooks |
| `tools/` | Repository checks and benchmark infrastructure scripts |
| `agent-context/` | Shared decisions, open work, and pointers for coding agents |

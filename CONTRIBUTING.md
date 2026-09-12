# Contributing to LLGM

The [development guide](development/README.md) covers environment setup,
local commands, and package builds. From a checkout, start with:

```sh
make setup
make check
```

Keep changes focused on an observable behavior. Describe the problem, the
resulting behavior, and the checks you ran when submitting a change.

- Follow the [code standards](development/standards.md) for Python and docstrings.
- Use the [testing guide](development/testing.md) to choose the appropriate checks.
- Follow [documentation ownership](development/documentation.md) when updating guides or moving pages. Run `make docs` for documentation changes.

For work with a coding agent, [AGENTS.md](AGENTS.md) is the entry point to the
repository instructions and shared project context.

Report reproducible bugs or propose changes through [GitHub issues](https://github.com/simjay/llgm/issues).

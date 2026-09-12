# Development

These repository guides are for contributors and maintainers changing the
library, its documentation, or its evaluation tooling. Library users start with
the [user documentation](../docs/index.md). The setup and command reference
below apply from the repository root.

| Guide | Purpose |
| --- | --- |
| [Code standards](standards.md) | Write useful comments, docstrings, and small interfaces. |
| [Testing](testing.md) | Select checks, measure coverage, and configure integration gates. |
| [Documentation conventions](documentation.md) | Place content, maintain navigation, and record changes. |
| [Publishing](publishing.md) | Configure PyPI Trusted Publishing, release tested distributions, and set up GitHub Pages or Read the Docs hosting. |
| [Brand assets](branding.md) | Maintain the project logo, icons, and repository design assets. |

Run `make` from the repository root to see the available commands. The default
target prints help grouped by everyday development, documentation and packages,
local data checks, the current LongMemEval pilot, and remote ColBERT development.
Installing dependencies and running checks are explicit steps. Paid API and
Modal commands run only when their targets are selected.

Make targets cover repeated development tasks and the current evaluation.
Historical experiments keep their direct reproduction commands in
[the experiment guides](../experiments/README.md).

## Set up a checkout

Use Python 3.11 or newer, Make and [uv](https://docs.astral.sh/uv/). After cloning
the repository and entering its directory:

```sh
make setup
make check
```

`make setup` creates `.venv` only when that directory is absent, then
installs LLGM in editable mode, the `dev` dependency group and
`docs/requirements.txt`. Its additive `uv pip install` preserves installed
optional extras. It does not use environment synchronization or require pip
inside the selected environment. Dependency installation can access package
indexes.

Every Python command uses `PYTHON`, which defaults to `.venv/bin/python`.
Select an existing environment explicitly when needed:

```sh
make setup PYTHON=/path/to/environment/bin/python
make check PYTHON=/path/to/environment/bin/python
```

CI can use `make check PYTHON=python` after installing its dependencies. `UV`
selects the installer executable for `setup`. The package build also requires
`uv` on `PATH` for its isolated build environment.

Install only the optional integrations needed for the work. For example, adding
the OpenAI adapter to the default environment is an explicit installation:

```sh
uv pip install --python .venv/bin/python -e '.[openai]'
```

See [configuration](../docs/guide/configuration.md) for adapter settings and
[testing](testing.md) for integration prerequisites. Core imports and ordinary
checks require neither provider credentials nor model downloads.

For remote ColBERTv2/PLAID development, `make setup-colbert` adds the pinned Modal
SDK without installing local GPU dependencies. The separately invoked
`make colbert-prepare` and `make test-colbert` use the authenticated Modal
workspace. See [integration testing](testing.md) for setup,
artifacts and manual CI. Frozen commands and protocol details live in the
[experiment index](../experiments/README.md).

## Find the owning code

The [user architecture](../docs/guide/architecture.md) explains how the library
fits into an application. This map identifies where to change the implementation.

| Responsibility | Owning code |
| --- | --- |
| Application construction, seed admission, and public operations | [`LLGM`](../src/llgm/llgm.py) |
| Shared records, settings, errors, and time semantics | [`core/`](../src/llgm/core/) |
| Atomic source publication and index ownership | [`memory/workspace.py`](../src/llgm/memory/workspace.py) |
| Evidence resolution, search, and graph access | [`memory/evidence.py`](../src/llgm/memory/evidence.py) |
| Effective reads and query-local evidence | [`memory/query.py`](../src/llgm/memory/query.py) |
| Edge and journal proposals | [`memory/maintenance.py`](../src/llgm/memory/maintenance.py) |
| Seed scheduling, recursive branches, and final synthesis | [`inference/nodes.py`](../src/llgm/inference/nodes.py) |
| Isolated Python execution and container cleanup | [`inference/repl.py`](../src/llgm/inference/repl.py) |
| Shared resource admission and usage accounting | [`inference/budget.py`](../src/llgm/inference/budget.py) |
| Replaceable provider, retrieval, and persistence adapters | [`models/`](../src/llgm/models/), [`retrieval/`](../src/llgm/retrieval/), [`storage/`](../src/llgm/storage/) |
| Benchmark preparation, generation, judging, and reporting | [`evaluation/`](../src/llgm/evaluation/) |

The primary LongMemEval runner freezes inputs and scheduled trials, records
physical model calls, and separates generation from judging. Its priced admission
and token pacing are evaluation controls. Ordinary `LLGM` calls use the runtime's
call, context, and time budgets without enforcing a dollar limit. Framework
transport tests do not establish a completed Mem0 or Graphiti integration. See
[evaluation testing](testing.md#longmemeval-evaluation) and the
[experiment index](../experiments/README.md) for the owning methods and commands.

## Check a change

| Command | Scope and output |
|---|---|
| `make test` | Deterministic pytest suite, excluding every `integration` test |
| `make format` | Apply Ruff formatting and import ordering across Python code |
| `make lint` | Check formatting, imports and unused code across library, tests, tools, examples and Sphinx configuration |
| `make docstrings` | AST audit of modules, classes and functions, including private and nested definitions |
| `make check` | Lint, docstrings and deterministic tests |
| `make test-data` | Local checksum-pinned LongMemEval ingest, retrieval, immutable-source, and current-journal checks |
| `make coverage` | Deterministic tests under branch coverage, with terminal, JSON, XML, and HTML reports in `runs/coverage/unit/` |

`check` does not install dependencies or invoke hosted providers, Docker, S3 or
ColBERT. Live integration tests have separate explicit gates documented in
[testing](testing.md). Keep those flags local to their intended command.

`test-data` uses the complete pinned dataset already on disk and never downloads
it. If the optional dataset is absent, these tests skip. To select another local
copy:

```sh
LLGM_TEST_LONGMEMEVAL_PATH=/path/to/longmemeval_s_cleaned.json make test-data
```

These checks establish storage and retrieval contracts over real histories.
They do not score model answers or establish benchmark superiority. Coverage
reports also describe executed code paths, not answer quality.

For a focused edit, run the owning test file directly before the broader check:

```sh
.venv/bin/python -m pytest tests/test_application.py -q
```

Prefer tests of observable contracts: canonical source attribution, immutable
references, current journal visibility, resource limits, and retained failures. The
[architecture guide](../docs/guide/architecture.md) identifies the application and runtime
boundaries. The [implementation status](../docs/reference/implementation-status.md) separates
supported contracts from unimplemented capabilities.

## Build and preview documentation

```sh
make docs
make serve-docs
```

The docs target performs a fresh Sphinx HTML build with warnings treated as
errors, then runs `tools/check_docs.py` to audit rendered API anchors,
local navigation, public authored links, prose punctuation, and the public
documentation boundary.
Published prose is authored in `docs/`. API reference is extracted from library
docstrings. The build does not publish development guides, benchmark runbooks,
repository research, agent context, or external Markdown indexes. HTML output is
in `docs/_build/html/`.
The build does not read or validate research, agent context, or maintainer guides.
Use `make docs-links` for a separate repository-wide Markdown link audit.
Override that location for a separate build, including in CI:

```sh
make docs PYTHON=python DOCS_BUILD_DIR=/tmp/llgm-docs
```

After moving or deleting a site page, use a fresh output directory or remove
only the generated `docs/_build/` directory before rebuilding. Sphinx can leave
old HTML in an existing output directory. The documentation audit rejects it.

The preview target builds first and serves only on `127.0.0.1:8000`. Stop the
server with Ctrl-C. Use `make serve-docs PORT=8010` to choose another local port.
`serve-docs` uses the same `DOCS_BUILD_DIR` override as `docs`.
This command previews the site locally. Pushing `main` builds a documentation
artifact in GitHub Actions. Deployment requires Pages setup and the repository
variable `DOCS_PUBLISH_ENABLED=true`. See [publishing](publishing.md) for setup
and the alternative Read the Docs hosting route.

Write documentation in Markdown, use relative links between repository files,
and keep API examples aligned with the implemented signatures. Public pages
should explain behavior and limits without requiring internal project context.
The [documentation conventions](documentation.md) define where each kind of
content belongs and how navigation is maintained.

## Build a distribution

```sh
make build
```

This creates a source distribution and builds its wheel in isolation, then
checks all wheel and source-distribution metadata in `dist/` with strict Twine
validation. The build may fetch its declared build dependencies. It does not
publish packages or replace the development environment. Installed-package
checks run in CI. Publishing a GitHub release runs those checks again and uploads
the tested distributions after they pass. See [publishing](publishing.md) for
the required PyPI account setup and version-tag convention.

`make clean` removes exactly `build/`, `dist/` and `docs/_build/`. It preserves
the virtual environment, local datasets, memory workspaces and experiment or
coverage artifacts under `runs/`.

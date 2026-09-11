<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/brand/llgm-logo-dark.svg">
    <img src="docs/_static/brand/llgm-logo-light.svg" alt="LLGM" width="400">
  </picture>
</p>

<h1 align="center">Large Language Graphical Model</h1>

<p align="center">Persistent evidence and recursive context access for LLM applications.</p>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-14866D?style=flat-square" alt="MIT license"></a>
</p>

<p align="center">
  <a href="#getting-started">Getting started</a> ·
  <a href="docs/index.md">Documentation</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

LLGM is a Python library for evidence that grows across conversations. It keeps
original sources in an evidence graph with independent primary edges and local
journals for corrections. A question retrieves seed nodes, smaller-model Python
delegates investigate them with bounded concurrency, and the root combines their
cited findings into one answer.

The default searches with BM25 and starts delegates at the first three distinct
source nodes in passage rank order. ColBERTv2 with PLAID is an optional retrieval
backend. Combining BM25 and ColBERT has been tested experimentally and is not a
configured default. The [node-search guide](docs/guide/node-search.md) explains
selection, follow-up discovery and how to locate missing evidence.

The graphical-model motivation is local computation and bounded node-to-node
messages. Original sources remain addressable, and explicit journal patches are
applied before selected text reaches the model. The library is an unpublished
experimental alpha. It does not claim mathematically minimal messages,
probabilistic factorization, convergence, or an established advantage over flat
retrieval.

Hosted checks have exercised retrieval, recursive execution and canonical
citations. They have also exposed missed facts within selected nodes and
incorrect synthesis. A working retriever or a valid citation does not establish
a correct answer. See [current capabilities and limits](docs/reference/implementation-status.md).

For a design refresher, start with [Concepts](docs/guide/concepts.md), then
[Node search](docs/guide/node-search.md) and [Architecture](docs/guide/architecture.md).

## Getting started

From a checkout, install with Python 3.11 or newer:

```sh
python -m pip install '.[openai]'
export LLGM_ROOT_MODEL=YOUR_ROOT_MODEL_ID
export LLGM_SIDECAR_MODEL=YOUR_SMALLER_MODEL_ID
```

Set `OPENAI_API_KEY` and replace both model placeholders. Start Docker and make
the trusted `python:3.12-slim` image available locally, or select another image
with `LLGM_NODE_REPL_IMAGE`. The runtime never pulls an image automatically.
This example makes hosted calls and stores evidence in `./memory` by default:

```python
import asyncio

from llgm import Conversation, LLGM, Settings


async def main():
    """Store a conversation and answer a question from its evidence."""
    async with LLGM.from_settings(Settings.from_env()) as memory:
        await memory.ingest(
            Conversation.from_turns([
                {"role": "user", "text": "Atlas production uses PostgreSQL."},
            ]),
            idempotency_key="example-atlas-conversation",
        )
        result = await memory.answer("What database does Atlas production use?")
        print(result.answer, result.status, result.references)


asyncio.run(main())
```

For local configuration, copy `.env.example` to `.env`, fill in the model IDs
and credentials, and call `load_env_file(".env")` from `llgm` before creating
settings. The CLI accepts `llgm ask "Your question" --env-file .env`. Existing
exported variables take precedence. The
[configuration guide](docs/guide/configuration.md) covers providers, file syntax,
and TOML overrides.

Sources receive random UUIDs when `node_id` is omitted. Each source is immutable.
A later observation gets a new node. Primary edges guide discovery, while
precise journal amendments connect replaced passages to their new evidence. The [user guide](docs/guide/index.md) explains setup, current-read
semantics, resource ownership, and result handling. For a credential-free run of the full pipeline, use `python examples/offline.py`
with the core package and Docker. The [walkthrough](docs/guide/walkthrough.md)
also includes source and amendment operations that need no Docker.

## Repository layout

| Directory | Contents |
| --- | --- |
| [docs](docs/index.md) | End-user guides and API reference, published as the documentation site |
| [development](development/README.md) | Repository-only contributor setup, testing, documentation maintenance and release instructions |
| [src/llgm](src/llgm) | Python library and command-line interface |
| [examples](examples/README.md) | Runnable offline and hosted examples |
| [tests](tests) | Deterministic contracts and opt-in integration tests |
| `research/` | Internal design notes and measurements |
| `experiments/` | Repository benchmark configurations, source pins, and run instructions |
| [tools](tools) | Documentation checks and local experiment utilities |

## Contributing

See [Contributing](CONTRIBUTING.md) for development setup, standards, testing,
and documentation ownership. Use [GitHub issues](https://github.com/simjay/llgm/issues)
for reproducible bugs and proposed changes.

GitHub Actions builds and checks the documentation when committed changes are
pushed. The `main` documentation workflow also uploads the built site. GitHub
Pages deployment requires an explicit publishing switch. See
[Publishing](development/publishing.md).

Licensed under the [MIT license](LICENSE).

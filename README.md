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

LLGM is a Python library for answering questions across stored conversations.
It keeps conversations as connected sources, uses smaller models to read relevant
passages, and gives a root model their findings to compose an answer. Each answer
can cite the original text, so you can inspect the evidence behind it.

- **Persistent evidence:** Keep original conversations and apply corrections without erasing their history.
- **Recursive reading:** Let a model follow related sources and ask focused questions of another node.
- **Flexible models:** Use OpenAI, Anthropic, or an explicitly configured compatible endpoint.
- **Replaceable search:** Start with local BM25 or supply a retriever such as ColBERTv2 with PLAID.

LLGM is an experimental alpha. See [capabilities and limits](docs/reference/implementation-status.md)
before choosing it for an application.

## Getting started

You need Python 3.11 or later, Docker, and model API credentials. The package is
not on PyPI yet. From a cloned or downloaded copy, install the OpenAI extra:

```sh
python -m pip install '.[openai]'
```

Start Docker, prepare the interpreter image, and set model IDs available to your
account. Replace both placeholders and set `OPENAI_API_KEY` in your environment:

```sh
docker pull python:3.12-slim
export LLGM_ROOT_MODEL=YOUR_ROOT_MODEL_ID
export LLGM_SIDECAR_MODEL=YOUR_SMALLER_MODEL_ID
```

Save this as `ask_atlas.py` and run `python ask_atlas.py`. It makes hosted model
calls and stores the conversation in `./memory`:

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
        print(result.answer)
        print(result.status, result.references)
        print(result.evidence.unresolved)


asyncio.run(main())
```

The [quickstart](docs/guide/quickstart.md) explains result handling and custom
clients. For storage without model calls or Docker, try the
[evidence walkthrough](docs/guide/walkthrough.md#follow-an-update-and-a-correction).

## Documentation

- [Concepts](docs/guide/concepts.md): Learn sources, links, corrections, and recursive reading.
- [Architecture](docs/guide/architecture.md): Follow a conversation from ingestion to an answer.
- [Node search](docs/guide/node-search.md): Understand how LLGM chooses where to read.
- [Configuration](docs/guide/configuration.md): Choose providers, storage, and resource limits.
- [API reference](docs/reference/api.md): Find public interfaces and signatures.

## Contributing

See [Contributing](CONTRIBUTING.md) to get involved, or open a
[GitHub issue](https://github.com/simjay/llgm/issues) for a bug or proposed change.

Licensed under the [MIT license](LICENSE).

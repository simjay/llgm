<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/brand/llgm-logo-dark.svg">
    <img src="docs/_static/brand/llgm-logo-light.svg" alt="LLGM" width="400">
  </picture>
</p>

<h1 align="center">Large Language Graphical Model</h1>

<p align="center">Persistent memory with traceable sources for LLM applications.</p>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-14866D?style=flat-square" alt="MIT license"></a>
</p>

<p align="center">
  <a href="#getting-started">Getting started</a> ·
  <a href="https://llgm.readthedocs.io/en/latest/">Documentation</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

LLGM is a Python library for answering questions from stored conversations.
Models read relevant passages, follow related sources when needed, and return
their findings to a root model that writes the answer. References point back to
the original text so you can inspect what supports an answer.

- **Persistent evidence:** Keep original conversations and apply corrections without erasing their history.
- **Recursive reading:** Let models investigate related sources without sharing every conversation in full.
- **Flexible models:** Use OpenAI, Anthropic, or an explicitly configured compatible endpoint.
- **Replaceable search:** Start with local BM25 or supply a retriever such as ColBERTv2 with PLAID.

LLGM is an experimental alpha. See [capabilities and limits](https://llgm.readthedocs.io/en/latest/reference/implementation-status.html)
before choosing it for an application.

## Getting started

Install from the repository with Python 3.11 or later and Git. LLGM is not on
PyPI yet:

```sh
python -m pip install 'llgm[openai] @ git+https://github.com/simjay/llgm.git'
```

The [Quickstart](https://llgm.readthedocs.io/en/latest/guide/quickstart.html) walks
through Docker setup and configuring `OPENAI_API_KEY`, `LLGM_ROOT_MODEL`, and
`LLGM_SIDECAR_MODEL`. Once configured, save this as `quickstart.py` and run
`python quickstart.py`. It makes hosted model calls and stores data in `./memory`:

```python
import asyncio

from llgm import Conversation, LLGM, Settings


async def main():
    """Store a conversation and answer a question from its evidence."""
    async with LLGM.from_settings(Settings.from_env()) as memory:
        await memory.ingest(
            Conversation.from_turns([
                {"role": "user", "text": "The production database is PostgreSQL."},
            ]),
            idempotency_key="quickstart-production-database",
        )
        result = await memory.answer("Which database does production use?")
        print(result.answer)
        print(result.status, result.references)
        print(result.evidence.unresolved)


asyncio.run(main())
```

For storage without model calls or Docker, try the
[evidence update example](https://llgm.readthedocs.io/en/latest/guide/walkthrough.html#follow-an-update-and-a-correction).

## Documentation

- [Concepts](https://llgm.readthedocs.io/en/latest/guide/concepts.html): Learn sources, links, corrections, and recursive reading.
- [Architecture](https://llgm.readthedocs.io/en/latest/guide/architecture.html): Follow a conversation from ingestion to an answer.
- [Node search](https://llgm.readthedocs.io/en/latest/guide/node-search.html): Understand how LLGM chooses where to read.
- [Configuration](https://llgm.readthedocs.io/en/latest/guide/configuration.html): Choose providers, storage, and resource limits.
- [API reference](https://llgm.readthedocs.io/en/latest/reference/api.html): Find public interfaces and signatures.

## Contributing

See [Contributing](CONTRIBUTING.md) to get involved, or open a
[GitHub issue](https://github.com/simjay/llgm/issues) for a bug or proposed change.

Licensed under the [MIT license](LICENSE).

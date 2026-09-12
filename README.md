<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/simjay/llgm/main/docs/_static/brand/llgm-logo-dark.svg">
    <img src="https://raw.githubusercontent.com/simjay/llgm/main/docs/_static/brand/llgm-logo-light.svg" alt="LLGM" width="400">
  </picture>
</p>

<h1 align="center">Large Language Graphical Model</h1>

<p align="center">Connect what was said. Read what matters. Answer with evidence.</p>

<p align="center">
  <a href="https://pypi.org/project/llgm/"><img src="https://img.shields.io/badge/PyPI-llgm-3776AB?style=flat-square" alt="LLGM on PyPI, first release pending"></a>
  <a href="https://github.com/simjay/llgm/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square" alt="Python 3.11+"></a>
  <a href="https://github.com/simjay/llgm/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-14866D?style=flat-square" alt="MIT license"></a>
</p>

<p align="center">
  <a href="#getting-started">Getting started</a> ·
  <a href="https://llgm.readthedocs.io/en/latest/">Documentation</a> ·
  <a href="https://github.com/simjay/llgm/blob/main/CONTRIBUTING.md">Contributing</a>
</p>

**LLGM (Large Language Graphical Model)** is a Python library that
gives language models a persistent collection of conversations to investigate.
It keeps the original words, connects related conversations, and lets models
work through the evidence before answering.

## Why LLGM?

Useful knowledge rarely arrives in one message. A team chooses a database in
one conversation, discusses its backup policy in another, and changes the
deployment a month later. A question about the current setup may depend on all
three. A summary can lose a qualification. A search result can point to the right
conversation without containing the fact you need.

LLGM starts from a simple idea: **let the model investigate connected evidence
where it lives, and bring back the parts that support the answer.**

The design draws on two ideas. [Graphical models](https://www.isiweb.ee.ethz.ch/papers/arch/aloe-2001-1.pdf)
motivate local computation and passing messages between related parts.
[Recursive Language Models](https://arxiv.org/abs/2512.24601) motivate keeping
long context in an external environment that a model can inspect with code and
delegate in smaller pieces. LLGM brings these ideas to persistent conversation
memory. Its messages contain findings and source excerpts, rather than
probability distributions.

## How it works

Suppose one conversation says that Atlas uses PostgreSQL and follows the
platform backup policy. Another says that policy keeps backups for seven days.
You ask: **Which database does Atlas use, and how long are its backups kept?**

1. **Store the conversations.** Each becomes a source node with its original
   text. Organization can add links between related nodes.
2. **Find places to start reading.** Search selects relevant nodes for the
   question.
3. **Investigate locally.** Reader models inspect passages using Python in
   isolated Docker containers. They can search again or ask a reader at another
   node a focused question, such as which backup policy applies.
4. **Combine the evidence.** Readers return selected excerpts and findings. A
   final model combines them into an answer with references to the stored text.

In this example, the answer can connect PostgreSQL with the seven-day policy,
and you can inspect the source for each fact. Readers keep their own working
context, so the final model receives their selected evidence instead of every
conversation they visited.

Explicit journal amendments can also point from an old statement to a correction
while preserving both originals. Automatic organization proposes relationships.
Your application decides when to record an exact correction. The
[concepts guide](https://llgm.readthedocs.io/en/latest/guide/concepts.html) walks
through that distinction.

## Getting started

LLGM requires Python 3.11 or later. The first
[PyPI release](https://pypi.org/project/llgm/) is in progress. Once available,
install the alpha with the OpenAI provider:

```sh
python -m pip install --pre 'llgm[openai]'
```

Until that release is available, install from the repository:

```sh
python -m pip install 'llgm[openai] @ git+https://github.com/simjay/llgm.git'
```

Start Docker, prepare the `python:3.12-slim` image, and set `OPENAI_API_KEY`,
`LLGM_ROOT_MODEL`, and `LLGM_SIDECAR_MODEL`. The
[Quickstart](https://llgm.readthedocs.io/en/latest/guide/quickstart.html) gives the
setup commands. Then save this as `quickstart.py` and run `python quickstart.py`:

```python
import asyncio

from llgm import LLGM


async def main():
    """Save a note and ask a question about it."""
    async with LLGM.from_settings() as memory:
        await memory.ingest("Atlas production uses PostgreSQL.")
        result = await memory.answer("Which database does Atlas production use?")
        print(result.answer)


asyncio.run(main())
```

That is the application flow: open memory, add evidence, ask a question.
The block closes its connections and keeps the saved evidence in `./memory`.
Ingestion and answering can make hosted model calls. Pass a chat message list
to `ingest()` when you have a complete conversation. The
[Quickstart](https://llgm.readthedocs.io/en/latest/guide/quickstart.html) also shows
how to avoid duplicate ingestion and inspect the answer's sources.

OpenAI, Anthropic, and compatible endpoints can supply the models. Local SQLite
and BM25 search work by default, with optional storage and retrieval adapters.
For storage without model calls or Docker, try the
[evidence update example](https://llgm.readthedocs.io/en/latest/guide/walkthrough.html#follow-an-update-and-a-correction).

LLGM is an experimental alpha. Its evidence trail makes answers inspectable,
but models can still miss or misinterpret a fact. Performance and cost depend on
the workload and model choices. See
[capabilities and limits](https://llgm.readthedocs.io/en/latest/reference/implementation-status.html)
for the current boundaries.

## Documentation

- [Concepts](https://llgm.readthedocs.io/en/latest/guide/concepts.html): Learn sources, links, corrections, and recursive reading.
- [Architecture](https://llgm.readthedocs.io/en/latest/guide/architecture.html): Follow a conversation from ingestion to an answer.
- [Node search](https://llgm.readthedocs.io/en/latest/guide/node-search.html): Understand how LLGM chooses where to read.
- [Configuration](https://llgm.readthedocs.io/en/latest/guide/configuration.html): Choose providers, storage, and resource limits.
- [API reference](https://llgm.readthedocs.io/en/latest/reference/api.html): Find public interfaces and signatures.

## Contributing

See [Contributing](https://github.com/simjay/llgm/blob/main/CONTRIBUTING.md) to get involved, or open a
[GitHub issue](https://github.com/simjay/llgm/issues) for a bug or proposed change.

Licensed under the [MIT license](https://github.com/simjay/llgm/blob/main/LICENSE).

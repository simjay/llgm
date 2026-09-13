# Large Language Graphical Model

**LLGM stands for Large Language Graphical Model.** It gives language models a persistent memory they can investigate.
Store conversations as they happen, ask a question later, and follow the
answer's references back to what was actually said.

A useful answer often spans several conversations. The database choice might
be in a release discussion, its backup policy in an operations thread, and a
later correction somewhere else. Finding one matching passage is a starting
point. The model still has to connect the facts and understand which ones apply.

LLGM organizes those conversations as a graph and gives reader models a way to
investigate it. Each reader works with local evidence, asks focused questions
of other readers when needed, and returns selected findings and source text.
A final model brings those contributions together.

## Start here

- [Quickstart](guide/quickstart.md): install LLGM, store a conversation, ask a
  question, and read the cited source.
- [Concepts](guide/concepts.md): learn how stored conversations, links, and
  model readers work together.
- [Evidence walkthrough](guide/walkthrough.md): store, search, link, and correct
  evidence in Python without calling a model.

LLGM requires Python 3.11 or later. The first
[PyPI release](https://pypi.org/project/llgm/) is in progress. The Quickstart
covers installation from the repository while it is pending, with configured
models, DSPy readers and an explicit local-search setting.

## How an answer comes together

Imagine asking which database Atlas uses and how long it keeps backups. One
reader can inspect the deployment conversation while another checks the backup
policy. If the deployment points to a separate registry, its reader can ask a
child reader to investigate that source.

Readers use Python to inspect stored passages without putting an entire history
into each model prompt. They return the excerpts and findings needed by the
question. The final answer can then cite the database decision and the retention
policy separately.

The name reflects the design's inspiration from local computation and message
passing in graphical models. Recursive language-model reading provides a way
to investigate the evidence at each node. [Concepts](guide/concepts.md) explains
both ideas through an example, and [Architecture](guide/architecture.md) follows
the complete process from ingestion to the answer.

## Your first program

After the [Quickstart setup](guide/quickstart.md), the application needs only a
memory context, some evidence, and a question:

```python
import asyncio

from llgm import LLGM


async def main():
    """Save a message and answer a follow-up from the same conversation."""
    async with LLGM.from_settings() as memory:
        await memory.answer("Atlas production uses PostgreSQL.")
        result = await memory.answer("Which database does Atlas production use?")
        print(result.answer)


asyncio.run(main())
```

Messages and nonempty replies remain in `./memory` when the program exits.
Each repeated call adds a new turn. The model selects its answer wording. [Quickstart](guide/quickstart.md) shows how to inspect the
cited text and read incomplete results.

## Configure your application

Use [Conversations](guide/conversations.md) for chat IDs, imports and read-only
questions. Use [Configuration](guide/configuration.md) for providers, storage,
search and limits. Metadata uses local SQLite. Configured hybrid search combines
BM25 and ColBERT on Modal, while the quickstart explicitly selects local BM25.
The [Graph viewer](guide/graph-viewer.md) lets you inspect the saved evidence.

Use the [API reference](reference/api.md) when you need a signature or a precise
contract. LLGM is an experimental alpha, and its
[capabilities and limits](reference/implementation-status.md) describe what to
expect from models, storage, and resource controls.

```{toctree}
:caption: User guide
:maxdepth: 2
:hidden:

Guide overview <guide/index>
guide/quickstart
guide/conversations
guide/concepts
guide/architecture
guide/node-search
guide/graph-viewer
guide/walkthrough
guide/configuration
```

```{toctree}
:caption: Reference
:maxdepth: 2
:hidden:

Reference overview <reference/index>
reference/api
reference/implementation-status
```

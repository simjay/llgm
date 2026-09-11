# User guide

Use `LLGM` to store source histories, connect related evidence, and answer
questions with references to original text.

To learn how it works, start with [Concepts](concepts.md), then read
[Architecture](architecture.md) and [Node search](node-search.md). Use
[Quickstart](quickstart.md) when ready to run code.

| Guide | Purpose |
| --- | --- |
| [Quickstart](quickstart.md) | Install the library, create memory, and handle an answer. |
| [Configuration](configuration.md) | Select providers, storage, retrieval, and resource limits. |
| [Concepts](concepts.md) | Learn what nodes, passages, edges, journals, and delegates mean through an example. |
| [Architecture](architecture.md) | Follow a question through storage, search, model reading, and final synthesis. |
| [Node search](node-search.md) | Understand passage retrieval, node selection, follow-up discovery and evidence loss. |
| [Evidence walkthrough](walkthrough.md) | Follow a complete answer and apply an explicit correction to stored evidence. |

```{toctree}
:maxdepth: 1
:hidden:

quickstart
configuration
concepts
architecture
node-search
walkthrough
```

See the [reference](../reference/index.md) for signatures and capability limits.

Complete scripts are available as {download}`offline.py <../../examples/offline.py>`
and {download}`recursive_memory.py <../../examples/recursive_memory.py>`. Both inference scripts require Docker and the configured local Python image.
The hosted script additionally requires configured models and credentials.

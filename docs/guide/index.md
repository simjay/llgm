# User guide

Use `LLGM` to store source histories, connect related evidence, and answer
questions with references to original text.

Start with [Quickstart](quickstart.md) to run the library, or [Concepts](concepts.md)
to understand its design. Then follow Architecture, Node search, and the Evidence
walkthrough in order. Configuration is a reference you can return to as needed.

| Guide | Purpose |
| --- | --- |
| [Quickstart](quickstart.md) | Install the library, create memory, and handle an answer. |
| [Concepts](concepts.md) | Learn what nodes, passages, edges, journals, and delegates mean through an example. |
| [Architecture](architecture.md) | Follow a question through storage, search, model reading, and final synthesis. |
| [Node search](node-search.md) | Understand passage retrieval, node selection, follow-up discovery and evidence loss. |
| [Evidence walkthrough](walkthrough.md) | Follow a complete answer and apply an explicit correction to stored evidence. |
| [Configuration](configuration.md) | Select providers, storage, retrieval, and resource limits. |

See the [reference](../reference/index.md) for signatures and capability limits.

Complete scripts are available as {download}`offline.py <../../examples/offline.py>`
and {download}`recursive_memory.py <../../examples/recursive_memory.py>`. Both inference scripts require Docker and the configured local Python image.
The hosted script additionally requires configured models and credentials.

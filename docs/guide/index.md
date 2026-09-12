# User guide

Start with [Quickstart](quickstart.md) to run your first answer. For a closer look
at the storage layer, the [Evidence walkthrough](walkthrough.md) runs without
model credentials or Docker. Read Concepts and Architecture when you want to
understand how the pieces fit together.

| Guide | Purpose |
| --- | --- |
| [Quickstart](quickstart.md) | Install the library, create memory, and handle an answer. |
| [Concepts](concepts.md) | Learn nodes, passages, links, corrections, and model readers through an example. |
| [Architecture](architecture.md) | Follow a question through storage, search, model reading, and final synthesis. |
| [Node search](node-search.md) | See how passage rankings select starting nodes, and tune or inspect the search. |
| [Evidence walkthrough](walkthrough.md) | Store and search sources, connect them, and apply a correction with preserved citations. |
| [Configuration](configuration.md) | Select providers, storage, retrieval, and resource limits. |

See the [reference](../reference/index.md) for signatures and capability limits.

## Complete examples

After the tutorials, try these downloadable scripts:

- {download}`recursive_memory.py <../../examples/recursive_memory.py>` stores two
  conversations and answers a question with your configured models. It needs the
  provider setup and Docker image from Quickstart.
- {download}`offline.py <../../examples/offline.py>` demonstrates two starting
  nodes, a recursive child, and a source correction. It uses scripted model
  responses to show the execution flow and needs Docker, but no API credentials.

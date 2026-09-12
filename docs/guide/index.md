# User guide

LLGM stands for **Large Language Graphical Model**. It stores your conversations,
connects related ones, and lets model readers investigate the parts that matter
to a question. Start with [Concepts](concepts.md) for the idea behind that design,
or [Quickstart](quickstart.md) to store a conversation and ask your first question.

If you want to explore the stored evidence before connecting a model, the
[Evidence walkthrough](walkthrough.md) runs without credentials or Docker.

| Guide | Purpose |
| --- | --- |
| [Quickstart](quickstart.md) | Install the library, create memory, and handle an answer. |
| [Concepts](concepts.md) | See why LLGM keeps conversations in a graph and how model readers use it. |
| [Architecture](architecture.md) | Follow a question through storage, search, model reading, and final synthesis. |
| [Node search](node-search.md) | See how passage rankings select starting nodes, and tune or inspect the search. |
| [Evidence walkthrough](walkthrough.md) | Store and search sources, connect them, and apply a correction with preserved citations. |
| [Configuration](configuration.md) | Select providers, storage, retrieval, and resource limits. |

See the [reference](../reference/index.md) for signatures and capability limits.

## Complete examples

These scripts show a complete program around the examples in the guides:

- {download}`recursive_memory.py <../../examples/recursive_memory.py>` continues a
  persistent conversation and answers a follow-up with your configured models. It needs the
  provider setup and Docker image from Quickstart.
- {download}`offline.py <../../examples/offline.py>` demonstrates two starting
  nodes, a recursive child, and a source correction. It uses scripted model
  responses to show the execution flow and needs Docker, but no API credentials.

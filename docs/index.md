# LLGM documentation

LLGM is a Python library for answering questions from stored conversations.
It keeps the original text, searches for relevant passages, and lets models
investigate the sources they need. The answer includes references to the evidence
used to produce it.

The library is an unpublished alpha installed from source. Full inference needs
Docker and configured model access. See [Quickstart](guide/quickstart.md) to run
your first example and [Capabilities and limits](reference/implementation-status.md)
before choosing features for an application.

| Section | What you will find |
| --- | --- |
| [User guide](guide/index.md) | Install and configure the library, understand its architecture, and follow working examples. |
| [Reference](reference/index.md) | Public Python interfaces and the limits of the current implementation. |

## Start here

To understand how it works, follow these tutorials in order:

1. [Concepts](guide/concepts.md): what is stored, what a node does, and how an
   answer comes together, starting with a concrete example.
2. [Architecture](guide/architecture.md): how stored evidence becomes an answer
   and how storage, retrieval and model access fit together.
3. [Node search](guide/node-search.md): how passages become starting nodes and
   how the models find more information when needed.
4. [Evidence walkthrough](guide/walkthrough.md): follow a database decision
   and a later correction back to their original sources.

Use [Configuration](guide/configuration.md) to change providers, storage, or
resource limits. The [API reference](reference/api.md) lists Python interfaces.

```{toctree}
:maxdepth: 2
:hidden:

guide/index
reference/index
```

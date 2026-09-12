# LLGM documentation

LLGM is a Python library for answering questions across stored conversations.
It keeps the original text, finds relevant passages, and lets smaller models
investigate the evidence before a root model writes an answer with references.

## Start here

- **Run your first example:** [Quickstart](guide/quickstart.md) takes you from
  installation to storing a conversation and asking a question.
- **Understand the design:** [Concepts](guide/concepts.md) introduces the pieces
  through an example, then [Architecture](guide/architecture.md) follows an answer
  from search to synthesis.

LLGM is an alpha installed from source. Inference requires Python 3.11 or later,
Docker, and model API credentials.

## Explore the library

Learn how [node search](guide/node-search.md) chooses where to read, or follow the
[evidence walkthrough](guide/walkthrough.md) to trace an answer and apply a
correction. Use [Configuration](guide/configuration.md) to choose providers,
storage, and resource limits.

The [API reference](reference/api.md) documents Python interfaces.
[Capabilities and limits](reference/implementation-status.md) explains what the
current implementation supports.

```{toctree}
:caption: User guide
:maxdepth: 2
:hidden:

Overview <guide/index>
guide/quickstart
guide/concepts
guide/architecture
guide/node-search
guide/walkthrough
guide/configuration
```

```{toctree}
:caption: Reference
:maxdepth: 2
:hidden:

Overview <reference/index>
reference/api
reference/implementation-status
```

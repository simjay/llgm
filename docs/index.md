# LLGM documentation

Build LLM applications that can answer questions using earlier conversations.
LLGM stores the original text, connects related sources, and lets models read the
passages they need. Answers include references you can open and inspect.

## Start here

- [Quickstart](guide/quickstart.md): install LLGM, store a conversation, ask a
  question, and read the cited source.
- [Concepts](guide/concepts.md): learn how stored conversations, links, and
  smaller-model readers work together.
- [Evidence walkthrough](guide/walkthrough.md): store, search, link, and correct
  evidence in Python without calling a model.

LLGM is an experimental alpha installed from source. It requires Python 3.11 or
later. Answer generation also needs Docker and configured model access.

## How an answer comes together

1. **Find a starting point.** Search identifies relevant passages and the
   conversations that contain them.
2. **Read the evidence.** Smaller models inspect those sources and can ask for
   more information from linked sources or another search.
3. **Write the answer.** A final model combines the findings and their references.

Follow this process in [Architecture](guide/architecture.md), then see
[Node search](guide/node-search.md) for how starting sources are selected.

## Configure your application

Use [Configuration](guide/configuration.md) to choose model providers, storage,
search, and limits. Local SQLite storage and BM25 search are the defaults.

The [API reference](reference/api.md) covers Python classes and methods.
[Capabilities and limits](reference/implementation-status.md) explains the
current support boundaries.

```{toctree}
:caption: User guide
:maxdepth: 2
:hidden:

Guide overview <guide/index>
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

Reference overview <reference/index>
reference/api
reference/implementation-status
```

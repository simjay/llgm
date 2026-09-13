# Examples

Choose an example by what you want to learn. Run these scripts from the
repository root after [setting up the checkout](../docs/contributing/README.md#set-up-a-checkout).
For installation without a checkout, start with the [Quickstart](../docs/guide/quickstart.md).

| Example | What you will see | Requirements |
| --- | --- | --- |
| [recursive_memory.py](recursive_memory.py) | Start an Orion deployment conversation, ask a follow-up, and print its topic ID, citations and usage | Provider credentials, three model IDs, the provider and `rlm` extras, and configured search |
| [offline.py](offline.py) | Two scripted readers apply a correction, consult a child and return a cited answer | The `rlm` extra. Real storage and Deno/Pyodide, without hosted model calls |
| [modal_retrieval.py](modal_retrieval.py) | Search a saved ColBERTv2/PLAID index and check its canonical passage identities | The `modal` extra, an authenticated deployment and artifacts from a completed retrieval check |

For just storage, search and corrections, the
[evidence walkthrough](../docs/guide/walkthrough.md) has complete scripts that
need only the core package. They do not start a model or sandbox.

## Continue a conversation with models

Follow the [Quickstart model setup](../docs/guide/quickstart.md#2-configure-models-and-search),
then run:

```sh
.venv/bin/python examples/recursive_memory.py
```

Or copy `.env.example` to `.env`, fill in the model IDs and provider credentials,
and select that file explicitly:

```sh
.venv/bin/python examples/recursive_memory.py --env-file .env
```

Existing exported values take precedence over file values. The template selects
hybrid search, which uses the Modal deployment. Set
`LLGM_RETRIEVER_BACKEND=sqlite_fts5` for local BM25, as in Quickstart. See
[search configuration](../docs/guide/configuration.md#search-backend) for both paths.

This example can incur provider and retrieval charges. It keeps its messages in
the configured workspace. Each rerun adds new turns to the `orion` conversation.

## Follow a scripted answer

```sh
.venv/bin/python examples/offline.py
```

The first interpreter startup may download runtime assets. Model decisions are
scripted, but the code executes in real DSPy sandboxes over a temporary workspace.
It checks source corrections, recursive evidence delivery and final citations.
Its repeatable output demonstrates the mechanism, not model answer quality.

## Query a fixed remote index

Follow the [ColBERT runbook](../experiments/colbert.md) to prepare the index and
its local passage records, then use `modal_retrieval.py` with that run directory.
This example opens a fixed index. Configured application search separately
refreshes growing workspace snapshots.

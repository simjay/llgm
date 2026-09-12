# Quickstart

This example stores a conversation and answers a question about it using hosted
models. The smaller model reads the evidence, and the root model writes the
answer. Both roles use OpenAI in this example.

To try storage without model calls or Docker, use the
[evidence update example](walkthrough.md#follow-an-update-and-a-correction).

## Install and configure

LLGM requires Python 3.11 or later. It is not published on PyPI yet, so run this
command from a downloaded or cloned copy of the package:

```sh
python -m pip install '.[openai]'
```

Inference uses Docker to run the Python written by the smaller model. Start
Docker and download the default interpreter image:

```sh
docker pull python:3.12-slim
```

LLGM requires that image to be present locally. It does not download images
during a query.

Set `OPENAI_API_KEY` in your environment and choose model IDs available to your
account. Replace the placeholders before running these commands:

```sh
export LLGM_ROOT_MODEL=YOUR_ROOT_MODEL_ID
export LLGM_SIDECAR_MODEL=YOUR_SMALLER_MODEL_ID
```

See [configuration](configuration.md) for other providers and explicit
[`.env` loading](configuration.md#local-environment-file).

## Store a conversation and ask a question

The example stores a note about Atlas, a fictional project. Save it as
`quickstart.py` and run `python quickstart.py`. It makes hosted model calls and
stores data in `./memory` by default.

```python
import asyncio

from llgm import Conversation, LLGM, Settings


async def main():
    """Store a source and answer a question using its evidence."""
    async with LLGM.from_settings(Settings.from_env()) as memory:
        outcome = await memory.ingest(
            Conversation.from_turns([
                {"role": "user", "text": "Atlas production uses PostgreSQL."},
            ]),
            idempotency_key="quickstart-atlas-source",
        )
        result = await memory.answer(
            "What database does Atlas production use?",
        )
        print("maintenance:", outcome.maintenance.status)
        print("answer:", result.status, result.answer)
        print("references:", result.references)
        print("unresolved:", result.evidence.unresolved)


asyncio.run(main())
```

The answer should identify PostgreSQL and cite the stored source. Exact wording
depends on the models. LLGM keeps the original text so you can inspect the
evidence behind the answer.

`ingest()` creates an immutable source node with a generated ID. Retrying the
same input with the same `idempotency_key` reuses that source. Changed information
belongs in a new node. Maintenance, which looks for links to existing nodes,
may run again on a retry.

`answer()` searches for matching passages, chooses their source nodes, and
starts smaller-model readers called *node delegates*. Their findings go to the
root model for the final answer. To start from a known source, pass
`node_id=outcome.source.node_id`. That skips the initial search. The delegate can
still search or follow links if it needs more evidence.

## Understand the result

| Status | Meaning |
| --- | --- |
| `completed` | The root returned an answer with no reported unresolved evidence or required gaps. |
| `partial` | The result records missing evidence, a skipped or failed branch, or an empty initial search. |
| `budget_exhausted` | A resource limit stopped the run without a final answer. |
| `failed` | An error handled by the inference runtime stopped the run without a final answer. |

Status describes how the run ended. It does not certify that an answer is
correct. Read `result.evidence.unresolved` for missing information,
`result.references` for source locations, and `result.usage` for resource usage.
Invalid arguments, preparation errors, unexpected programming errors, and
cancellation can raise exceptions instead of returning a status.

Source storage and maintenance have separate outcomes. A failed maintenance
step does not remove a stored source. By default, maintenance uses the smaller
model to propose and publish allowed relationships after checking their
structure and references. To review proposals first, pass
`MaintenancePolicy(mode="propose")` to `LLGM.from_settings()`. Use
`mode="disabled"` to skip maintenance. Import `MaintenancePolicy` from `llgm`.

## Use your own search backend

The default search index updates from new workspace records automatically. To
use a different source retriever, provide a factory that opens an `Evidence`
handle. This helper assumes the retriever already indexes sources in the
workspace:

```python
from llgm import Evidence, LLGM


async def ask_with_search(settings, retriever, question):
    """Search this workspace through a caller-owned retriever."""
    async def evidence_factory(workspace, *, passage_chars):
        """Open evidence access using the supplied source retriever."""
        return await Evidence.open(
            workspace, retriever, passage_chars=passage_chars,
        )

    async with LLGM.from_settings(
        settings, evidence_factory=evidence_factory,
    ) as memory:
        return await memory.answer(question)
```

LLGM uses this factory for both maintenance and answers. Search results must
carry valid references to sources in this workspace. LLGM reads their original
text instead of treating returned search snippets as evidence. Invalid references
raise an error. Inline journal notes remain searchable in the local index.

You own the supplied retriever, including index updates and cleanup. LLGM closes
each evidence handle after use. See [node search](node-search.md) for backend
options and how passage results become starting nodes.

## Own directly constructed clients

`LLGM.from_settings()` closes the workspace and model clients it creates. If you
construct `LLGM` directly, you own those resources. Register cleanup as each
resource is acquired so it also runs if a later step fails:

```python
from contextlib import AsyncExitStack

from llgm import LLGM, Workspace
from llgm.models import create_model


async def ask_with_clients(root_model_id, sidecar_model_id, question):
    """Use separate providers and close their clients after the answer."""
    async with AsyncExitStack() as stack:
        root = create_model("openai", root_model_id)
        stack.push_async_callback(root.aclose)
        sidecar = create_model("anthropic", sidecar_model_id)
        stack.push_async_callback(sidecar.aclose)
        workspace = await stack.enter_async_context(Workspace.open("./memory"))
        memory = LLGM(workspace, root, sidecar)
        return await memory.answer(question)
```

This helper needs both provider extras and credentials, plus the Docker image
used above. A complete hosted script is available as
{download}`recursive_memory.py <../../examples/recursive_memory.py>`.

# Quickstart

You will store a short conversation, ask a question about it, and read the
source text cited by the answer. This example uses OpenAI for model calls and
saves its data locally.

You need Python 3.11 or later, Git, Docker, and an OpenAI API key with access to
the models you select. The shell commands below use Bash or Zsh.

## 1. Install LLGM

In a directory where you want to keep the example, create and activate a virtual
environment. LLGM is not published on PyPI yet, so install it from the repository:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'llgm[openai] @ git+https://github.com/simjay/llgm.git'
```

The `[openai]` extra installs the provider client. See
[configuration](configuration.md) for other providers.

## 2. Prepare Docker

Start Docker, then download the default Python image:

```sh
docker pull python:3.12-slim
```

The reading model writes Python to inspect stored evidence. LLGM runs that
Python inside Docker. Model generation still happens through the provider API.
LLGM requires the image to be present before answering and does not download it
during a query.

## 3. Configure the models

Set these three environment variables in the same shell where you will run the
example. Replace every value inside angle brackets with your own value:

```sh
export OPENAI_API_KEY='<your OpenAI API key>'
export LLGM_ROOT_MODEL='<model ID for writing the answer>'
export LLGM_SIDECAR_MODEL='<model ID for reading evidence>'
```

The **root model** combines findings into the final answer. The **sidecar model**
reads sources, follows related evidence, and helps organize new conversations.
You can use the same model ID for both roles. If you choose a smaller sidecar,
it still needs to follow the reading instructions and produce Python reliably.

Use actual API model IDs available to your account. LLGM has no default model
IDs and does not replace these placeholders for you. The example uses the
default `openai` provider for both roles. If you already configured another
provider, set `LLGM_ROOT_PROVIDER=openai` and `LLGM_SIDECAR_PROVIDER=openai`.

Environment variables apply to the current shell. For a local `.env` file,
follow [explicit environment-file loading](configuration.md#local-environment-file).
LLGM does not discover or load that file automatically.

## 4. Store evidence and ask a question

Save this complete script as `quickstart.py`. It stores a note about a production
database, asks which database is used, and prints the cited source text:

```python
import asyncio

from llgm import Conversation, LLGM, Settings


async def main():
    """Store a conversation, answer a question, and inspect its sources."""
    async with LLGM.from_settings(Settings.from_env()) as memory:
        outcome = await memory.ingest(
            Conversation.from_turns([
                {"role": "user", "text": "The production database is PostgreSQL."},
            ]),
            idempotency_key="quickstart-production-database",
        )
        result = await memory.answer("Which database does production use?")
        print("maintenance:", outcome.maintenance.status)
        print("answer:", result.answer)
        print("status:", result.status)
        print("unresolved:", result.evidence.unresolved)
        for reference in result.references:
            source = await memory.workspace.resolve(reference)
            print("source:", source.text)


asyncio.run(main())
```

Run it from the activated environment:

```sh
python quickstart.py
```

This makes hosted model calls. The answer should identify PostgreSQL, and the
source output should contain the original statement. Exact wording depends on
the models. Inspect the status and unresolved reasons if a source or answer
is missing.

Your data stays in `./memory` after the script finishes. Run from the same working
directory to reuse it, or set `LLGM_WORKSPACE_PATH` to choose another location.
`LLGM.from_settings()` closes its connections when the `async with` block ends.

## 5. Add more conversations

Call `ingest()` for each new conversation. Each call creates an immutable source
node. Retrying the same input with the same `idempotency_key` reuses that source,
so rerunning this example does not duplicate its note. Use a new key for new
content. Reusing a key with changed content raises a conflict.

Maintenance looks for relationships to existing nodes and may run again on a
retry. A failed maintenance step does not remove the stored source. The
[configuration guide](configuration.md) explains how to review or disable
automatic maintenance.

`answer()` searches the stored conversations and sends the selected nodes to
readers called *node delegates*. Their findings go to the root model for the
final answer. If you already know where to start, pass
`node_id=outcome.source.node_id` to skip the initial search. That reader can
still search or follow links for more evidence.

## Understand the result

| Status | Meaning |
| --- | --- |
| `completed` | The root returned an answer with no reported unresolved evidence. |
| `partial` | The result records missing evidence, a skipped or failed branch, or an empty initial search. |
| `budget_exhausted` | A resource limit stopped the run without a final answer. |
| `failed` | An error handled by the inference runtime stopped the run without a final answer. |

Status describes how the run ended. Check the cited text when assessing the
answer. `result.evidence.unresolved` explains missing information,
`result.references` lists source locations, and `result.usage` records resource usage.
Invalid arguments, preparation errors, unexpected programming errors, and
cancellation can raise exceptions instead of returning a status.

## Troubleshoot setup

| Problem | What to check |
| --- | --- |
| Model IDs are missing | Export both `LLGM_ROOT_MODEL` and `LLGM_SIDECAR_MODEL` before starting Python. |
| Authentication or model-access error | Check the API key, model IDs, and access for that provider account. |
| Docker cannot start an interpreter | Start Docker and confirm that `python:3.12-slim` is available locally. |
| A rerun reports an ingestion conflict | Keep the original note unchanged or assign a new idempotency key to changed content. |
| An answer is `partial` | Read `result.evidence.unresolved` and inspect the source references before using the answer. |

## Continue learning

- [Concepts](concepts.md) explains nodes, evidence, and recursive reading.
- [Evidence walkthrough](walkthrough.md) follows a complete answer and a source correction.
- [Custom search](configuration.md#use-your-own-search-backend) connects another retriever.
- [Client ownership](configuration.md#own-directly-constructed-clients) explains cleanup when you construct clients yourself.

To try storage without model calls or Docker, use the
[evidence update example](walkthrough.md#follow-an-update-and-a-correction).

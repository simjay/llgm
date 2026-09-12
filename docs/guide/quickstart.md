# Quickstart

Start a conversation with `answer()`. LLGM saves new turns, keeps related
discussion in one topic node, and retrieves evidence for follow-up questions.

You need Python 3.11 or later, Git, Docker, and an OpenAI API key with access to
the models you select. The shell commands below use Bash or Zsh.

## 1. Install LLGM

In a directory where you want to keep the example, create and activate a virtual
environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

The first [PyPI release](https://pypi.org/project/llgm/) is in progress. Its
installation command will be:

```sh
python -m pip install --pre 'llgm[openai]'
```

Until that release is available, install the current code from the repository:

```sh
python -m pip install 'llgm[openai] @ git+https://github.com/simjay/llgm.git'
```

The `[openai]` extra includes the provider client used here. Other providers are
covered in [configuration](configuration.md).

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
export LLGM_MAIN_MODEL='<model ID for writing the answer>'
export LLGM_READER_MODEL='<model ID for reading evidence>'
export LLGM_GRAPH_MODEL='<small model ID for organizing topics>'
```

The **main model** combines findings into the final answer. The **reader model**
reads sources and follows related evidence. The **graph model** chooses
whether to append to an existing topic or start a clearly different one, and
proposes generic connections. You can use the same model ID for multiple roles.
If you choose a smaller reader,
it still needs to follow the reading instructions and produce Python reliably.

Use actual API model IDs available to your account. LLGM has no default model
IDs and does not replace these placeholders for you. The example uses the
default `openai` provider for all three roles. If you already configured another
provider, set `LLGM_MAIN_PROVIDER=openai`, `LLGM_READER_PROVIDER=openai`,
and `LLGM_GRAPH_PROVIDER=openai`.

Environment variables apply to the current shell. For a local `.env` file,
follow [explicit environment-file loading](configuration.md#local-environment-file).
LLGM does not discover or load that file automatically.

## 4. Start a conversation

Save this as `quickstart.py`:

```python
import asyncio

from llgm import LLGM


async def main():
    """Continue a conversation with automatic memory."""
    async with LLGM.from_settings() as memory:
        await memory.answer("Atlas production uses PostgreSQL.", conversation_id="atlas")
        result = await memory.answer("Which database does it use?", conversation_id="atlas")
        print(result.answer)


asyncio.run(main())
```

Run it from the activated environment:

```sh
python quickstart.py
```

The answer should identify PostgreSQL. Each call saves your new message and the
returned assistant text. The same `conversation_id` resumes the active topic
across calls and application restarts. Its default is `"default"`.

LLGM asks its organizing model whether a message continues the active topic,
returns to a retrieved topic, or clearly starts a different topic. Follow-ups,
corrections, subtopics, time gaps and large histories should stay together.
The model sees bounded routing previews. Its judgment can be wrong. There is
no size-triggered splitting or fixed target node size.

`from_settings()` reads configuration when the block opens and closes owned
clients when it ends. Data stays in `./memory`, or the location selected by
`LLGM_WORKSPACE_PATH`. History is stored locally. Evidence presented to hosted
models is sent to their provider.

## 5. Send messages or catch up on earlier history

You may send new messages as text or as role/content records:

```python
result = await memory.answer(
    [{"role": "user", "content": "Keep its backups for seven days."}],
    conversation_id="atlas",
)
print(result.answer)
print(result.node_id)
```

Send only new turns. Do not resend the complete transcript on each answer call.
The list must end with a nonempty user message and accepts user and assistant
roles. System instructions, tools, streaming and multimodal input are not part
of this application interface. It returns `AnswerResult`, not a provider SDK
response. General conversation can produce a reply without citations. Personal
memory claims should be supported by the stored evidence.

Use `ingest()` when catching up on conversations that happened elsewhere:

```python
outcome = await memory.ingest(
    [
        {"role": "user", "content": "Atlas production uses PostgreSQL."},
        {"role": "assistant", "content": "How long should its backups be kept?"},
        {"role": "user", "content": "Keep them for seven days."},
    ],
    conversation_id="atlas",
    idempotency_key="atlas-backup-import",
)
```

An imported batch is routed as one unit. Related batches can share a node even
when they came from different sessions. LLGM preserves their exact text, roles
and per-turn source dates. Automatic splitting inside an imported batch is not
implemented. `Conversation.from_turns()` accepts source metadata and timestamps.
An explicit `Conversation.node_id` requests an exact immutable source import
and bypasses topic routing.

An ingestion retry key reuses the saved batch. Changed content under the same
key raises a conflict. `answer()` treats repeated calls as new messages and does
not provide response replay or automatic retries. Stored input survives an
inference failure. Only a nonempty returned reply is appended as assistant text.
Calls sharing a Workspace object serialize conversation updates. Applications
using independent workspace handles must serialize a shared chat themselves.

Routing and answer generation share the answer budget. Generic connection
maintenance runs for newly created answer topics with its separate allowance,
reported under `result.usage["maintenance"]`. Imports report routing usage under
`outcome.maintenance.usage["topic_routing"]`. Maintenance failure does not undo
stored evidence. `organize=False` skips import connection discovery but still
routes the batch. `MaintenancePolicy(mode="disabled")` also disables model topic
routing and keeps appending to that conversation's active node.

For a read-only memory question:

```python
result = await memory.answer("Which database does Atlas use?", remember=False)
```

This leaves stored conversation history unchanged. Add `node_id` in read-only
mode to choose the starting node explicitly. Conversation answers always include
the active topic among their bounded initial readers, which helps resolve
follow-ups such as "what about that?".

Existing schema-3 workspaces require an explicit copy before opening them with
this version. See [workspace migration](configuration.md#storage-choices).

## Understand the result

An answer includes the evidence the model selected. Inside the same memory block,
you can open those references and inspect the original text:

```python
for reference in result.references:
    source = await memory.workspace.resolve(reference)
    print(source.text)

print(result.status)
print(result.evidence.unresolved)
```

This is useful when an answer is surprising or incomplete. For the backup
question, look for the statement about seven days and the conversation that
identifies Atlas production.

| Status | Meaning |
| --- | --- |
| `completed` | The main model returned an answer with no reported unresolved evidence. |
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
| Model IDs are missing | Export `LLGM_MAIN_MODEL`, `LLGM_READER_MODEL`, and `LLGM_GRAPH_MODEL` before starting Python. |
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

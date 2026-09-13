# Quickstart

Tell LLGM something, then ask about it. `answer()` saves each new message and
reply so you can continue the conversation later.

This example uses OpenAI models and local BM25 search. You need Python 3.11 or
later, Git, and an OpenAI API key. The commands use Bash or Zsh.

## 1. Install LLGM

Create and activate a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

The first [PyPI release](https://pypi.org/project/llgm/) is in progress. Until it
is available, install from the repository:

```sh
python -m pip install 'llgm[openai,rlm] @ git+https://github.com/simjay/llgm.git'
```

After the first release, the package installation command will be:

```sh
python -m pip install --pre 'llgm[openai,rlm]'
```

The `openai` extra supplies the provider client. The `rlm` extra installs DSPy
and Deno for running the reader's Python in a Pyodide sandbox. The first reader
startup may download runtime assets, so allow network access on that first run.

## 2. Configure models and search

Set these values in the same shell where you will run Python. Replace the
angle-bracket placeholders with your API key and model IDs:

```sh
export OPENAI_API_KEY='<your OpenAI API key>'
export LLGM_MAIN_MODEL='<model ID for writing the answer>'
export LLGM_READER_MODEL='<model ID for reading evidence>'
export LLGM_GRAPH_MODEL='<model ID for organizing topics>'
export LLGM_RETRIEVER_BACKEND=sqlite_fts5
```

The **main model** writes the final answer. The **reader model** investigates
sources by writing Python. The **graph model** places messages into topics and
proposes connections. All three roles default to the OpenAI provider, and you
can use the same model ID for each. LLGM has no default model IDs.

The last line selects local word-based search for this example. The configured
default is `hybrid`, which combines BM25 with ColBERT on an authenticated Modal
deployment. See [retrieval configuration](configuration.md#search-backend)
when you want to use that backend.

For another provider or a local `.env` file, follow
[configuration](configuration.md). LLGM loads environment files only when your
application explicitly requests one.

## 3. Start a conversation

Save this as `quickstart.py`:

```python
import asyncio

from llgm import LLGM


async def main():
    """Save a message and answer a follow-up from the same conversation."""
    async with LLGM.from_settings() as memory:
        await memory.answer("Atlas production uses PostgreSQL.")
        result = await memory.answer("Which database does Atlas production use?")
        print(result.answer)


asyncio.run(main())
```

Run it:

```sh
python quickstart.py
```

The answer should identify PostgreSQL. Its wording depends on your models.
Both calls can make hosted model requests. The input messages and nonempty
replies remain in `./memory` after the program exits. Running the script again
adds new messages.

`from_settings()` reads your configuration when the block opens and closes its
workspace and model clients when the block ends. Keep this block open while
handling messages. Set `LLGM_WORKSPACE_PATH` to store data elsewhere.

## 4. Keep separate conversations

The first program uses the default conversation ID, `"default"`. Give each chat
an ID to resume its current topic across calls and restarts. Inside your
open memory block:

```python
result = await memory.answer(
    "Keep Atlas backups for seven days.",
    conversation_id="atlas",
)
print(result.conversation_id)
print(result.node_id)
```

The conversation ID belongs to your application. The node ID identifies the
topic LLGM selected. A conversation can move between topics, and related
conversations can share one topic. IDs do not isolate access to the workspace.

Send only new messages on each call. Use `ingest()` to import earlier history,
or `remember=False` to ask without saving another turn. The
[conversation guide](conversations.md) shows those workflows and explains retries.

## Understand the result

Inside the memory block, inspect the answer's references against the stored text:

```python
for reference in result.references:
    source = await memory.workspace.resolve(reference)
    print(source.text)

print(result.status)
print(result.evidence.unresolved)
```

For the database question, look for the original PostgreSQL statement.
A valid reference identifies stored evidence. Read it to check whether it
actually supports the answer.

| Status | Meaning |
| --- | --- |
| `completed` | The main model returned an answer with no reported unresolved evidence. |
| `partial` | Evidence is missing, a branch failed or was skipped, or no starting node was available. An explicit abstention can also return this status. |
| `budget_exhausted` | A resource limit stopped the run without a final answer. |
| `failed` | An error handled by inference stopped the run without a final answer. |

`result.usage` records work performed, and `result.trace` shows the reading
steps. Invalid inputs and preparation errors can raise exceptions. Unexpected
errors and cancellation can also propagate to your application.

## Browse what was saved

After running the script, open the [graph viewer](graph-viewer.md):

```sh
llgm view --workspace ./memory
```

Select a node to inspect its turns, journal and connections. The viewer uses
your configured search backend and does not run answer models. Stop it with Ctrl-C.

## Troubleshoot setup

| Problem | What to check |
| --- | --- |
| Missing model IDs | Set all three `LLGM_*_MODEL` variables before starting Python. |
| Authentication or model-access error | Check your key and the model IDs available to that provider account. |
| An error mentions Modal or hybrid retrieval | Set `LLGM_RETRIEVER_BACKEND=sqlite_fts5` for this tutorial, or configure the hybrid service. |
| DSPy cannot start an interpreter | Install the `rlm` extra and allow the first runtime download. |
| An answer is `partial` | Read `result.evidence.unresolved` and inspect its references. |
| An older workspace will not open | Follow [workspace migration](configuration.md#existing-databases) for the supported formats. |

Continue with [conversations](conversations.md) for application behavior,
[concepts](concepts.md) for the method, or the [evidence walkthrough](walkthrough.md)
to try storage and corrections without models or a sandbox.

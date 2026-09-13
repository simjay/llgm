# Configuration

Configure three model roles. The **graph model** chooses topic placement
and proposes connections. The **reader model** reads evidence recursively.
The **main model** writes the final answer. Maintenance can use a small model
that makes conservative structured decisions, while readers need to generate
Python reliably. Each role has its own configured client. Model IDs may be shared.
The [quickstart](quickstart.md) shows the initial setup with local storage and
explicit BM25 search.

| To configure | Read |
| --- | --- |
| Credentials and a local file | [Environment files](#local-environment-file) |
| Providers and model IDs | [Providers and settings](#providers-and-settings) |
| Local or remote search | [Search backend](#search-backend) |
| Recursion and resource allowances | [Application limits](#application-and-maintenance-limits) and [usage](#runtime-limits-and-usage) |
| Storage or an older workspace | [Storage choices](#storage-choices) |

Once you have set the model IDs and credentials in your environment, open LLGM
without constructing a separate settings object:

```python
from llgm import LLGM


async def ask(question):
    """Answer from the workspace selected by the environment."""
    async with LLGM.from_settings() as memory:
        return await memory.answer(question)
```

The environment is read when the `async with` block starts. LLGM opens the
workspace and provider clients, then closes them when the block ends. Keep that
block open to store sources or ask several questions with the same application.

Use an explicit `Settings` object when you need a TOML file or Python overrides.
`Settings.from_env()` reads environment variables, and
`Settings.load(config_file="llgm.toml", overrides={...})` adds those other inputs.
These methods only parse settings. They do not create storage or connect to a
provider. Pass the resulting object to `LLGM.from_settings(settings)` to use it
without reloading configuration.

Values take precedence in this order:

1. Explicit Python overrides or CLI options such as `--workspace`
2. The selected TOML file
3. Existing process environment variables
4. Values added from the selected environment file
5. Library defaults

Only supplied fields replace lower-priority values. Settings capture model IDs
when they are created, and provider clients read credentials when they are
created. Changing a file or the environment does not update an open application.

## Local environment file

If your application uses a local `.env` file, create it with the required
credentials and model IDs. For OpenAI, the minimum is:

```text
OPENAI_API_KEY=your-api-key
LLGM_MAIN_MODEL=your-main-model-id
LLGM_READER_MODEL=your-reader-model-id
LLGM_GRAPH_PROVIDER=openai
LLGM_GRAPH_MODEL=your-graph-model-id
LLGM_RETRIEVER_BACKEND=sqlite_fts5
```

This selects local BM25, as in Quickstart. Use `hybrid` with a configured Modal
service when you want semantic retrieval as well.

Replace the placeholders and keep this file out of version control. You can
export the same variables directly instead.

Load the selected file once at application startup:

```python
from llgm import LLGM, load_env_file

load_env_file(".env")


async def ask(question):
    """Answer using the environment configured at application startup."""
    async with LLGM.from_settings() as memory:
        return await memory.answer(question)
```

Call `load_env_file()` before entering `LLGM.from_settings()`, or before
`Settings.load()` when also using TOML. The helper adds missing
variables to the process environment. Existing variables win, including empty
values. It does not search for files or load them automatically. Omit the call
when your environment is already configured.

The [command-line interface](#ask-from-the-command-line) loads a selected file
with `--env-file`.

The file accepts one `KEY=value` assignment per line, optional `export`, blank
lines, and comments. Keys use letters, digits, and underscores, with no leading
digit. Single or double quotes preserve a single-line literal value. Unquoted
inline comments must be preceded by whitespace. Variables, commands, and escape
sequences are not expanded. Multiline values are unsupported. A missing file,
malformed assignment, or duplicate key raises a configuration error before any
variables are added.

## Ask from the command line

Continue a conversation from the shell. This saves your question and a nonempty
reply, using the same behavior as `answer()` in Python:

```sh
llgm ask "What database does Atlas use?" --conversation-id atlas --env-file .env
```

This uses the same model configuration and DSPy sandbox requirements as the Python
API. Add `--config llgm.toml` to select a TOML file or `--workspace ./memory`
to override its storage path. `--conversation-id` defaults to `default`.

To query without saving turns, add `--read-only`:

```sh
llgm ask "How long are backups kept?" --conversation-id atlas --read-only --env-file .env
```

`--node-id` requires `--read-only` and selects one starting node. `--scope`
accepts a JSON object selecting applicable edges and amendments. It does not
filter all source searches or enforce access control. `--as-of-ms` accepts a
validity instant in Unix milliseconds, and `--query-date` supplies separate
human-readable date context. See [query time](#query-and-validity-time).

When the application returns a result, the command writes JSON containing `answer`, `status`,
`references`, `unresolved`, and `usage`. The exit code is zero for `completed`
and two for other result statuses. Handled configuration, storage and provider
errors also exit with two and write an error to standard error. Unexpected
exceptions may propagate. Check both the exit code and result status in scripts.

## Providers and settings

Configure model IDs explicitly. This example uses OpenAI for the main model and
Anthropic for the reader:

```text
LLGM_WORKSPACE_PATH=./memory
LLGM_MAIN_PROVIDER=openai
LLGM_MAIN_MODEL=your-main-model-id
LLGM_READER_PROVIDER=anthropic
LLGM_READER_MODEL=your-reader-model-id
LLGM_GRAPH_PROVIDER=openai
LLGM_GRAPH_MODEL=your-graph-model-id
```

Provider names are `openai`, `anthropic`, and `openai_compatible`. Install the
`openai` extra for OpenAI and compatible endpoints, or `anthropic` for Anthropic.
The defaults use OpenAI for all three roles. The mixed-provider example needs both
extras and both `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`.

`LLGM_MAIN_API_KEY_ENV`, `LLGM_READER_API_KEY_ENV`, and
`LLGM_GRAPH_API_KEY_ENV` can name alternative credential variables. They contain
variable names, never credential values. `LLGM_MAIN_BASE_URL`, `LLGM_READER_BASE_URL`, and
`LLGM_GRAPH_BASE_URL` select individual endpoints. A compatible endpoint
requires an explicit base URL.

OpenAI uses the Responses API, Anthropic uses the Messages API, and compatible
endpoints use Chat Completions. Compatible clients default to plain-text output.
If an endpoint supports native JSON Schema or needs `max_completion_tokens`
instead of `max_tokens`, construct `OpenAICompatibleModelClient` directly with
those options. See [client ownership](#own-directly-constructed-clients) and
[model adapters](../reference/api.md#models-and-embeddings).

The adapters return complete responses rather than streaming tokens. Try your
chosen models on representative questions. A working connection alone does not
show that a model can write the inspection code or use the evidence correctly.

A minimal TOML file uses lowercase field names:

```toml
[llgm]
workspace_path = "./memory"
retriever_backend = "sqlite_fts5"
main_provider = "openai"
main_model = "your-main-model-id"
reader_provider = "openai"
reader_model = "your-reader-model-id"
graph_provider = "openai"
graph_model = "your-graph-model-id"
max_model_calls = 12
```

Pass the resulting settings to `LLGM.from_settings()` to open the application.
For example, this reads the file and changes only the workspace location:

```python
from llgm import Settings

settings = Settings.load(
    config_file="llgm.toml",
    overrides={"workspace_path": "./project-memory"},
)
```

`settings.redacted()` reports values and field provenance with credential-bearing
URL components removed. Values loaded from an environment file have
`environment` provenance because they are read from the process environment.
Provider API keys are not stored in `Settings`.

Unknown application `LLGM_` keys, malformed values, unsupported providers, and
incompatible URI/backend combinations fail explicitly. `LLGM_TEST_*` variables
belong to integration tests and are ignored when reading the application's
environment. They are not accepted as application fields in
TOML or Python overrides.

### Native OpenAI reasoning

Direct construction of `OpenAIModelClient` accepts an optional
`reasoning_effort`. This is a native Responses API option, with no corresponding
`Settings`, TOML, or `LLGM_` environment field:

```python
from llgm.models.base import Message, ModelRequest
from llgm.models.hosted import OpenAIModelClient


async def reason_with_model(model_id, prompt):
    """Request medium reasoning from a model that supports it."""
    async with OpenAIModelClient(model_id, reasoning_effort="medium") as model:
        return await model.complete(ModelRequest(
            (Message("user", prompt),),
            max_output_tokens=2048,
            temperature=None,
        ))
```

The default `reasoning_effort=None` omits the effort parameter and leaves the
provider default in effect. An explicit value is sent to the provider and
retained in the client's descriptor. The adapter accepts nonblank strings but
does not infer which efforts a model supports. Use a value accepted by the
selected model.

For an effort other than `None` or `"none"`, requests must keep
`temperature=None`. The native adapter rejects a conflicting temperature before
dispatch. `max_output_tokens` bounds reasoning tokens and visible answer tokens
together. Enabling reasoning does not add another output allowance or model call.

## Python node execution

Install the `rlm` extra alongside your provider extra using the
[Quickstart installation](quickstart.md#1-install-llgm). DSPy runs generated Python in its default
Deno/Pyodide sandbox. The first startup can fetch runtime assets. No host paths,
environment variables or network access are granted to generated code. Model
requests and evidence callbacks run on the host.

- `LLGM_RETRIEVAL_K`, default 12. Maximum passages in the initial search.

- `LLGM_MAX_SEED_NODES`, default 3. Maximum starting nodes, including the current topic.

- `LLGM_MAX_CONCURRENCY`, default 3. Concurrent seed branches and active model calls.

- `LLGM_MAX_JOURNAL_BYTES`, default 65536. Serialized operational journal bytes per node.

`LLGM_RETRIEVAL_K` bounds initial passage retrieval, while `LLGM_MAX_SEED_NODES`
bounds distinct admitted node owners. Require `max_seed_nodes <= retrieval_k <= 40`.
The current topic takes the first slot, and search can fill the rest. Without a
current topic, search supplies all seeds. See [node selection](node-search.md)
for read-only overrides and skipped retrieval.
`LLGM_MAX_CONCURRENCY` queues excess admitted branches and bounds active model
calls. It does not silently select a smaller seed set. Explicit seed-limit skips
and operational failures remain visible as unresolved outcomes.

The complete operational journal is limited by `LLGM_MAX_JOURNAL_BYTES` per node.
`QueryEvidence` also limits its cached journals to 4 MiB of serialized data by
default. Irreducible overflow raises a budget error. Full historical storage can
continue growing. Appended conversation turns are stored in separate immutable
blobs. Metadata paging reads their coordinates, and a span read loads its owning
turn. Explicit imports and legacy base sources still load their complete source blob.

## Application and maintenance limits

`LLGM.from_settings()` owns its configured workspace and clients. Its `Budget`
uses the inference limits in settings. Pass constructor options such as
`max_depth`, `max_steps`, and `max_operations` to control recursion. Maintenance
uses the separately configured graph client. Topic routing in `answer()`
shares its answer budget. Catch-up routing and connection proposals use
`MaintenancePolicy.budget`. Usage and trace events distinguish maintenance
calls from reader calls.

These controls are Python constructor options, rather than `LLGM_` environment
variables. They also work as keyword arguments to `LLGM.from_settings()`:

| Option | Default | Controls |
| --- | --- | --- |
| `max_depth` | 3 | Recursive depth beyond each starting node, which has depth zero |
| `max_steps` | 16 | DSPy action iterations per node, followed by an extraction call if needed |
| `max_operations` | 128 | Shared operation allowance across the answer |
| `passage_chars` | 2048 | Character bound for default source indexing passages |
| `capture_text` | `False` | Additional model-output and code capture in traces |

Pass `SandboxConfig` as `repl_config` to bound startup and execution time,
context and code bytes, printed output, and returned findings. These are admission
limits. They do not provide operating-system CPU or memory quotas. The fields
are in the [Python execution reference](../reference/api.md#dspy-python-execution).
All DSPy action, extraction and plain subquery calls consume the shared model
budget, including the final extraction after `max_steps` iterations.

For tests or another sandbox, `interpreter_factory` accepts a zero-argument
factory returning a synchronous DSPy `CodeInterpreter`. It must own its isolation
and finish shutdown within a bounded time. The default uses `PythonInterpreter`
with no extra permissions.

```python
from llgm import Budget, LLGM, MaintenancePolicy


async def ask_with_limits(settings, question):
    """Answer with a shallow recursion limit and proposals that need review."""
    policy = MaintenancePolicy(
        mode="propose",
        max_candidates=4,
        budget=Budget(max_model_calls=4, max_graph_calls=4),
    )
    async with LLGM.from_settings(
        settings, maintenance_policy=policy, max_depth=2,
    ) as memory:
        return await memory.answer(question)
```

The policy's budget governs import routing and connection discovery. Routing
within `answer()` uses the answer budget instead. `propose` still permits topic
routing, while `disabled` turns off both model routing and connection discovery.
See [organization controls](conversations.md#control-automatic-organization)
for when each operation runs and where its results appear.

Source writes survive a later maintenance failure. Connection validation checks
structure and source references, not whether the proposed relationship is true.

### Change limits for one answer

An explicit `budget` replaces the complete application budget. It does not merge
only the fields you supply. `Budget()` also has smaller defaults than the
integrated application. Use `replace()` to change one limit while preserving
the others:

```python
from dataclasses import replace


async def ask_with_more_time(memory, question):
    """Allow one answer more time while retaining the configured call and text limits."""
    budget = replace(memory.inference_budget, timeout_seconds=180)
    return await memory.answer(question, budget=budget)
```

## Search backend

`LLGM.from_settings()` and `llgm view` default to hybrid BM25 and ColBERTv2.
Install the `modal` extra and authenticate the Modal SDK. The selected deployment
must expose `prepare_workspace` and `search_workspace` with LLGM's pinned
ColBERT checkpoint. Installing the client alone does not deploy that service.
This path is for applications with that service already available.

Before publication, install the client extra from the repository:

```sh
python -m pip install 'llgm[modal] @ git+https://github.com/simjay/llgm.git'
```

Authenticate using your Modal account's SDK configuration, then set the app
and environment names below. See [Modal's deployed-function guide](https://modal.com/docs/guide/trigger-deployed-functions)
for calling an existing deployment.

- `LLGM_RETRIEVER_BACKEND`, default `hybrid`. Hybrid source ranking or explicit `sqlite_fts5` for local BM25.

- `LLGM_RETRIEVAL_MODAL_APP`, default `llgm-colbert`. Authenticated Modal application.

- `LLGM_RETRIEVAL_MODAL_ENVIRONMENT`, default empty. Modal's configured default environment.

The first source search uploads the workspace's source snapshot for indexing.
After source appends, the next search uploads a complete updated source
snapshot and prepares another generation. This includes new questions and
replies saved by `answer()`. Source text, metadata and derived indexes remain
on the deployment's volume. Unchanged searches reuse the index.
This work can incur Modal charges. Graph browsing alone does not contact Modal.
See [node search](node-search.md) for fusion, small-corpus behavior, and indexing limits.

For a local setup without a semantic service, explicitly select BM25:

```sh
export LLGM_RETRIEVER_BACKEND=sqlite_fts5
```

Storage primitives such as `Evidence.open(workspace)` and direct `LLGM(...)`
construction retain local lexical retrieval unless supplied an evidence factory.
The settings factory owns configuration of the default hybrid application path.

## Use your own search backend

The configured indexes update from new workspace records automatically. To use
another source retriever, supply an async factory that returns an open `Evidence`
handle. This helper assumes the retriever already indexes sources in the workspace:

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

LLGM uses the factory for both maintenance and answers. Search results must carry
valid references to this workspace. LLGM reads their original text rather than
treating returned snippets as evidence. Inline journal notes remain searchable
in the local index.

You own the supplied retriever, including index updates and cleanup. LLGM closes
each returned evidence handle after use. Setting `LLGM_RETRIEVER_BACKEND` to a
name other than `hybrid` or `sqlite_fts5` requires this factory. An explicit
factory also overrides either built-in selection.
See [node search](node-search.md) for backend choices and passage references.

## Own directly constructed clients

`LLGM.from_settings()` closes the workspace and model clients it creates.
Construct `LLGM` directly to supply custom clients, configure provider-specific
options, or inject a graph model. You then own those resources.

Register cleanup as each resource is acquired so it also runs if a later step
fails:

```python
from contextlib import AsyncExitStack

from llgm import LLGM, Workspace
from llgm.models import create_model


async def ask_with_clients(main_model_id, reader_model_id, graph_model_id, question):
    """Use separate providers and close their clients after the answer."""
    async with AsyncExitStack() as stack:
        main = create_model("openai", main_model_id)
        stack.push_async_callback(main.aclose)
        reader = create_model("anthropic", reader_model_id)
        stack.push_async_callback(reader.aclose)
        workspace = await stack.enter_async_context(Workspace.open("./memory"))
        graph = create_model("openai", graph_model_id)
        stack.push_async_callback(graph.aclose)
        memory = LLGM(workspace, main, reader, graph_model=graph)
        return await memory.answer(question)
```

This helper needs both provider extras and credentials, plus the `rlm` extra and its
sandbox. When constructing a native adapter around an existing SDK client, the
adapter borrows it. Closing that adapter does not close the supplied SDK client.

## Storage choices

The default workspace stores source blobs locally and metadata in SQLite.
`LLGM_WORKSPACE_PATH` defaults to `./memory`. Unless overridden, blob files go in
`./memory/blobs/` and metadata in `./memory/metadata.sqlite3`. Relative paths are
resolved from the process working directory.

`LLGM_BLOB_BACKEND=s3` with `LLGM_BLOB_URI=s3://bucket/prefix/` selects the S3 blob
adapter, installed with the `s3` extra. S3 holds immutable objects. The SQLite
database still needs supported local storage.

`LLGM_DATABASE_URL=sqlite:////absolute/path/metadata.sqlite3` selects another
SQLite location. Postgres and distributed metadata are unsupported. An attempt
to open Postgres raises a capability error. S3 stores source objects only, so it
does not make the workspace a shared database for multiple machines.

The derived evidence index lives beside the metadata database in a directory
such as `metadata.sqlite3.indexes`. If restoring a metadata backup, close all
workspace users and remove the old derived index directory before reopening.
It will rebuild from authoritative records. Automated backup/restore, orphan
blob cleanup, and index compaction are not implemented.

### Existing databases

The current workspace metadata format is schema 5, with independent primary
edges and a compact operational journal. Schema-2 source blobs and journal record
identities remain valid within it. Opening an older workspace does not silently
change its graph meaning. Schema-1 versioned-source workspaces remain unsupported.
Schema 4 workspaces are rejected without changing their data and must be rebuilt
from their original inputs.

To copy an existing local schema-3 workspace into a new directory while
preserving original evidence:

```sh
llgm migrate ./old-memory ./new-memory
```

The source stays unchanged. Point `LLGM_WORKSPACE_PATH` at the new directory.
No automatic in-place migration runs when a workspace opens.

For a local schema-2 workspace, classify each reference-valued journal record
explicitly as `edge`, `amendment`, or `unresolved`, then copy it to a new location:

```sh
llgm migrate OLD_WORKSPACE NEW_WORKSPACE --journal-roles classification.json
```

The JSON file maps journal entry IDs to those roles. Only uncorrected whole-node
assertion links can become primary edges. Unresolved classifications preserve
history but block operational reads until resolved. The converter preserves the
original workspace, source IDs, original bytes, journal references, and retry
records. It refuses an existing destination and never guesses a pointer's role.
The [API reference](../reference/api.md#storage-transition) exposes the equivalent
Python function for the schema-2 copy. These converters handle only the described
local workspace formats.

## Query and validity time

Pass `as_of_ms` to `LLGM.answer()` and `valid_from_ms`/`valid_until_ms` in journal
or edge applicability. These machine fields use integer Unix milliseconds.
`parse_instant_ms("2026-09-11T12:00:00Z")` converts an explicit ISO instant.
Date-only conversion requires a supplied timezone and denotes a civil-day start.
Keep original source dates and question text separately when their precision is
unknown. `query_date` is model context and never implicitly sets `as_of_ms`.

The `scope` argument selects which edge and journal applicability rules are
relevant to the question. For example, `scope={"environment": "production"}`
can select a production-only correction. It does not filter all source searches
by metadata or enforce access control. Use separate workspaces or enforce access
in your application when source visibility must be restricted.

Append order decides which matching overwrite takes precedence. Its audit time
does not. Selecting a date for evidence also does not freeze the workspace.
An answer may observe records added while it runs.

## Runtime limits and usage

All branches and recursive children share one answer budget. These settings
control the answer limits:

- `LLGM_MAX_MODEL_CALLS`, default 40. Total model calls, including final synthesis.

- `LLGM_MAX_READER_CALLS`, default 36. Calls used to investigate evidence.

- `LLGM_MAX_GRAPH_CALLS`, default 8. Topic routing calls within an answer.

- `LLGM_MAX_SEARCHES`, default 8. Retrieval calls.

- `LLGM_MAX_EVIDENCE_TOKENS`, default 65536. Evidence exposed during the answer.

- `LLGM_MAX_BUNDLE_TOKENS`, default 8000. Selected evidence returned for synthesis.

- `LLGM_MAX_CONTEXT_TOKENS`, default 65536. Accounted context for a model call.

- `LLGM_MAX_OUTPUT_TOKENS`, default 2048. Requested output allowance per call.

- `LLGM_TIMEOUT_SECONDS`, default 120. Deadline for preparation and inference.

Use constructor options for depth, steps and operations, as shown above.
Connection proposals have a separate budget. Ordinary `LLGM` answers have no spending cap
in dollars.

Connection maintenance after a new conversational topic uses a separate
allowance, so the answer deadline is not a wall-clock bound for the entire
application call. Cleanup also runs after deadlines or cancellation and can
extend the time until the call returns.

The default text allowance counts UTF-8 bytes. It is a conservative accounting
unit rather than a provider tokenizer, despite the `_TOKENS` setting names.
The context check includes message formatting, an output schema when supplied,
and the requested output allowance. Provider-specific overhead can differ.
Actual provider usage is recorded separately. Missing usage and unpriced cost
remain unknown. A canceled request may still be billed by the provider.

Node traces identify seed selection, branches, parent/child invocations, Python
execution, returned evidence, and final synthesis. `capture_text=True` additionally
retains model output and code. Referenced evidence and metadata can still contain
source data when text capture is disabled. Treat traces as application data.

See [result handling](quickstart.md#understand-the-result) for answer status and
exception behavior. The [node execution API](../reference/api.md#effective-reads-and-node-execution)
accepts caller-selected seeds when your application owns retrieval.

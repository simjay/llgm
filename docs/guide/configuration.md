# Configuration and adapters

Start with local storage and configure the two model roles. The sidecar model
reads evidence and can investigate related nodes. The root model combines the
findings into an answer. Both roles can use the same provider or different ones.

`Settings.from_env()` reads environment variables. Use
`Settings.load(config_file="llgm.toml", overrides={...})` to add a TOML file or
Python overrides. These methods only parse settings. They do not create storage
or connect to a provider.

Values take precedence in this order:

1. Explicit Python overrides or CLI options such as `--workspace`
2. The selected TOML file
3. Existing process environment variables
4. Values added from the selected environment file
5. Library defaults

Only supplied fields replace lower-priority values. Model IDs and credentials
are captured when settings and clients are created. Changing a file or the
environment does not update an existing application.

## Local environment file

If your application uses a local `.env` file, create it with the required
credentials and model IDs. For OpenAI, the minimum is:

```text
OPENAI_API_KEY=your-api-key
LLGM_ROOT_MODEL=your-root-model-id
LLGM_SIDECAR_MODEL=your-sidecar-model-id
```

Replace the placeholders and keep this file out of version control. You can
export the same variables directly instead.

Load the selected file once at application startup:

```python
from llgm import Settings, load_env_file

load_env_file(".env")
settings = Settings.from_env()
```

Call `load_env_file()` before `Settings.load()` when also using TOML, and before
creating provider clients so they can read credentials. The helper adds missing
variables to the process environment. Existing variables win, including empty
values. It does not search for files or load them automatically. Omit the call
when your environment is already configured.

The equivalent CLI option is explicit and can be combined with a TOML file or
workspace override:

```sh
llgm ask "What database does Atlas production use?" --env-file .env
llgm ask "What database does Atlas production use?" --env-file .env --config llgm.toml --workspace ./memory
```

The file accepts one `KEY=value` assignment per line, optional `export`, blank
lines, and comments. Keys use letters, digits, and underscores, with no leading
digit. Single or double quotes preserve a single-line literal value. Unquoted
inline comments must be preceded by whitespace. Variables, commands, and escape
sequences are not expanded. Multiline values are unsupported. A missing file,
malformed assignment, or duplicate key raises a configuration error before any
variables are added.

## Providers and settings

Configure model IDs explicitly. This example uses OpenAI for the root and
Anthropic for the sidecar:

```text
LLGM_WORKSPACE_PATH=./memory
LLGM_ROOT_PROVIDER=openai
LLGM_ROOT_MODEL=your-root-model-id
LLGM_SIDECAR_PROVIDER=anthropic
LLGM_SIDECAR_MODEL=your-sidecar-model-id
```

Provider names are `openai`, `anthropic`, and `openai_compatible`. Install the
`openai` extra for OpenAI and compatible endpoints, or `anthropic` for Anthropic.
The defaults use OpenAI for both roles. The mixed-provider example needs both
extras and both `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`.

`LLGM_ROOT_API_KEY_ENV` and
`LLGM_SIDECAR_API_KEY_ENV` can name alternative credential variables. They contain
variable names, never credential values. `LLGM_ROOT_BASE_URL` and
`LLGM_SIDECAR_BASE_URL` select individual endpoints. A compatible endpoint
requires an explicit base URL.

The adapters return complete responses rather than streaming tokens. Try your
chosen models on representative questions. A working connection alone does not
show that a model can write the inspection code or use the evidence correctly.

A minimal TOML file uses lowercase field names:

```toml
[llgm]
workspace_path = "./memory"
root_provider = "openai"
root_model = "your-root-model-id"
sidecar_provider = "openai"
sidecar_model = "your-sidecar-model-id"
max_model_calls = 12
```

Pass the resulting settings to `LLGM.from_settings()` to open the application.
`settings.redacted()` reports values and field provenance with credential-bearing
URL components removed. Values loaded from an environment file have
`environment` provenance because they are read from the process environment.
Provider API keys are not stored in `Settings`.

Unknown application `LLGM_` keys, malformed values, unsupported providers, and
incompatible URI/backend combinations fail explicitly. `LLGM_TEST_*` and
`LLGM_REPL_*` belong to test and REPL configuration and are ignored when reading
the application's environment. They are not accepted as application fields in
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

Inference requires a running Docker daemon and a trusted local Python image.
`LLGM_NODE_REPL_IMAGE` selects the image, preferably by immutable digest. The
runtime never pulls it automatically. Generated code runs in isolated containers
with no host workspace mount or network access. Model generation and evidence
callbacks run on the host.

`LLGM_RETRIEVAL_K` bounds initial passage retrieval, while `LLGM_MAX_SEED_NODES`
bounds distinct admitted node owners. Require `max_seed_nodes <= retrieval_k <= 40`.
`LLGM_MAX_CONCURRENCY` queues excess admitted branches and bounds active model
calls. It does not silently select a smaller seed set. Explicit seed-limit skips
and operational failures remain visible as unresolved outcomes.

The complete operational journal is limited by `LLGM_MAX_JOURNAL_BYTES` per node.
`QueryEvidence` also limits its cached journals to 4 MiB of serialized data by
default. Irreducible overflow raises a budget error. Full historical storage can
continue growing. The current source-blob adapter loads each owning JSON source
on the host, even when the model requests only metadata or a small span.

## Application and maintenance limits

`LLGM.from_settings()` owns its configured workspace and clients. Its `Budget`
uses the inference limits in settings. Pass constructor options such as
`max_depth`, `max_steps`, and `max_operations` to control recursion. Maintenance
uses the sidecar client by default and has its own `MaintenancePolicy.budget`.
A directly constructed application can use a separate maintenance client.

```python
from llgm import Budget, LLGM, MaintenancePolicy


async def ask_with_limits(settings, question):
    """Answer with a shallow recursion limit and proposals that need review."""
    policy = MaintenancePolicy(
        mode="propose",
        max_candidates=4,
        budget=Budget(max_model_calls=4, max_sidecar_calls=4),
    )
    async with LLGM.from_settings(
        settings, maintenance_policy=policy, max_depth=2,
    ) as memory:
        return await memory.answer(question)
```

Maintenance settings apply when ingesting or organizing sources. They do not
change the inference budget. LLGM normally retrieves seeds before node inference.
An explicit `answer(..., node_id=...)` bypasses retrieval and supplies one seed.

An alternative retrieval backend requires an explicit `evidence_factory`.
The application uses the same factory for maintenance and answering. See
[custom search](quickstart.md#use-your-own-search-backend).

## Storage choices

The default workspace stores source blobs locally and metadata in SQLite.
`LLGM_BLOB_BACKEND=s3` with `LLGM_BLOB_URI=s3://bucket/prefix/` selects the S3 blob
adapter, installed with the `s3` extra. S3 holds immutable objects. The SQLite
database still needs supported local storage.

`LLGM_DATABASE_URL=sqlite:////absolute/path/metadata.sqlite3` selects another
SQLite location. Postgres and distributed metadata are unsupported. An attempt
to open Postgres raises a capability error. The S3 adapter has not been validated
against a live service.

The derived evidence index lives beside the metadata database in a directory
such as `metadata.sqlite3.indexes`. If restoring a metadata backup, close all
workspace users and remove the old derived index directory before reopening.
It will rebuild from authoritative records. Automated backup/restore, orphan
blob cleanup, and index compaction are not implemented.

### Existing databases

The current workspace metadata format is schema 3, with independent primary
edges and a compact operational journal. Schema-2 source blobs and journal record
identities remain valid within it. Opening an older workspace does not silently
change its graph meaning. Schema-1 versioned-source workspaces remain unsupported.

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
Python function. This is a narrow schema-2 copy operation, not general migration
or automatic source-version conversion.

## Query and validity time

Pass `as_of_ms` to `LLGM.answer()` and `valid_from_ms`/`valid_until_ms` in journal
or edge applicability. These machine fields use integer Unix milliseconds.
`parse_instant_ms("2026-09-11T12:00:00Z")` converts an explicit ISO instant.
Date-only conversion requires a supplied timezone and denotes a civil-day start.
Keep original source dates and question text separately when their precision is
unknown. `query_date` is model context and never implicitly sets `as_of_ms`.

Append order decides which matching overwrite takes precedence. Its audit time
does not. Selecting a date for evidence also does not freeze the workspace.
An answer may observe records added while it runs.

## Runtime limits and usage

All branches and recursive children share one answer budget. These settings
control the main limits:

| Environment variable | Default | Limits |
| --- | --- | --- |
| `LLGM_MAX_MODEL_CALLS` | 40 | Total model calls, including final synthesis |
| `LLGM_MAX_SIDECAR_CALLS` | 36 | Calls used to investigate evidence |
| `LLGM_MAX_SEARCHES` | 8 | Retrieval calls |
| `LLGM_MAX_EVIDENCE_TOKENS` | 65536 | Evidence exposed during the answer |
| `LLGM_MAX_BUNDLE_TOKENS` | 8000 | Selected evidence returned for synthesis |
| `LLGM_MAX_CONTEXT_TOKENS` | 65536 | Accounted context for a model call |
| `LLGM_MAX_OUTPUT_TOKENS` | 2048 | Requested output allowance per call |
| `LLGM_TIMEOUT_SECONDS` | 120 | Total answer time, including preparation |

Use constructor options for depth, steps and operations, as shown above.
Maintenance has a separate budget. Ordinary `LLGM` answers have no spending cap
in dollars.

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
exception behavior, and [execution mechanisms](architecture.md#execution-mechanisms)
for advanced execution interfaces.

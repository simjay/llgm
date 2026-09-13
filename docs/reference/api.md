# Public API

Start with `LLGM` to store conversations and answer questions. Use the lower-level
interfaces when your application needs to inspect sources, supply a retriever,
or own provider clients. The [quickstart](../guide/quickstart.md) shows a complete
application, and [configuration](../guide/configuration.md) explains the defaults.

| To do this | Start with |
| --- | --- |
| Open an application from environment or TOML settings | [`Settings`](#llgm.Settings) and [`LLGM.from_settings()`](#llgm.LLGM.from_settings) |
| Continue a chat or ask without saving turns | [`LLGM.answer()`](#llgm.LLGM.answer) and the [conversation guide](../guide/conversations.md) |
| Import earlier history | [`LLGM.ingest()`](#llgm.LLGM.ingest), with [`Conversation`](#llgm.Conversation) for metadata |
| Inspect an answer's status and sources | [`AnswerResult`](#llgm.AnswerResult) |
| Change limits for one answer | [`Budget`](#llgm.Budget) and the [override example](../guide/configuration.md#change-limits-for-one-answer) |
| Read an original source or record an explicit correction | [`Workspace`](#llgm.Workspace) |
| Search passages or inspect relationships | [`Evidence`](#llgm.Evidence) |
| Choose a provider or add your own client | [Models and embeddings](#models-and-embeddings) |
| Supply a different search backend | [Retrieval](#retrieval) and the [factory example](../guide/configuration.md#use-your-own-search-backend) |
| Browse stored topics and search results | [Graph viewer](#graph-viewer) |

## Application and settings

`LLGM.from_settings()` owns the workspace and clients it creates. Direct
construction borrows caller-owned resources. `Settings` only parses and validates
configuration. `Budget` describes a complete set of limits, so passing a new one
replaces the application's budget for that answer.

```{eval-rst}
.. autoclass:: llgm.LLGM
   :members:

.. autoclass:: llgm.Settings
   :members:

.. autofunction:: llgm.load_env_file

.. autoclass:: llgm.Budget
   :members:

.. autoclass:: llgm.AnswerResult
   :members:

.. autoclass:: llgm.IngestionOutcome
   :members:

.. autoclass:: llgm.MaintenancePolicy
   :members:

.. autoclass:: llgm.MaintenanceResult
   :members:
```

## Sources and evidence records

Use `Workspace` for original records and explicit changes. `Evidence` adds search
and graph access. A `SourceSpan` points into an original turn, while `NodeRef`
identifies a whole source and `JournalRef` identifies an appended journal entry.
The [evidence walkthrough](../guide/walkthrough.md) shows how these references
survive a correction.

```{eval-rst}
.. autoclass:: llgm.Workspace
   :members:

.. autoclass:: llgm.Evidence
   :members:

.. autoclass:: llgm.Conversation
   :members:

.. autoclass:: llgm.Turn
   :members:

.. autoclass:: llgm.SourceNode
   :members:

.. autoclass:: llgm.SourceSpan
   :members:

.. autoclass:: llgm.NodeRef
   :members:

.. autoclass:: llgm.JournalRef
   :members:

.. autoclass:: llgm.JournalEntry
   :members:

.. autoclass:: llgm.Edge
   :members:

.. autoclass:: llgm.Provenance
   :members:

.. autoclass:: llgm.ResolvedEvidence
   :members:

.. autoclass:: llgm.IngestResult
   :members:

.. autofunction:: llgm.parse_instant_ms
```

## Models and embeddings

`ModelClient` accepts a `ModelRequest` and returns a complete `ModelResponse`.
Native adapters translate that request to the provider API. To add another
provider, implement this interface or wrap an async function with
`CallableModelClient`. See [client ownership](../guide/configuration.md#own-directly-constructed-clients)
for cleanup and [provider settings](../guide/configuration.md#providers-and-settings)
for endpoint selection.

### Provider clients

```{eval-rst}
.. autoclass:: llgm.models.ModelClient
   :members:

.. autofunction:: llgm.models.create_model

.. autoclass:: llgm.models.OpenAIModelClient
   :members:
   :inherited-members:

.. autoclass:: llgm.models.AnthropicModelClient
   :members:
   :inherited-members:

.. autoclass:: llgm.models.OpenAICompatibleModelClient
   :members:
   :inherited-members:

.. autoclass:: llgm.models.CallableModelClient
   :members:
```

### Requests and responses

```{eval-rst}
.. autoclass:: llgm.models.Message
   :members:

.. autoclass:: llgm.models.ModelRequest
   :members:

.. autoclass:: llgm.models.ModelResponse
   :members:

.. autoclass:: llgm.models.ModelCapabilities
   :members:

.. autoclass:: llgm.models.Usage
   :members:
```

### Embedding clients

Dense retrieval accepts the small `Embedder` interface. `EmbeddingClient` also
exposes configuration details through a descriptor.

```{eval-rst}
.. autoclass:: llgm.retrieval.base.Embedder
   :members:

.. autoclass:: llgm.models.EmbeddingClient
   :members:

.. autoclass:: llgm.models.OpenAIEmbeddingClient
   :members:

.. autoclass:: llgm.models.EmbeddingEvent
   :members:
```

### Scripted responses

`ScriptedModelClient` is useful for testing application handling of chosen
responses. It does not call a model.

```{eval-rst}
.. autoclass:: llgm.models.ScriptedModelClient
   :members:
```

## Retrieval

A `Retriever` returns ranked `SearchHit` objects containing source passages and
references. It does not generate an answer. An injected retriever must index
sources from the application's workspace and remain current as sources change.
Configured applications use `WorkspaceHybridRetriever` for BM25 and ColBERT
search on Modal. The local `sqlite_fts5` option uses `SQLiteBM25Retriever`.
See [node search](../guide/node-search.md) for ranking and seed selection.

```{eval-rst}
.. autoclass:: llgm.retrieval.Retriever
   :members:

.. autoclass:: llgm.retrieval.SearchPassage
   :members:

.. autoclass:: llgm.retrieval.SearchHit
   :members:

.. autoclass:: llgm.retrieval.SQLiteBM25Retriever
   :members:

.. autoclass:: llgm.retrieval.ExactDenseRetriever
   :members:

.. autoclass:: llgm.retrieval.HybridRetriever
   :members:

.. autofunction:: llgm.retrieval.split_nodes

.. autofunction:: llgm.retrieval.corpus_fingerprint
```

### Configured workspace search

The settings factory and CLI viewer share a workspace retriever across their
evidence handles. It refreshes source generations on search. Direct
`Evidence.open(workspace)` keeps local BM25 as its default. See
[search configuration](../guide/configuration.md#search-backend).

```{eval-rst}
.. autoclass:: llgm.retrieval.workspace.WorkspaceHybridRetriever
   :members: search, descriptor
```

### ColBERT and PLAID

The local adapter requires its optional dependencies and a prepared checkpoint.
The Modal adapter calls an already deployed retrieval service. Neither an import
nor a fixed-index connection builds an index. This differs from configured
workspace search, which prepares a generation lazily. Keep the checkpoint,
tokenizer, passage mapping and configuration consistent when reopening an index.
`ExactColBERTRetriever` uses the same encoder for corpora below 64 passages.

```{eval-rst}
.. autoclass:: llgm.retrieval.colbert.ColBERTConfig
   :members:

.. autoclass:: llgm.retrieval.colbert.ColBERTRetriever
   :members: build, open, search, descriptor

.. autoclass:: llgm.retrieval.modal.ModalColBERTRetriever
   :members: connect, search, descriptor

.. autoclass:: llgm.retrieval.colbert_exact.ExactColBERTRetriever
   :members: search, descriptor

.. autoclass:: llgm.retrieval.ColBERTTokenizer
   :members:

.. autoclass:: llgm.retrieval.DiagnosticTokenizer
   :members:

.. autofunction:: llgm.retrieval.validate_encoder_text
```

## Storage adapters

`BlobStore` operations are synchronous. `Workspace` invokes them on its storage
worker, with immutable bytes addressed by SHA-256 digest. S3 changes where blobs
are stored. It does not replace SQLite metadata or provide distributed access to
a workspace.

```{eval-rst}
.. autoclass:: llgm.storage.BlobStore
   :members:

.. autoclass:: llgm.storage.LocalBlobStore
   :members:

.. autoclass:: llgm.storage.S3BlobStore
   :members:
```

## Journal interpretation and graph operations

Use these functions when you need to inspect journal interpretation or review
proposed links before publishing them. `accept_link()` checks the proposal's
structure and references. Your application remains responsible for deciding
whether the relationship is meaningful.

```{eval-rst}
.. autoclass:: llgm.memory.evidence.EvidencePassage
   :members:

.. autoclass:: llgm.memory.evidence.EvidenceSearchHit
   :members:

.. autoclass:: llgm.memory.evidence.TraversalResult
   :members:

.. autoclass:: llgm.memory.evidence.JournalInterpretation
   :members:

.. autofunction:: llgm.memory.evidence.interpret_journal

.. autoclass:: llgm.memory.evidence.LinkProposal
   :members:

.. autofunction:: llgm.memory.evidence.propose_links

.. autofunction:: llgm.memory.evidence.accept_link
```

## Effective reads and node execution

`QueryEvidence.initialize_node()` loads complete local read metadata.
`source_info()` provides paginated turn handles, while `read_segments()` returns
original or replacement evidence with its own references. `read()` accepts only
results with one such segment. Keep separate references for multi-segment reads.

```{eval-rst}
.. autoclass:: llgm.memory.query.QueryEvidence
   :members:

.. autoclass:: llgm.inference.nodes.NodeRuntime
   :members:

.. autoclass:: llgm.inference.nodes.NodeSeed
   :members:
```

The integrated application uses `NodeRuntime`. Generated Python can call
`source_info`, `read`, `edges`, `search`, and `query_node` through host callbacks.
See [architecture](../guide/architecture.md) for the complete question flow.

## Storage transition

```{eval-rst}
.. autofunction:: llgm.memory.migration.copy_schema3_workspace

.. autofunction:: llgm.memory.migration.copy_schema2_workspace
```

Both converters preserve the original local workspace and copy it to schema 5.
The schema-2 converter additionally requires explicit journal classifications. See
[existing databases](../guide/configuration.md#existing-databases).

## Errors

Configuration and preparation failures can raise these exceptions. An error
handled inside inference may instead appear in the returned answer status.
See [result handling](../guide/quickstart.md#understand-the-result).

```{eval-rst}
.. autoclass:: llgm.core.errors.LLGMError

.. autoclass:: llgm.core.errors.ConfigurationError

.. autoclass:: llgm.core.errors.CapabilityError

.. autoclass:: llgm.core.errors.ConflictError

.. autoclass:: llgm.core.errors.ReferenceResolutionError

.. autoclass:: llgm.core.errors.BudgetExceeded

.. autoclass:: llgm.core.errors.ProviderError

.. autoclass:: llgm.core.errors.SchemaError
```

## DSPy Python execution

`SandboxConfig` controls byte admission and interpreter deadlines for the
integrated application. `interpreter_factory` accepts a DSPy `CodeInterpreter`
factory. Each factory call must return a fresh interpreter. The default is
DSPy's Deno/Pyodide `PythonInterpreter` without host access permissions.

Node readers use `query_node(node_id, question)` to start another reader with
isolated context. DSPy's `llm_query(prompt)` makes a plain reader-model call.
All calls, including extraction after `max_steps` action iterations, share the
answer budget. See [the answer flow](../guide/architecture.md#how-an-answer-runs).

```{eval-rst}
.. autoclass:: llgm.inference.repl.SandboxConfig
   :members:

.. autoclass:: llgm.inference.repl.REPLError

.. autoclass:: llgm.inference.repl.REPLTimeoutError
```

## Graph viewer

See the [graph viewer guide](../guide/graph-viewer.md) for CLI usage and browsing.

```{eval-rst}
.. autoclass:: llgm.viewer.GraphViewer
   :members: from_application, search, serve_forever
```

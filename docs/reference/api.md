# Public API

Use `LLGM` for the integrated application. The [quickstart](../guide/quickstart.md)
shows setup and result handling, while [concepts](../guide/concepts.md) explains
sources, primary edges, local amendments, and node delegates.

## Application and evidence records

```{eval-rst}
.. automodule:: llgm
   :members:
   :imported-members:
```

## Models and embeddings

Model adapters translate provider-neutral requests and responses. Embedding
clients expose vector generation and a descriptor for reproducibility.

```{eval-rst}
.. automodule:: llgm.models
   :members:
   :imported-members:
```

## Retrieval

`Retriever` describes ranked source-passage search. `Embedder` is the minimal
vector-generation protocol accepted by dense retrieval. The richer
`EmbeddingClient` above also describes its encoder configuration.

```{eval-rst}
.. automodule:: llgm.retrieval
   :members:
   :imported-members:

.. autoclass:: llgm.retrieval.base.Embedder
   :members:

.. autoclass:: llgm.retrieval.modal.ModalColBERTRetriever
   :members: connect, search, descriptor
```

## Storage adapters

`BlobStore` operations are synchronous. `Workspace` invokes them on its storage
worker, with immutable bytes addressed by SHA-256 digest.

```{eval-rst}
.. automodule:: llgm.storage
   :members:
   :imported-members:
```

## Journal interpretation and graph operations

The following functions expose journal interpretation and explicit proposal
publication. They preserve evidence history and do not establish semantic truth.

```{eval-rst}
.. automodule:: llgm.memory.evidence
   :members: EvidencePassage, EvidenceSearchHit, TraversalResult, JournalInterpretation, interpret_journal, LinkProposal, propose_links, accept_link
```

## Effective reads and node execution

`QueryEvidence.initialize_node()` loads complete local read metadata.
`source_info()` provides paginated turn handles, while `read_segments()` returns
canonical original or replacement evidence. `read()` accepts only results with
one canonical segment. Multi-segment effective reads must retain their separate
references.

```{eval-rst}
.. automodule:: llgm.memory.query
   :members: QueryEvidence

.. automodule:: llgm.inference.nodes
   :members: NodeRuntime, NodeSeed
```

The integrated application uses this Python node runtime. Generated code can call
`source_info`, `read`, `edges`, `search`, and `query_node` through host callbacks.
See the [walkthrough](../guide/walkthrough.md) for examples of these operations.

## Storage transition

```{eval-rst}
.. automodule:: llgm.memory.migration
   :members: copy_schema2_workspace
```

The converter preserves the original local schema-2 workspace and requires
explicit classifications before copying its records into schema 3. See
[existing databases](../guide/configuration.md#existing-databases).

## Errors

```{eval-rst}
.. automodule:: llgm.core.errors
   :members:
```

## Specialized execution interfaces

Use these interfaces directly when you need to supply context or control retrieval
outside the integrated application. They require caller-owned clients and have
their own operations and stopping rules. Use `LLGM.from_settings()` for the
ordinary ingestion and answering workflow.

### Structured node queries

```{eval-rst}
.. automodule:: llgm.inference.recursive
   :members: RecursiveRuntime
```

### Iterative retrieval

Construct `EvidenceSidecar` with its model, retriever, and a declared search
policy, then pass it and the root model to `IterativeRuntime`. The caller closes
the model and retrieval resources.

```{eval-rst}
.. automodule:: llgm.inference.iterative
   :members: IterativeRuntime, EvidenceSidecar
```

### Docker Python execution

```{eval-rst}
.. automodule:: llgm.inference.repl
   :members:

.. automodule:: llgm.inference.rlm
   :members: RLMRuntime, RLMResult
```

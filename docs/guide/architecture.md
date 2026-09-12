# Architecture

This tutorial follows an answer through LLGM. It builds on the fictional Atlas
project from [concepts](concepts.md): Database records the original database choice,
Update changes that choice, Backups gives the retention period, and Registry
records the deployment region.

Your application uses `LLGM` to store conversations and ask questions. Models
choose what to read and what to conclude. The library supplies the stored text,
applies explicit amendments, enforces limits and tracks citations.

## Set up the parts your application uses

Four objects determine most of the behavior:

| Object | What it controls |
| --- | --- |
| `Workspace` | Stored conversations, relationships and journals. |
| `Settings` | Storage and model configuration. |
| `Budget` | Limits on work performed for one answer. |
| `LLGM` | Ingestion, optional organization and answering. |

`LLGM.from_settings()` creates a workspace and model clients from your settings.
Using it as an async context manager also closes those resources when you leave
the block. If you construct `LLGM` with your own workspace and clients, your
application is responsible for closing them.

The default answer runtime uses Docker to run the Python written by delegates.
The model calls can use hosted providers. This separates where a model generates
instructions from where those instructions execute. See the
[quickstart](quickstart.md) for the prerequisites and a complete program.

## Store and organize the conversations

Calling `LLGM.ingest()` first saves a conversation as an immutable source.
Store Database, Update, Backups and Registry separately so each has its own
identity and original text. A later observation needs a new node. Reusing an
existing ID with different content raises a conflict.

After saving the source, ingestion can ask a maintenance model to find related
nodes and propose edges. For example, it might connect Database to Registry
because both describe the Atlas deployment. `MaintenancePolicy` controls this
step:

| Mode | Behavior |
| --- | --- |
| `validated` | Publish proposals whose references and relationship types pass the configured checks. This is the default. |
| `propose` | Return proposals for your application to review. |
| `disabled` | Skip relationship discovery. |

These checks establish that a proposal has valid references and an allowed
relationship type. The model can still misunderstand the relationship.

The ingestion result reports source storage and maintenance separately. If
maintenance fails, the source stays saved. `ingest(..., organize=False)` skips
that step for one call. You can use `organize()` later. Retrying ingestion may
run maintenance again, even when the source was already saved.

Explicit journal amendments are separate from this edge discovery. An amendment
on Database can point to Update as the replacement for the PostgreSQL statement.
Ordinary automatic organization does not create that amendment.

## How an answer runs

For the question "Which database does Atlas production use, and how long are
backups kept?", the flow is:

```{mermaid}
flowchart TD
    Question[Your question] --> Search[Search passages and choose seed nodes]
    Search --> Database[Delegate reads Database]
    Search --> Backups[Delegate reads Backups]
    Database --> Child[Optional query to another node]
    Child --> Database
    Database --> Evidence[Selected excerpts and findings]
    Backups --> Evidence
    Evidence --> Root[Root writes the answer]
    Root --> Result[Answer, references and unresolved needs]
```

### 1. Find starting nodes

LLGM retrieves ranked passages before making an inference model call. It then
walks that ranking and selects distinct node owners until it reaches
`max_seed_nodes`.

If the top passages belong to Database, Database, Backups and Registry, a seed
limit of two selects Database and Backups. Multiple high-ranking passages
from Database do not consume both slots. The selector does not add their
passage scores into a new node score.

`retrieval_k` controls how many passage hits enter this selection. It is separate
from the seed limit. Passing `answer(..., node_id=...)` bypasses initial search
and supplies that node as the only seed. The
[node search guide](node-search.md) explains this stage in more detail.

### 2. Start a delegate for each selected node

Every selected seed gets a delegate. `max_concurrency` limits how many seed
branches can run at once, so extra branches wait for a slot.

Each delegate starts with the question, source references, a small page of turn
metadata and the node's complete operational journal. The metadata gives turn
roles and coordinates for choosing what to read. It does not contain the full
conversation text.

The delegate writes Python to request source slices and print observations.
It can keep data in its interpreter without putting all of it into the model's
context. In the example, Database's delegate reads the database choice with its
applicable amendment, while Backups' delegate reads the retention period.

Reading a small span limits how much text reaches the model. The current storage
adapter still loads the owning source into application memory to resolve that
span. A large source can therefore require substantial application memory even
when the delegate prints only a short excerpt.

### 3. Investigate a related node when needed

Inside the interpreter, `edges()` exposes relationships and their target nodes.
`search()` can find more source references. A delegate can read a reference
directly, or use `query_node(node_id, question)` to start a child delegate for
a focused investigation.

If the question also asks for the deployment region, Database's delegate could
query Registry. The child uses the same reading mechanism and returns selected
findings, excerpts and unresolved needs to its parent. Its Python variables and
working messages remain local. Only `query_node` starts a child. Looking up an
edge does not start another model by itself.

All children share the answer's limits. A stored edge is a route for finding
evidence, not a requirement to visit every connected node. No separate runtime
graph has to be built or stored before answering.

The [evidence walkthrough](walkthrough.md) shows the delegate's read and query
operations in a working example.

### 4. Return evidence to the root

Each seed branch returns selected findings with exact supporting excerpts.
The root receives those excerpts with speaker roles, dates and source
references, followed by the branch summaries. It does not receive every
passage that was retrieved or read along the way.

The root makes one final model call to combine the returned evidence. In the
current `LLGM` pipeline, that call has no further query phase. If neither a seed
nor a child returned the backup policy, the root has no supported retention
period to use.

The library checks that cited evidence was returned by a branch and resolves
its references to the stored sources. It cannot establish that a quotation
supports every claim the root makes.

### 5. Handle the result

`AnswerResult` contains the answer, evidence references, unresolved needs,
usage and a trace of execution. Read the status alongside the answer text.
A fluent answer can still be partial.

For example, Database might establish the database choice while Backups' delegate
fails before reading the policy. The successful findings remain available, and
the failed work is recorded. The result may establish MySQL while leaving
retention unresolved. A delegate's local missing-fact note can be resolved by
another branch. An unresolved amendment or unrecovered operation failure remains
visible in the result even if the root omits it from its prose.

An empty retrieval has no starting evidence and produces a partial result.
An explicit abstention can also have empty answer text with an explanation.
See [result handling](quickstart.md#understand-the-result) for application code.

## Keep work within a budget

Seeds and their children share one `Budget`. More seeds give the system more
places to investigate, but leave less work available per branch unless you
also raise the limits.

The runtime reserves capacity for delegates to return their findings and for
the root to write the answer. Those reservations use the existing budget.
They do not guarantee completion when a provider fails, Docker stops or the
deadline expires.

Evidence and model context also have size limits. If a required journal or
selected evidence return is too large, the runtime reports the problem instead
of silently cutting it to fit. These work limits do not enforce a dollar
spending cap. See [runtime limits](configuration.md#runtime-limits-and-usage)
for the configurable units and usage fields.

## Read updates while preserving their sources

Before returning source text, the evidence reader applies explicit journal
amendments that match the query's scope and time. A replacement of Database's
PostgreSQL statement returns the selected MySQL text from Update with Update's
reference. Unchanged parts of Database keep their original references.

Missing replacement text, overlapping amendments and replacement cycles leave
an unresolved result. The reader does not silently present overwritten text
as current when the amendment cannot be resolved. The full journal history
remains available, and `Workspace.resolve()` can still inspect the original
source.

The operational journal removes redundant working entries while retaining the
history. It has a finite size limit, so a journal that cannot fit fails
explicitly. This does not bound the amount of history stored on disk.

Queries read currently stored evidence. They can observe new appends while
running. `as_of_ms` selects explicitly declared validity periods, but it does
not reconstruct the workspace as it existed at an earlier moment. Human date
text in `query_date` is separate from that machine-readable time selector.
See [updates and corrections](walkthrough.md#follow-an-update-and-a-correction)
for an example with scope and time.

## Replaceable interfaces

Start with the default components, then replace the part your application needs:

| Interface | Purpose |
| --- | --- |
| `ModelClient.complete()` | Call a generation model through a provider adapter. |
| `Retriever.search()` | Rank source passages for a query. |
| `BlobStore.put()` and `BlobStore.get()` | Store and retrieve immutable source bytes. |

The default retriever uses a local SQLite FTS5 index. It reuses the index and
refreshes it for new evidence. The first build still reads existing sources,
and storage and search costs grow with the corpus.

To attach a different retriever to `LLGM`, supply an async `evidence_factory`
that opens evidence access for the workspace. Returned source references must
belong to that workspace. LLGM resolves their text from the original stored
sources and keeps local journal retrieval available. Injected retrievers remain
owned by your application. See
[custom search](quickstart.md#use-your-own-search-backend).

The optional Modal ColBERT adapter sends queries to a remote GPU service. Index
construction is a separate explicit operation. Connecting the retriever does
not upload the workspace or build an index. The
[search guide](node-search.md) explains when to use this backend.

## Execution mechanisms

`LLGM` uses `NodeRuntime` for the pipeline described above. The package also
exposes lower-level executors for applications that need to supply their own
context or control retrieval directly:

| Executor | What it runs |
| --- | --- |
| `NodeRuntime` | Seed delegates, recursive Python investigation and final root synthesis. |
| `RecursiveRuntime` | Structured node operations expressed as JSON. |
| `IterativeRuntime` | Retrieval under a caller-supplied search policy. |
| `RLMRuntime` | Recursive Python queries over caller-supplied context. |

These are separate interfaces with different operations and stopping rules.
They are not modes selected through `LLGM` settings. Their callers own the
model clients they supply. The
[advanced API](../reference/api.md#specialized-execution-interfaces)
documents their signatures.

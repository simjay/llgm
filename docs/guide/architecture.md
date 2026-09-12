# Architecture

LLGM turns a question into a set of local investigations. Search finds starting
conversations, model readers inspect their evidence and ask follow-up questions,
and a final model combines the returned excerpts into an answer. The stored
history can grow without requiring every answer to place that entire history in
one prompt.

There are two kinds of work here. **Ingestion** preserves conversations and can
connect them to related sources. **Answering** chooses what to investigate for a
particular question. The relationships persist between questions. The readers
and their working contexts last for one answer.

We will follow the team conversations from [concepts](concepts.md). Database
records the database choice, Backups records retention, and Registry records
the hosting region. A later Update conversation changes the database choice.
The application stores these through `LLGM.ingest()` and asks about them through
`LLGM.answer()`.

## Store first, then organize

`ingest()` saves the original conversation before attempting to organize it.
Each source has its own identity. Submitting different content under an existing
node ID raises a conflict rather than replacing the stored conversation.

Organization searches for candidate conversations and asks a maintenance model
to propose relationships supported by their evidence. For example, it might
connect Database to Registry because both describe the same deployment. As new
conversations arrive, this builds routes that later readers can investigate.

`MaintenancePolicy` controls what happens to those proposals:

| Mode | Behavior |
| --- | --- |
| `validated` | Publish proposals that pass reference and allowed-relationship checks. This is the default. |
| `propose` | Return proposals for your application to review. |
| `disabled` | Skip relationship discovery. |

The default checks establish that a proposal uses resolvable references and an
allowed relationship label. The relationship itself is a model judgment. Choose
`propose` if your application needs to review that judgment before publication.

The ingestion result reports storage and maintenance separately. For example,
Registry can be stored successfully even if the maintenance model is unavailable.
Its text is still searchable, and you can organize it later.

Use `ingest(..., organize=False)` to skip organization for one call, or call
`organize()` later. Retrying ingestion can run maintenance again even if the
source was already saved. These operations discover edges. Your application
records exact journal amendments separately. Answering can use these records
and search for additional sources, but does not automatically publish new edges.

## How an answer runs

Consider this question:

> Which database does production use, where is it hosted, and how long are backups kept?

```{mermaid}
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 24}}}%%
flowchart TD
    Question[Question] --> Seeds[Search and select starting nodes]
    Seeds --> Database[Database delegate]
    Seeds --> Backups[Backups delegate]
    Database --> Child[Registry child delegate]
    Child --> Database
    Database --> Root[Root combines evidence and findings]
    Backups --> Root
    Root --> Result[Answer and evidence]
```

The diagram shows one possible investigation. A different question or passage
ranking can produce different seeds, and a delegate only asks a child when it
decides that child will help.

### 1. Choose the starting nodes

The library searches passages, then selects their distinct owning nodes in
ranking order. These starting nodes are called seeds. With a limit of two,
a ranking of Database, Database, Backups and Registry selects Database and
Backups.

`retrieval_k` limits the passage pool and `max_seed_nodes` limits the number
of starting nodes. Passing `answer(..., node_id=...)` bypasses this search and
uses the named node as the only seed. The [node search guide](node-search.md)
explains how to tune selection and inspect skipped nodes.

### 2. Read each seed locally

Every selected seed gets a node delegate. `max_concurrency` limits concurrent
seed branches, so additional branches wait for a slot.

A delegate starts with the question, references, a small page of turn metadata
and the node's complete operational journal. Imagine a long Database conversation
with hundreds of turns about setup, staging and production. The initial metadata
offers speaker roles and coordinates for choosing spans, without sending all
those turns to the model.

The delegate writes Python to read evidence and print observations. The model
sees the output, decides whether it has the needed fact, and can write another
operation. Its interpreter can retain text and perform calculations without
placing every intermediate value in the model's context. Database's delegate
can inspect the production decision while Backups' delegate reads the retention
period.

LLGM runs this generated Python in isolated Docker interpreters. Evidence access
and model requests remain in the host runtime. When you use a hosted provider,
the evidence presented to a model is sent to that provider. The
[quickstart](quickstart.md) covers setup.

Before returning a source span, LLGM applies relevant journal amendments. If
Database's PostgreSQL statement has an explicit replacement pointing to Update,
an effective read returns the MySQL text with Update's reference. Unchanged
text keeps its original references.

### 3. Ask another node when needed

Inside a delegate's interpreter, `edges()` lists applicable outgoing
relationships and `search()` finds additional references. The delegate can read
a reference directly or call `query_node(node_id, question)` to start a child.

Database might find the database choice but need Registry to establish the
hosting region. It can ask that node, "Where is production hosted?" The child
uses the same reading mechanism and returns its region finding, selected source
evidence and any unresolved needs to its parent. Its Python variables stay in
its interpreter, and its model messages stay local to that child in the host
runtime.

This is how the graphical-model inspiration becomes an execution method. Each
reader does local work, then passes a bounded result along a useful relationship.
A child can ask its own child. The resulting delegation tree belongs to this
question and can use only a small portion of the persistent evidence graph.

Only `query_node()` starts a child. Searching or inspecting an edge does not
start one automatically. Children share the answer's budget. There is no
requirement to visit every connected node or to construct a separate graph
before answering.

The [walkthrough](walkthrough.md#choose-a-related-node) shows these interpreter
operations and distinguishes them from application Python.

### 4. Combine the returned evidence

The root receives exact excerpts selected by the branches, with source
references and available speaker and date metadata. Branch summaries follow
those excerpts, so the root can check a finding against what was actually said.
Retrieved or read passages that a branch did not return are not available to the
root.

In our example, the branches need to return the current database choice, the
region statement and the seven-day backup policy. Finding a conversation called
Backups is not enough. A delegate has to read and return the relevant policy text.

The root makes one final model call. It has no further tool phase in the
current `LLGM` pipeline. If no branch returns the backup policy, the root has
no supported retention period to use.

The library checks that the root's citations identify evidence returned by a
branch. It cannot establish that the cited text supports every claim in the
answer.

### 5. Check the outcome

`AnswerResult` includes answer text, evidence references, unresolved needs,
usage and an execution trace. These let your application show the answer,
identify the original statements behind it, and distinguish a completed
investigation from one that ran out of time or left a source unread.

For example, Database may return the database choice while Backups fails before
reading its policy. Successful findings remain available, and unrecovered
failures remain visible. Another branch can resolve a delegate's missing-fact
note, but the root cannot silently remove a required failure report.

An empty search produces a partial result without starting delegates. Nodes
skipped by the initial seed limit also make the result partial. The
[quickstart](quickstart.md#understand-the-result) shows result handling, including
budget exhaustion and explicit abstention.

## Understand the limits on work and storage

Seeds and children share one `Budget`. Increasing the number of seeds does
not increase that budget. More branches leave less available work per branch
unless you also raise the limits.

The runtime reserves capacity for delegates to return findings and for the root
to write an answer. A provider failure, interpreter failure or expired deadline
can still prevent completion. Required journals and selected evidence must fit
their limits. The runtime reports a size problem rather than silently cutting
those records to fit.

These limits control model calls, evidence, context and time. They do not impose
a dollar spending cap. See [runtime limits](configuration.md#runtime-limits-and-usage)
for the units and usage fields.

Small model inputs also do not guarantee small storage reads. Resolving a short
span currently loads its owning source into application memory. Operational
journals are loaded in full, and retained source and journal history can keep
growing on disk.

## Understand what a query sees

Reads use currently stored evidence. New appends can become visible during an
answer. The query is not a snapshot of the workspace at the moment it started.

`as_of_ms` selects declared validity periods for edges and amendments. It does
not reconstruct an earlier version of the workspace. `query_date` is separate
human-readable date text for the model and does not set that machine-readable
selector.

Missing replacement evidence, overlapping amendments and replacement cycles
leave unresolved evidence. The reader does not silently use an overwritten
value as current when its replacement cannot be resolved.
`Workspace.resolve()` still reads the original stored source. The
[correction walkthrough](walkthrough.md#follow-an-update-and-a-correction)
demonstrates the difference with scope and time.

## The parts your application owns

The normal entry point handles the pieces above together. These objects become
useful when you want to configure the pipeline or provide a component yourself:

| Object | Responsibility |
| --- | --- |
| `LLGM` | Coordinates ingestion, optional organization and answering. |
| `Workspace` | Stores original conversations, directed edges and journals. |
| `Settings` | Selects storage, model providers and runtime limits. |
| `Budget` | Limits the work performed for one answer. |

`LLGM.from_settings()` creates the workspace and model clients. Its async context
manager closes them when your application is done with that instance. If you
pass your own workspace and clients to `LLGM`, your application owns and closes
them. See [resource ownership](configuration.md#own-directly-constructed-clients).

## Replace one component at a time

| Interface | What you can replace |
| --- | --- |
| `ModelClient.complete()` | Generation through a different model provider. |
| `Retriever.search()` | Passage ranking for the stored evidence. |
| `BlobStore.put()` and `BlobStore.get()` | Storage of immutable source bytes. |

The default retriever uses a reusable SQLite FTS5 index. It refreshes newly
published evidence. Its first build reads existing sources, and corpus growth
still increases storage and indexing work.

A custom retriever enters through an async `evidence_factory`. Its references
must resolve to evidence in the workspace. LLGM uses the stored text when
reading those references and keeps local journal retrieval available. Your
application owns the injected retriever and keeps its index up to date. See
[custom search](configuration.md#use-your-own-search-backend).

`LLGM` uses `NodeRuntime` for the pipeline described here. The
[advanced API](../reference/api.md#specialized-execution-interfaces) also exposes
lower-level execution interfaces for applications that supply their own context
or retrieval policy. They have separate contracts and are not settings that
change the default answer pipeline.

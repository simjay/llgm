# Architecture

LLGM turns a question into a set of local investigations. The current conversation
topic is the first reading target, and search can add other starting nodes.
Model readers inspect their evidence and ask follow-up questions,
and a final model combines the returned excerpts into an answer. The stored
history can grow without requiring every answer to place that entire history in
one prompt.

`answer()` is the primary application entry point. It stores new user turns,
continues or selects a topic, runs local readers, and stores the returned reply.
`ingest()` catches up on earlier conversation batches without generating a reply.
The [conversation guide](conversations.md) covers inputs, retries and persistence.
This page follows how LLGM investigates a question after topic selection.

## Keep a topic together

Each conversation ID remembers the node for its current topic. The ID belongs
to the chat, so it stays the same when the topic changes. Topic routing sees a
bounded preview of recent active turns and retrieved candidate nodes. It prefers
the active topic, can return to another existing topic, and creates a node only
for a clear topic change. Length, elapsed time, a session boundary, and a related
subtopic are not reasons to split. The routing model can still misclassify a topic.
Imported batches route as complete units, without internal segmentation.

Each new turn is stored in an immutable blob with stable coordinates. Appending
does not rewrite earlier turns or create another graph node. Turn metadata can
be paged without loading the entire appended conversation. A span read loads its
own turn. A single enormous turn still requires loading that turn's blob.

The active topic is always the first default seed, even when a follow-up lacks
searchable keywords or uses `remember=False`. The pointer belongs to the selected
`conversation_id`, so another chat's activity does not change it. Additional seeds
and recursive children can investigate other nodes. Only selected evidence
reaches the final main model context.

## Model responsibilities

A small graph model chooses whether incoming turns extend an existing
topic or begin a clearly unrelated topic. It also proposes generic connections
between nodes. The reader model performs recursive reading and interprets
what those connections mean for the current question. The main model produces
the final answer. All three roles have independent model and provider settings.
`MaintenancePolicy` controls automatic organization, and `MaintenanceResult`
reports its outcome. Each recursive child uses the reader model with a fresh
working context.

## Connect topics

New answer topics and ordinary imports can trigger bounded connection discovery.
Connections carry source references explaining where they came from. For
example, a connection from Database to Registry helps a reader find a deployment
record. Its relevance to the question is decided during reading.

By default, LLGM checks proposed connections for valid structure and source
references before publishing them. The application can request proposals for
review or disable automatic organization through
[maintenance controls](conversations.md#control-automatic-organization).
These checks do not establish that a connection is meaningful. Exact journal
amendments remain separate, explicit operations.

## Preserve conversation and evidence history

Incoming turns are saved before inference. Generation failure does not remove
them. A nonempty returned reply is appended with the assistant role. Previous
assistant replies remain fallible history, not independent factual confirmation.
Read-only `answer(..., remember=False)` performs no conversation writes.
An explicit `node_id` is available in that read-only mode.

Conversation mutation is serialized for callers sharing a Workspace object.
Independent handles must be coordinated by the application. The workspace
persists active topic IDs across restarts. It does not provide response replay,
whole-answer transactions, or a snapshot of the graph.

## How an answer runs

Consider this question:

> Which database does production use, where is it hosted, and how long are backups kept?

```{mermaid}
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 24}}}%%
flowchart TD
    Question[Question] --> Seeds[Current topic first, then search for other seeds]
    Seeds --> Database[Database delegate]
    Seeds --> Backups[Backups delegate]
    Database --> Child[Registry child delegate]
    Child --> Database
    Database --> Main[Main combines evidence and findings]
    Backups --> Main
    Main --> Result[Answer and evidence]
```

The diagram shows one possible investigation. A different question or passage
ranking can produce different seeds, and a delegate only asks a child when it
decides that child will help.

### 1. Choose the starting nodes

The library selects the current topic first, then fills remaining slots from
distinct owners of ranked passages. These starting nodes are called seeds.
If Database is current, a two-node limit selects Database and the highest-ranked
other owner, even if Database has no matching passage. Its reader receives recent
turn references directly from the session pointer.

A one-node limit skips initial retrieval when the current topic is known.
Reading that topic can also proceed when routing has spent the search allowance.
A read-only question without a current topic uses passage retrieval alone.

`retrieval_k` limits the passage pool and `max_seed_nodes` limits the number
of starting nodes. Passing `answer(..., remember=False, node_id=...)` bypasses this search and
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

Each delegate is a [DSPy RLM](https://dspy.ai/api/modules/RLM/) with its own
Deno/Pyodide interpreter. DSPy handles Python actions, observations, persistent
variables and `SUBMIT` results. LLGM validates each submitted citation against
evidence accessed by that invocation. Evidence access
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
a reference directly or call `query_node(node_id, question)` to start a child. The
child is another DSPy RLM with a fresh context. DSPy also supplies `llm_query`
and `llm_query_batched` for ordinary model calls on selected text. These calls
use the same reader model and shared budget. They do not start recursive node
readers.

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

The main model receives exact excerpts selected by the branches, with source
references and available speaker and date metadata. Branch summaries follow
those excerpts, so the main model can check a finding against what was actually said.
Retrieved or read passages that a branch did not return are not available to the
main.

In our example, the branches need to return the current database choice, the
region statement and the seven-day backup policy. Finding a conversation called
Backups is not enough. A delegate has to read and return the relevant policy text.

The main model makes one final model call. It has no further tool phase in the
current `LLGM` pipeline. If no branch returns the backup policy, the main model has
no supported retention period to use.

The library checks that the main model's citations identify evidence returned by a
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
note, but the main model cannot silently remove a required failure report.

In read-only mode without a current topic, an empty search produces a partial
result without starting delegates. Nodes skipped by the initial seed limit also
make the result partial. The
[quickstart](quickstart.md#understand-the-result) shows result handling, including
budget exhaustion and explicit abstention.

## Understand the limits on work and storage

Seeds and children share one `Budget`. Increasing the number of seeds does
not increase that budget. More branches leave less available work per branch
unless you also raise the limits.

The runtime reserves capacity for delegates to return findings and for the main model
to write an answer. A provider failure, interpreter failure or expired deadline
can still prevent completion. Required journals and selected evidence must fit
their limits. The runtime reports a size problem rather than silently cutting
those records to fit.

These limits control model calls, evidence, context and time. They do not impose
a dollar spending cap. See [runtime limits](configuration.md#runtime-limits-and-usage)
for the units and usage fields.

Small model inputs also do not guarantee small storage reads. Appended source
spans load one turn. Explicit immutable imports and legacy sources still load
their base source blob into application memory. Operational
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
| `LLGM` | Saves conversations, routes topics, organizes links and coordinates answers. |
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

Configured applications use hybrid BM25 and ColBERT search through Modal.
Source appends prepare a new source generation on the next search. Inline
journals retain local SQLite retrieval. See [node search](node-search.md) for
indexing costs, exact scoring for small corpora, and the explicit offline option.

A custom retriever enters through an async `evidence_factory`. Its references
must resolve to evidence in the workspace. LLGM uses the stored text when
reading those references and keeps local journal retrieval available. Your
application owns the injected retriever and keeps its index up to date. See
[custom search](configuration.md#use-your-own-search-backend).

`LLGM` uses `NodeRuntime` for the pipeline described here. Applications that
already own evidence access and seed selection can call
[`NodeRuntime.answer()`](../reference/api.md#effective-reads-and-node-execution)
directly with `NodeSeed` records. Both entry points use the same DSPy reader
loop, shared budgets and citation validation.

# Conversations and imports

Use `answer()` for an ongoing chat and `ingest()` for a conversation that already
happened elsewhere. Both preserve original messages and let LLGM group related
discussion into topic nodes.

The snippets below belong inside an open `async with LLGM.from_settings()`
block. See [Quickstart](quickstart.md) for a complete program and setup.

## Continue a chat

```python
await memory.answer("Atlas production uses PostgreSQL.", conversation_id="atlas")
result = await memory.answer("What database does it use?", conversation_id="atlas")
print(result.answer)
```

The first call stores the statement and a reply. The second stores your
follow-up, reads relevant evidence and stores its reply. Follow-ups start at
the conversation's current topic, so "What about its backups?" does not have
to repeat searchable topic words.

| Identity | What it means |
| --- | --- |
| `conversation_id` | Your stable chat ID. It defaults to `"default"` and remembers the current topic across restarts. |
| `node_id` | A stored topic selected by LLGM. Several sessions or chats can contribute related turns to it. |

Use different conversation IDs when chats need separate current-topic pointers.
They still search a shared workspace. For separate source visibility, use
separate workspaces or enforce access in your application.

## Send only new turns

`answer()` accepts text or a sequence of new user and assistant turns:

```python
result = await memory.answer(
    [{"role": "user", "content": "Keep Atlas backups for seven days."}],
    conversation_id="atlas",
)
```

Mappings accept `content` or `text`. Typed `Turn` records also work. The sequence
must end with a nonempty user message, which becomes the question to answer.
Do not resend the full transcript or replies LLGM already saved. Each supplied
turn is treated as new evidence.

The interface accepts text, without system instructions, tool messages,
multimodal content or streaming. It returns an `AnswerResult`, with status,
references and usage. General conversation can produce a reply without
citations. Stored assistant replies remain fallible history and are not
independent confirmation of a fact.

## How topics grow

The graph model compares incoming turns with a bounded preview of the current
topic and retrieved candidates. It can continue that topic, return to another
existing topic, or create a topic for a clearly unrelated discussion.

A database decision, its backup policy and a later correction can stay together.
A question about a holiday itinerary may begin another topic. Length, time gaps,
session boundaries and related subtopics are not intended reasons to split.
Routing is a model decision and can be wrong.

New turns append without rewriting earlier text or source references. There is
no target topic size or automatic splitting of a very long turn. Readers inspect
selected passages and can request more turn metadata when needed. See
[architecture](architecture.md#keep-a-topic-together) for storage behavior.

## Import earlier history

Use `ingest()` to save supplied history without generating a reply:

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
print(outcome.source.node_id)
print(outcome.maintenance.status)
```

Each imported batch is routed as one unit. Split a mixed-topic transcript into
appropriate batches yourself when boundaries matter. LLGM preserves exact text
and roles, and can reuse a topic across related sessions.

Use `Conversation.from_turns()` for source metadata, turn IDs and a timestamp:

```python
from llgm import Conversation, parse_instant_ms

history = Conversation.from_turns(
    [{"role": "user", "text": "Atlas backups are retained for seven days."}],
    metadata={"source": "operations-chat"},
    timestamp_ms=parse_instant_ms("2026-09-11T12:00:00Z"),
)
outcome = await memory.ingest(history, conversation_id="atlas")
```

Metadata and the timestamp apply to that batch's turns. Import dated sessions
separately to retain their distinct dates. Appended turn IDs receive a unique
batch prefix, so repeated source-local IDs do not collide within a topic.

An explicit `Conversation.node_id` takes the exact source-import path instead.
It bypasses topic routing, preserves the supplied source identity and does not
move a conversation's pointer. `Workspace.ingest()` provides low-level source
storage without model organization. See the
[evidence walkthrough](walkthrough.md#store-search-and-read-without-a-model).

## Retry an import or handle a failed answer

An ingestion `idempotency_key` prevents the same batch from being appended again.
Keep the input and conversation ID unchanged when retrying. Changed input under
the same key raises a conflict. Connection maintenance may run again even when
storage reuses the saved batch.

`answer()` has no retry key or response replay. Repeating a call adds another
message. Once input has been stored, an inference failure does not undo it.
Only nonempty returned answer text is appended as an assistant reply. Validation
and routing can fail before storage, so an exception alone does not establish
whether the input was saved.

Conversation updates serialize for callers sharing one `Workspace` object.
Independent handles need application coordination for writes to the same chat.
An entire answer is not one storage transaction, and reads are not snapshots.

## Ask without adding to history

```python
result = await memory.answer(
    "How long are its backups kept?",
    conversation_id="atlas",
    remember=False,
)
```

This reads from the current Atlas topic without saving the question, a reply,
or a new pointer. Search can add other starting nodes. Without a current topic,
retrieval supplies all starting nodes.

Read-only questions must be plain text. To choose one starting node explicitly,
also pass `node_id`. It overrides the current topic and skips initial retrieval.
Its reader can still search and follow links. See [node search](node-search.md)
for the full selection rule.

## Control automatic organization

Topic placement and connection discovery are separate operations. Imports can
propose connections after storage. Conversational answers run connection
discovery after answering when they created a new topic.

Set a `MaintenancePolicy` mode for the application, or use `organize=False`
for an individual import:

| Control | Effect |
| --- | --- |
| `mode="validated"` | Default. Validate proposed connections structurally and publish accepted links. |
| `mode="propose"` | Return connection proposals for review. Topic routing still runs. |
| `mode="disabled"` | Skip model routing and connection discovery. Append to this conversation's active topic, or create one if needed. |
| `organize=False` on `ingest()` | Skip this import's connection discovery. Topic routing still runs. |

Organization does not create exact journal amendments. Record those explicitly
when your application knows which statement a correction replaces. See the
[correction example](walkthrough.md#follow-an-update-and-a-correction).

Routing in `answer()` shares the answer budget. Import routing and connection
discovery use maintenance allowances. Inspect `result.usage["maintenance"]` when
connection maintenance ran, or `outcome.maintenance` after an import. Import
routing usage appears under its `usage["topic_routing"]` key. A maintenance
failure leaves stored evidence intact. See
[configuration](configuration.md#application-and-maintenance-limits) to set limits.

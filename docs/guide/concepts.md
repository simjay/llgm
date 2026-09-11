# Concepts

LLGM helps a language model answer questions using stored conversations. It keeps
the original text, finds useful parts when a question arrives, and lets models
investigate related conversations before writing an answer.

This tutorial follows one question through those ideas. You do not need to know
graph theory or have a model account to follow it. The
[quickstart](quickstart.md) shows how to use the library in Python.

## Start with a growing conversation history

Suppose you have these conversations about a project called Atlas:

| Conversation | What it records |
| --- | --- |
| Database | Atlas production uses PostgreSQL. Staging uses SQLite. |
| Update | Atlas production has moved to MySQL. |
| Backups | Atlas production retains backups for seven days. |
| Registry | Atlas production runs in eu-west-1. |

Later, you ask:

> Which database does Atlas production use, and how long are backups kept?

With four short statements, you could send everything to one model. As the
history grows, that approach sends more unrelated text with every question.
A model has a limited **context**, which is the text available to it while
producing a response.

LLGM stores the history outside that context. Models request useful excerpts
as they work. The challenge becomes finding and reading enough evidence to
answer the question within a chosen budget.

## 1. Store each conversation as a node

A **source node** holds one conversation, including its turns, speaker roles,
metadata and optional event time. Database, Update, Backups and Registry become
four nodes. These names are labels for the example. The library can generate
an ID for each node.

Sources are **immutable**, meaning their original text stays unchanged after
ingestion. Update is a new node even though it changes a fact from Database.
This lets you inspect what was originally said and where a later correction
came from.

LLGM refers to an exact excerpt with a **source span**. Think of it as an address
containing the node ID, the turn ID and the character range within that turn.
This address is what makes a quotation traceable to stored text.

## 2. Search for passages, then choose starting nodes

Search ranks **passages**, which are short pieces of stored evidence. A passage
about Atlas can be useful even when most of its conversation discusses something
else.

LLGM uses the ranked passages to choose their owning nodes as **seeds**.
Seeds are the starting points for this answer. For our question, search might
choose Database and Backups. A seed is a conversation to investigate. It does
not have to contain the whole answer.

The [node search guide](node-search.md) explains how passage rankings become
seeds and how to choose a search backend.

## 3. Give each seed a reader

A **node delegate** is a model assigned to investigate one node for a question.
It can make several model calls as it reads evidence and decides what to do next.
The delegate at Database looks for the current database choice. The delegate
at Backups looks for the retention period. Each keeps its own working context.

A delegate can ask to read a particular source span instead of receiving the
whole conversation. It does this by writing short Python instructions that
LLGM runs in an isolated interpreter. The model sees the resulting printed
observations and decides what to inspect next.

The model assigned to this role is called the **sidecar model** in configuration.
You can choose a smaller model for repeated reading tasks and a stronger model
for the final answer. Both roles can also use the same model.

## 4. Follow a relationship when more context is needed

Suppose you also ask which region hosts Atlas. A stored **edge** from Database
to Registry can tell the Database delegate where to investigate that part of
the question. The delegate can ask Registry a focused question:

> Which deployment region is recorded?

A child delegate reads Registry and returns its findings with the supporting
excerpts. That child can ask another node a question if needed. This is the
recursive part of LLGM. A delegate can also search for more evidence when there
is no useful edge.

Only selected findings and evidence return to the parent. The child's whole
working conversation stays local. These delegates are temporary model calls.
You do not need a running service for every stored node.

## 5. Record corrections without losing the original

An edge helps a reader find another conversation. To apply a precise correction,
LLGM uses a **journal**, which is a node's record of notes and explicit amendments.

Your application can record an amendment on Database saying that its PostgreSQL
statement should now be read using the MySQL statement in Update. When that
amendment applies, LLGM returns the replacement text with Update's source
reference. The original PostgreSQL statement remains available for inspection.
The unrelated staging statement still says SQLite.

An edge from Database to Update alone does not apply this correction. Automatic
organization can propose or publish relationships, but it does not turn every
new statement into a journal amendment. The
[evidence walkthrough](walkthrough.md#follow-an-update-and-a-correction) shows
how to record an exact replacement in Python.

Before a delegate reads a node, LLGM loads its **operational journal**. This is
the set of journal entries needed to interpret current reads, including notes
and applicable changes. Redundant entries can leave this working set while
the full journal history stays stored. LLGM does not currently delete old
sources or implement an automatic forgetting policy.

## 6. Combine the findings into an answer

The **root model** receives the selected original excerpts and the delegates'
findings. With the explicit amendment above, those include Update's MySQL
statement and Backups' seven-day retention period. The root can combine them:

> Atlas production uses MySQL and retains backups for seven days.

The result includes references to the supporting evidence. A valid reference
shows which stored text was used. You should still check whether that text
supports the answer, especially when the sources disagree.

Each stage can lose useful information. Search might miss Backups, a delegate
might read the wrong turn, or the root might combine facts incorrectly. LLGM
reports unresolved evidence and unfinished work alongside the answer so your
application can handle incomplete results.

## How this relates to graphical models

The stored nodes and edges form an evidence graph. The design borrows the idea
of doing work locally and passing limited information between related parts.
That is the connection to graphical models.

LLGM does not define probabilities or factor functions over its nodes. Its
messages are findings and source references, and model-generated findings can
be wrong. You can use the library without treating it as a probabilistic model
or assuming that repeated messages converge to a correct answer.

Continue with [architecture](architecture.md) to follow the same example through
storage, search, model calls and the returned result.

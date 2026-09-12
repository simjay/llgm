# Concepts

LLGM helps a language model answer questions from stored conversations. It keeps
the original text outside the model's prompt, searches for useful evidence, and
lets models read related conversations before writing an answer.

This page introduces the pieces through a team's database discussions. The
[architecture tutorial](architecture.md) then follows a request through the
library, and the [evidence walkthrough](walkthrough.md) shows the operations in
Python.

## Store the conversation

Suppose your team has three conversations:

| Conversation | What it records |
| --- | --- |
| Database | Production uses PostgreSQL. Staging uses SQLite. |
| Backups | Production backups are retained for seven days. |
| Registry | Production runs in eu-west-1. |

You later ask:

> Which database does production use, and how long are backups kept?

You could send all three conversations to a model. With hundreds of longer
conversations, that also sends a lot of unrelated text. A model's **context**
is the text available to it while producing a response, and that context has a
size limit.

LLGM stores each conversation as a **source node**. A node contains the turns,
speaker roles and metadata, with an ID that identifies it in storage. The
Database and Backups names here are labels for the example.

Sources are **immutable**. Their original text stays unchanged after ingestion.
If the team changes databases later, you store the new conversation as another
node. You can still inspect what the team said earlier.

## Find a starting point

Search works with **passages**, which are small pieces of stored evidence.
A useful paragraph can rank well even when the rest of its conversation is
about something else.

LLGM uses the passage ranking to choose a few owning nodes as **seeds**. A seed
is a starting point for investigation. For this question, Database and Backups
would be useful seeds.

A passage is the unit of search. A node is the unit of conversation storage and
delegation. Searching passages helps find the relevant part of a long node
without sending the whole node to a model.

## Give each starting node a reader

A **node delegate** is a model assigned to investigate one node for the question.
The Database delegate looks for the production database. The Backups delegate
looks for the retention period. Each has its own working context.

A delegate writes short Python instructions to request evidence. LLGM runs the
instructions in an isolated interpreter and sends the printed observations back
to the model. The delegate can repeat this process as it decides what to read.
You configure the model for this work as the **sidecar model**.

When a delegate reads text, LLGM retains a **source span** identifying the node,
turn and character range it came from. That reference lets your application
trace a quotation back to its original source.

The **root model** receives the delegates' selected excerpts and findings, then
writes the final answer. You can use a smaller model for repeated reading and
a stronger model for synthesis, or the same model for both roles.

## Connect conversations when one points to another

Now suppose the question also asks where production is hosted. An **edge** from
Database to Registry records a relationship between those nodes. The edge has
a direction and a relationship label, such as `deployment_registry`.

The Database delegate can inspect that edge and ask a child delegate:

> Which deployment region is recorded in Registry?

The child reads Registry and returns selected findings with their evidence.
It can ask another node a question in the same way. This is **recursive
inference**. The child's entire working conversation does not enter its
parent's context.

An edge provides a route to evidence. It does not start a model call by itself.
Delegates can also search for other nodes, so a useful source does not need an
edge from the initial seed. Delegates exist only while answering. Stored nodes
do not each need their own running model service.

## Apply a correction without erasing the old statement

The team later says that production has moved to MySQL. Store that statement
in a new Update node.

A node's **journal** records notes and explicit amendments to how its evidence
should be read. Your application can add an amendment on Database that replaces
its exact PostgreSQL span with the MySQL span from Update. An effective read
then returns MySQL with Update's reference. The staging statement still says
SQLite, and the original PostgreSQL text remains available.

An edge from Database to Update would help a delegate find the update, but
would not apply that replacement. Edges establish relationships. Explicit
journal amendments change effective reads. Automatic organization currently
creates relationship proposals, not these exact amendments.

Before a delegate reads a node, LLGM loads its **operational journal**, the working set
of entries needed to interpret its evidence. Redundant entries can leave that
working set while the full journal history remains stored. This does not delete
old conversations or provide an automatic forgetting policy.

## Check the answer and its evidence

With the correction recorded and both facts returned by delegates, the root
can answer:

> Production uses MySQL and retains backups for seven days.

The result contains the answer, supporting references and unresolved needs.
A reference tells you which stored text was cited. It does not by itself prove
that the text supports the claim.

Search can miss a useful conversation, a delegate can overlook a passage, and
the root can draw the wrong conclusion. Read the result's status and evidence
alongside its answer. The [quickstart](quickstart.md#understand-the-result)
shows how to do that in an application.

## Why a graph?

The nodes and edges form an evidence graph. The design borrows an idea from
graphical models: investigate related parts locally and exchange limited
messages between them.

LLGM does not assign probabilities or factor functions to its nodes. Its
messages are model-generated findings and source references. It does not
guarantee that repeated messages converge to a correct answer.

Continue with [architecture](architecture.md) to see which parts the library
handles and which decisions the models make.

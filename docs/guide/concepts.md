# Concepts

LLGM stands for **Large Language Graphical Model**. It explores a way for language
models to work with a history that is larger than the context they can read at
once: keep the conversations, connect related evidence, and let local readers
ask one another focused questions.

The motivation starts with an ordinary experience. A decision made in one
conversation is qualified in another and changed weeks later. Answering a
question about that decision requires more than finding a sentence that uses the
right words. You may need to follow the discussion, distinguish two environments,
and work out which statement still applies.

Sending the entire history on every question makes the model input grow with
the archive. A running summary is smaller, but its author has to choose what to
keep before knowing every future question. Passage search gives you relevant
excerpts, but the first matches may only point toward the answer. LLGM keeps the
original evidence available and makes retrieval the beginning of an investigation.

This page develops that idea through a team's database discussions. The
[architecture tutorial](architecture.md) follows the same example through the
runtime, and the [evidence walkthrough](walkthrough.md) shows the operations in
Python.

## Why a graph?

Think of the team's conversations as places where different parts of an answer
live. The database discussion knows which system was chosen. A deployment
discussion knows where it runs. A later migration discussion explains what
changed. Relationships between these conversations give a reader a reason to
visit one after reading another.

The graphical-model inspiration is **local computation and passing messages
along relationships**, as developed in
[factor graphs and the sum-product algorithm](https://www.isiweb.ee.ethz.ch/papers/arch/aloe-2001-1.pdf).
In LLGM, a local reader investigates one conversation and can ask a reader at
another conversation for help. It receives a bounded set of findings and source
excerpts, rather than the other reader's entire working history. A final reader
combines what the branches established.

These messages contain language-model findings and evidence references. LLGM
does not define probabilistic factors or perform belief propagation with a
convergence guarantee. The useful idea is how to divide the reading work while
keeping its conclusions connected to their sources.

The persistent graph is also distinct from the work done for one question.
Conversations and their relationships stay in storage. Readers are created when
needed and finish when they return their findings. A stored conversation does
not require a model service that runs continuously.

## Store the conversation

Suppose your team has three conversations:

| Conversation | What it records |
| --- | --- |
| Database | Production uses PostgreSQL. Staging uses SQLite. |
| Backups | Production backups are retained for seven days. |
| Registry | Production runs in eu-west-1. |

You later ask:

> Which database does production use, and how long are backups kept?

LLGM keeps related discussion in a **topic node**. Several sessions can share
a node. A node contains the turns,
speaker roles and metadata, with an ID that identifies it in storage. The
Database and Backups names here are labels for the example.

Published turns are **immutable**. New turns append to the same topic while
earlier text and source-span references stay unchanged.
This matters when the team changes databases later. You can add the new decision
and still recover the earlier statement, who made it, and the conversation
around it.

## Find a starting point

Search works with **passages**, which are small pieces of stored evidence.
A useful paragraph can rank well even when the rest of its conversation is
about something else.

LLGM uses the passage ranking to choose a few owning nodes as **seeds**. A seed
is a starting point for investigation. For this question, Database and Backups
would be useful seeds.

A passage is the unit of search. A node is the unit of topic storage and delegation. Its size does not trigger
automatic splitting. The RLM can inspect selected portions of a very long topic. Searching passages helps find the relevant part of a long node
without sending the whole node to a model.

## Give each starting node a reader

A **node delegate** is a model assigned to investigate one node for the question.
The Database delegate looks for the production database. The Backups delegate
looks for the retention period. Each has its own working context.

A delegate works in a read, compute and inspect loop. It writes Python to request
evidence, LLGM runs that code in an isolated interpreter, and the model reads the
printed observations before choosing its next step. For example, the Database
reader can inspect production statements without filling its input with a long
discussion about staging.

This draws on [Recursive Language Models](https://arxiv.org/abs/2512.24601), which
treat context as something a program can inspect. Text can stay in interpreter
variables until the delegate prints the portions it wants to consider. The
model's **context** is its current input, including those printed observations,
and still has a size limit. You configure the model used for local reading as the
**reader model**.

When a delegate reads text, LLGM retains a **source span** identifying the node,
turn and character range it came from. That reference lets your application
trace a quotation back to its original source.

The **main model** receives the delegates' selected excerpts and findings, then
writes the final answer. The roles let you choose one model for repeated local
reading and another for synthesis, or use the same model for both.

## Connect conversations when one points to another

Now suppose the question also asks where production is hosted. An **edge** from
Database to Registry records a relationship between those nodes. The edge has
a direction, supporting evidence, and no relationship type. It records a connection.
The RLM reader determines what the connection means for its question.

The Database delegate can inspect that edge and ask a child delegate:

> Which deployment region is recorded in Registry?

The child reads Registry and returns the region statement with its source
reference. It can ask another node a question in the same way. This is
**recursive inference**. The parent receives the selected findings and evidence,
while the child's intermediate reads and model conversation stay local.

An edge provides a route to evidence. The delegate decides whether that route
helps answer its question. It can also search for other nodes when the existing
relationships do not lead to what it needs. Search, direct reading and recursive
questions work together, so each answer does not have to traverse the whole
graph.

## Apply a correction without erasing the old statement

The team later says that production has moved to MySQL. Store that statement
in a new Update node.

A node's **journal** records notes and explicit amendments to how its evidence
should be read. Your application can add an amendment on Database that replaces
its exact PostgreSQL span with the MySQL span from Update. An effective read
then returns MySQL with Update's reference. The staging statement still says
SQLite, and the original PostgreSQL text remains available.

An edge from Database to Update helps a delegate find the new discussion.
An explicit journal amendment does something more precise: it tells the reader
which original span to replace and where to find its replacement. Automatic
organization currently proposes relationships between nodes. Your application
records these exact amendments separately.

Before a delegate reads a node, LLGM loads its **operational journal**, the working
set of entries needed to interpret its evidence. Redundant entries can leave
that working set while the full journal history remains stored. The original
conversations remain in storage as well.

## Check the answer and its evidence

With the correction recorded and both facts returned by delegates, the main model
can answer:

> Production uses MySQL and retains backups for seven days.

The result contains the answer, supporting references and unresolved needs.
A reference tells you which stored text was cited. It does not by itself prove
that the text supports the claim.

Search can miss a useful conversation, a delegate can overlook a passage, and
the main model can draw the wrong conclusion. Read the result's status and evidence
alongside its answer. The [quickstart](quickstart.md#understand-the-result)
shows how to do that in an application.

Continue with [architecture](architecture.md) to see which parts the library
handles and which decisions the models make.

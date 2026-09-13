# Evidence walkthrough

Start by storing two conversations, finding a passage and following a link.
Then apply an exact correction without changing the original text. Both examples
need only the installed core package. The final section shows how delegates use
these operations while answering through DSPy sandboxes.

The examples use a team's production database and backup policy.
[Concepts](concepts.md) introduces the terms used here.

## Store, search and read without a model

With Python 3.11 or later and Git, install the core package in your Python
environment:

```sh
python -m pip install 'llgm @ git+https://github.com/simjay/llgm.git'
```

Save the complete example below as `evidence.py` and run `python evidence.py`.
It creates a temporary workspace with two sources, adds an explicit relationship,
then searches and reads the stored evidence. It needs no credentials or sandbox runtime.

```python
import asyncio
from tempfile import TemporaryDirectory

from llgm import Conversation, Provenance, Workspace
from llgm.memory.evidence import Evidence


async def main():
    """Store two notes and inspect their evidence without model calls."""
    with TemporaryDirectory() as directory:
        async with Workspace.open(directory) as workspace:
            database = await workspace.ingest(Conversation.from_turns([
                {"role": "user", "text": "Production database uses PostgreSQL."},
            ]))
            backups = await workspace.ingest(Conversation.from_turns([
                {"role": "user", "text": "Production backups are kept for seven days."},
            ]))
            await workspace.publish_edge(
                database.node_id,
                backups.node_id,
                provenance=Provenance("user", "evidence-example"),
            )
            async with await Evidence.open(workspace) as evidence:
                hits = await evidence.search("production database", k=1)
                for reference in hits[0].passage.refs:
                    record = await evidence.read(reference)
                    print("found:", record.text)
                targets = await evidence.neighbors(database.node_id)
                print("linked conversations:", len(targets))


asyncio.run(main())
```

Expected output:

```text
found: Production database uses PostgreSQL.
linked conversations: 1
```

`Workspace.ingest()` stores a source without running model-based organization.
`Evidence.search()` ranks passages, while `Evidence.read()` returns the original
text at a reference. The edge you published connects Database to Backups.
`neighbors()` returns its target reference without reading that target or
starting a model.

The temporary workspace is removed when the example finishes. For persistent
storage, open a directory your application keeps, as in the quickstart.

## Follow an update and a correction

Suppose an original conversation says:

> Production uses PostgreSQL. Staging uses SQLite.

A later conversation says:

> Production now uses MySQL.

We want production queries from September 11 onward to use MySQL while retaining
the original conversation and leaving the staging statement unchanged.

Save the following **standalone application script** as `correction.py` and run
it with `python correction.py`. It needs only the installed core package. It
creates a temporary workspace and makes no model calls or sandbox requests.

```python
import asyncio
from tempfile import TemporaryDirectory

from llgm import Conversation, Provenance, SourceSpan, Workspace, parse_instant_ms
from llgm.memory.evidence import Evidence
from llgm.memory.query import QueryEvidence


async def main():
    """Apply a scoped replacement and inspect both effective and original text."""
    with TemporaryDirectory() as directory:
        async with Workspace.open(directory) as workspace:
            old_text = "Production uses PostgreSQL. Staging uses SQLite."
            new_text = "Production now uses MySQL."
            old = await workspace.ingest(Conversation.from_turns([
                {"role": "user", "turn_id": "decision", "text": old_text},
            ]))
            new = await workspace.ingest(Conversation.from_turns([
                {"role": "user", "turn_id": "update", "text": new_text},
            ]))

            old_start = old_text.index("PostgreSQL")
            new_start = new_text.index("MySQL")
            subject = SourceSpan(
                old.node_id, "decision", old_start, old_start + len("PostgreSQL")
            )
            replacement = SourceSpan(
                new.node_id, "update", new_start, new_start + len("MySQL")
            )
            await workspace.append_journal(
                old.node_id,
                subject=subject,
                record_kind="overwrite",
                relation="replace",
                value=replacement,
                provenance=Provenance("user", "correction-example", (replacement,)),
                applicability={
                    "scope": {"env": "production"},
                    "valid_from_ms": parse_instant_ms("2026-09-11T00:00:00Z"),
                },
            )

            async with await Evidence.open(workspace) as evidence:
                query = QueryEvidence(
                    evidence,
                    scope={"env": "production"},
                    query_date=None,
                    as_of_ms=parse_instant_ms("2026-09-11T12:00:00Z"),
                )
                parts = await query.read_segments(
                    SourceSpan(old.node_id, "decision", 0, len(old_text))
                )
                effective = "".join(part.text for part in parts)
                assert effective == "Production uses MySQL. Staging uses SQLite."
                assert any(part.reference == replacement for part in parts)
                original = await workspace.resolve(subject)
                assert original.text == "PostgreSQL"
                print("effective:", effective)
                print("original:", original.text)


asyncio.run(main())
```

Expected output:

```text
effective: Production uses MySQL. Staging uses SQLite.
original: PostgreSQL
```

### What the replacement identifies

`subject` identifies the exact PostgreSQL span in the old conversation.
`replacement` identifies the MySQL span in the new one. `SourceSpan` offsets
follow Python string indexing, starting at zero and excluding the end position.

The journal entry records who supplied the correction through `Provenance`.
Its `value` points to the replacement evidence. `QueryEvidence.read_segments()`
returns the unchanged parts with their original references and the replacement
with its new reference. It does not create a modified copy of the old source.
`workspace.resolve(subject)` always reads the original PostgreSQL text.

### When it applies

The example uses an explicit scope of `env=production` and a validity start of
September 11. A different scope or an earlier `as_of_ms` leaves this patch
inactive. Omitting scope or time needed to decide applicability produces
unresolved evidence. Missing replacement evidence is also unresolved rather
than silently returning the old value as current.

`as_of_ms` selects a declared validity period. It is not a snapshot of what
was stored on that date. `query_date` is optional date text for a model, so this
model-free example leaves it unset.

To revise the replacement, append another `overwrite` with the same exact
subject, relation and scope. Among matching entries that apply to the query,
the latest append wins. A `correction` can instead target a previous
`JournalRef`, for example to retract that patch. Neither operation deletes
original sources or journal history. See the
[journal API](../reference/api.md#journal-interpretation-and-graph-operations)
for the record types and methods.

## Follow one answer

Download {download}`offline.py <../../examples/offline.py>` and save it as
`offline.py`. It uses real SQLite storage and DSPy sandboxes with scripted
model responses, so the same reads and returns happen each time. It makes no
hosted calls and needs no model credentials.

Install the sandbox extra and run the saved script. The first run can download
Deno/Pyodide runtime assets:

```sh
python -m pip install 'llgm[rlm] @ git+https://github.com/simjay/llgm.git'
python offline.py
```

The script creates these sources in a temporary workspace. The names below
are explanatory labels. Actual node IDs are generated identities.

| Source | Text and role |
| --- | --- |
| Database | Production uses PostgreSQL. Staging uses SQLite. Selected as a seed. |
| Backups | Production backups are retained for seven days. Selected as another seed. |
| Registry | The deployment region is eu-west-1. Reached through an edge from Database. |
| Update | MySQL. Used as the evidence for an exact replacement of Database's PostgreSQL span. |

The script asks for the production database, backup retention and region.
Follow these steps through its answer:

1. This read-only example has no current conversation pointer, so search selects
   Database and Backups. Each gets a delegate.
2. Database's delegate reads its source with the journal amendment applied.
   Production now says MySQL, while staging still says SQLite. The replacement
   text points back to Update.
3. Database follows its generic connection to Registry and asks a child to read
   Registry. The child returns eu-west-1 with its source reference.
4. Backups returns the seven-day retention statement independently.
5. The main model combines the selected excerpts and findings into a cited answer.

The script checks those behaviors, including that the original PostgreSQL
statement remains readable. It removes its temporary workspace when done.
The model decisions in this example are scripted to make the sequence visible.
Hosted models choose their own reads and can miss useful evidence. The
[quickstart](quickstart.md) shows how to configure those models.

## Choose a related node

The following snippets show **code inside a delegate's interpreter**. LLGM
provides the functions and `context` variable there. They are not standalone
application scripts.

An edge gives a delegate a generic connection and a target node. The reader
interprets why Database and Registry matter to the current question and can ask
the target for more evidence:

```python
links = edges()
for edge in links["edges"]:
    finding = query_node(
        edge["reference"]["node_id"],
        "Which deployment region is recorded?",
    )
    print(finding)
```

`edges()` returns descriptions under `edges` and unique target references under
`references`. It omits withdrawn edges and those that do not apply to the query's
scope or time. `query_node()` starts the child investigation. Looking up the
edge alone does not read its target.

The child has its own interpreter and working conversation. The parent receives
only the returned findings, evidence and unresolved needs.

## Inspect a large node without printing it all

Every delegate receives `context` with the question, references, the journal
used to interpret the node, and a first page of turn metadata under `source_page`.
The metadata describes turns without loading all their text into the prompt.

`source_info()` supplies turn IDs, speaker roles, lengths and span coordinates.
This interpreter snippet reads at most 400 characters from the first turn:

```python
page = source_info(limit=1)
reference = dict(page["turns"][0]["reference"])
reference["end"] = min(reference["end"], reference["start"] + 400)
selected = read(reference)
print(selected)
```

The model sees printed output. Assigning a variable or leaving a bare expression
does not display it. `read()` takes one reference, so a delegate uses a loop to
read several passages. It can request later metadata pages with
`source_info(offset=...)`, using the preceding response's `next_offset`.

This controls the text sent to the model. Reading an appended turn still loads
that turn's blob into application memory. Explicit imports and legacy base
sources load their complete source blob. The complete operational journal must
also fit its byte limit.

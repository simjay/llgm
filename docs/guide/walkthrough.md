# Evidence walkthrough

This tutorial follows evidence from storage to an answer, then shows how a
correction changes what a later query reads. It uses the same fictional Atlas
project as the [concepts guide](concepts.md).

## Follow one answer

The complete {download}`offline.py example <../../examples/offline.py>` uses
SQLite storage and Docker interpreters. Its model responses are scripted so
you can follow the same reads and returns each time. It needs no credentials
and makes no hosted calls.

With Python 3.11 or later and Docker running, install from a copy of the package,
download the interpreter image, and run the example:

```sh
python -m pip install .
docker pull python:3.12-slim
python examples/offline.py
```

The script creates four sources in a temporary workspace. The labels below are
for explanation. Actual node IDs are generated opaque identities.

| Source | Evidence and role |
| --- | --- |
| Database | Atlas production uses PostgreSQL. Staging uses SQLite. Retrieved as a seed. |
| Backups | Atlas production retains backups for seven days. Retrieved as another seed. |
| Registry | The deployment region is eu-west-1. Reached through a primary edge from Database. |
| Update | MySQL. An exact journal patch replaces only Database's PostgreSQL span with this evidence. |

The script asks about the production database, backups, and region. Follow the
five steps from that question to its answer:

1. Search finds Database and Backups. The example allows two starting nodes, so
   both receive a delegate.
2. The Database delegate reads its source with applicable journal patches. It
   sees MySQL for production and SQLite for staging. The MySQL passage points
   back to Update as its evidence.
3. Database follows its `deployment_registry` edge and asks a child delegate to
   inspect Registry. The child returns eu-west-1 and its source reference.
4. The Backups delegate reads the seven-day retention statement. It returns
   that finding independently of the Database branch.
5. The root receives the selected source quotes and the branch findings. It
   combines them into one answer with references.

The original PostgreSQL text remains readable. The MySQL claim cites Update,
so you can distinguish the earlier statement from the evidence that replaced it.
The example removes its temporary workspace when it finishes.

With hosted models, the models choose what to read and may miss useful evidence.
For a separate example using hosted models and two related sources, download
{download}`recursive_memory.py <../../examples/recursive_memory.py>`.
See the [quickstart](quickstart.md) for credentials and model configuration.

### Choose a related node

An edge tells a delegate why another node might help. In the example, the
application creates an edge from Database to Registry with the relation
`deployment_registry`. Inside its interpreter, the delegate can find that
relation and ask the target node a focused question:

```python
links = edges(relation="deployment_registry")
for edge in links["edges"]:
    finding = query_node(
        edge["reference"]["node_id"],
        "Which deployment region is recorded?",
    )
    print(finding)
```

`edges()` returns relationship descriptions under `edges` and unique target
references under `references`. It omits withdrawn edges and edges that do not
apply to the query's scope or time. Finding an edge does not read its target.
`query_node()` starts that separate investigation and returns selected findings
to the caller.

### Inspect a large node without printing it all

Every delegate receives a Python `context` with the question, available
references, the journal entries used to interpret the node, and a first page
of turn metadata under `source_page`. Metadata tells the model where to read
without putting all the text into its prompt.

The delegate can call `source_info()` to inspect more turn IDs, speaker roles,
lengths, and span coordinates. This code reads at most 400 characters from the
first turn:

```python
page = source_info(limit=1)
reference = dict(page["turns"][0]["reference"])
reference["end"] = min(reference["end"], reference["start"] + 400)
selected = read(reference)
print(selected)
```

The model sees printed output. Assigning a value to a variable or leaving a
bare expression does not show it. `read(reference)` takes one reference, so use
a loop to read several passages. `source_info(offset=...)` retrieves later pages
using the preceding response's `next_offset`.

Variables stay in that delegate's interpreter. A child gets its own interpreter,
and its parent sees only what it returns. This keeps the entire child conversation
out of the parent's prompt.

These reads limit the text sent to models. The storage adapter still loads the
owning source into host memory to resolve a span. The journal used for current
reads is also loaded in full and must fit its configured byte limit.

## Follow an update and a correction

Suppose the original record says `PostgreSQL | SQLite`, with the first value for
production and the second for staging. A new source says `MySQL`. We want
production queries to use that update while preserving the original record and
leaving staging unchanged.

This independent script needs only the core package. It makes no model calls
and does not start Docker. It creates two sources, points an exact replacement
at the new evidence, then compares an effective read with an original read:

```python
import asyncio
from tempfile import TemporaryDirectory

from llgm import Conversation, Provenance, SourceSpan, Workspace, parse_instant_ms
from llgm.memory.evidence import Evidence
from llgm.memory.query import QueryEvidence


async def main():
    """Inspect effective references while retaining exact original source history."""
    with TemporaryDirectory() as directory:
        async with Workspace.open(directory) as workspace:
            old_text = "PostgreSQL | SQLite"
            old = await workspace.ingest(Conversation.from_turns([
                {"role": "user", "turn_id": "decision", "text": old_text},
            ]))
            new = await workspace.ingest(Conversation.from_turns([
                {"role": "user", "turn_id": "update", "text": "MySQL"},
            ]))
            subject = SourceSpan(old.node_id, "decision", 0, len("PostgreSQL"))
            replacement = SourceSpan(new.node_id, "update", 0, len("MySQL"))
            await workspace.append_journal(
                old.node_id, subject=subject, record_kind="overwrite",
                relation="replace", value=replacement,
                provenance=Provenance("user", "walkthrough", (replacement,)),
                applicability={
                    "scope": {"env": "production"},
                    "valid_from_ms": parse_instant_ms("2026-09-11T00:00:00Z"),
                },
            )
            async with await Evidence.open(workspace) as evidence:
                query = QueryEvidence(
                    evidence, {"env": "production"}, "September 11, 2026",
                    as_of_ms=parse_instant_ms("2026-09-11T12:00:00Z"),
                )
                await query.initialize_node(old.node_id)
                parts = await query.read_segments(
                    SourceSpan(old.node_id, "decision", 0, len(old_text))
                )
                assert parts[0].reference == replacement
                assert "".join(part.text for part in parts) == "MySQL | SQLite"
                assert (await workspace.resolve(subject)).text == "PostgreSQL"
                print("effective:", "".join(part.text for part in parts))
                print("original:", (await workspace.resolve(subject)).text)


asyncio.run(main())
```

Expected output:

```text
effective: MySQL | SQLite
original: PostgreSQL
```

`SourceSpan` offsets use Python string indexing, starting at zero and excluding
the end position. The subject spans only `PostgreSQL`, so the patch
leaves ` | SQLite` alone. `workspace.resolve(subject)` always reads the original
text. `QueryEvidence.read_segments()` applies the query's relevant patches and
preserves the reference of each original or replacement segment.

The patch applies to `env=production` from September 11 onward. A query with a
different scope or an earlier `as_of_ms` sees the unpatched passage. If a query
omits the scope or time needed to decide whether the patch applies, the read
reports unresolved evidence. Missing replacement evidence is also unresolved.
It does not silently return the old value as current.

To revise this patch later, append another applicable `overwrite` for the same
exact subject, relation, and scope. The latest append wins. A `correction` can
instead target a previous `JournalRef`, for example to retract the patch.
Neither operation deletes the original source or journal history. The
[journal API](../reference/api.md#journal-interpretation-and-graph-operations)
documents these records and methods.

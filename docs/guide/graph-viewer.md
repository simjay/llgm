# Graph viewer

Open a saved workspace as an interactive graph:

```sh
llgm view --workspace ./memory --conversation-id default
```

The command opens a browser and keeps the viewer running until you press Ctrl-C.
It uses port 8765 on your own computer. Use `--port 0` to choose an available port,
and `--no-browser` to print the URL without opening it. The workspace must already
exist. Viewing it does not require model credentials or the `rlm` extra.
Add `--env-file .env` if your storage or search settings are in that file.
The CLI loads it only when selected.

Graph browsing and source inspection do not use the retrieval service. Searching
uses the configured backend, which defaults to hybrid search on Modal. To keep
search local as well, set `LLGM_RETRIEVER_BACKEND=sqlite_fts5`. See
[search configuration](configuration.md#search-backend).

For example, after discussing an Atlas database migration, the graph might show
an Atlas topic connected to a backup-policy topic. The conversation selector
chooses which saved conversation's current topic to highlight. Selecting a node
in the viewer does not change that saved pointer.

## Explore the graph and its records

Each circle represents a topic node. Labels show node IDs, with long IDs shortened
on the map. Hover or open a node to see its full ID. Arrows show the direction of
published connections that have not been withdrawn. The graph shows stored
connectivity, including connections with applicability conditions. It does not
interpret those conditions for a particular question.

The green node is the selected conversation's current topic. **Go to current**
opens that node. The graph and conversation pointers refresh every five seconds.
Use **Refresh** to also reload the selected node's records. Drag nodes to arrange
them, drag the canvas to pan, and use the zoom controls or **Fit graph** to navigate.

Click a node on the map or in the workspace list to open its inspector:

- **Turns** lists original conversation turns. Expand a turn to read it. Large
  turns are read in sections, and turn listings have a **Load more** control.
- **Journals** lists the node's retained amendment history. Expand an entry to
  inspect it, or open its complete JSON in the browser.
- **Connections** lets you follow incoming and outgoing edges to another node.
- **Files** shows the source manifest's storage location and a link to open it.

Source manifests and appended turns live in content-addressed blob files when
using local storage. Each appended turn also has an **Open source file** link.
Imported base turns live inside their source manifest. Journals are rows in
`metadata.sqlite3`, not separate files. The viewer exposes those records as JSON
and identifies their owning node and database table.

Opening a file displays its contents in the browser. It does not launch a local
editor or modify the file. Use the browser's Back button to return to the graph.
With a remote blob store, the viewer reads through that store and shows the blob
identity instead of a local path.

## Search the same evidence

The search bar calls the same `Evidence.search()` backend used by LLGM. With the
default CLI configuration, this is hybrid BM25 and ColBERTv2 source search through
Modal, with local inline journal search. See [search setup](configuration.md#search-backend).
Results are grouped by node in passage-rank order. Clicking a
result opens its node, and matching nodes are marked in amber on the map.
The displayed scores are backend-specific, not similarity percentages.

Search does not run Main, Reader, or Graph answer-model calls. A configured
semantic retriever can still perform encoding or remote retrieval work. Searching
never moves the current-topic pointer. It may upload a source snapshot and build
a semantic index, as well as refresh the local journal index.
Unlike answer seed admission, the viewer displays every distinct owner in the
returned passage pool without applying `max_seed_nodes` or adding the current
node to the search results. See [Node search](node-search.md) for that distinction.

### Share an application's custom retriever

When an application supplies an evidence factory, create its viewer through
`GraphViewer.from_application()`. This shares the exact workspace, factory,
passage window, and retrieval pool size:

```python
from llgm.viewer import GraphViewer


async def inspect_memory(memory):
    """Browse an already-open LLGM instance using its configured evidence backend."""
    async with GraphViewer.from_application(memory, conversation_id="atlas") as viewer:
        print(viewer.url)
        await viewer.serve_forever()
```

The application and its custom retriever must stay open for the viewer's lifetime.
Closing the viewer closes its request handles and HTTP server. It leaves the
application's workspace and retriever open.

A storage-only program can use `GraphViewer(workspace)` directly with local BM25.
The CLI accepts `--config` and `--env-file` for storage and retrieval settings.
It supports configured hybrid search and explicit `sqlite_fts5`. Use the Python
factory above for other custom retrieval backends.

## Large topics and live updates

The graph overview reads only node IDs, edge metadata, journal counts, and
conversation pointers. It does not load topic text. The inspector lists 32 turns
or journal entries at a time and previews at most 16,384 characters per read.
Opening a complete file is an explicit full-file read. Imported base sources still
require loading their original blob to obtain turn coordinates. An individual
appended turn is also loaded by the storage adapter before its preview is sliced.

The viewer loads the whole graph's metadata. It is intended for a modest number
of large topic nodes, and it does not yet virtualize very large graphs. It shows
original stored evidence and journal records, rather than an effective view after
applying amendments. Refreshes observe current writes without freezing a
workspace-wide snapshot across separate requests.

The server listens only on `127.0.0.1`. It generates a new private URL for each
run and serves its own assets locally. It exposes read operations for records in
the selected workspace, with no arbitrary filesystem browsing or graph edits.

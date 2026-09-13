# Node search

When you ask, "What about its backups?", LLGM starts at the current topic for
your `conversation_id`. You do not need to repeat the topic's searchable words.
Search can add other relevant conversations. These starting conversations are
called **seed nodes**.

This page explains that selection rule, the settings that affect it, and how
to tell whether search missed needed evidence. The later model calls are
covered in [architecture](architecture.md#how-an-answer-runs).

## Start at the current topic

Each conversation has a persistent pointer to its current topic. That node is
always the first default seed, including when `remember=False`. The pointer
survives application restarts and is separate from other conversations' activity.
The current reader receives recent turn references even when search finds no
matching passages.

Search fills remaining seed slots. With `max_seed_nodes=1`, the current topic is
the sole seed and initial retrieval is skipped. If topic routing has already
spent the answer's search allowance, reading still starts at the known current
node. Readers may search or follow connections within the remaining budget.

When a read-only question has no current topic, initial retrieval supplies the
seeds. An explicit `node_id` overrides the session pointer and becomes the sole
seed. A read-only question never moves that pointer or appends conversation turns.

## Follow a passage ranking

Configured applications default to hybrid source search: BM25 word matching
and ColBERTv2 semantic matching over the same passages, fused by rank. The
authenticated Modal worker indexes source snapshots and runs both components.
Inline journal notes stay in a local SQLite FTS5 index. Search makes no Main,
Reader, or Graph calls, but semantic encoding and indexing use remote compute.

When initial retrieval runs, selection follows four steps:

1. Retrieve up to `retrieval_k` passages for the original question.
2. Reserve the first slot for the current topic when it exists. Walk the ranking
   and fill the remaining slots with distinct owners, up to `max_seed_nodes` total.
3. Give each selected node all of its distinct matching references from the
   returned pool. Remove exact duplicates, but keep overlapping spans separate.
4. Record other owners in the pool as skipped because of the seed limit.

For example, suppose Database is the current topic, the question asks about
its backups, and the ranking is:

| Passage rank | Owning conversation | Selection with a three-node limit |
| --- | --- | --- |
| 1 | Database | Add a reference to the current Database seed |
| 2 | Database | Add a reference to Database |
| 3 | Backups | Select Backups |
| 4 | Database | Add another reference to Database |
| 5 | Registry | Select Registry |
| 6 | Deployment notes | Record this node as skipped |

Database gets three references and one delegate. Its repeated hits do not
consume additional seed slots or combine into a new node score. No LLM reranks
the initial seeds. The selected delegates decide what to read afterward.

## Choose how much to retrieve

| Setting | Default | Effect |
| --- | ---: | --- |
| `retrieval_k` | 12 | Maximum initial passage hits, up to 40 |
| `max_seed_nodes` | 3 | Maximum starting nodes, no greater than `retrieval_k` |
| `max_concurrency` | 3 | Limit on concurrent seed branches and on concurrent model calls |

These settings have matching `LLGM_` environment variables. For example,
`LLGM_RETRIEVAL_K=20` expands the initial pool to at most twenty passages.
See [configuration](configuration.md) for loading settings in your application.

Increasing the passage limit does not necessarily change the seeds. In the
ranking above, a larger pool still starts with Database, Backups and Registry.
It may give them more references. Raise `max_seed_nodes` if you want to admit
additional conversations. Selected branches beyond `max_concurrency` wait
for a slot.

All branches share one answer budget. More seeds add work without automatically
raising the model-call, context or time allowances.

Two other controls affect search:

- `passage_chars` sets the local journal and offline BM25 passage size. It defaults
  to 2,048 characters. Hybrid sources use 180-token windows with 32-token overlap,
  measured using the pinned ColBERT tokenizer, including metadata and special tokens.
- `max_searches` limits search calls across the whole answer. Its default of
  eight includes the initial retrieval. Set it through the answer budget or
  `LLGM_MAX_SEARCHES`.

Both conversation and read-only answers reserve the first seed slot for their
current topic and supply recent turn references along with matching passages.
Remaining slots follow the same ranking rule. No score threshold is implemented.

If you already know which node to investigate, pass `node_id` and
`remember=False` to `answer()`.
That node becomes the sole seed and initial retrieval is skipped. The delegate
still chooses what to read within it.

## Choose a retrieval method

Changing the retriever changes the passage ranking. It keeps the same seed
selection rule and delegate workflow.

BM25 works from word matches. ColBERT compares learned vectors for query tokens
and passage tokens, allowing matches beyond identical words. PLAID is the
search engine used to find promising passages efficiently in a ColBERT index.
LLGM uses equal-weight reciprocal-rank fusion. Each source component retrieves
up to 40 passages. A passage contributes `1 / (60 + rank)` from each component
that finds it. LLGM sums those contributions and returns up to `retrieval_k` hits.
The score is a ranking value, not a calibrated similarity or probability.

Workspace corpora with fewer than 64 passages use exact ColBERT MaxSim scoring.
This keeps semantic retrieval available before there is enough text to train
PLAID's clusters. Larger corpora use the official PLAID index. The descriptor
reports which engine ran. Passages are search windows inside topic nodes and
do not create additional graph nodes.

| Option | How it connects to LLGM |
| --- | --- |
| Hybrid, the configured default | `LLGM_RETRIEVER_BACKEND=hybrid`, with BM25 and ColBERT on Modal |
| Local BM25 | `LLGM_RETRIEVER_BACKEND=sqlite_fts5`, refreshed from new publications |
| Official ColBERTv2 and PLAID | Optional local adapter or authenticated Modal transport over an explicitly prepared index |
| Lexical and dense hybrid | `HybridRetriever` combines passage rankings over matching corpora |

Supply other custom retrievers through `evidence_factory`. A custom backend name
does not build or deploy an index. Your application maintains that index and
provides references that resolve in the workspace. LLGM reads their text from
the stored evidence. See
[custom retrieval](configuration.md#use-your-own-search-backend).

With an external source retriever, inline journal notes remain searchable in
the local index. Their ranking can be combined with external results. That
local journal search is separate from combining BM25 and ColBERT over source
passages.

A ColBERT worker may need to load its model and index before serving its first
search. This startup delay is a **cold start**. Later searches can reuse the
loaded state. The configured hybrid backend uploads source records on the first
search and after source appends, then builds or reopens a content-addressed
generation. Unchanged searches send only its ID and the query. A changed corpus
currently requires a complete snapshot upload and a new semantic index. This
can be expensive for large topics. Index compaction and incremental PLAID updates
are not implemented. Journals and graph-only edits do not rebuild the source index.

Search waits for preparation and fails explicitly if the worker is unavailable.
It never substitutes BM25 or an older generation. Indexing time is part of the
answer deadline. Use explicit local BM25 when remote search is not configured.
The lower-level `ModalColBERTRetriever.connect()` still opens an existing fixed
index without uploading sources. See [search setup](configuration.md#search-backend).

Local and remote ColBERT adapters bind the index to a canonical passage corpus.
They reject unknown or duplicate passage IDs, invalid ranks and nonfinite scores.
Returned text and references come from that corpus. Editing result metadata does
not change later searches or the index identity.

For the underlying methods, see the [ColBERTv2 paper](https://aclanthology.org/2022.naacl-main.272/)
and [PLAID paper](https://arxiv.org/abs/2205.09707).

## Find more evidence after selection

The seeds determine where investigation starts, not every node it may reach.
Inside its interpreter, a delegate can use these operations:

| Operation | Effect |
| --- | --- |
| `search("production hosting region", k=5)` | Find more candidate references with the configured retrieval backend |
| `edges()` | Inspect applicable outgoing relationships and their targets |
| `query_node(node_id, question)` | Ask a child delegate to investigate a node |

These are delegate-interpreter operations, not functions to import into your
application. The [walkthrough](walkthrough.md#choose-a-related-node) shows them
in context.

Search can reach any node in the configured index, including nodes without an
edge from the current node. It uses the shared search allowance and accepts
at most forty hits per call. Searching or inspecting edges does not start
children automatically. A delegate can also read a discovered reference directly.

The main model gets the selected findings and source quotes after investigation.
It has one final synthesis call and no further search phase.

## Diagnose missing evidence

Inspect `result.trace` for a `seed_selection` event. It reports selected and
skipped nodes. Later node operations, reads and branch returns show what
happened after selection.

| Observation | What to inspect |
| --- | --- |
| Useful conversation absent from the returned pool | Question wording, backend, passage size and retrieval limit |
| Useful conversation found but skipped | Seed limit and passage ranking |
| Node selected, useful turn never read | Delegate's initial and follow-up reads |
| Fact read but absent from branch findings | Delegate's evidence selection and return limits |
| Fact returned but used incorrectly | Main's interpretation of scope, dates and quantities |
| Answer missing a supporting reference | Final citation selection |

A node skipped because of the seed limit makes the current pipeline's result
`partial`, even if a later read reaches it or the answer text appears complete.
When there is no current topic or explicit seed, empty retrieval also produces
a partial result. Read
`result.evidence.unresolved`, the answer and its references together. Valid
references identify source text but do not prove that it supports the model's
claim. See [capabilities and limits](../reference/implementation-status.md)
for the other result constraints.

## Inspect search visually

The [graph viewer](graph-viewer.md) lets you search the same evidence backend,
see matching topic nodes, and open their stored turns and journals. It keeps
the selected conversation's current node highlighted without moving its pointer.

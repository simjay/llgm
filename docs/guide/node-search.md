# Node search

When you ask, "Which database does production use?", LLGM first needs useful
places to read. It searches passages and selects the conversations that own
them. Those starting conversations are called **seed nodes**.

This page explains that selection rule, the settings that affect it, and how
to tell whether search missed needed evidence. The later model calls are
covered in [architecture](architecture.md#how-an-answer-runs).

## Follow a passage ranking

The default retriever uses BM25 in a local SQLite FTS5 index. BM25 ranks text
using word matches. The index includes source passages and inline journal
notes, and refreshes as workspace records are published. This default search
and seed selection make no model calls.

Selection follows four steps:

1. Retrieve up to `retrieval_k` passages for the original question.
2. Walk that ranking and select the first `max_seed_nodes` distinct owners.
3. Give each selected node all of its distinct matching references from the
   returned pool. Remove exact duplicates, but keep overlapping spans separate.
4. Record other owners in the pool as skipped because of the seed limit.

For example, suppose the question asks about the database and its backups,
and the ranking is:

| Passage rank | Owning conversation | Selection with a three-node limit |
| --- | --- | --- |
| 1 | Database | Select Database |
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

- `passage_chars` sets the default local passage size, in characters. It defaults
  to 2,048 and is a Python runtime option, not an environment setting.
- `max_searches` limits search calls across the whole answer. Its default of
  eight includes the initial retrieval. Set it through the answer budget or
  `LLGM_MAX_SEARCHES`.

If you already know which node to investigate, pass `node_id` to `answer()`.
That node becomes the sole seed and initial retrieval is skipped. The delegate
still chooses what to read within it.

## Choose a retrieval method

Changing the retriever changes the passage ranking. It keeps the same seed
selection rule and delegate workflow.

BM25 works from word matches. ColBERT compares learned vectors for query tokens
and passage tokens, allowing matches beyond identical words. PLAID is the
search engine used to find promising passages efficiently in a ColBERT index.
LLGM's official adapter uses the released ColBERTv2 model with PLAID.

| Option | How it connects to LLGM |
| --- | --- |
| Local BM25 | Default workspace index, refreshed from new publications |
| Official ColBERTv2 and PLAID | Optional local adapter or authenticated Modal transport over an explicitly prepared index |
| Lexical and dense hybrid | `HybridRetriever` combines passage rankings over matching corpora |
| BM25 and ColBERT together | Requires a caller-supplied composite retriever. There is no built-in application setting for this combination |

Supply a custom retriever through `evidence_factory`. A backend name alone
does not build or deploy an index. Your application maintains the index and
provides references that resolve in the workspace. LLGM reads their text from
the stored evidence. See
[custom retrieval](configuration.md#use-your-own-search-backend).

With an external source retriever, inline journal notes remain searchable in
the local index. Their ranking can be combined with external results. That
local journal search is separate from combining BM25 and ColBERT over source
passages.

A ColBERT worker may need to load its model and index before serving its first
search. This startup delay is a **cold start**. Later searches can reuse the
loaded state. Connecting the Modal adapter does not upload the workspace or
build a remote index.

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

The root gets the selected findings and source quotes after investigation.
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
| Fact returned but used incorrectly | Root's interpretation of scope, dates and quantities |
| Answer missing a supporting reference | Final citation selection |

A node skipped because of the seed limit makes the current pipeline's result
`partial`, even if a later read reaches it or the answer text appears complete.
Empty retrieval also produces a partial result. Read
`result.evidence.unresolved`, the answer and its references together. Valid
references identify source text but do not prove that it supports the model's
claim. See [capabilities and limits](../reference/implementation-status.md)
for the other result constraints.

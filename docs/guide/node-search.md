# Node search

Suppose you have stored hundreds of conversations and ask, "Which database does
Atlas use in production?" LLGM needs a few useful places to start reading.
It searches small passages, then selects the nodes that contain those passages.
Those starting nodes are called *seeds*.

Each seed receives a smaller-model reader called a *node delegate*. The delegate
reads relevant evidence and returns findings to the root model, which writes
the answer. A passage search result points to possible evidence. The delegate
still needs to inspect and interpret it.

```{mermaid}
flowchart LR
    Q[Question] --> P[Find passages]
    P --> N[Choose nodes]
    N --> R[Read evidence]
```

## The current selection rule

The default search uses BM25 in a local SQLite FTS5 index. BM25 ranks passages
by matching the question's words against stored text. The index includes source
passages and inline journal notes, and updates as workspace records are
published. The initial search and node selection make no model calls.

Selection follows four steps:

1. Retrieve up to `retrieval_k` passages for the original question.
2. Follow the passage ranking and select the first `max_seed_nodes` distinct
   source nodes. Repeated hits from one node do not increase its priority.
3. Give each selected delegate every distinct matching reference for its node
   from the returned pool. Exact duplicate references are removed. Overlapping
   spans remain separate.
4. Record the remaining source nodes as skipped because of the seed limit.

For example, imagine this ranking:

| Passage rank | Node containing the passage | Selection at a three-node limit |
| --- | --- | --- |
| 1 | Database | Start a Database delegate |
| 2 | Database | Add a reference to the same delegate |
| 3 | Backups | Start a Backups delegate |
| 4 | Database | Add another reference to Database |
| 5 | Registry | Start a Registry delegate |
| 6 | Deployment notes | Record this node as skipped |

Database receives three references and one delegate. No model reranks these
starting nodes. The delegates choose how to investigate after selection.

## Adjust the starting work

| Setting | Default | Meaning |
| --- | ---: | --- |
| `retrieval_k` | 12 | Initial passage limit, at most 40 |
| `max_seed_nodes` | 3 | Maximum starting nodes, no greater than `retrieval_k` |
| `max_concurrency` | 3 | Maximum concurrent seed branches and model calls |
| `passage_chars` | 2,048 | Default local passage character limit, supplied as a runtime option |
| `max_searches` | 8 | Maximum search calls across the answer, including initial retrieval |

The first three settings also have `LLGM_` environment variables, as does
`LLGM_MAX_SEARCHES`. See [configuration](configuration.md) for settings precedence
and budget units.

Increasing `retrieval_k` can expose more references without changing the selected
nodes. In the example, retrieving more lower-ranked passages still starts with
Database, Backups, and Registry. Increase `max_seed_nodes` to allow more starting
nodes. Excess branches wait if there are more selected seeds than concurrency
slots.

All branches share one inference budget. Allowing more starting nodes adds work
without increasing that budget automatically. See [configuration](configuration.md)
for model-call, context, and time limits.

If you already know where to start, `answer(question, node_id=...)` uses that
node as the sole seed and skips initial retrieval. The delegate receives turn
metadata and references, then uses Python to read selected spans. The
[walkthrough](walkthrough.md#inspect-a-large-node-without-printing-it-all)
shows those reads.

## What the retriever changes

The retriever decides which passages enter this process and in what order.
Changing it keeps the same node-selection rule and delegate workflow.

BM25 uses word matches. ColBERT compares query tokens with passage tokens using
learned vectors, which can capture similarities beyond matching words.
PLAID is a search engine for ColBERT representations. It reduces the work needed
to find promising passages in an index. LLGM's official adapter uses ColBERTv2
with PLAID. See the [ColBERTv2 paper](https://aclanthology.org/2022.naacl-main.272/)
and [PLAID paper](https://arxiv.org/abs/2205.09707) for the underlying methods.

| Option | Current integration |
| --- | --- |
| Local BM25 | Default workspace index, refreshed from new publications |
| Official ColBERTv2 and PLAID | Optional local adapter or authenticated Modal transport over an explicitly prepared index |
| Lexical and dense hybrid | `HybridRetriever` combines passage rankings over matching corpora |
| BM25 and ColBERT together | Requires a caller-supplied composite retriever. No built-in application setting combines them |

To use an external retriever, provide it through `evidence_factory` and keep its
index up to date. Setting a backend name alone does not build or deploy an index.
Returned references must point to evidence in the workspace. LLGM resolves the
stored text before using it as evidence. See
[custom retrieval](quickstart.md#use-your-own-search-backend) for the interface.

Inline journal notes remain searchable locally when you supply an external
source retriever. Their ranking may be combined with external results. That
does not combine BM25 and ColBERT searches over source text.

A ColBERT worker may need to load its model and index before its first search.
This startup delay is called a *cold start*. Later searches may be faster
because the worker reuses the loaded model and index.

## How discovery continues

A Database delegate might discover that the deployment region lives elsewhere.
It has three ways to continue:

| Operation inside the delegate | What it does |
| --- | --- |
| `search("Atlas deployment region", k=5)` | Finds more candidate references using the answer's configured search backend |
| `edges()` | Lists applicable relationships and their target nodes |
| `query_node(node_id, question)` | Starts a child delegate to investigate a node and return selected findings |

Search and edge inspection do not start children automatically. The delegate
can read a discovered target directly or ask a child to investigate it. Search
can reach any node in the configured index. It consumes the shared search
allowance and accepts at most forty hits per call.

The root receives the delegates' selected source quotes and findings for one
final synthesis call. It has no further tool phase. The initial seeds limit
where reading starts, while follow-up discovery can reach more sources before
synthesis. See [the answer lifecycle](architecture.md#how-an-answer-runs).

## Diagnose the stage that lost the evidence

| Observation | Where to investigate |
| --- | --- |
| Useful source missing from search results | Question wording, search backend, passage size, and retrieval limit |
| Useful source found but skipped | The starting-node limit and the order of passage results |
| Node selected, but its useful turn was not read | The delegate's initial and follow-up reads |
| Fact read but omitted from the delegate's return | The delegate's choice of findings and its message limits |
| Fact returned but used incorrectly | The root's interpretation of scope, dates, and quantities |
| Correct claim missing its citation | Whether the final answer cites the supporting evidence |

Inspect `result.trace` for the `seed_selection` event. It lists selected and
skipped nodes. Subsequent node operations, evidence reads, and branch returns
help locate later losses.

A node skipped because of the seed limit makes the answer `partial`, even if
the answer text appears complete. Empty retrieval also produces a partial
result. Inspect `result.evidence.unresolved`, the answer, and its references
together. Valid references locate source text but do not prove that the text
supports the model's claim. See [capabilities and limits](../reference/implementation-status.md)
for practical constraints.

# Capabilities and limits

LLGM is an alpha library. The [quickstart](../guide/quickstart.md) covers
installation, model setup and an explicit local-search configuration. The first
[PyPI release](https://pypi.org/project/llgm/) is still in progress.

You can store conversations, search their text and apply explicit corrections
without a model or sandbox. To ask LLGM a question, configure model clients and
the `rlm` extra. The tables below describe what you can use today and
where an application still needs its own decisions or checks.

## Supported capabilities

| Capability | Current behavior |
| --- | --- |
| Integrated application | `LLGM.from_settings()` owns configured resources. `answer()` saves new messages, resumes a conversation's topic, investigates evidence and saves a reply. `ingest()` imports earlier text, chat turns or a `Conversation`. |
| Source storage | `Workspace` stores original turns and stable Unicode spans. Related turns append to the same topic node without rewriting earlier evidence. Idempotency supports ingestion retries. |
| Graph viewer | `llgm view` draws the saved graph and current conversation topic. Nodes open paged source and journal records. Search reuses the evidence backend. The viewer is local and does not edit memory. See [Graph viewer](../guide/graph-viewer.md). |
| Primary graph | Directed edges retain provenance, applicability, and withdrawal state. Delegates can inspect relationships and investigate their targets. |
| Journals and corrections | Explicit journal writes retain history and can change effective reads. Original and replacement references remain available. Maintenance does not generate these corrections automatically. |
| Time | Sources can carry event timestamps. Queries and amendments accept numeric validity instants with explicit timezone conversion. |
| Initial node selection | The conversation's current topic is the first default seed, including for read-only answers. Ranked passages fill remaining slots when search capacity is available. Without a current topic, retrieval supplies the seeds. A supplied `node_id` in read-only mode overrides the pointer and bypasses initial search. See [node search](../guide/node-search.md). |
| Node inference | Concurrent DSPy RLM branches use isolated Deno/Pyodide Python, lazy source reads, recursive children, and selected citation returns. The main model combines their evidence in one final call. |
| Answer outcomes | Results include status, references, usage, and unresolved needs. Failed or skipped work remains visible even when another branch supplies an answer. |
| Local retrieval | A reusable incremental SQLite FTS5 index covers source and inline-journal passages. |
| Retrieval adapters | Configured applications and the CLI viewer default to BM25 plus ColBERT through Modal. Small source corpora use exact ColBERT and larger ones use PLAID. Explicit local BM25 and custom evidence factories remain available. |
| Model adapters | Native OpenAI, Anthropic, and OpenAI-compatible endpoints support complete-response generation. Each role can use a separately configured model. |
| Storage | Local blobs and SQLite metadata are the default. An optional S3 blob adapter is available. Metadata remains local SQLite. |
| Resource limits | One answer shares its call, search, evidence, context, operation and time allowances. Recursion depth and per-node steps are also bounded. Maintenance has a separate budget. |
| Existing workspaces | Metadata schema 5 is supported. Schema 4 requires rebuilding from inputs. Copy schema 3 explicitly. Schema 2 additionally requires classifying its journal pointers. |

## Conversation management

`answer()` stores new user turns and nonempty replies, with a persistent
conversation ID and conservative model topic routing. Related history appends
to the same topic without a size threshold. `ingest()` imports earlier batches
through the same routing policy. It does not split inside a batch. Topic
classification quality and hosted reasoning on gigantic nodes remain unmeasured.
`remember=False` preserves the read-only answer contract. The interface is text
only and does not implement streaming, tool calling or response replay.
See [conversations and imports](../guide/conversations.md) for input formats,
retry behavior and the distinction between chat IDs and topics.

## Answer quality and citations

A successful API call or `completed` answer does not guarantee factual accuracy.
A delegate can miss a relevant passage, omit a useful fact, or repeat work.
The main model can misinterpret returned evidence. A compatible provider interface
alone does not establish that a model can reliably perform these tasks.

The main model receives the selected original excerpts before branch summaries.
Citation validation restricts its references to evidence actually returned by
those branches. This establishes the cited source's identity, not whether it
supports every claim in the answer. Review representative questions from your
application before relying on a model configuration.

Inspect the answer, references, unresolved reasons, and status together.
An empty answer with an explicit unresolved reason is a valid partial result.
An unsuccessful local read does not establish that the fact is absent from the
whole workspace. See [result handling](../guide/quickstart.md#understand-the-result)
and [troubleshooting evidence loss](../guide/node-search.md#diagnose-missing-evidence).

## Storage and scale

Models see selected source spans. Reading an appended span loads its one
immutable turn, while listing turn coordinates does not load appended text. Explicit imports and legacy base blobs still
load their whole source. A single giant turn is not streamed.

Operational journals have finite byte limits. Redundant entries compact without
deleting history, and necessary entries that exceed the limit cause an explicit
failure. Historical storage continues growing. Automatic forgetting, source-copy
materialization, and historical deletion policies are not provided.

Queries read currently published records and may observe new appends between
operations. A past-date query interprets available evidence at that date. It
does not reconstruct a past workspace snapshot or ranking.

Query scope selects applicable edges and journal entries. It does not restrict
which sources a search can return or provide an authorization boundary. Keep
source access restrictions in your application or use separate workspaces.

Postgres and distributed metadata are unsupported. The S3 adapter has not been
validated against a live service. Automated backup and restore, orphan-blob
cleanup, and index compaction are not provided. See
[storage configuration](../guide/configuration.md#storage-choices) before choosing
a backend or restoring a workspace.

## External services and cost

Inference requires the optional `rlm` extra, which installs DSPy and Deno.
DSPy's default interpreter runs Python in Pyodide/WASM. Runtime assets can be
downloaded on first use. Model calls and evidence access happen on the host.
Each interpreter stays owned until cleanup finishes, including after a deadline
or cancellation. Byte limits apply to admitted inputs and outputs. This sandbox
does not impose operating-system CPU, memory or process quotas.

Hosted model calls require provider credentials. Default hybrid search needs
the authenticated Modal deployment and its pinned ColBERT assets. It uploads
source snapshots and prepares a generation on first search and after source
appends. Unchanged searches reuse the generation. Full reindexing and cold starts
can be expensive and are included in answer deadlines. Exact ColBERT scoring
serves corpora with fewer than 64 passages, with PLAID used from 64 passages
onward.
The fixed-index Modal adapter remains available independently. See
[search setup](../guide/configuration.md#search-backend).

Ordinary answers have no dollar-denominated spending cap. Call and text limits
constrain work but do not set a provider billing limit. Provider usage is recorded
when available, and missing usage remains unknown. Canceling a local wait cannot
guarantee that a dispatched hosted request stops or incurs no charge. See
[runtime limits and usage](../guide/configuration.md#runtime-limits-and-usage).

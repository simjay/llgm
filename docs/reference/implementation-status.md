# Capabilities and limits

LLGM is an alpha library. Its first [PyPI release](https://pypi.org/project/llgm/)
is in progress. Use the checkout installation in the
[quickstart](../guide/quickstart.md) until that release is available.

You can store conversations, search their text and apply explicit corrections
without a model or Docker. To ask LLGM a question, configure model clients and
local Docker execution. The tables below describe what you can use today and
where an application still needs its own decisions or checks.

## Supported capabilities

| Capability | Current behavior |
| --- | --- |
| Integrated application | `LLGM.from_settings()` opens memory using environment configuration. `ingest()` accepts plain text, chat turns or a `Conversation` with metadata. `answer()` finds starting conversations, delegates reading and combines the findings. |
| Source storage | `Workspace` stores original turns and stable Unicode spans. Idempotency supports ingestion retries. Changed information becomes a new node. |
| Primary graph | Directed edges retain provenance, applicability, and withdrawal state. Delegates can inspect relationships and investigate their targets. |
| Journals and corrections | Explicit journal writes retain history and can change effective reads. Original and replacement references remain available. Maintenance does not generate these corrections automatically. |
| Time | Sources can carry event timestamps. Queries and amendments accept numeric validity instants with explicit timezone conversion. |
| Initial node selection | Ranked passages select the first distinct node owners, up to the configured cap. A supplied `node_id` bypasses initial search. See [node search](../guide/node-search.md). |
| Node inference | Concurrent branches use isolated Docker Python, lazy source reads, recursive children, and selected citation returns. The root combines their evidence in one final call. |
| Answer outcomes | Results include status, references, usage, and unresolved needs. Failed or skipped work remains visible even when another branch supplies an answer. |
| Local retrieval | A reusable incremental SQLite FTS5 index covers source and inline-journal passages. |
| Retrieval adapters | BM25, dense cosine, hybrid fusion, and optional official ColBERTv2/PLAID adapters. A custom source retriever requires an evidence factory and valid workspace references. |
| Model adapters | Native OpenAI, Anthropic, and OpenAI-compatible endpoints support complete-response generation. Each role can use a separately configured model. |
| Storage | Local blobs and SQLite metadata are the default. An optional S3 blob adapter is available. Metadata remains local SQLite. |
| Resource limits | One answer shares its call, search, evidence, context, operation and time allowances. Recursion depth and per-node steps are also bounded. Maintenance has a separate budget. |
| Existing workspaces | Metadata schema 3 is supported. A local schema-2 workspace can be explicitly copied after classifying its journal pointers. |

## Answer quality and citations

A successful API call or `completed` answer does not guarantee factual accuracy.
A delegate can miss a relevant passage, omit a useful fact, or repeat work.
The root can misinterpret returned evidence. A compatible provider interface
alone does not establish that a model can reliably perform these tasks.

The root receives the selected original excerpts before branch summaries.
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

Large source text is exposed to models through selected spans. The current JSON
blob adapter still loads each owning source in host memory. It does not stream
an arbitrarily large source from storage.

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

Inference requires a running Docker daemon and a trusted Python image already
available locally. Each interpreter remains owned by the runtime until cleanup
completes. An unavailable daemon or an expired cleanup deadline can fail an
answer. Model calls and evidence access happen on the host, outside that container.

Hosted model calls require provider credentials. ColBERT/PLAID requires a
separately prepared index. The Modal adapter connects to an authenticated remote
service, but connecting does not upload sources or build the index. BM25 and
ColBERT source rankings are not automatically combined by application settings.
Index creation and initial model loading are separate work from searching an
already loaded index.

Ordinary answers have no dollar-denominated spending cap. Call and text limits
constrain work but do not set a provider billing limit. Provider usage is recorded
when available, and missing usage remains unknown. Canceling a local wait cannot
guarantee that a dispatched hosted request stops or incurs no charge. See
[runtime limits and usage](../guide/configuration.md#runtime-limits-and-usage).

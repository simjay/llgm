# Node runtime hardening and hosted validation

Date: September 11, 2026.

The observed node-pipeline failures are addressed and the declared diagnostic
gates pass. The final original-cohort run answers all 15 LLGM cases correctly.
The focused run answers all 12 autonomous LLGM cases correctly and passes five
separately controlled mechanism checks. Correctness here includes requested
conflict reporting and abstention. These are small, exposed development cases,
not evidence of general benchmark superiority.

The user authorized fixes and real-model testing with the local credential.
This report follows the [earlier live diagnostic](node-pipeline-live-2026-09-11.md).
That report and its runs remain unchanged, including their failures.

## Changes and their purpose

| Observed issue | Implemented behavior | Verification |
| --- | --- | --- |
| Branches spent their shared allowance exploring before returning local facts | Each active frame reserves one final model call. Queued seeds reserve inspection and finalization calls. Child admission preserves other reservations. Final-only schemas constrain the last allowed response. | Deterministic queue/admission checks, and the original two-hop case completing with 12 sidecar calls plus one root call |
| Every seed tried to answer the whole question | The prompt leads with the local contribution and demonstrates one Python step combining a lazy source read with edge discovery. Each branch receives the admitted seed count. | Separate local contributions in the three-seed live cases |
| Node-only children spent calls discovering basic coordinates | Initial context contains one bounded page of source metadata and span references. Actual source text still requires a read. | Deterministic metadata checks and actual hosted child reads |
| A branch's local missing fact became an unavoidable final gap | Model-authored local notes stay in branch results. Only attributed host-owned `required_gaps` are forced into the final result. The root can resolve local omissions using siblings. | Three distinct contributors, local-absence and negative-approval live cases |
| Edge callbacks exposed UUIDs without enough meaning to choose a neighbor | Applicable edge descriptors include relation, provenance and applicability alongside target references. Multiple relationships to one target remain distinct. | Edge contract tests and a live irrelevant-first ordering check |
| Repairing an invalid operation or range still left the answer permanently partial | Callback schema and reference-resolution errors become recoverable observations. Successful recovery leaves the error in the trace without forcing a stale gap. | Real models repaired separately injected invalid-operation and invalid-range failures |
| Models finished before reading available source text | An unsupported premature finish receives a corrective observation within the existing step and call limits. | Deterministic rejection checks and four observed premature-finish rejections in the focused live protocol cases |
| Empty explicit abstention failed validation | Empty answer text is accepted when a meaningful unresolved reason accompanies it. The flat runner follows the same contract. Blank explanation strings remain invalid. | Deterministic schema tests. Live abstentions used explanatory answer text, so they do not establish the empty-text path |
| Completed child evidence could be lost when the parent exhausted its budget | Child-selected evidence is retained only after successful callback transport admission. Later parent budget exhaustion can return those findings within existing bundle/context limits. | Earlier deterministic two-ancestor tests and a new controlled live failure after actual completed child delivery |

The implementation remains in the existing evidence/query boundary and node
runtime. The changes do not introduce a new production controller. The optional
fault and ordering controls live in `evaluation/node_protocol.py` and are enabled
only by explicit experiment protocols. Production execution does not inject them.

Reservations cover model calls and preserve the existing root reserves. Python
execution and callbacks still consume the ordinary shared operation allowance.
Reservations do not guarantee success under every time, context, source or
operation limit. Missing replacement evidence and journal interpretation gaps
remain required gaps. Unselected local reads, undelivered child payloads,
cancellation and unrecovered schema failure do not acquire a new salvage path.

## Frozen conditions

All paths in this section are relative to the repository. Artifacts are ignored
local outputs. The experiment commands and control definitions are maintained in
[node-pipeline instructions](../../experiments/node-pipeline.md).

| Setting | Value |
| --- | --- |
| Root | OpenAI `gpt-6-astra` |
| Node delegate | OpenAI `gpt-5.6-terra` |
| Provider API | Responses with native JSON schemas |
| OpenAI SDK | 2.54.0 |
| Host Python | 3.11.13 |
| Docker image | `sha256:ec7d6c95cd3692a2e2d228a8b1ca74e4025b54121fcc4c5da6f09cfa473315ad` |
| Execution | Local isolated Docker Python, hosted generation, BM25 seed retrieval |
| Original per-answer limits | 12 sidecar calls, 16 total calls, 180 seconds |
| Focused per-answer limits | 20 sidecar calls, 24 total calls, 180 seconds |
| Seed/model concurrency | At most three |
| Maximum recursive depth | Three, with seed depth zero |
| Maximum model steps per frame | Twelve |
| Output allowance | 2,048 tokens per call |
| Preparation | Independent opaque node IDs, curated primary edges and exact journal patches, no model maintenance |
| Repetition policy | One attempt per declared case/arm per run, no automatic retries |

Each run freezes all 52 package Python files plus the exact cohort and
configuration before model dispatch. Evaluator expectations, source labels and
mechanism requirements stay outside model input. Actual requests, responses,
provider request IDs, usage, Python operations, callback returns and final
predictions are retained. Protocol interventions are recorded immediately and
again in the final trial record.

| Identity | SHA-256 |
| --- | --- |
| Original cohort | `31c83240f47baeb7b201e70299eec57d09343cefb5fea315d792e3b8ff964c04` |
| Original configuration | `c6c76fb1717470b7add0863ef12fa488d0abef22f7893e06bb2555251a457576` |
| Focused cohort | `b458bfe5a93454feff76c7966a914f68e5acd89b9f85249ff555f9bf35aa2771` |
| Focused configuration | `12bc94ee33b24748e2463aa53e7bfa1febec269eab41767aa9a1f71d6b671d89` |
| Final `inference/nodes.py` | `ea08e1bd6fc8dfc22b08a55ecbdb2e659e38ecf413eb6306d3bef5771d3ed5d1` |

The final `full-v2` and `focused-v1` package copies match the working package
source hashes. The preceding `full-v1` differs in the node prompt. Its failed
attempt is retained rather than overwritten by the repeat.

## Answer results and retained failures

Run root: `runs/node-hardening-20260911/`.

| Run and scope | LLGM correct | Flat correct | Trials | Actual generation calls |
| --- | --- | --- | --- | --- |
| `full-v1`, first hardening revision, original cases | 14/15 | 12/15 | 30 | 113 |
| `full-v2`, final revision, same cases and limits | 15/15 | 12/15 | 30 | 98 |
| `focused-v1`, autonomous paired cases | 12/12 | 10/12 | 24 | 71 |
| `focused-v1`, controlled LLGM-only cases | 5/5 | Not run | 5 | 36 |

The first hardening revision still missed the Aster two-hop credential.
Separately generating the read and edge operations used exploration calls before
the grandchild could be admitted. Its three-seed answer was correct but retained
a legitimate operational gap after redundant child admission was denied. The
final prompt clarified local collection, demonstrated batched callbacks, and
replaced unrelated seed UUIDs with a seed count. The repeat used the original
budgets and fixed both observed outcomes. There was no answer injection or
per-case production prompt.

The final original run has 14 completed LLGM outcomes and one correct partial
abstention. The autonomous focused run has ten completed outcomes and two correct
partial outcomes for conflict and missing evidence. Controlled cases have four
completed outcomes and one correct partial outcome retaining the injected parent
failure. No final trial failed or was left unstarted.

Flat retrieval misses the graph-only credential cases because a single lexical
search does not retrieve the isolated credential strings. It abstains rather
than inventing them. These deliberately constructed cases demonstrate navigation
access, not general retrieval superiority or a matched-compute advantage.

Every final answer, unresolved reason and canonical citation was reviewed against
the frozen case evidence. The primary coding agent performed this post-run
semantic review. It was not blinded and was not an independent human or hosted
judge. Per-trial reasoning is retained in each run's `semantic-review.jsonl`.
Lexical diagnostics remain separate in the original predictions. In particular,
the unanswerable case with no required answer substring was assessed for a real
abstention rather than accepted solely because its lexical check passed.

## Execution evidence

The original Aster case is an autonomous recursive success. Its three seeds
include an entry and two distractors. The entry invokes routing node `n4`, which
invokes credential node `n5`. Each opens an isolated interpreter and performs
actual model generation and source reads. Citation `e5` returns through `n5` to
`n4`, then seed `n2`, then the root. The leaf and routing delegate express local
uncertainty about the credential's purpose. The entry has the connecting evidence
and resolves that uncertainty. The root returns `JASPER`, with no unresolved gap.
All twelve sidecar calls fit the original allowance, with the thirteenth call
reserved for root synthesis.

The focused Verbena two-hop case also returns the correct credential, but its
delegate follows both edges using direct remote reads. It creates no child.
That case validates graph discovery and reading, not recursive execution.

The three-source Sorrel and Bluebell cases return distinct local evidence from
all three seeds. Each seed contributes its own source and the root combines the
results. Local omissions do not become global gaps. Bluebell's explicit lack of
approval is preserved as the answer `UNAPPROVED`, rather than treated as a
failure to access evidence. One delegate inaccurately called the destination
code a recipient code in a local note. The final answer follows the separately
cited facts correctly. This illustrates why correct final answers do not imply
every intermediate sentence is reliable.

| Controlled case | Observed mechanism |
| --- | --- |
| `protocol-v1-01-grandchild` | Prescribed `query_node` traversal creates a child and grandchild. Actual `OBSIDIAN` evidence reaches every ancestor and the root. A premature finish is rejected before inspection. |
| `protocol-v1-02-child-return` | A real child returns `GARNET` and citation `e2`. After its payload is admitted and Python execution returns, the protocol injects parent exhaustion. The parent retains `e2`, and the root answers correctly while retaining the attributed budget gap. |
| `protocol-v1-03-recovery` | One read is changed to an out-of-bounds range. The model eventually performs a valid read and returns `VIOLET`. The failure remains traceable but is absent from final required gaps. |
| `protocol-v1-04-recovery` | One read is changed to an unknown callback operation. The model subsequently reads the source and returns `QUARTZ` without a stale final gap. |
| `protocol-v1-05-edge-order` | Actual relationship descriptors are presented with incident contact first. The model selects the authorization target, reads its credential, and returns `CORUNDUM` without reading the contact. |

The two recovery cases consume eight and seven total calls respectively. They
include unnecessary metadata/search steps and premature-finish attempts. The
runtime makes recovery possible and preserves its cost. It does not make the
model's repair sequence optimal.

`mechanism-audit.json` verifies parent-child IDs, depth, interpreter opens, real
model request IDs, selected citations at each return boundary and final canonical
evidence. It also checks single fault injection, successful reads after errors,
local contribution counts and the controlled edge order. Audit scripts are
retained beside the output. Controlled cases are not pooled into an autonomous
success-rate claim.

## Accounting and local verification

| Run | Input tokens | Output tokens | Calls | Canonical final citations checked |
| --- | --- | --- | --- | --- |
| `full-v1` | 208,740 | 6,163 | 113 | 46 |
| `full-v2` | 176,387 | 4,733 | 98 | 45 |
| `focused-v1` | 181,187 | 5,907 | 107 | 44 |
| Total for this task | 566,314 | 16,803 | 318 | 135 |

All 89 scheduled trials have a start record, terminal record and prediction.
All 318 requests have corresponding responses and usage records, with distinct
provider request IDs. Provider usage agrees with per-trial aggregation and has
zero unknown-usage calls. Frozen Python hashes and cohort/configuration identities
verify. All 135 final citations match exact Unicode spans in the independently
loaded frozen source text. Valid offsets alone are not semantic support, which
is assessed in the separate review above.

Actual currency cost is unknown. No price estimate, dollar-cap enforcement or
provider invoice amount is inferred from token counts. The final original LLGM
arm uses 83 calls compared with 15 for flat retrieval. This is not a cost saving.
The separate focused cohort has a larger per-answer allowance and must not be
treated as an equal-budget repeat of the original cohort.

`make check` passes 776 tests and 230 subtests. Two installed-distribution checks
are skipped in the editable environment, and 20 integration tests are excluded
by the deterministic command. Ruff passes for 107 files, and the docstring audit
finds 1,783 documented definitions across 105 files. The hosted experiment runs
exercise actual Docker execution and model adapters separately from that command.
Strict Sphinx generation and the documentation navigation/API/publication-boundary
audit pass. Public architecture, walkthrough and implementation limits describe
the changed contracts.

## Scope that remains unmeasured

- Repeated independent histories, larger distractor sets, other model pairs and
  providers are needed to estimate general answer and navigation reliability.
  Models can still repeat a search or over-investigate a simple case.
- Curated edges and patches isolate inference. These runs do not establish
  learned link precision, journal proposal correctness or a forgetting policy.
- Empty-text abstention, hard context overflow, cancellation and several storage
  failure boundaries are covered by deterministic contracts, not each by a
  separately injected live trial in this task.
- Local Docker concurrency is exercised. Distributed metadata, S3 service
  operation, large-source streaming and scalability remain separate contracts.
- The package and documentation remain unpublished. This task did not dispatch
  CI, deploy a site or run additional GPU infrastructure.

The current implementation has no remaining failure in the declared final
diagnostic cases. Further research should use independent data and explicit
comparators instead of treating these development passes as a benchmark result.

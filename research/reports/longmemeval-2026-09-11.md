# LongMemEval: five-question pilot and preparation

September 11, 2026. The final candidate answered **all five development questions
correctly** in one fresh end-to-end run with the unchanged official evaluator.
It uses a GPT-5.4 root with medium reasoning and GPT-4.1 delegates and maintenance.
The full 500-question run remains stopped. Operator commands and fixed settings
belong in [the experiment guide](../../experiments/longmemeval.md).

## Goal and current scope

We want LLGM to beat reproduced Mem0 OSS and Graphiti on LongMemEval-S by at
least five percentage points at no higher total cost per question. The initial
comparison used LLGM, BM25 and full-context readers. Follow-up development runs
use LLGM only. A later comparison with matched reader models is needed to assess
whether the architecture adds value over simpler approaches. The named-framework
target remains untested.

The user authorized up to $500 API spend, then narrowed execution to five
questions first. The pilot completed within its separate $5 generation and
$0.10 judging allowances. The full run must not resume automatically. Framework
integration is paused while these failures are reviewed. Costs below use retained
token usage and declared prices. Local compute and storage remain unpriced.

## Final five-question result

| Question | Final answer | Official verdict |
| --- | --- | --- |
| Degree | Business Administration | Correct |
| Clothing | 1 blazer + 2 boots = 3 physical pieces, corresponding to 2 pickups | Correct |
| Updated personal-best 5K time | 25:50, superseding the earlier 27:12 statement | Correct |
| Museum visit interval | 7 days between January 8 and January 15 | Correct |
| Unstated hamster name | Explicit abstention | Correct |

All five answers come from `five-question-v6/`, with one scheduled attempt each.
They are not assembled from the best answers across iterations. Each question
received a fresh isolated history and ordinary automatic maintenance. The
reference answers and official judge remained unchanged throughout development.

The independent audit passed 555 integrity/accounting checks. All 61 executed
Python modules match the frozen candidate and all ten final citations match
stored source text, role, date and canonical coordinates. All final claims were
reviewed against their supplied histories. The five root responses reported
positive reasoning-token counts, and their retained requests omit temperature.
The exact medium setting is established by the frozen protocol and executed
adapter path. The record does not contain a separate raw HTTP payload.

V6 completed all 237 maintenance outcomes, making 231 maintenance model calls
and accepting 67 primary edges. Its 15 seed delegates made 53 read callbacks,
with no query or Python errors or truncated responses. No delegate queried a
recursive child or inspected edges, and no journal amendment was applied.
Consequently this run does not demonstrate a benefit from recursive graph
traversal or journal correction. All five runtime outcomes remain `partial`
because the declared three-seed policy skipped 76 other seed candidates in total.
The official correctness verdict does not erase those coverage limits.


| Frozen candidate | Root configuration | Correct | API cost including judging |
| --- | --- | ---: | ---: |
| v1 | GPT-4.1, initial three-arm comparison | 2/5 LLGM | $3.139091 across all three arms |
| v2 | GPT-4.1, failed prompt-only reading repair | 1/5 | $1.761923 |
| v3 | GPT-4.1, printed reads and turn metadata | 3/5 | $1.631284 |
| v4 | GPT-4.1, date and unit instructions | 3/5 | $1.639703 |
| v5 | GPT-5.4, source-first evidence, default reasoning none | 4/5 | $1.6602105 |
| v6 | GPT-5.4, same prompt and presentation, medium reasoning | 5/5 | $1.656449 |

V6 cost $1.342948 for construction, $0.3104735 for answering, and $0.0030275
for judging. All usage is known. Prices are token-based API estimates, not
invoices. Local CPU, Docker and storage remain unpriced. Root-only replays are
separate costs and do not establish end-to-end accuracy.

This meets the requested five-question development gate. It does not establish
benchmark superiority, an independent accuracy estimate, or a framework win.
The questions and failure traces were exposed during development. Earlier BM25
and full-context controls used GPT-4.1 readers, so comparing them with the new
reasoning root would mix model allocation with architecture effects. The clothing
annotation also remains ambiguous about counting units, even though the final
answer explicitly distinguishes physical pieces from pickups.

## The concrete repair

The search already found and admitted the necessary nodes for the original
failures. The facts were lost after retrieval. Three small changes target that
observed path:

| Failure | Implemented change | What it fixes |
| --- | --- | --- |
| A delegate read only the first passage, or read text without printing it | Show bounded printed reads of individual references, and supply role/coordinate metadata to every seed | The delegate can inspect the relevant user turns and actually see the returned text |
| The root received branch conclusions before their supporting quotes, with dates buried in metadata | Present deduplicated exact source quotes first, with speaker and source date beside each quote | The root can compare original statements instead of trusting a delegate's incomplete summary |
| Even with both values present, the root selected the older one or confused pairs with individual objects | Qualify a GPT-5.4 final root, then explicit medium reasoning with unchanged source-first prompt | Test whether the final model can interpret the collected facts consistently |

For example, the running-time sources report 27:12 on May 23 and refer to an
existing personal best of 25:50 on May 30. The first pilot never delivered the
newer number to the root. The reading repair delivered it, but GPT-4.1 still
selected 27:12. The GPT-5.4 v5 root selected 25:50 and cited both statements.
This separates retrieval, local reading and synthesis, rather than changing
all three because the final answer was wrong.

The clothing question has a separate ambiguity. One pair of boots and one
blazer represent two pickup groups and three physical pieces. The completed
exchange does not establish a third pending errand. A passing answer must make
its units clear without inventing a return. The unchanged official reference
is the scalar three and does not explain its counting convention.

These changes use the existing node delegates and single final root call.
They add no graph controller, verifier agent, retry loop or answer-specific rule.
The new native model option configures reasoning within the final call. Its
reasoning tokens share the existing output limit and are included in API cost.

## Initial five-question comparison

Selection took the first question in released dataset order for each of five
categories, before this pilot generated answers. These are exposed development
questions, including the previously used single-session case. They are not a
representative accuracy sample or an independent holdout. All supplied sessions
were ingested, 237 per arm and 711 source occurrences across the three arms.

| Question | Category | LLGM | BM25 | Full context |
| --- | --- | --- | --- | --- |
| `e47becba` | Single-session recall | Correct | Correct | Correct |
| `0a995998` | Multi-session count | Incorrect | Incorrect | Incorrect |
| `6a1eabeb` | Knowledge update | Incorrect | Correct | Correct |
| `gpt4_59149c77` | Temporal reasoning | Incorrect | Correct | Incorrect |
| `0862e8bf_abs` | Abstention | Correct | Correct | Correct |

All 15 scheduled answers and 15 official-prompt judgments are present. There were
no failed physical API calls, missing usage records or unresolved judgments.
The pilot made 276 generation calls and 15 judge calls. Strict and upstream
substring verdicts agree. Each answer was attempted once in this pilot, with no
replacement selected after inspection.

| Arm | Correct | Construction API cost | Answer API cost | Combined API cost |
| --- | ---: | ---: | ---: | ---: |
| LLGM | 2/5 | $1.359618 | $0.151554 | $1.511172 |
| BM25 | 4/5 | $0 | $0.113094 | $0.113094 |
| Full context | 3/5 | $0 | $1.507740 | $1.507740 |

Judging cost another $0.007085, making the pilot total **$3.139091**. BM25 was
both more accurate and cheaper on these five questions. This small result does
not estimate the size of an advantage on the full benchmark.

LLGM's 237 maintenance outcomes completed and published 80 primary edges.
Its five answers are marked `partial` because initial retrieval found more nodes
than the three-seed allowance admitted. All admitted delegates returned without
an operational failure. Some delegates also reported missing local facts.
The `partial` status does not turn an incorrect answer into a correct one.

The run used actual Docker Python delegates and final root synthesis, but no
delegate invoked a recursive child. Twelve edge inspections exposed eight
neighbor references, followed by no `query_node` calls or direct cross-node
reads. No journal amendments were generated or applied. Therefore
this pilot does not demonstrate useful recursive traversal or journal correction.
It does show that automatic edge construction incurred substantial cost without
recursive use in these attempts.

The independent audit reconciled all 291 unique physical call records and all
15 scheduled answers and judgments. All 60 frozen Python source files match the
executed candidate. All 24 final cited spans match stored text, role, date and
coordinates. For each of LLGM's three misses, all annotated source nodes were
among the admitted seeds. These integrity checks do not certify the generated
claims or the semantics of the 80 proposed edges.

### What the misses reveal

**Multi-session count:** LLGM returned one pair of boots. Two admitted delegates
read repeated references to the Zara boots, while the third read generic
assistant advice instead of the user turn about collecting a blazer. All three
annotated source histories were admitted, so the failure includes reading within
nodes, not only the seed limit. BM25 included the boots and blazer but did not
give the reference count of three. All arms received the official incorrect
verdict. The annotation deserves a separate note: its marked turns mention a
blazer and apparently the same exchanged boots twice. The scalar reference does
not explain whether individual boots or repeated mentions account for three.
Preserve that ambiguity alongside the official score, without silently rescoring.

**Knowledge update:** LLGM answered 27:12 from the older May 23 session. The
newer May 30 session was also admitted and contains 25:50 in its opening user
turn. Its delegate instead read a later turn that omitted the number and returned
that the personal best was unspecified. The root therefore received the old
value without the new one. Both simple controls found 25:50. The failure occurs
while collecting the relevant facts from an admitted node.

**Temporal reasoning:** The relevant January 8 MoMA and January 15 Met passages
were read and returned. Each delegate nonetheless answered the entire interval
question using its local date, claiming zero days. The root repeated the wrong
summary and cited one passage that did not establish either named event pair.
The reference interval is seven days. Exact citation offsets passed, while the
interpretation and claim support failed. Full context abstained despite having
the necessary history. BM25 answered seven days.

### Next repair to evaluate

Keep the next change small and centered on evidence delivery. A node delegate
should collect the facts it can establish locally, including the named event,
value, date and supporting span. It should leave cross-node counting, comparison
and date arithmetic to the root when it lacks the other facts. Before returning
an omission, it should search or inspect other relevant turns in its own node.
The root should compare dated facts and require support for each part of a
combined claim, rather than accept agreement between unsupported branch answers.

Use the retained traces to verify exactly where facts were dropped, then evaluate
one separately frozen candidate on the same five development questions. Keep this
failed candidate intact. Do not add more frameworks or launch 500 questions merely
to obtain a larger failure count. Recursive traversal needs an explicitly observed
child invocation before its contribution can be claimed. These are proposed
repairs, not changes verified by this run.

### Five-question improvement pass

The user requested improvement until all five development questions are correct,
with full test suites deferred until that point. Trace inspection found that
11 of 15 delegates copied the runnable first-reference example exactly. All
15 finished on their second model response despite having exploration calls
available. The missed blazer and newer running time were already among the
supplied passage handles. This gives a concrete first change: remove the example
and the instruction that encourages finishing immediately after a read.

Candidate v2 changes the existing node and root prompts. Delegates are instructed
to inspect relevant passage candidates in bounded batches and return supported
local subjects, values, dates and units. The root must verify the quoted facts
before counting or comparing dates. Counts must state their units and distinguish
repeated mentions from additional objects. No benchmark answer, event name or
case-specific date was added to these instructions. No new class, controller or
search backend was added.

The separate `longmemeval_pilot_v2.json` protocol runs LLGM on the same five
questions. The unchanged comparison readers are not rerun. Each trial still
builds a fresh memory, so construction cost remains visible. The candidate is
frozen under `five-candidate-v2/`, and attempts are retained under
`five-question-v2/`. The 62 affected runtime and evaluation tests passed before
dispatch. It scored **1/5**, with $1.759808 generation and $0.002115 judging,
for $1.761923 total and no unknown usage. Only abstention was correct.

This revision regressed the model interface. Delegates called `read` without
printing its return value, leaving model observations empty despite successful
source access. Some also invented unsupported batch-reference wrappers. The
Python interpreter executes statements without automatically displaying bare
expression values. Removing the runnable example had also removed the models'
most reliable demonstration of that boundary. These failed attempts are retained.

Candidate v3 restores a bounded loop example that prints each individual
`read(reference)` result, explicitly documents the single-reference API and empty
stdout behavior, and supplies the existing bounded metadata page to every seed.
Ranked passage handles are preserved. The metadata identifies user and assistant
turns without loading source text into model context. Overlapping prompt text
was shortened, and the per-observation instruction no longer urges an immediate
return. The existing huge-source test now covers both node-only and retrieved
seeds. All 63 affected checks passed. The separately frozen run at
`five-question-v3/` scored **3/5**, costing $1.628924 for generation and $0.002360
for judging, or $1.631284 total with no unknown usage. Degree, temporal reasoning
and abstention were correct. All relevant reading observations were now visible,
and no query callback errors occurred. One maintenance proposal failed schema
validation during temporal-history construction.

The remaining running-time failure moved to synthesis: a delegate correctly
returned 25:50 with May 30 evidence, but the root still chose the May 23 value
27:12. The clothing answer now included both clear pending pickups, boots and
blazer, and excluded an old-boots return that appeared already completed. Its
count of two pickups was rejected against the scalar reference three.

Candidate v4 leaves the corrected reading path unchanged and clarifies two root
rules. For a changing attribute, compare statements about the same subject in
date order and use the latest supported value at the query date unless an earlier
time is requested. A future-plan sentence can still report a current fact. For
counts involving known-size groups, distinguish groups or pending actions from
individual objects and state both counts when the question leaves the unit
ambiguous. This avoids treating repeated mentions as new objects. The model
instructions contain no benchmark values, names, question IDs or reference answers.
The separately frozen five-question run at `five-question-v4/` completed **3/5**,
with the same two failures. Generation cost $1.637338 and judging cost $0.002365,
for $1.639703 total. All 272 physical calls have usage and all maintenance
outcomes completed. All 62 reads were printed, with no query callback errors.
The root still chose the older running time despite receiving the update.
The changed instructions are present in its actual request.

Candidate v5 changes the private root input presentation. The exact selected
quotes now appear before the branch summaries. Each has its citation ID, speaker,
source date and timestamp beside the text, while retaining complete metadata and
canonical references. Shared citation IDs are deduplicated. The branch summaries
remain attributed and explicitly fallible. Both empty-root reservation and
branch-admission accounting use the new representation. Public result records,
recursive child payloads and the canonical citation registry are unchanged.
The focused runtime and evaluation checks passed 107 tests with one optional skip
and seven subtests. Full test suites were deferred until the five-case goal.

Root-only replays reuse the two failed v4 requests to isolate synthesis from
memory construction. They retain every call, response and cost. They neither
replace earlier answers nor count as a completed five-question run. Presentation
alone did not reliably fix GPT-4.1. One clothing response reached three by
inventing an outstanding old-boots return, which is not accepted as a semantic
success. A later response omitted the blazer altogether.

The model comparison then holds the two inputs fixed and compares GPT-4.1 with
`gpt-5.4-2026-03-05`. The latter correctly interprets the newer personal-best
statement. Counting remains sensitive to units, so the candidate instructions
neutrally require labeled numeric totals and short arithmetic for each plausible
unit. They do not prioritize the unit matching the scalar reference. Some replay
responses still open with an ambiguous two-item count before showing three
physical pieces. This is retained as a limitation rather than a clean pass.

The frozen v5 protocol keeps GPT-4.1 delegates and maintenance, but uses GPT-5.4
for the final root with default reasoning effort none and temperature zero.
The fresh five-question run at `five-question-v5/` completed **4/5** under the
unchanged official evaluator. Degree, the updated running time, the museum
interval and abstention are correct. Clothing still returns two groups and calls
them individual items, failing to give the alternate three-piece count. Generation
cost $1.656963 and judging cost $0.0032475, for $1.6602105 total with no unknown
usage. The six root-only diagnostic directories preceding v5 cost $0.150355
combined, separately from all end-to-end runs. Their records are itemized in
`root-replay-costs.json`. Earlier BM25/full-context results used GPT-4.1 final readers, so their
comparison with v5 is unmatched and cannot isolate an architecture contribution.

Candidate v6 leaves the v5 root prompt and evidence presentation unchanged and
sets the existing final call to medium reasoning. The native adapter exposes
this option explicitly, rejects conflicting temperature parameters before
provider dispatch, and records it in its descriptor. The benchmark protocol
records the effort, while final-reader requests retain temperature None. The
52 model/accounting tests and 24 runner tests passed, with 15 subtests.
The two retained v5 root inputs then passed a semantic replay for $0.0480175,
with known usage. The fresh `five-question-v6/` run completed 5/5, as reported above. All previous
failed attempts and replay responses remain intact. After the five-case gate, `make check` passed 1,039 tests and 245 subtests, with
two skipped and 21 opt-in integration cases deselected. Lint and all-definition
docstring checks passed. The default CLI and Make protocol now select v6. This
command-default edit and a recorder docstring punctuation edit postdate the
frozen executed package without changing its inference behavior.
Strict `make docs` and rendered configuration, architecture and walkthrough
checks passed. The default benchmark preparation command validated exactly five
questions with zero model calls. Logs are retained as `check-after-five-v6.log`,
`docs-after-five-v6.log` and `default-v6-preflight.log`.

The entire recorded LongMemEval sequence, including preparation, the stopped
full run, six five-question candidates and seven root-only diagnostics, has
$28.3853245 known estimated API cost. Older calls retain $0.039778 in unknown
reservations. `development-costs.json` records this scope separately from the
successful final run's $1.656449. No additional paid run remains active.


## Earlier runner and cleanup changes

Nine retired experiment-specific test suites were removed. Meaningful budget,
cancellation and unknown-usage checks moved into shared accounting tests.
A remaining import-identity-only suite and incidental fixture assertions were
also removed. Product storage, journal, retrieval, interpreter isolation and
evidence-provenance tests remain. The final local gate passed 1,016 tests and
238 subtests, with two skipped and 21 opt-in integration cases deselected.
The strict documentation build and rendered publication checks also passed.

One runner now owns ordinary ingestion, answering, common reader controls and
separate official-prompt judging. It freezes the exact imported package sources,
dataset/evaluator hashes, models, prompts, budgets and complete attempt schedule.
Every provider request is durably recorded before dispatch. Unknown usage retains
its reservation. Unstarted attempts and missing judgments stay in the denominator.
Offline reporting recovers individual calls even when a final aggregate was lost.
The pinned judge function is validated before any paid generation.

Three observed defects were repaired before freezing the candidate:

1. Root synthesis could ignore a sibling's finding or cite a node identifier as
   evidence. Its native schema now permits only selected branch evidence IDs.
   Host validation remains, and the prompt distinguishes a local omission from
   absence across all branches.
2. Maintenance copied whole passage objects where exact references were required,
   sometimes truncating its output. The request now has a structural native
   schema, and the maintenance client forwards that schema into budget admission
   and the provider request. The prompt asks for only the inner reference value.
3. A controlled Docker overlap reproduced removal already in progress while the
   container was in fact being removed. Cleanup now confirms that exact container's
   absence within the existing deadline. It does not repeat removal. The original
   smoke failure's precise cause remains unproved.

The Docker reproduction failed all eight controlled closes before the fix and
passed all eight afterward. Eight ordinary post-fix closes also passed. Every
final inspection confirmed the intended container was absent.

## Retained paid development results

These are diagnostics, not estimates of LongMemEval accuracy. Each changed
candidate has a separate retained run. Earlier attempts were not overwritten.

| Run | Scope | LLGM correct | BM25 correct | Full context correct | Known API cost |
| --- | --- | ---: | ---: | ---: | ---: |
| `smoke-v2` | Ten independently authored plumbing questions | 8/10 | 10/10 | 10/10 | $0.2095785 |
| `smoke-v3` | Same questions after root repair | 9/10 | 10/10 | 10/10 | $0.2049765 |
| `cost-pilot-v1` | First released LongMemEval case, 53 supplied sessions | 1/1 | 1/1 | 0/1 | $0.7109935 |
| `cost-pilot-v2` | Same exposed case after maintenance and cleanup repair | 1/1 | 1/1 | 1/1 | $0.6630455 |
| `smoke-v4` | Final candidate, including token pacing | 10/10 | 10/10 | 10/10 | $0.2169475 |

The initial `smoke-v1` stopped before any API call because judge-client setup
omitted its required base URL. One separate provider qualification call returned
but its local wrapper lost usage while incorrectly awaiting a synchronous parse.
Its $0.00425 reservation remains unknown. A distinct corrected qualification
cost $0.000044. Neither provider qualification contained a benchmark question.

The full-history maintenance check improved from 42 completed and 11 failed
outcomes to all 53 completed outcomes. The repaired run published 16 primary
edges with exact endpoint provenance and used seven query calls. LLGM's outcome
was marked partial because the declared three-seed cap left other candidates
unvisited. Its answer was still judged correct. The full-context answer changed
between repetitions without a reader change, illustrating why one case cannot
establish an accuracy advantage.

An independent audit checked all 60 frozen Python files for both final diagnostics,
219 stored source occurrences, 39 final cited spans and 50 edge-provenance spans.
Roles, text, dates and supplied order matched the inputs. Original session aliases
and gold metadata were absent from generation, and maintenance received no
evaluation question. All 174 physical call IDs were unique, and recomputed costs
matched durable allowances. These are integrity checks, not a claim that every
generated edge or interpretation was semantically correct.

## Stopped full comparison

`full-v1` schedules all 500 LongMemEval-S questions for three arms, giving 1,500
attempts. Each arm receives a fresh workspace and its full supplied history.
LLGM uses automatic primary-edge maintenance and the ordinary retrieval-first
Python node runtime. No gold-selected seeds, manual edges or patches are supplied.
Automatic journal amendment generation is absent, so this run does not measure
a journal-generation benefit.

All generation roles use `gpt-4.1-2025-04-14`. Judging uses the pinned official
LongMemEval prompts with `gpt-4o-2024-08-06`, strict yes/no parsing, and the upstream
substring verdict retained separately. Failed answers score zero. Missing judge
responses leave accuracy incomplete.

Four cases run concurrently, with rotated serial arm order within each case.
Generation roles share a 400,000-token sliding 60-second admission window against
a measured 450,000-token provider limit. Queue time remains in measured latency
and existing deadlines. SDK retries are disabled. There is one declared answer
attempt, with no replacement answer selected after inspection.

The complete full-context serializer was checked on all 500 histories before
dispatch. All fit the 250,000-token limit, with a maximum of 169,474 and median
158,237 `cl100k_base` tokens. None require clipping. The histories contain 23,867
source-session occurrences per arm. Source order is preserved even where dates
are not chronological.

The user stopped this run before judging. It retained 74 attempted trials, with
70 finished answers and four interrupted trials, from the original 1,500-attempt
schedule. There were 1,340 physical calls with $14.692706 known estimated API
cost. One canceled call retains $0.035528 of unknown-usage reservation. Python
and its process-scoped sleep inhibitor both exited. `stop.json` records the
reason and preserves the instruction not to restart automatically.

No full-benchmark accuracy follows from these incomplete records. Public
histories overlap and include previously exposed diagnostic questions. They
are not 500 independent, previously untouched histories.

Across preparation, the stopped run and the five-question pilot, known estimated
API cost is $19.8373825. The two unknown-usage calls retain $0.039778 of
reservations, giving $19.8771605 of local cost plus unresolved liability. These
figures cover this evaluation sequence, not older research runs or an invoice.

## Artifact locations

All local artifacts are under `runs/longmemeval-20260911/`:

- `five-question-v1/` retains all 15 pilot answers, judgments, traces and costs.
- `five-question-v1/independent-audit.json` verifies schedule, source identity,
  citations, accounting, recursive activity and the three failure traces.
- `five-question-selection.json` records selection before pilot generation.
- `full-v1/` retains the interrupted full comparison and its stop record.
- `preparation-costs.json` reconciles preparatory physical call files.
- `final-audit.json` records independent diagnostic and protocol checks.
- `full-reader-preflight.json` records complete-input admission for all 500 cases.
- `cleanup/` retains before/after Docker reproductions.
- `check-prefreeze.log` and `docs-prefreeze.log` retain local validation output.
- `five-default-preflight/` confirms the current CLI default schedules five
  questions without model calls. `docs-five-question.log` and
  `lint-five-question.log` retain the final documentation and static checks.

The pilot executed `full-candidate-v1/llgm` through `PYTHONPATH`, then copied that
imported package into its own source artifact. Subsequent checkout changes to
shared reader helpers, optional framework recording and the five-question CLI
default were not part of this frozen model run.

Mem0 and Graphiti adapters are paused subsequent work. Their extraction, embeddings,
reranking, internal fallbacks, temporal preparation and resource costs must be
observable before a framework comparison is trustworthy. No framework score or
claim that the numerical goal has been met follows from these controls.

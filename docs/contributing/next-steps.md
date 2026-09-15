# Next steps: one clear conversation pipeline

Status: proposed refactor plan, recorded September 13, 2026. This document defines
the next implementation review. It does not change the library's current behavior.
The [conversation guide](../guide/conversations.md) describes the existing API.

## Objective

Make LLGM's normal conversation flow understandable as one sequence. The execution
trace, implementation, tests and tutorials should describe that same sequence.
Pause feature expansion while establishing this contract and repairing the gaps.

Refactor the orchestration around the contract. Keep storage, retrieval, provider
adapters and DSPy execution where they already satisfy it. Preserve one answer
controller and avoid building another framework around the existing components.

## Proposed execution contract

The default flow is:

```text
Place and save the question
    -> select useful starting nodes
    -> prepare each Reader with its journal
    -> read locally and investigate further when needed
    -> produce one Main answer
    -> save the reply and maintain the affected topic
```

### 1. Place and save the question

The Graph model receives the incoming turns, the conversation's current topic,
and bounded candidate previews. It continues that topic, returns to an existing
topic, or creates a new one for a clear topic change. LLGM appends the incoming
turns and records the current-topic pointer.

Output: a current node and references to the saved turns. Earlier turns retain
their text, roles and source identities. A new Q&A does not require a new node.

### 2. Select the starting nodes

Always include the current node. Search retrieves candidate passages from other
nodes. An explicit relevance decision admits zero or more additional node owners
within the seed limit. An available slot is not a reason to admit a candidate.

Output: the selected node IDs, relevant passage references, and admission reasons.
Ranking proposes candidates. Admission decides whether investigating them is
useful. Choose and validate the admission method before changing the default.

Selected nodes may be connected or disconnected. A connection between distinct
topics can be useful. Prevent unnecessary repeated investigation without treating
connectivity as proof of redundancy.

### 3. Prepare each Reader

The runtime starts an independent Reader RLM invocation for every admitted node,
subject to bounded concurrency. It loads the complete operational journal before
source inspection. Readers receive the question, source coordinates and the
applicable read instructions without loading the full topic into model context.

Output: a local reading context with its amendments ready. Source reads apply
those amendments before exposing text and preserve canonical provenance.

### 4. Investigate locally, then expand when useful

Each Reader inspects relevant source text in its own node before expanding to
other nodes. This requires actual local evidence access, not a request to read
the entire topic. The runtime must enforce that ordering.

The Reader can inspect outgoing connections, search for a missing fact, read a
discovered reference directly, or ask a child Reader to investigate another node.
It chooses useful paths and can stop without inspecting every connection.
Each child follows the same preparation and local-reading contract. Child
findings return to its parent for selection.

Output: selected findings, cited source excerpts and explicit unresolved needs.
Shared budgets bound exploration and reserve finalization capacity. The protocol
must define how repeated investigations within one answer are detected without
suppressing distinct questions about the same node.

### 5. Produce the answer

After the admitted branches settle, Main receives their selected evidence,
findings and unresolved needs. It performs one final synthesis call. Complete
Reader histories and unselected intermediate observations stay outside that call.

Output: an answer with references and an honest completion status. Main does not
start another retrieval phase. Handled branch failures can support an explicitly
partial answer when enough evidence remains.

### 6. Save the reply and maintain the affected topic

Append the nonempty assistant reply to the same topic, preserving its assistant
role. Run bounded maintenance after each completed exchange, including an append
to an existing topic. Use the newly appended turns to identify relevant changes.
Do not rely only on the beginning of a long topic.

The Graph model proposes useful connections and supported amendments. The host
validates and publishes accepted changes. Amendment generation needs a defined
evidence policy before it is enabled. Stored assistant text remains fallible
history and cannot become independent confirmation of its own claims.

Output: saved conversation turns and a visible maintenance outcome. Preserve
committed turns if maintenance fails. Use the same update rules for imported
turns and live conversation updates, without generating replacement replies
during historical replay.

## Rules that apply across the pipeline

- Nodes hold information. Journals describe amendments to that information.
  A correction identifies what it changes and the source evidence supporting it.
  Unrelated new facts belong in source turns.
- Keep general forgetting deferred. Operational reconciliation does not imply
  physical history deletion. Preserve existing journal records and references
  while defining any transition away from standalone journal assertions.
- The host enforces stage ordering, admission limits, persistence, citation access
  and cleanup. Prompts guide relevance judgments and interpretation within those
  boundaries.
- Define one failure policy for committed input, missing evidence, exhausted
  limits, cancellation and unfinished maintenance. Report what completed and
  preserve committed evidence. Do not hide required failures behind fluent text.
- Read-only queries and exact imports remain explicit workflows with documented
  write behavior. Their implementation should reuse the relevant stages without
  obscuring the default conversation flow.

## Implementation gaps to review

The following observations were checked against revision `fead6a9`. They are a
starting checklist, not a claim that the target contract is implemented.

| Area | Current behavior | Required work |
| --- | --- | --- |
| Topic placement and persistence | Current-topic routing and append storage exist. | Preserve and test them as the entry stage. |
| Additional seeds | Ranked owners fill available slots without a relevance gate. | Add an admission policy that can select no extras. |
| Journal preparation | The operational journal loads before Reader execution and effective reads. | Retain this behavior for seeds and children. |
| Local reading | Local-first behavior is prompted, but recursion can happen first. | Enforce local inspection before expansion. |
| Journal purpose | The API accepts standalone assertions as well as amendments. | Define supported amendment rules and a transition that preserves old evidence. |
| Main synthesis | Selected branch output reaches one final Main call. | Retain this boundary and make partial outcomes clear. |
| Maintenance timing | Conversation answering organizes connections only for a new topic. Imports organize after a batch. | Apply one update contract to existing-topic appends and imported turns. |
| Maintenance input | Connection discovery starts from a bounded topic prefix. | Include the appended evidence that triggered the update. |

Start with [application orchestration](../../src/llgm/llgm.py),
[topic routing](../../src/llgm/memory/topics.py),
[node execution](../../src/llgm/inference/nodes.py),
[effective reads](../../src/llgm/memory/query.py),
[workspace persistence](../../src/llgm/memory/workspace.py), and
[maintenance policy](../../src/llgm/memory/maintenance.py).

## Work order

1. Finalize each stage's input, output, owner and completion condition. Select the
   relevance admission method, supported amendment policy, within-answer reuse
   rules, and maintenance timing and failure behavior. Keep these choices in one
   contract rather than scattering them across prompts and settings.
2. Review the gap table against the current checkout. Classify each contract as
   implemented, missing or conflicting, and identify its owning code and tests.
3. Refactor the default path in small changes. Keep application coordination
   separate from storage and maintenance policy. Reuse existing components and
   remove redundant branches as their replacements become verified.
4. Run complete conversation checks using real storage and retrieval. Use
   controlled model decisions to test enforcement, real sandbox execution to
   test Reader callbacks, and a small hosted run to assess actual Reader behavior.
5. Update the user tutorials, API reference, contributor code map and shared
   context once behavior changes. Remove superseded guidance. Resume broader
   experiments only after the normal pipeline is consistent.

## Acceptance scenarios

| Scenario | Required observation |
| --- | --- |
| Ordinary follow-up | The current topic is reused, the question and reply append, and maintenance runs. |
| Clear topic change and return | Routing creates a topic only when needed and can return to the earlier node. |
| Irrelevant search candidates | The current node remains selected and all unnecessary extras are rejected. |
| Useful additional topic | Its Reader contributes evidence that reaches Main with valid references. |
| Connected starting nodes | Connectivity does not disqualify a useful seed, and repeated work follows the declared reuse policy. |
| Useful recursive query | Local inspection occurs before child admission, and selected child evidence returns through its parent. |
| Explicit correction | Effective reads use supported replacement evidence while original source references remain readable. |
| Late update to a large topic | Maintenance sees the new evidence even when it is absent from the topic prefix. |
| Import and live update | Equivalent turns follow the same maintenance rules without replay generating new replies. |
| Failure or cancellation | Committed input remains, resources close, and incomplete answering or maintenance stays visible. |
| Read-only question | Relevant evidence is inspected without appending turns or moving the conversation pointer. |

Inspect the resulting traces and intermediate memory state, not only final answer
text. A passing deterministic test establishes a contract. A sandbox check proves
execution at that boundary. A hosted run measures model behavior for its stated
inputs and configuration. These are separate forms of evidence.

Follow [testing](testing.md) for commands and integration prerequisites. Run
focused contract tests during the refactor, then `make check`. Use `make coverage`
to inspect untested branches, `make docs` for the user site, and `make docs-links`
for repository guidance. Benchmark protocols and execution commands stay in
[experiments](../../experiments/README.md).

## First deliverable

A reviewed execution contract and an updated implementation gap list come first.
The refactor is complete when the normal workflow can be explained in the six
steps above and the implementation, traces, tests and tutorials agree with it.
This establishes a coherent product baseline. It does not establish benchmark
superiority or a mathematical graphical-model inference guarantee.

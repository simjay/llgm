# Agent context

Shared project knowledge for coding agents working on LLGM. Start with the root
[AGENTS.md](../AGENTS.md), which points to the canonical contributor instructions.
This directory helps an agent find the relevant code, retain design decisions,
and choose useful validation. It is repository guidance, outside the user site.

## Format and reading order

[AGENTS.md](https://agents.md/) is the open Markdown format for coding-agent
instructions. The files below are LLGM's organization of supporting knowledge,
not an additional standardized schema or a dependency on a particular agent.

The linked [agent-context article](https://www.puppyone.ai/en/blog/what-is-agent-context)
distinguishes durable stored knowledge from the working context selected for a
task. This directory holds the former. Load only the files and source sections
needed for the current request, along with fresh observations from tools.

| File | Read it when |
| --- | --- |
| [Project context](PROJECT_CONTEXT.md) | Starting work and checking the current product scope. |
| [Architecture](ARCHITECTURE.md) | Locating implementation boundaries and the tests that protect them. |
| [Decisions](DECISIONS.md) | Considering a change that could alter an established design. |
| [Testing](TESTING.md) | Choosing checks or interpreting what their results establish. |
| [Remaining tasks](REMAINING_TASKS.md) | Selecting unfinished work within the user's request. |

Conversation history and terminal output describe a particular task. They are
not repository instructions. Confirm the checkout, dependency availability, and
relevant behavior before relying on older observations.

## What belongs here

Keep stable project facts, design rationale, code pointers, and actionable open
work. Link the owning source or guide instead of copying its full contents.
Do not store chat transcripts, raw tool output, credentials, account details,
machine paths, private datasets, or local experiment results in tracked context.

The six Markdown files listed here, including this README, are explicitly allowed
by `.gitignore`. Other files in this directory stay ignored. Local scratch notes
can live under `local/` and may be absent in a fresh clone. Never force-add them.
Research and retained run artifacts also remain private and outside this context.

## Updating context

Update the relevant file when a code boundary, settled decision, or task status
changes. A decision should state its rationale and point to its implementation.
An open task should state its current status and an observable completion check.
Remove stale session summaries instead of accumulating a chronological log.

Record validation against the actual revision checked. A test being available
does not mean it ran. Keep proposed behavior separate from implemented behavior,
and leave unverified outcomes explicit. Git records revisions to this shared
knowledge. The user's current request determines what work to perform.

Follow [documentation ownership](../development/documentation.md) for updates.
Library users should find all installation, usage, and API information in
[the user documentation](../docs/index.md) without reading these files.

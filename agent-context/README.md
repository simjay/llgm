# Agent context

Shared project knowledge for coding agents working on LLGM. Start with the root
[AGENTS.md](../AGENTS.md), which points to the canonical contributor instructions.
This directory retains design decisions and links to canonical code and
validation guides. Architecture and testing files are navigation pointers.
It is repository guidance, outside the user site.

## Reading order

Read project context when starting a task, then select the supporting files
needed for that change. These files store durable repository knowledge. Fresh
code and tool observations establish the current state of the checkout.

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

Follow [documentation ownership](../docs/contributing/documentation.md) for updates.
Library users should find all installation, usage, and API information in
[the user documentation](../docs/index.md) without reading these files.

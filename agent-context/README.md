# Agent context

This directory is working context for agents. Supported library behavior lives
in [the end-user docs](../docs/index.md). Contributor and maintainer instructions
live in [development](../development/README.md). Neither audience should need
these agent notes to use or maintain the library.

## Read for the current task

1. Read [project context](PROJECT_CONTEXT.md) for settled decisions and constraints.
2. Consult [remaining tasks](REMAINING_TASKS.md) for unfinished work relevant to the request.
3. Use [code boundaries](ARCHITECTURE.md) to find the owning implementation and tests.
4. Read [local state](LOCAL_STATE.md) only when the task depends on this checkout's environment or artifacts.

The user's current request takes precedence over the backlog. Read the owning
guide and code before editing. These notes do not replace either. Start with
`git status --short` and preserve existing changes, including untracked files.

## Maintain the context

Keep decisions in project context, unfinished work in the backlog, and dated
machine observations in local state. Replace obsolete state rather than appending
a session narrative. Store measurements in dated research reports and link them
from here. `research/` is internal and must not be included in the published
documentation or its navigation. Repository experiment commands belong in
[experiments](../experiments/README.md). The
[documentation maintenance guide](../development/documentation.md)
defines ownership and validation for the rest of the repository.

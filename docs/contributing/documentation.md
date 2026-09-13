# Documentation conventions

The published documentation serves people using the LLGM library. It explains
installation, configuration, behavior, examples, and the public API. Repository
development, testing, release operations, and research have separate homes.

## Content ownership

| Location | Audience and responsibility |
| --- | --- |
| Repository `README.md` | Project introduction and links to the user documentation and contributor entry point. |
| `CONTRIBUTING.md` | Contributor entry point for setup, standards, and checks. |
| `docs/guide/` | Library users: installation, conversations and imports, configuration, concepts, architecture, search, graph inspection and worked examples. |
| `docs/reference/` | Library users: generated public API signatures and current capability limits. |
| `docs/contributing/` | Repository-only contributor and maintainer guides: checkout setup, code ownership, standards, testing, documentation, branding, and publication. |
| `examples/` | Runnable examples, indexed by its `README.md`. |
| `experiments/` | Repository-only benchmark configurations, frozen protocols, source pins, and run commands. |
| `AGENTS.md` | Tracked instructions for agents working in a fresh clone. Contributor guidance remains in `CONTRIBUTING.md` and `docs/contributing/`. Excluded from the documentation site and source distribution. |
| Local `research/` | Ignored, untracked designs, hypotheses, original briefs, and dated measurements. Never committed, pushed, or published. |
| `agent-context/` | Six tracked Markdown files for shared coding-agent knowledge, decisions, code pointers, testing, and remaining work. Excluded from the user site and packages. All other files in this directory stay private. |

Give each fact a canonical home. Instructions for running the library belong in
`docs/guide/` and `docs/reference/`. Instructions for changing, testing, or releasing this repository belong
in `docs/contributing/`. Agent notes may refer to either without copying them. Keep
machine paths, transient credential availability, and current task state in
ignored local agent notes. Never store credentials in any of these documents.

Keep `research/`, private run artifacts, and machine-specific context local.
Do not stage, commit, push, or copy their contents into tracked files, package
archives, or published assets. `.gitignore` allows only `README.md`,
`PROJECT_CONTEXT.md`, `ARCHITECTURE.md`, `DECISIONS.md`, `TESTING.md`, and
`REMAINING_TASKS.md` under `agent-context/`. Other entries there, including `local/`
and `LOCAL_STATE.md`, remain ignored. Never force-add them.

[Agent context](../../agent-context/README.md) follows the root
[AGENTS.md](../../AGENTS.md) entry point. It stores concise project knowledge, not
conversation transcripts or private experiment reports. Repository guidance may
link these shared files. Tracked documents must not depend on private notes or
research. Contributor instructions, tests, and builds must work in a fresh clone
without those local files.

The user documentation must stand alone. Its explanations, examples, navigation
and build checks must work without `research/`, `agent-context/`, `AGENTS.md`,
or maintainer guides. Verify behavior against the implementation. Internal notes
can help locate code, but are not required reading or build inputs for users.

## Navigation exposes every page

The root README introduces the project and links to section entry points.
Directory indexes explain their own contents. In the Sphinx site,
`docs/index.md` owns the complete sidebar through two captioned `toctree` blocks:
User guide and Reference. Each block lists its overview and every page directly.
Keep these page links visible on the home page, guide pages, reference pages and
search results. Only headings within an inactive page may collapse. Section
indexes are reading guides and must not introduce another nested navigation tree.
Cross-links may lead directly to a page or heading when they explain a concept.
They do not replace sidebar navigation.

The root `CONTRIBUTING.md` leads to `docs/contributing/README.md`. Contributor pages
use ordinary Markdown and relative links, without Sphinx directives. They are
read in the repository. Sphinx excludes `contributing/` entirely, including
its source exports and search entries. Removing pages from navigation alone
is insufficient. The public audit skips contributor sources, while
`make docs-links` checks their repository links.

Do not link, include, mirror, or offer downloads of repository-only development,
research, experiment runbooks, or agent notes in published pages. A library user
must not need maintainer instructions to understand a supported workflow.

Use descriptive link labels and relative repository paths. Moving a page also
requires updating incoming links, its section navigation, image paths, download
directives, and any validation code that names its generated HTML. Avoid
workstation-specific links. Sphinx API directives remain executable directives
in public reference pages, not code examples displayed as plain text.

`docs/index.md`, `docs/guide/`, and `docs/reference/` own the published prose. Sphinx also extracts public API
reference from library docstrings. Files under `docs/_static/` are selected site
assets. Brand explorations belong in `docs/contributing/brand-explorations/`.
Do not mirror or include contributor Markdown or Markdown from elsewhere in
the repository in public pages. Benchmark
run instructions stay in `experiments/`, and the examples README remains a
repository index. Public guides may link to or offer downloads of individual
example scripts when those files support the documented workflow.

Source distributions include the public documentation, contributor guides, and
required JSON inputs for experiment tools and tests. They exclude brand
explorations, `research/`, `AGENTS.md`, `agent-context/`, and experiment Markdown.
Archive inclusion does not make contributor guides part of the documentation
website. A local documentation build must not copy repository-only material into
published HTML, search indexes, or downloadable assets.

## Prose and historical records

Write direct statements about behavior, prerequisites, and limitations. Avoid
session narration, conversational quotations, repeated disclaimers, and claims
of completeness without evidence. Do not use em dashes or semicolons in prose.
Syntax inside code blocks and verbatim source documents keeps its original form.
This rule also applies to docstrings rendered in the public API reference.

Teach concepts and architecture through a concrete example. Introduce each term
when the reader first needs it, explain one step at a time, and show how each
step changes the example. Keep field lists and detailed contracts in reference
sections. Keep the first runnable program consistent across README, site home
and Quickstart. Put import and retry details in the conversation guide, settings
in configuration, and ranking details in node search. State when a tutorial
explicitly chooses a different backend from the library default.

Check examples against current signatures and remove obsolete advice instead
of keeping a history of previous designs in user pages. Installation commands
must identify an available release or a repository checkout. A pending release
command must not be the only way to run a guide.

Keep original briefs and dated research reports in local `research/`. Preserve
frozen experiment protocols in `experiments/` and generated run artifacts in
ignored `runs/`. A current public guide describes supported behavior and
limitations without reproducing internal reports or task history.

## Update the document that owns the change

| Change | Documentation to review |
| --- | --- |
| Public behavior, signature, or configuration | Relevant user guide and API docstrings. Implementation status when support changes. |
| Internal module boundary or resource ownership | Contributor code map and relevant implementation docstrings. Update user architecture only when observable behavior or a supported extension contract changes. |
| Validation command or contributor requirement | Contributor setup, testing guide, or standards, plus any command wrappers and CI that implement it. |
| Benchmark method or frozen inputs | Repository-only experiment instructions and the applicable protocol. Create a new protocol version for a changed frozen run. |
| Completed measurement | Local dated report with code/input identities, denominators, limitations, and artifact locations. Update public capability limits only when the evidence supports them. |
| Shared design decision or public task status | Relevant curated agent-context file, with implementation pointers and an observable completion check. |
| Session progress, private measurement or machine prerequisite | Ignored local notes, outside shared agent context. |

State implemented and verified behavior separately. Code and completed checks
determine what is supported. A later code change does not rerun an earlier
experiment or strengthen its conclusions.

## Verify the result

Run `make docs` after documentation or navigation changes. It performs a strict
Sphinx build and audits the rendered API, public authored and HTML links, prose
punctuation, and exclusion of repository-only material from published output.
It also requires every guide and reference page as a direct sidebar entry on
all authored pages, search, and the generated index. Links in the page body or
inside collapsed branches cannot satisfy that check. Verify desktop navigation
and the mobile menu in a browser after changing the sidebar.
It does not read or validate agent context. Use `make docs-links` separately
to check links across repository Markdown, including the six curated context
files. Private agent notes, archives, and research are excluded even when they
exist in the checkout. Their absence is tolerated in a fresh clone.
That optional check is not a documentation build prerequisite. Keep the
publication exclusion checks even when those local directories are absent.

Inspect the affected pages,
including their sidebar hierarchy, images, downloads, code blocks, and diagrams.
Check narrow screens as well as desktop layouts. Setting names in tables and
API signatures must not push explanations outside the viewport. Run standalone
storage examples as written, and distinguish application scripts from snippets
that require a model interpreter. In validation reports, distinguish provider
examples executed with hosted models from installation and syntax checks.
When moving or deleting a published page, remove only the generated
`docs/_build/` directory first, or choose a fresh `DOCS_BUILD_DIR`. A Sphinx
environment rebuild does not delete obsolete HTML. The audit rejects stale
pages that could otherwise remain publishable.
See [documentation checks](testing.md#documentation-checks) for what the audit
establishes and [publishing](publishing.md) for hosting configuration.

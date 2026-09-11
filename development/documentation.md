# Documentation conventions

The published documentation serves people using the LLGM library. It explains
installation, configuration, behavior, examples, and the public API. Repository
development, testing, release operations, and research have separate homes.

## Content ownership

| Location | Audience and responsibility |
| --- | --- |
| Repository `README.md` | Project introduction and links to the user documentation and contributor entry point. |
| `CONTRIBUTING.md` | Contributor entry point for setup, standards, and checks. |
| `docs/guide/` | Library users: installation, configuration, concepts, architecture, and worked examples. |
| `docs/reference/` | Library users: generated public API signatures and current capability limits. |
| `development/` | Repository-only contributor and maintainer guides: checkout setup, code ownership, standards, testing, documentation, branding, and publication. |
| `examples/` | Runnable examples, indexed by its `README.md`. |
| `experiments/` | Repository-only benchmark configurations, frozen protocols, source pins, and run commands. |
| `research/` | Internal designs, hypotheses, original briefs, and dated measurements. Excluded from the documentation site and source distribution. |
| `AGENTS.md` and `agent-context/` | Internal agent instructions, decisions, code ownership, and task state. Excluded from the documentation site and source distribution. |

Give each fact a canonical home. Instructions for running the library belong in
`docs/`. Instructions for changing, testing, or releasing this repository belong
in `development/`. Agent notes may refer to either without copying them. Keep
machine paths, transient credential availability, and current task state in
agent context. Never store credentials in any of these documents.

The user documentation must stand alone. Its explanations, examples, navigation
and build checks must work without `research/`, `agent-context/`, `AGENTS.md`,
or maintainer guides. Verify behavior against the implementation. Internal notes
can help locate code, but are not required reading or build inputs for users.

## Navigation follows the directory tree

The root README introduces the project and links to section entry points.
Directory indexes explain their own contents. In the Sphinx site,
`docs/index.md` owns the User guide and Reference indexes. Each section's
`index.md` owns its pages through a local `toctree`. Every public page must be
reachable through this hierarchy. Cross-links may lead directly to a page or
heading when they explain a specific concept. They do not replace section
navigation.

The root `CONTRIBUTING.md` leads to `development/README.md`. Development pages
use ordinary Markdown and relative links, without Sphinx directives. They are
read in the repository and have no entry in the published site's navigation.

Do not link, include, mirror, or offer downloads of repository-only development,
research, experiment runbooks, or agent notes in published pages. A library user
must not need maintainer instructions to understand a supported workflow.

Use descriptive link labels and relative repository paths. Moving a page also
requires updating incoming links, its section navigation, image paths, download
directives, and any validation code that names its generated HTML. Avoid
workstation-specific links. Sphinx API directives remain executable directives
in public reference pages, not code examples displayed as plain text.

`docs/` is the only published prose source. Sphinx also extracts public API
reference from library docstrings. Files under `docs/_static/` are selected site
assets. Brand explorations belong in `development/brand-explorations/`.
Do not mirror or include Markdown from elsewhere in the repository. Benchmark
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
sections. Check examples against current signatures and remove obsolete advice
instead of keeping a history of previous designs in user pages.

Keep historical records in their owning repository directories. Preserve
original briefs, frozen experiment inputs, and measured conditions. A current
public guide describes supported behavior and limitations without reproducing
internal reports or task history.

## Update the document that owns the change

| Change | Documentation to review |
| --- | --- |
| Public behavior, signature, or configuration | Relevant user guide and API docstrings. Implementation status when support changes. |
| Internal module boundary or resource ownership | Development code map and relevant implementation docstrings. Update user architecture only when observable behavior or a supported extension contract changes. |
| Validation command or contributor requirement | Development setup, testing guide, or standards, plus any command wrappers and CI that implement it. |
| Benchmark method or frozen inputs | Repository-only experiment instructions and the applicable protocol. Create a new protocol version for a changed frozen run. |
| Completed measurement | Internal dated report with code/input identities, denominators, limitations, and artifact locations. Update public capability limits only when the evidence supports them. |
| Internal task progress or local prerequisite | Agent backlog or local state, outside the published documentation. |

State implemented and verified behavior separately. Code and completed checks
determine what is supported. A later code change does not rerun an earlier
experiment or strengthen its conclusions.

## Verify the result

Run `make docs` after documentation or navigation changes. It performs a strict
Sphinx build and audits the rendered API, public authored and HTML links, prose
punctuation, and exclusion of repository-only material from published output.
It does not read or validate internal writing. Use `make docs-links` separately
to check links across repository Markdown, including research and agent context.
That optional check is not a documentation build prerequisite.

Inspect the affected pages,
including their sidebar hierarchy, images, downloads, code blocks, and diagrams.
When moving or deleting a published page, remove only the generated
`docs/_build/` directory first, or choose a fresh `DOCS_BUILD_DIR`. A Sphinx
environment rebuild does not delete obsolete HTML. The audit rejects stale
pages that could otherwise remain publishable.
See [documentation checks](testing.md#documentation-checks) for what the audit
establishes and [publishing](publishing.md) for hosting configuration.

# Publishing

This maintainer guide covers releasing the Python package to PyPI and publishing
the user documentation. Package releases and documentation deployments use
separate workflows.

## Release the package to PyPI

`.github/workflows/publish.yml` runs when a GitHub release is published, including
prereleases. It requires an exact match between the release tag and the version
in `pyproject.toml`, prefixed with `v`. For example, version `0.1.0a1` requires
tag `v0.1.0a1`. Pushing a tag alone or saving a draft release does not upload.

The release workflow calls `.github/workflows/ci.yml` from the same commit.
All CI jobs must pass, including Python 3.11 through 3.14, provider SDK contracts,
lint, coverage, documentation, and tests against fresh wheel and source archive
installations. CI retains those tested distributions as the
`python-distributions` artifact. The publish job downloads that artifact and
uploads it without rebuilding the package.

Publishing uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
Only the publish job has `id-token: write`. It runs in the `pypi` GitHub
environment and uses short-lived OpenID Connect credentials. No PyPI API token
or GitHub Actions secret is required. The publishing action also produces
digital attestations by default.

### One-time account setup

1. In `simjay/llgm`, create a GitHub environment named `pypi` under
   **Settings → Environments**. Set its deployment rules to allow tags matching
   `v*`.
2. On PyPI, configure a GitHub Trusted Publisher with these exact values:

   | Field | Value |
   | --- | --- |
   | PyPI project name | `llgm` |
   | Repository owner | `simjay` |
   | Repository name | `llgm` |
   | Workflow filename | `publish.yml` |
   | Environment name | `pypi` |

   If the project does not exist, add a
   [pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
   in your PyPI account's publishing settings. A pending publisher does not
   reserve the name. If the project already exists, an owner must add the
   publisher in its publishing settings. Verify ownership or name availability
   before the first release.

These account settings are separate from the workflow files. Committing the
workflow does not configure a PyPI account or create a package release.

### Publish a version

1. Set a new version in `pyproject.toml` and review package metadata and release
   notes. For the first release, update the installation guides and README that
   currently describe LLGM as unpublished. README assets and repository links
   use absolute URLs so the package index can resolve them.
2. Run `make check`, `make docs`, and `make build`. Commit and push the release
   changes, including both workflow files. Use that reviewed commit for the tag.
3. Create and publish a GitHub release with the matching `v`-prefixed tag.
   Mark alpha, beta, and release-candidate versions as prereleases in GitHub.
4. Check **Actions → Publish to PyPI**. After successful publication, inspect the
   PyPI files and rendered description, then install the exact released version
   in a fresh environment and check `llgm --help`.

`make build` only builds and validates local distributions. It never uploads.
Published versions cannot be overwritten. The workflow keeps duplicate-file
errors visible instead of enabling `skip-existing`. If an upload partially
succeeds, inspect the files on PyPI before retrying or choosing a new version.

## What happens when you push

`.github/workflows/ci.yml` validates pushes and pull requests. Its documentation
job runs `make docs` with warnings treated as errors. It also accepts manual
runs and calls from the package release workflow.

`.github/workflows/docs.yml` builds and audits the user site on pushes to `main`
and manual runs from `main`. It uploads only generated HTML as a Pages artifact.
Deployment runs only when the repository variable `DOCS_PUBLISH_ENABLED` equals
`true`. Without that setting, the build and artifact upload run, but deployment
is skipped. The ordinary CI workflow never deploys.

| Location | What is available |
| --- | --- |
| Repository on GitHub | Markdown files are readable after they are committed and pushed |
| GitHub Actions | CI validates changes and retains tested distributions. Published GitHub releases trigger the PyPI workflow. The documentation workflow builds an artifact and conditionally deploys it |
| PyPI | Requires a matching Trusted Publisher and a successful release workflow |
| GitHub Pages | Requires Actions as the Pages source and `DOCS_PUBLISH_ENABLED=true` |
| Read the Docs | `.readthedocs.yaml` is prepared, but the repository must be connected to a Read the Docs project |

Git pushes include committed files. Untracked or uncommitted documentation and
workflow files stay on the local machine. The intended publishing source must
include the source docs, package, build tools, dependencies and linked assets.

## Publishing at a GitHub URL

GitHub Pages can host this same Sphinx output while retaining the Read the Docs
appearance. It does not require a Read the Docs account.

To enable publication:

1. In the GitHub repository, choose **Settings → Pages → Build and deployment →
   Source → GitHub Actions**. Keep the source repository's visibility unchanged.
2. Under **Settings → Secrets and variables → Actions → Variables**, set
   `DOCS_PUBLISH_ENABLED` to `true`. This is a publishing switch, not a secret.
3. Push `main` or run **Publish documentation** manually from `main`. Confirm
   that both build and deployment succeed. The deployment reports the site URL.

For `simjay/llgm`, the standard project URL is `https://simjay.github.io/llgm/`.
The standard Pages site is public even when its source repository is private.
A private source repository requires a GitHub plan that supports Pages. Enabling
publication does not change the repository's visibility or publish other folders.
The workflow cannot automatically enable Pages. Set the publishing variable to
`false` to pause future deployments. That does not remove an existing site.

The deployment job needs `pages: write` and `id-token: write` permissions and a
`github-pages` environment. Pull requests can build for validation without
publishing. See
[GitHub's custom Pages workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

Read the Docs is the alternative hosting route described below. Both hosting
options build from `docs/`. Neither publishes repository development guides,
research, benchmark runbooks, or agent context.

The repository uses Sphinx with the Read the Docs theme. That theme controls
appearance. It does not put the site online.

## Build and preview locally

From the repository root:

```sh
make setup
make docs
make serve-docs
```

Open [the local preview](http://127.0.0.1:8000) and stop it with Ctrl-C. HTML is
written to `docs/_build/html/`. Publish from the source configuration rather than
committing generated output. See [development](README.md) for command overrides.
When a page has moved or been deleted, remove only the generated `docs/_build/`
directory before rebuilding, or select a fresh `DOCS_BUILD_DIR`. Rebuilding the
Sphinx environment does not remove obsolete HTML. The audit rejects stale pages.

`make docs` treats Sphinx warnings as errors, then checks rendered API anchors,
public source and HTML links, portability, prose punctuation, and the public
documentation boundary. It does not read research or agent context. The separate
`make docs-links` command audits repository links when needed.
Diagrams use a pinned Mermaid browser script from jsDelivr, so diagram rendering
requires browser network access. Text and tables explain the same flows.

## Published sources

`docs/` is the only published prose source. Its index owns two sections:
User guide and Reference. Sphinx extracts public Python API reference from
the installed package's docstrings. Images and shared site assets live under
`docs/_static/`.

Repository development guides, brand explorations, research, agent context,
and benchmark runbooks are not published.
Do not import, mirror, link or provide downloads of internal Markdown in the
site. Example scripts may be linked or downloaded from the public guides when
they support a documented workflow. The repository examples index is not
automatically turned into a site page.

The source distribution contains user documentation, contributor guides, and
required package resources. It excludes brand explorations, `research/`,
`agent-context/`, `AGENTS.md`, and experiment Markdown. Archive inclusion does
not add contributor guides to the website. Experiment JSON inputs remain
available for tools and tests.

## Connect the repository to Read the Docs

Read the Docs pulls from the Git repository. It is not another Git push target.
Its free Community service supports public repositories. Connecting a private
repository requires Read the Docs Business. See
[Business hosting](https://docs.readthedocs.com/platform/stable/commercial/index.html).

1. Push the package source, `docs/`, documentation tools, linked example scripts,
   `Makefile` and `.readthedocs.yaml` to the intended Git repository.
2. Sign in to Read the Docs and import that repository. Select the intended
   default branch and retain `.readthedocs.yaml` at the repository root.
3. Inspect the first build log. The configuration selects Ubuntu 24.04 and
   Python 3.12, installs `docs/requirements.txt` and the package, and builds
   `docs/conf.py` with warnings treated as errors. Its post-build job runs
   the same rendered API and publication-boundary audit as `make docs`.
4. Check the quickstart, diagrams, API search, script downloads and section
   navigation at the assigned project URL.
   Enable the intended release versions and pull-request previews in the project
   settings.

See the [Read the Docs configuration reference](https://docs.readthedocs.com/platform/stable/config-file/v2.html)
and [theme documentation](https://sphinx-rtd-theme.readthedocs.io/en/stable/installing.html)
for the hosting interfaces. Read the Docs assigns the project URL when the
repository is imported.

The build imports the installed library for API documentation. It needs no
provider credentials, model downloads, Docker, or GPU. Keep API keys out of the
documentation build environment.

## Maintain the site

Update direct documentation pins in both `docs/requirements.txt` and the `docs`
dependency group in `pyproject.toml`. Transitive dependencies are resolved at
installation, so these files do not form a complete environment lock. Run
`make docs` after changes. CI uses the same strict build and rendered-output
audit. Read the Docs runs its configured strict Sphinx build over `docs/` followed by
`tools/check_docs.py` on the generated HTML.

The displayed version comes from the installed package. Release tags should
include matching package and documentation versions. Rebuilding documentation
does not validate a changed implementation. Run the checks appropriate to the
underlying code change before expanding a public support claim.

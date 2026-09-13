"""Regression checks for successful-looking but unusable documentation builds."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

CHECKER = Path(__file__).resolve().parents[1] / "tools/check_docs.py"


def write_file(root, relative, content):
    """Create a small authored or rendered fixture at its actual relative path."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def run_check(root, output, *, repository_links=False):
    """Run the same offline documentation checker used after the Sphinx build."""
    return subprocess.run(
        [sys.executable, str(CHECKER), str(output), "--root", str(root)]
        + (["--repository-links"] if repository_links else []),
        capture_output=True,
        text=True,
        check=False,
    )


def sidebar(*targets):
    """Render theme sidebar links separately from the document body."""
    links = "".join(
        f'<li class="toctree-l1"><a href="{target}">Page</a></li>' for target in targets
    )
    return f'<div class="wy-menu wy-menu-vertical"><ul>{links}</ul></div>'


@pytest.fixture
def rendered_docs(tmp_path):
    """Provide actual symbol targets and index links at the nested API location."""
    output = tmp_path / "html"
    anchors = (
        "llgm.Workspace",
        "llgm.Workspace.resolve",
        "llgm.LLGM",
        "llgm.LLGM.answer",
        "llgm.LLGM.from_settings",
        "llgm.Edge",
        "llgm.Workspace.publish_edge",
        "llgm.Workspace.compact_journal",
        "llgm.parse_instant_ms",
        "llgm.inference.nodes.NodeRuntime",
        "llgm.inference.nodes.NodeRuntime.answer",
        "llgm.Evidence.read",
        "llgm.retrieval.base.Embedder.embed",
        "llgm.storage.BlobStore.get",
    )
    write_file(
        output,
        "reference/api.html",
        sidebar("../guide/quickstart.html", "#")
        + "".join(f'<section id="{anchor}">Symbol documentation</section>' for anchor in anchors),
    )
    write_file(
        output,
        "genindex.html",
        sidebar("guide/quickstart.html", "reference/api.html")
        + "".join(f'<a href="reference/api.html#{anchor}">{anchor}</a>' for anchor in anchors),
    )
    for page in ("index.html", "search.html"):
        write_file(output, page, sidebar("guide/quickstart.html", "reference/api.html"))
    write_file(output, "guide/quickstart.html", sidebar("#", "../reference/api.html"))
    write_file(tmp_path, "README.md", "# Example\n")
    write_file(tmp_path, "docs/index.md", "# LLGM\n")
    write_file(tmp_path, "docs/guide/quickstart.md", "# Quickstart\n")
    write_file(tmp_path, "docs/reference/api.md", "# Public API\n")
    return output


@pytest.mark.parametrize(
    "local_link",
    [
        "[status](/Users/alice/project/docs/status.md)",
        "[status](C:\\Users\\alice\\project\\docs\\status.md)",
        "[status](file:///home/alice/project/docs/status.md)",
        "[status](/%55sers/alice/project/docs/status.md)",
        '<img src="%2FUsers/alice/project/docs/image.svg">',
    ],
)
def test_docs_check_rejects_unparsed_api_and_workstation_links(tmp_path, local_link):
    """An exit-zero Sphinx artifact with raw directives and empty index is rejected."""
    (tmp_path / "README.md").write_text(local_link, encoding="utf-8")
    output = tmp_path / "html"
    write_file(output, "reference/api.html", "<p>.. py:class:: Workspace</p>")
    write_file(output, "genindex.html", "<h1>Index</h1>")
    result = run_check(tmp_path, output, repository_links=True)
    assert result.returncode == 1
    assert "README.md:1: nonportable local link" in result.stdout
    assert "unparsed Python-domain directive" in result.stdout
    assert "missing symbol anchor llgm.Workspace.resolve" in result.stdout
    assert "missing API link llgm.LLGM" in result.stdout


def test_docs_check_requires_rendered_artifacts(tmp_path):
    """Source checks alone cannot claim a successful documentation rendering audit."""
    write_file(tmp_path, "README.md", "[Guide](docs/guide/quickstart.md)")
    write_file(tmp_path, "docs/guide/quickstart.md", "# Guide\n")
    result = run_check(tmp_path, tmp_path / "absent")
    assert result.returncode == 1
    assert "Documentation check failed" in result.stdout
    assert "nonportable local link" not in result.stdout


@pytest.mark.parametrize(
    "source, content, target",
    [
        ("README.md", '<a href="docs/quickstart.md">Guide</a>', "docs/quickstart.md"),
        ("AGENTS.md", "[Guide](docs/quickstart.md)", "docs/quickstart.md"),
        ("CONTRIBUTING.md", "[Guide](docs/quickstart.md)", "docs/quickstart.md"),
        ("docs/contributing/README.md", "[Guide](../quickstart.md)", "../quickstart.md"),
        ("docs/index.md", "[Guide](quickstart.md)", "quickstart.md"),
        ("experiments/README.md", "[Guide](../docs/quickstart.md)", "../docs/quickstart.md"),
        ("agent-context/README.md", "[Guide](../docs/quickstart.md)", "../docs/quickstart.md"),
        ("examples/README.md", "[Guide](../docs/quickstart.md)", "../docs/quickstart.md"),
        (
            "tests/fixtures/README.md",
            "[Guide](../../docs/quickstart.md)",
            "../../docs/quickstart.md",
        ),
        (
            "docs/reference/status.md",
            "{download}`Report <../../research/REPORT.md>`",
            "../../research/REPORT.md",
        ),
        (
            "docs/contributing/branding.md",
            "![Logo](../_static/missing.svg)",
            "../_static/missing.svg",
        ),
        ("docs/_static/gallery.html", '<img src="missing.svg">', "missing.svg"),
        (
            "docs/_static/gallery.html",
            '<source srcset="present.svg 1x, missing.svg 2x">',
            "missing.svg",
        ),
        ("docs/index.md", "[guide]: quickstart.md\nRead [guide].", "quickstart.md"),
    ],
)
def test_docs_check_rejects_broken_source_links(tmp_path, rendered_docs, source, content, target):
    """Moved guides, downloads and images must be updated at every authored entry point."""
    write_file(tmp_path, "docs/guide/quickstart.md", "# Guide\n")
    write_file(tmp_path, "docs/_static/present.svg", "<svg></svg>")
    write_file(tmp_path, source, content)
    result = run_check(tmp_path, rendered_docs, repository_links=True)
    assert result.returncode == 1
    assert f"{source}:1: missing local target {target}" in result.stdout


@pytest.mark.parametrize("target", ["../agent-context/README.md", "../AGENTS.md"])
@pytest.mark.parametrize(
    "content",
    [
        "[Context](TARGET)",
        '<a href="TARGET">Context</a>',
        "{download}`Context <TARGET>`",
        "```{include} TARGET\n```",
        "```{eval-rst}\n.. include:: TARGET\n```",
    ],
)
def test_docs_check_keeps_agent_context_out_of_public_navigation(
    tmp_path,
    rendered_docs,
    content,
    target,
):
    """Agent instructions and context are invalid navigation targets for public readers."""
    write_file(tmp_path, "docs/index.md", content.replace("TARGET", target))
    write_file(tmp_path, "AGENTS.md", "[Context](agent-context/README.md)")
    write_file(tmp_path, "agent-context/README.md", "# Agent context\n")
    result = run_check(tmp_path, rendered_docs)
    assert result.returncode == 1
    assert "public documentation links to agent context" in result.stdout
    assert "missing local target" not in result.stdout


def test_docs_check_accepts_hierarchy_and_ignores_examples(tmp_path, rendered_docs):
    """Valid nested links and agent entry points pass without inspecting example code."""
    write_file(
        tmp_path,
        "README.md",
        """# Example
[Guide](docs/guide/quickstart.md#installation)
[Public site](https://example.com/missing)
[Email](mailto:docs@example.com)
[Introduction](#example)
```markdown
[Example path](docs/missing.md)
<img src="/Users/alice/missing.svg">
```
~~~markdown
[Other example](missing.md)
~~~
""",
    )
    write_file(tmp_path, "AGENTS.md", "[Context](agent-context/README.md)")
    write_file(
        tmp_path,
        "agent-context/README.md",
        "[Tasks](REMAINING_TASKS.md)\n[Instructions](../AGENTS.md)",
    )
    write_file(tmp_path, "agent-context/REMAINING_TASKS.md", "# Tasks\n")
    write_file(tmp_path, "docs/guide/quickstart.md", "# Installation\n")
    write_file(tmp_path, "docs/_static/logo light.svg", "<svg></svg>")
    write_file(tmp_path, "examples/offline.py", '"""Run an offline example."""\n')
    write_file(
        tmp_path,
        "docs/index.md",
        """[Guide](guide/quickstart.md "Getting started")
[Logo](<_static/logo light.svg>)
![Logo](_static/logo%20light.svg)
<img src="_static/logo%20light.svg">
<source srcset="_static/logo%20light.svg 1x, _static/logo%20light.svg 2x">
[guide]: guide/quickstart.md
{download}`Offline example <../examples/offline.py>`
""",
    )
    write_file(tmp_path, "docs/_build/generated.md", "[Ignored](missing.md)")
    result = run_check(tmp_path, rendered_docs)
    assert result.returncode == 0, result.stdout
    assert "source links, publication boundary, rendered navigation and API passed" in result.stdout


@pytest.mark.parametrize(
    "content",
    [
        "[Research](../research/README.md)",
        '<a href="../research/README.md">Research</a>',
        "{download}`Research <../research/README.md>`",
        "```{include} ../research/README.md\n```",
        "```{eval-rst}\n.. include:: ../research/README.md\n```",
    ],
)
def test_docs_check_prevents_research_publication(tmp_path, rendered_docs, content):
    """Internal research cannot enter the site through links, downloads or includes."""
    write_file(tmp_path, "research/README.md", "# Internal research\n")
    write_file(tmp_path, "docs/index.md", content)
    result = run_check(tmp_path, rendered_docs)
    assert result.returncode == 1
    assert "public documentation links to internal research" in result.stdout


def test_docs_check_keeps_external_markdown_out_of_site(tmp_path, rendered_docs):
    """Repository protocols remain outside the published documentation hierarchy."""
    write_file(tmp_path, "experiments/README.md", "# Protocols\n")
    write_file(tmp_path, "docs/index.md", "{download}`Protocols <../experiments/README.md>`")
    result = run_check(tmp_path, rendered_docs)
    assert result.returncode == 1
    assert "published Markdown must live in docs" in result.stdout


@pytest.mark.parametrize(
    "content",
    [
        "[Publishing](contributing/publishing.md)",
        '<a href="contributing/publishing.md">Publishing</a>',
        "{download}`Publishing <contributing/publishing.md>`",
        "```{include} contributing/publishing.md\n```",
        "```{literalinclude} contributing/release.py\n```",
        "```{eval-rst}\n.. include:: contributing/publishing.md\n```",
        "[Contributing](../CONTRIBUTING.md)",
    ],
)
def test_docs_check_keeps_maintainer_guidance_out_of_user_docs(tmp_path, rendered_docs, content):
    """Site content cannot pull in repository publishing or contributor instructions."""
    write_file(tmp_path, "docs/contributing/publishing.md", "# Publishing\n")
    write_file(tmp_path, "docs/contributing/release.py", '"""Repository release command."""\n')
    write_file(tmp_path, "CONTRIBUTING.md", "[Contributing](docs/contributing/publishing.md)")
    write_file(tmp_path, "docs/index.md", content)
    result = run_check(tmp_path, rendered_docs)
    assert result.returncode == 1
    assert "end-user documentation links to repository guidance" in result.stdout


def test_docs_check_allows_repository_contributor_navigation(tmp_path, rendered_docs):
    """Repository entry points can link to curated context without publishing it in the site."""
    write_file(tmp_path, "README.md", "[Contributing](CONTRIBUTING.md)")
    write_file(tmp_path, "AGENTS.md", "[Context](agent-context/README.md)")
    write_file(
        tmp_path,
        "CONTRIBUTING.md",
        "[Contributing](docs/contributing/README.md)\n[Context](agent-context/README.md)",
    )
    write_file(
        tmp_path,
        "docs/contributing/README.md",
        "[Guide](../reference/api.md)\n[Architecture](../../agent-context/ARCHITECTURE.md)",
    )
    write_file(tmp_path, "agent-context/README.md", "[Instructions](../AGENTS.md)")
    write_file(
        tmp_path, "agent-context/ARCHITECTURE.md", "[Contributing](../docs/contributing/README.md)"
    )
    result = run_check(tmp_path, rendered_docs, repository_links=True)
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize(
    "path",
    [
        "docs/development/publishing.md",
        "docs/_static/brand/explorations/index.html",
        "docs/agent-context/README.md",
        "docs/AGENTS.md",
        "docs/_static/agent-context/README.md",
        "docs/_static/AGENTS.md",
    ],
)
def test_docs_check_rejects_reintroduced_maintainer_content(tmp_path, rendered_docs, path):
    """Moving a page out of navigation alone does not make it end-user documentation."""
    write_file(tmp_path, path, "Maintainer content")
    result = run_check(tmp_path, rendered_docs)
    assert result.returncode == 1
    assert f"{path}: repository-only content is inside docs" in result.stdout


def test_docs_check_ignores_private_notes_and_literal_includes(tmp_path, rendered_docs):
    """Ignored notes do not become audit inputs even when present beside curated context."""
    for source in (
        "research/README.md",
        "agent-context/LOCAL_STATE.md",
        "agent-context/local/archive/README.md",
        "agent-context/private/notes.md",
        "agent-context/old-decisions.md",
        "LOCAL_STATE.md",
    ):
        write_file(tmp_path, source, "[Missing](missing.md)")
    write_file(tmp_path, "agent-context/README.md", "# Context\n")
    write_file(
        tmp_path,
        "docs/index.md",
        """# Documentation
````markdown
```{include} ../research/README.md
```
````
""",
    )
    result = run_check(tmp_path, rendered_docs, repository_links=True)
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize(
    "name",
    [
        "README.md",
        "PROJECT_CONTEXT.md",
        "ARCHITECTURE.md",
        "DECISIONS.md",
        "TESTING.md",
        "REMAINING_TASKS.md",
    ],
)
def test_docs_check_validates_each_curated_context_file_without_git(tmp_path, rendered_docs, name):
    """Public context is audited from its declared file list in checkouts and source fixtures."""
    write_file(tmp_path, f"agent-context/{name}", "[Missing](missing.md)")
    result = run_check(tmp_path, rendered_docs, repository_links=True)
    assert result.returncode == 1
    assert f"agent-context/{name}:1: missing local target" in result.stdout


@pytest.mark.parametrize(
    "source, target, message",
    [
        ("AGENTS.md", "research/README.md", "internal research"),
        ("CONTRIBUTING.md", "agent-context/LOCAL_STATE.md", "local state"),
        ("docs/contributing/README.md", "../../agent-context/local/notes.md", "local state"),
        ("experiments/README.md", "../research/README.md", "internal research"),
        ("agent-context/README.md", "../research/README.md", "internal research"),
        ("agent-context/README.md", "LOCAL_STATE.md", "local state"),
        ("agent-context/README.md", "local/archive/README.md", "local state"),
        ("agent-context/README.md", "private/notes.md", "local state"),
        ("agent-context/README.md", "old-decisions.md", "local state"),
        ("agent-context/README.md", "../runs/result.json", "local state"),
        ("agent-context/README.md", "../private/notes.md", "local state"),
        ("agent-context/README.md", "../LOCAL_STATE.md", "local state"),
    ],
)
def test_docs_check_rejects_repository_dependencies_on_private_state(
    tmp_path, rendered_docs, source, target, message
):
    """Tracked guidance cannot depend on ignored research, local notes or run artifacts."""
    write_file(tmp_path, str(Path(source).parent / target), "Private material")
    write_file(tmp_path, source, f"[Details]({target})")
    result = run_check(tmp_path, rendered_docs, repository_links=True)
    assert result.returncode == 1
    assert f"links to {message}: {target}" in result.stdout
    assert "missing local target" not in result.stdout


@pytest.mark.parametrize(
    "path, message",
    [
        ("research/report.html", "published page has no source in docs"),
        ("_sources/research/report.md.txt", "published source is outside docs"),
        ("_downloads/old/report.md", "published Markdown download is outside docs"),
        ("contributing/publishing.html", "published page has no source in docs"),
        ("_sources/contributing/publishing.md.txt", "published source is outside docs"),
        ("_downloads/old/publishing.md", "published Markdown download is outside docs"),
        ("agent-context/README.html", "published page has no source in docs"),
        ("_sources/agent-context/README.md.txt", "published source is outside docs"),
        ("_sources/AGENTS.md.txt", "published source is outside docs"),
        ("_downloads/old/AGENTS.md", "published Markdown download is outside docs"),
        ("_static/agent-context/README.md", "repository-only asset is published"),
        ("_static/AGENTS.md", "repository-only asset is published"),
        ("_static/contributing/publishing.md", "repository-only asset is published"),
        ("_static/research/report.md", "repository-only asset is published"),
        ("_static/brand/explorations/index.html", "repository-only asset is published"),
        ("_static/brand/explorations/04-return.svg", "repository-only asset is published"),
        ("_static/retired-gallery.html", "published static page has no source in docs"),
    ],
)
def test_docs_check_rejects_retired_published_artifacts(tmp_path, rendered_docs, path, message):
    """Unlinked internal pages and downloads must not survive a documentation rebuild."""
    write_file(tmp_path, "docs/contributing/publishing.md", "Internal research content")
    write_file(rendered_docs, path, "Internal research content")
    result = run_check(tmp_path, rendered_docs)
    assert result.returncode == 1
    assert f"{path}: {message}" in result.stdout


@pytest.mark.parametrize("directory", ["development", "contributing"])
def test_docs_check_rejects_retired_search_entries(tmp_path, rendered_docs, directory):
    """Removed maintainer pages cannot remain discoverable through a stale search index."""
    write_file(tmp_path, "docs/contributing/publishing.md", "# Publishing\n")
    index = {
        "docnames": ["reference/api", f"{directory}/publishing"],
        "filenames": ["reference/api.md", f"{directory}/publishing.md"],
    }
    write_file(rendered_docs, "searchindex.js", f"Search.setIndex({json.dumps(index)})")
    result = run_check(tmp_path, rendered_docs)
    assert result.returncode == 1
    assert f"docnames entry is outside docs: {directory}/publishing" in result.stdout
    assert f"filenames entry is outside docs: {directory}/publishing.md" in result.stdout


def test_docs_check_accepts_current_search_entries(tmp_path, rendered_docs):
    """The pinned Sphinx search format remains valid when it indexes only public sources."""
    index = {"docnames": ["reference/api"], "filenames": ["reference/api.md"]}
    write_file(rendered_docs, "searchindex.js", f"Search.setIndex({json.dumps(index)});")
    result = run_check(tmp_path, rendered_docs)
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize(
    "content",
    ["Search.setIndex(null)", "Search.setIndex({})", "unrecognized search format"],
)
def test_docs_check_rejects_unauditable_search_index(tmp_path, rendered_docs, content):
    """An unreadable search export must not silently bypass the publication check."""
    write_file(rendered_docs, "searchindex.js", content)
    result = run_check(tmp_path, rendered_docs)
    assert result.returncode == 1
    assert "cannot audit search index" in result.stdout


def test_docs_check_rejects_broken_rendered_navigation(tmp_path, rendered_docs):
    """A valid API page cannot hide moved HTML pages, assets or missing section anchors."""
    write_file(rendered_docs, "guide/quickstart.html", '<h1 id="installation">Install</h1>')
    write_file(
        rendered_docs,
        "index.html",
        """<a href="quickstart.html">Old location</a>
<a href="guide/quickstart.html#removed">Removed heading</a>
<a href="guide/quickstart.html#installation">Valid heading</a>
<img src="_static/missing.svg">
""",
    )
    result = run_check(tmp_path, rendered_docs)
    assert result.returncode == 1
    assert "missing rendered target quickstart.html" in result.stdout
    assert "missing rendered fragment guide/quickstart.html#removed" in result.stdout
    assert "missing rendered target _static/missing.svg" in result.stdout
    assert "#installation" not in result.stdout


@pytest.mark.parametrize(
    "page, target",
    [
        ("index.html", "reference/api.html"),
        ("guide/quickstart.html", "../reference/api.html"),
        ("reference/api.html", "#"),
        ("search.html", "reference/api.html"),
        ("genindex.html", "reference/api.html"),
    ],
)
def test_docs_check_requires_page_links_inside_every_sidebar(tmp_path, rendered_docs, page, target):
    """Home, leaf and utility pages need full navigation even when body links work."""
    path = rendered_docs / page
    content = path.read_text()
    link = f'<a href="{target}">Page</a>'
    path.write_text(content.replace(link, "", 1) + link)
    result = run_check(tmp_path, rendered_docs)
    assert result.returncode == 1
    assert f"{page}: missing sidebar page link reference/api.html" in result.stdout
    assert "missing rendered target" not in result.stdout


def test_docs_check_rejects_heading_links_as_page_navigation(tmp_path, rendered_docs):
    """An API member link does not replace the API page entry in the sidebar."""
    path = rendered_docs / "index.html"
    content = path.read_text().replace(
        'href="reference/api.html"', 'href="reference/api.html#llgm.LLGM"'
    )
    path.write_text(content)
    result = run_check(tmp_path, rendered_docs)
    assert result.returncode == 1
    assert "index.html: missing sidebar page link reference/api.html" in result.stdout
    assert "missing rendered fragment" not in result.stdout


def test_docs_check_rejects_page_links_hidden_in_nested_navigation(tmp_path, rendered_docs):
    """A nested page link remains hidden by theme CSS on home and search pages."""
    path = rendered_docs / "index.html"
    link = '<a href="reference/api.html">Page</a>'
    path.write_text(
        path.read_text().replace(
            f'<li class="toctree-l1">{link}</li>',
            f'<li class="toctree-l1">Reference<ul><li class="toctree-l2">{link}</li></ul></li>',
        )
    )
    result = run_check(tmp_path, rendered_docs)
    assert result.returncode == 1
    assert "index.html: missing sidebar page link reference/api.html" in result.stdout
    assert "missing rendered target" not in result.stdout


def test_docs_check_requires_authored_pages_to_render(tmp_path, rendered_docs):
    """An omitted source page cannot disappear from both output and navigation unnoticed."""
    write_file(tmp_path, "docs/guide/architecture.md", "# Architecture\n")
    result = run_check(tmp_path, rendered_docs)
    assert result.returncode == 1
    assert "guide/architecture.html: missing rendered page" in result.stdout
    assert "index.html: missing sidebar page link guide/architecture.html" in result.stdout


def test_docs_check_needs_no_repository_documents(tmp_path, rendered_docs):
    """The user site builds without a repository README or any internal documentation."""
    (tmp_path / "README.md").unlink()
    write_file(tmp_path, "examples/offline.py", '"""A downloadable library example."""\n')
    write_file(
        tmp_path,
        "docs/index.md",
        "{download}`Example <../examples/offline.py>`",
    )
    result = run_check(tmp_path, rendered_docs)
    assert result.returncode == 0, result.stdout


def test_internal_broken_links_do_not_block_public_docs(tmp_path, rendered_docs):
    """Repository guidance is checked separately without becoming a user-site build input."""
    sources = (
        "README.md",
        "AGENTS.md",
        "CONTRIBUTING.md",
        "agent-context/README.md",
        "docs/contributing/README.md",
        "experiments/README.md",
        "examples/README.md",
    )
    for source in sources:
        write_file(tmp_path, source, "[Missing](missing.md)")
    result = run_check(tmp_path, rendered_docs)
    assert result.returncode == 0, result.stdout
    audit = subprocess.run(
        [sys.executable, str(CHECKER), "--repository-links", "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert audit.returncode == 1
    for source in sources:
        assert f"{source}:1: missing local target" in audit.stdout
    assert "Required rendered page" not in audit.stdout


@pytest.mark.parametrize("punctuation, message", [("—", "em dash"), (";", "semicolon")])
def test_docs_check_rejects_prohibited_authored_prose(
    tmp_path, rendered_docs, punctuation, message
):
    """Ordinary tutorial text must follow the project's explicit punctuation rule."""
    write_file(tmp_path, "docs/index.md", f"# Guide\nKeep the source{punctuation} read a passage.")
    result = run_check(tmp_path, rendered_docs)
    assert result.returncode == 1
    assert f"docs/index.md:2: {message} in prose" in result.stdout


@pytest.mark.parametrize("punctuation, message", [("—", "em dash"), (";", "semicolon")])
def test_docs_check_rejects_prohibited_rendered_api_prose(
    tmp_path, rendered_docs, punctuation, message
):
    """Generated API docstrings receive the same prose check as authored tutorials."""
    api = rendered_docs / "reference/api.html"
    with api.open("a", encoding="utf-8") as stream:
        stream.write(f"\n<p>Keep the source{punctuation} read a passage.</p>")
    result = run_check(tmp_path, rendered_docs)
    assert result.returncode == 1
    assert f"reference/api.html:2: {message} in prose" in result.stdout


def test_docs_check_ignores_code_and_html_entity_syntax(tmp_path, rendered_docs):
    """Code punctuation, URL parameters and entity terminators are not prose punctuation."""
    write_file(
        tmp_path,
        "docs/index.md",
        """# Guide
Read &amp; retain a source. Use `first(); second()` for this example.
[Link](https://example.com/?first=1;second=2)
```python
first(); second()  # — is literal example content
```
<p>Save&nbsp;the text.</p>
<pre><code>first(); second() — literal example</code></pre>
""",
    )
    api = rendered_docs / "reference/api.html"
    with api.open("a", encoding="utf-8") as stream:
        stream.write(
            "<p>Read &amp; retain a source.</p><code>first(); second()</code>"
            "<pre><span>first(); second() — example</span></pre>"
            "<script>first(); second();</script><style>p {color: red;}</style>"
        )
    result = run_check(tmp_path, rendered_docs)
    assert result.returncode == 0, result.stdout

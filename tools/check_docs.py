"""Check public documentation without depending on repository-only writing."""

from __future__ import annotations

import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

API_ANCHORS = (
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
    "llgm.inference.recursive.RecursiveRuntime",
    "llgm.inference.recursive.RecursiveRuntime.answer",
    "llgm.inference.rlm.RLMRuntime",
    "llgm.inference.rlm.RLMRuntime.answer",
)
LOCAL_LINK = re.compile(r"^(?:/(?:Users|home|private|tmp|var|Volumes)/|[A-Za-z]:[\\/]|file:)")
MARKDOWN_LINK = re.compile(r"!?\[[^\]\n]*\]\(\s*(<[^>\n]+>|[^\s()]+(?:\([^()]*\)[^\s()]*)*)")
REFERENCE_LINK = re.compile(r"^ {0,3}\[[^\]\n]+\]:\s*(<[^>\n]+>|\S+)", re.MULTILINE)
DOWNLOAD_LINK = re.compile(r"\{download\}`([^`]+)`")
INCLUDE_LINK = re.compile(r"^\s*\.\.\s+(?:include|literalinclude)::\s+(\S+)")
MYST_INCLUDE = re.compile(r"^\{(?:include|literalinclude)\}\s+(\S+)")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
REPOSITORY_DIRECTORIES = (
    "development",
    "experiments",
    "examples",
    "tests/fixtures",
)
PUBLIC_AGENT_CONTEXT = (
    "README.md",
    "PROJECT_CONTEXT.md",
    "ARCHITECTURE.md",
    "DECISIONS.md",
    "TESTING.md",
    "REMAINING_TASKS.md",
)
PROSE_EXCLUSIONS = {"code", "pre", "script", "style", "svg", "textarea"}
VOID_ELEMENTS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
PROSE_PUNCTUATION = {"—": "em dash", ";": "semicolon"}
EXCLUDED_SITE_PATHS = (
    "development",
    "research",
    "agent-context",
    "AGENTS.md",
    "experiments",
    "_static/research",
    "_static/agent-context",
    "_static/AGENTS.md",
    "_static/brand/explorations",
)


class Page(HTMLParser):
    """Collect real HTML anchors and links separately from displayed directive text."""

    def __init__(self, content: str):
        """Parse page contents and retain source lines for link diagnostics."""
        super().__init__()
        self.ids: set[str] = set()
        self.links: set[str] = set()
        self.sidebar_links: set[str] = set()
        self.targets: list[tuple[str, int]] = []
        self.text: list[str] = []
        self.prose: list[tuple[str, int]] = []
        self._excluded: list[str] = []
        self._sidebar_stack: list[bool] = []
        self.feed(content)

    def handle_starttag(self, tag, attrs):
        """Retain targets and links that browsers can actually navigate."""
        attributes = dict(attrs)
        classes = attributes.get("class", "").split()
        if tag not in VOID_ELEMENTS and (self._sidebar_stack or "wy-menu-vertical" in classes):
            self._sidebar_stack.append(tag == "li" and "toctree-l1" in classes)
        if tag in PROSE_EXCLUSIONS:
            self._excluded.append(tag)
        if attributes.get("id"):
            self.ids.add(attributes["id"])
        if tag == "a" and attributes.get("name"):
            self.ids.add(attributes["name"])
        if tag == "a" and attributes.get("href"):
            self.links.add(attributes["href"])
            if len(self._sidebar_stack) > 1 and self._sidebar_stack[-2]:
                self.sidebar_links.add(attributes["href"])
        for name in ("href", "src"):
            if attributes.get(name):
                self.targets.append((attributes[name], self.getpos()[0]))
        if attributes.get("srcset") and not attributes["srcset"].startswith("data:"):
            for candidate in attributes["srcset"].split(","):
                if candidate.strip():
                    self.targets.append((candidate.split()[0], self.getpos()[0]))

    def handle_endtag(self, tag):
        """Resume prose collection after code, style or other nonprose elements."""
        if tag in self._excluded:
            self._excluded.remove(tag)
        if self._sidebar_stack and tag not in VOID_ELEMENTS:
            self._sidebar_stack.pop()

    def handle_data(self, data):
        """Keep visible text so an unparsed autodoc directive cannot pass."""
        self.text.append(data)
        if not self._excluded:
            self.prose.append((data, self.getpos()[0]))


def prose_errors(path: Path, prose: list[tuple[str, int]]) -> list[str]:
    """Report prohibited punctuation in visible prose with a readable excerpt."""
    errors = []
    for text, number in prose:
        for character, name in PROSE_PUNCTUATION.items():
            if character in text:
                excerpt = " ".join(text.split())[:120]
                line = number + text.count("\n", 0, text.index(character))
                errors.append(f"{path}:{line}: {name} in prose: {excerpt}")
    return errors


def source_prose(path: Path) -> list[tuple[str, int]]:
    """Extract authored prose while leaving example syntax and URL punctuation alone."""
    content = path.read_text(encoding="utf-8")
    if path.suffix == ".md":
        content = without_fences(content)
        content = REFERENCE_LINK.sub("", content)
        content = re.sub(r"!?\[([^\]\n]*)\]\([^\n]*?\)", r"\1", content)
        content = re.sub(r"`+[^`\n]*`+", "", content)
    return Page(content).prose


def without_fences(content: str) -> str:
    """Hide fenced examples while preserving line numbers for authored links."""
    lines = []
    fence = ""
    for line in content.splitlines(keepends=True):
        match = FENCE.match(line)
        if match and not fence:
            fence = match[1]
        elif (
            match
            and fence
            and match[1][0] == fence[0]
            and len(match[1]) >= len(fence)
            and not match[2].strip()
        ):
            fence = ""
        elif not fence:
            lines.append(line)
            continue
        lines.append("\n" if line.endswith("\n") else "")
    return "".join(lines)


def source_targets(path: Path) -> list[tuple[str, int]]:
    """Collect local-link candidates from the Markdown, MyST and HTML we author."""
    content = path.read_text(encoding="utf-8")
    includes = include_targets(content) if path.suffix == ".md" else []
    if path.suffix == ".md":
        content = without_fences(content)
    targets = Page(content).targets + includes
    if path.suffix == ".md":
        for pattern in (MARKDOWN_LINK, REFERENCE_LINK, DOWNLOAD_LINK):
            for match in pattern.finditer(content):
                target = match[1]
                if pattern is DOWNLOAD_LINK and "<" in target:
                    target = target.rsplit("<", 1)[1]
                targets.append((target.strip("<>"), content.count("\n", 0, match.start()) + 1))
    return targets


def include_targets(content: str) -> list[tuple[str, int]]:
    """Inspect executable MyST and reStructuredText includes, leaving literal examples alone."""
    targets = []
    fence = ""
    directive = ""
    for number, line in enumerate(content.splitlines(), 1):
        match = FENCE.match(line)
        if match and not fence:
            fence, directive = match[1], match[2].strip()
            include = MYST_INCLUDE.match(directive)
            if include:
                targets.append((include[1], number))
        elif (
            match
            and fence
            and match[1][0] == fence[0]
            and len(match[1]) >= len(fence)
            and not match[2].strip()
        ):
            fence, directive = "", ""
        elif not fence or directive == "{eval-rst}":
            include = INCLUDE_LINK.match(line)
            if include:
                targets.append((include[1], number))
    return targets


def local_target(source: Path, target: str, root: Path) -> tuple[Path | None, str]:
    """Resolve a repository or site URL, leaving external URLs outside the audit."""
    url = urlsplit(target)
    if url.scheme or url.netloc:
        return None, ""
    decoded = unquote(url.path)
    if decoded.startswith("/"):
        destination = root / decoded.lstrip("/")
    else:
        destination = source.parent / decoded if decoded else source
    return destination.resolve(), unquote(url.fragment)


def check_sources(root: Path, *, repository_links: bool = False) -> list[str]:
    """Audit public sources, optionally checking separate repository documents too."""
    errors = []
    sources = []
    public_context = {root / "agent-context" / name for name in PUBLIC_AGENT_CONTEXT}
    if repository_links:
        sources.extend(
            root / name
            for name in ("README.md", "AGENTS.md", "CONTRIBUTING.md")
            if (root / name).exists()
        )
        sources.extend(path for path in public_context if path.is_file())
    directories = ("docs",) + (REPOSITORY_DIRECTORIES if repository_links else ())
    for directory in directories:
        sources.extend(
            path
            for path in (root / directory).rglob("*")
            if path.suffix in (".md", ".html") and "_build" not in path.parts
        )
    for path in sorted(sources):
        if path.is_relative_to(root / "docs"):
            errors.extend(prose_errors(path.relative_to(root), source_prose(path)))
        if any(path.is_relative_to(root / "docs" / name) for name in EXCLUDED_SITE_PATHS):
            errors.append(f"{path.relative_to(root)}: repository-only content is inside docs")
        for target, number in source_targets(path):
            location = f"{path.relative_to(root)}:{number}"
            if LOCAL_LINK.match(unquote(target)):
                errors.append(f"{location}: nonportable local link {target}")
                continue
            destination, _ = local_target(path, target, root)
            if destination is None:
                continue
            if not destination.is_relative_to(root):
                errors.append(f"{location}: link leaves repository: {target}")
            elif not destination.exists():
                errors.append(f"{location}: missing local target {target}")
            if (
                destination.is_relative_to(root / "agent-context")
                or destination == root / "AGENTS.md"
            ) and path.is_relative_to(root / "docs"):
                errors.append(f"{location}: public documentation links to agent context: {target}")
            if destination.is_relative_to(root / "research"):
                errors.append(
                    f"{location}: public documentation links to internal research: {target}"
                )
            if (
                destination.is_relative_to(root / "agent-context")
                and destination not in public_context
            ) or (
                destination.is_relative_to(root / "runs")
                or destination.is_relative_to(root / "private")
                or destination == root / "LOCAL_STATE.md"
            ):
                errors.append(
                    f"{location}: repository documentation links to local state: {target}"
                )
            if path.is_relative_to(root / "docs") and (
                destination.is_relative_to(root / "development")
                or destination.is_relative_to(root / "experiments")
                or destination == root / "CONTRIBUTING.md"
            ):
                errors.append(
                    f"{location}: end-user documentation links to repository guidance: {target}"
                )
            if (
                path.is_relative_to(root / "docs")
                and destination.suffix == ".md"
                and not destination.is_relative_to(root / "docs")
            ):
                errors.append(f"{location}: published Markdown must live in docs: {target}")
    return errors


def public_sources(docs: Path) -> list[Path]:
    """List authored site pages without generated output or repository-only content."""
    return [
        path
        for path in docs.rglob("*.md")
        if "_build" not in path.parts
        and not any(path.is_relative_to(docs / name) for name in EXCLUDED_SITE_PATHS)
    ]


def check_rendered(html: Path, docs: Path) -> list[str]:
    """Check generated page paths, fragments and actual API/index output offline."""
    pages = {path: Page(path.read_text(encoding="utf-8")) for path in html.rglob("*.html")}
    api_path = html / "reference/api.html"
    index_path = html / "genindex.html"
    for required in (api_path, index_path):
        if required not in pages:
            raise FileNotFoundError(f"Required rendered page is missing: {required}")
    errors = []
    for path, page in sorted(pages.items()):
        errors.extend(prose_errors(path.relative_to(html), page.prose))
        for target, number in page.targets:
            destination, fragment = local_target(path, target, html)
            if destination is None:
                continue
            location = f"{path.relative_to(html)}:{number}"
            if not destination.is_relative_to(html) or not destination.exists():
                errors.append(f"{location}: missing rendered target {target}")
            elif fragment and destination in pages and fragment not in pages[destination].ids:
                errors.append(f"{location}: missing rendered fragment {target}")
    api, index = pages[api_path], pages[index_path]
    if ".. py:" in "".join(api.text):
        errors.append("reference/api.html: unparsed Python-domain directive")
    for anchor in API_ANCHORS:
        if anchor not in api.ids:
            errors.append(f"reference/api.html: missing symbol anchor {anchor}")
        if f"reference/api.html#{anchor}" not in index.links:
            errors.append(f"genindex.html: missing API link {anchor}")
    authored = {path.relative_to(docs).with_suffix(".html") for path in public_sources(docs)}
    expected = {html / path for path in authored if path.parts[0] in {"guide", "reference"}}
    for relative in sorted(authored | {Path("search.html"), Path("genindex.html")}):
        path = html / relative
        if path not in pages:
            errors.append(f"{relative}: missing rendered page")
            continue
        linked = {
            destination
            for target in pages[path].sidebar_links
            for destination, fragment in [local_target(path, target, html)]
            if destination is not None and not fragment
        }
        for missing in sorted(expected - linked):
            errors.append(f"{relative}: missing sidebar page link {missing.relative_to(html)}")
    return errors


def check_publication(root: Path, html: Path) -> list[str]:
    """Reject leftover pages, source exports and Markdown downloads outside the public sources."""
    docs = root / "docs"
    sources = public_sources(docs)
    pages = {path.relative_to(docs).with_suffix(".html") for path in sources}
    pages.update(Path(name) for name in ("genindex.html", "py-modindex.html", "search.html"))
    exports = {Path("_sources") / (str(path.relative_to(docs)) + ".txt") for path in sources}
    markdown = {path.read_bytes() for path in sources}
    errors = []
    for path in sorted(html.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(html)
        if relative.parts[0] == "_static":
            if any(relative.is_relative_to(name) for name in EXCLUDED_SITE_PATHS):
                errors.append(f"{relative}: repository-only asset is published")
            elif path.suffix == ".html" and not (docs / relative).is_file():
                errors.append(f"{relative}: published static page has no source in docs")
            continue
        if path.suffix == ".html" and relative not in pages:
            errors.append(f"{relative}: published page has no source in docs")
        if relative.parts[0] == "_sources" and relative not in exports:
            errors.append(f"{relative}: published source is outside docs")
        if (
            relative.parts[0] == "_downloads"
            and path.suffix == ".md"
            and path.read_bytes() not in markdown
        ):
            errors.append(f"{relative}: published Markdown download is outside docs")
    search = html / "searchindex.js"
    if search.exists():
        errors.extend(check_search_index(search, docs, sources))
    return errors


def check_search_index(search: Path, docs: Path, sources: list[Path]) -> list[str]:
    """Reject stale search entries for pages that no longer belong in the user site."""
    try:
        content = search.read_text(encoding="utf-8").strip().removesuffix(";")
        if not content.startswith("Search.setIndex(") or not content.endswith(")"):
            raise ValueError("unexpected search-index wrapper")
        index = json.loads(content[len("Search.setIndex(") : -1])
        expected = {
            "docnames": {path.relative_to(docs).with_suffix("").as_posix() for path in sources},
            "filenames": {path.relative_to(docs).as_posix() for path in sources},
        }
        errors = []
        for key, permitted in expected.items():
            values = index[key]
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise ValueError(f"invalid {key}")
            for value in values:
                if value not in permitted:
                    errors.append(f"searchindex.js: {key} entry is outside docs: {value}")
        return errors
    except (KeyError, TypeError, ValueError) as error:
        return [f"searchindex.js: cannot audit search index: {error}"]


def check(root: Path, html: Path, *, repository_links: bool = False) -> list[str]:
    """Audit authored and rendered documentation without making network requests."""
    root, html = root.resolve(), html.resolve()
    return (
        check_sources(root, repository_links=repository_links)
        + check_rendered(html, root / "docs")
        + check_publication(root, html)
    )


def main() -> int:
    """Fail on broken source/navigation links, audience leaks or unusable API output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path, nargs="?", help="Sphinx HTML output directory")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--repository-links",
        action="store_true",
        help="Also audit repository-only source links, without requiring HTML output",
    )
    arguments = parser.parse_args()
    if arguments.html is None and not arguments.repository_links:
        parser.error("HTML output is required unless --repository-links is selected")
    try:
        if arguments.html is None:
            errors = check_sources(arguments.root.resolve(), repository_links=True)
        else:
            errors = check(
                arguments.root, arguments.html, repository_links=arguments.repository_links
            )
    except OSError as error:
        print(f"Documentation check failed: {error}")
        return 1
    for error in errors:
        print(error)
    if not errors and arguments.html is None:
        print("Documentation: repository source links passed")
    elif not errors:
        print(
            "Documentation: source links, publication boundary, rendered navigation and API passed"
        )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())

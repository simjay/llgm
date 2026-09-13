"""Require docstrings on Python modules, classes and functions, including helpers."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

DEFAULT_PATHS = ("src", "tests", "examples", "tools", "docs/conf.py")
DEFINITIONS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def python_files(paths: list[str]) -> list[Path]:
    """Expand explicit files or source directories without scanning generated output."""
    found = set()
    for name in paths:
        path = Path(name)
        if not path.exists():
            raise FileNotFoundError(f"Docstring check path does not exist: {path}")
        candidates = path.rglob("*.py") if path.is_dir() else [path]
        found.update(p for p in candidates if p.suffix == ".py" and "_build" not in p.parts)
    return sorted(found)


def inspect_tree(tree: ast.AST, label: str) -> tuple[int, list[str]]:
    """Count definitions and report absent or blank docstrings without importing code."""
    definitions = [node for node in ast.walk(tree) if isinstance(node, DEFINITIONS)]
    missing = [
        f"{label}:{getattr(node, 'lineno', 1)}: {getattr(node, 'name', '<module>')}"
        for node in definitions
        if not (ast.get_docstring(node) or "").strip()
    ]
    return len(definitions), missing


def inspect_file(path: Path) -> tuple[int, list[str]]:
    """Check the file and embedded sandbox setup source when present."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    total, missing = inspect_tree(tree, str(path))
    # Sandbox setup is shipped as source text. Inspecting only the assignment
    # would omit functions executed inside WASM.
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_SANDBOX_SETUP" for t in node.targets)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            count, absent = inspect_tree(ast.parse(node.value.value), f"{path}::_SANDBOX_SETUP")
            total += count
            missing.extend(absent)
    return total, missing


def main() -> int:
    """Exit nonzero for undocumented definitions or unreadable source inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=DEFAULT_PATHS)
    arguments = parser.parse_args()
    total = 0
    missing = []
    try:
        paths = python_files(arguments.paths)
        for path in paths:
            count, absent = inspect_file(path)
            total += count
            missing.extend(absent)
    except (OSError, SyntaxError) as error:
        print(f"Docstring check failed: {error}")
        return 1
    for location in missing:
        print(f"Missing docstring: {location}")
    print(f"Docstrings: {total - len(missing)}/{total} definitions across {len(paths)} files")
    return int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())

"""Execute the standalone storage tutorials against the installed public API."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

WALKTHROUGH = Path(__file__).resolve().parents[1] / "docs/guide/walkthrough.md"


@pytest.mark.parametrize(
    "section",
    ["Store, search and read without a model", "Follow an update and a correction"],
)
def test_storage_tutorial_runs_as_written(section, tmp_path):
    """Copied tutorial code preserves search results, original text and effective corrections."""
    text = WALKTHROUGH.read_text(encoding="utf-8").split(f"## {section}\n", 1)[1]
    text = text.split("\n## ", 1)[0]
    program = re.search(r"```python\n(.*?)```", text, re.DOTALL).group(1)
    expected = re.search(r"```text\n(.*?)```", text, re.DOTALL).group(1)
    script = tmp_path / "tutorial.py"
    script.write_text(program, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-I", str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected

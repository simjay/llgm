"""Installed-distribution checks; CI runs these against wheel and sdist builds.

Editable development installs skip the distribution checks. Set
VERIFY_LLGM_WHEEL=1 to make an editable or missing installation a test failure.
Every check imports LLGM in an isolated subprocess outside the checkout.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

_IMPORT_CHECK = r"""
import asyncio
import builtins
import importlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import sys

checkout_source = Path(sys.argv[1]).resolve() / 'src'
before = set(Path.cwd().iterdir())
original_open = builtins.open

def blocked(*args, **kwargs):
    raise AssertionError('Import attempted to open a network/DB connection or create a directory')

def read_only_open(file, mode='r', *args, **kwargs):
    if any(flag in mode for flag in 'wax+'):
        raise AssertionError('Import attempted to write a file')
    return original_open(file, mode, *args, **kwargs)

socket.socket = blocked
socket.create_connection = blocked
sqlite3.connect = blocked
os.mkdir = blocked
os.makedirs = blocked
builtins.open = read_only_open

import llgm
for name in ('llgm.llgm', 'llgm.core', 'llgm.memory', 'llgm.inference', 'llgm.models', 'llgm.retrieval', 'llgm.retrieval.colbert', 'llgm.retrieval.modal', 'llgm.evaluation', 'llgm.storage', 'llgm.cli', 'llgm.viewer'):
    importlib.import_module(name)

assert llgm.LLGM.__module__ == 'llgm.llgm'
origin = Path(llgm.__file__).resolve()
assert not origin.is_relative_to(checkout_source), origin
for package in ('openai', 'anthropic', 'boto3', 'torch', 'transformers', 'colbert', 'dspy', 'deno', 'litellm'):
    assert package not in sys.modules, f'Optional dependency imported eagerly: {package}'
assert set(Path.cwd().iterdir()) == before
print(json.dumps({'origin': str(origin), 'import_side_effects': False}))
"""


_RESOURCE_CHECK = r"""
from importlib.metadata import distribution
from importlib.resources import files
import json

package = files('llgm')
for asset in ('index.html', 'app.js', 'style.css', 'record.html', 'record.js'):
    assert package.joinpath('viewer', asset).is_file(), f'Missing graph viewer asset: {asset}'
assert package.joinpath('py.typed').is_file(), 'Wheel is missing the PEP 561 marker'
for module in ('core', 'memory', 'inference', 'models', 'retrieval', 'evaluation', 'storage'):
    assert package.joinpath(module, '__init__.py').is_file(), module
for retired in ('iterative.py', 'recursive.py', 'rlm.py'):
    assert not package.joinpath('inference', retired).is_file(), f'Retired runtime packaged: {retired}'

dist = distribution('llgm')
assert dist.metadata['Name'] == 'llgm'
assert dist.metadata['Requires-Python'] == '>=3.11'
assert dist.metadata['License-Expression'] == 'MIT'
assert any(str(path).endswith('/licenses/LICENSE') for path in dist.files), 'Missing license file'
entrypoints = [e for e in dist.entry_points if e.group == 'console_scripts' and e.name == 'llgm']
assert len(entrypoints) == 1
assert entrypoints[0].value == 'llgm.cli:main'
assert callable(entrypoints[0].load()), 'Console entry point cannot be imported'
for requirement in dist.requires or []:
    assert 'extra ==' in requirement, f'Unexpected mandatory dependency: {requirement}'
print(json.dumps({'version': dist.version, 'typing_marker': True, 'entrypoint': entrypoints[0].value}))
"""


class PackagingTests(unittest.TestCase):
    """Wheel-install probes for metadata, packaged resources and side-effect-free imports."""

    @classmethod
    def setUpClass(cls):
        """Require a non-editable installation, failing when wheel verification is mandatory."""
        required = os.environ.get("VERIFY_LLGM_WHEEL") == "1"
        try:
            dist = importlib.metadata.distribution("llgm")
        except importlib.metadata.PackageNotFoundError:
            if required:
                raise AssertionError(
                    "Install the built LLGM distribution before packaging tests"
                ) from None
            raise unittest.SkipTest(
                "Packaging checks require an installed LLGM distribution"
            ) from None
        direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
        if direct_url.get("dir_info", {}).get("editable"):
            if required:
                raise AssertionError(
                    "Packaging checks require a non-editable wheel or sdist installation"
                )
            raise unittest.SkipTest(
                "Packaging checks run against a non-editable installation in CI"
            )

    def run_isolated(self, script: str) -> dict:
        """Probe the installed distribution without project-directory or PYTHONPATH imports."""
        environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        with tempfile.TemporaryDirectory(prefix="llgm-installed-check-") as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    textwrap.dedent(script),
                    str(Path(__file__).resolve().parents[1]),
                ],
                cwd=directory,
                env=environment,
                text=True,
                capture_output=True,
                timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_import_is_lazy_and_has_no_external_side_effects(self):
        """Importing the installed package neither loads optional stacks nor touches external resources."""
        result = self.run_isolated(_IMPORT_CHECK)
        self.assertFalse(result["import_side_effects"])

    def test_installed_resources_metadata_and_console_entrypoint(self):
        """Installed metadata, resources and the console entry point are available from the wheel."""
        result = self.run_isolated(_RESOURCE_CHECK)
        self.assertTrue(result["typing_marker"])
        self.assertEqual(result["entrypoint"], "llgm.cli:main")


if __name__ == "__main__":
    unittest.main()

"""Runs the node:test suites in tests/js/ (static/*.js logic) so plain `pytest` covers them.

The frontend has no JS toolchain; Node's built-in test runner needs none.
"""
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_xr_logic_node_suite():
    result = subprocess.run(
        ["node", "--test", *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests" / "js").glob("*.test.js"))],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr

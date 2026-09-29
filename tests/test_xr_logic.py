"""Runs the node:test suite for static/xr_logic.js so plain `pytest` covers it.

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
        ["node", "--test", "tests/js/xr_logic.test.js"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr

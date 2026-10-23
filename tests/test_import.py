# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks what `import cuju` imports, in a fresh interpreter."""

import importlib.util
import subprocess
import sys

import pytest


def run(code):
    """Returns what `code` prints in a new interpreter that sees the same `cuju`."""
    result = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, check=True, timeout=120
    )
    return result.stdout.split()


def test_import_leaves_gym_and_gymnasium_out():
    found = run(
        "import sys, cuju; print('gym' in sys.modules, 'gymnasium' in sys.modules, "
        "'cuju.env' in sys.modules)"
    )
    assert found == ['False', 'False', 'False']


def test_import_of_gfootball_works_without_gym():
    # Each `None` in `sys.modules` makes its import fail, as an installation without it does.
    found = run(
        "import sys; sys.modules['gym'] = sys.modules['gymnasium'] = None; "
        'import cuju.gfootball; '
        "print('cuju.gfootball.env' in sys.modules, hasattr(cuju.gfootball, 'other'))"
    )
    assert found == ['False', 'False']


@pytest.mark.skipif(
    importlib.util.find_spec('gymnasium') is None and importlib.util.find_spec('gym') is None,
    reason='no gymnasium or gym',
)
def test_import_of_gfootball_binds_env():
    # GRF: `import gfootball` binds `gfootball.env`, so the attribute serves at once.
    found = run(
        'import sys, cuju.gfootball; '
        "print('cuju.gfootball.env' in sys.modules, "
        'callable(cuju.gfootball.env.create_environment), '
        "'cuju.gfootball.env' in sys.modules)"
    )
    assert found == ['False', 'True', 'True']


@pytest.mark.skipif(importlib.util.find_spec('gymnasium') is None, reason='no gymnasium')
def test_gymnasium_imported_later_gets_the_ids():
    found = run(
        'import cuju, gymnasium; '
        "print('GFootball-11_vs_11_kaggle-simple115v2-v0' in gymnasium.envs.registry)"
    )
    assert found == ['True']

# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks what `import cuju` imports, in a fresh interpreter."""

import importlib.util
import json
import os
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


def test_import_leaves_torch_out_until_a_batch_is_used():
    # cuju and its state reader start without importing PyTorch.
    found = run(
        "import sys, cuju, cuju.state; print('torch' in sys.modules); "
        "print('BatchEnvironment' in dir(cuju), sorted(cuju.__all__) == sorted(set(cuju.__all__)))"
        "; cuju.BatchEnvironment; print('torch' in sys.modules, cuju.BatchEnvironment.__module__, "
        'cuju.memory_estimate.__module__, cuju.library_path.__module__, '
        'cuju._MemoryEstimate.__module__)'
    )
    assert found == ['False', 'True', 'True', 'True', 'cuju', 'cuju', 'cuju', 'cuju']


def test_every_public_name_of_cuju_resolves():
    found = run(
        'import cuju; missing = [name for name in cuju.__all__ if not hasattr(cuju, name)]; '
        "print(missing or 'none'); from cuju import ActionLog, MultiDeviceBatch, digest, replay; "
        'import cuju.action_log; print(cuju.ActionLog is cuju.action_log.ActionLog)'
    )
    assert found == ['none', 'True']


def test_action_log_imports_first():
    # A submodule that builds on `BatchEnvironment` imports on its own, before any name of `cuju`.
    found = run("import cuju.multi_device, cuju.action_log; print('ok')")
    assert found == ['ok']


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


def test_only_the_renderer_of_cuju_render_joins_the_package(tmp_path):
    # A whole cuju later on the path, such as an older installation with its own renderer, stays
    # out of the package, and a folder `cuju` that holds `render` and no `__init__.py`, as an
    # editable cuju-render keeps, joins it as an absolute path, here the current directory's.
    stray = tmp_path / 'stray' / 'cuju'
    (stray / 'render').mkdir(parents=True)
    (stray / '__init__.py').write_text("raise SystemExit('the stray cuju was imported')\n")
    (stray / 'render' / '__init__.py').write_text('STRAY = True\n')
    portion = tmp_path / 'portion' / 'cuju'
    (portion / 'render').mkdir(parents=True)
    (portion / 'render' / '__init__.py').write_text('PORTION = True\n')
    home = os.path.dirname(os.path.dirname(importlib.util.find_spec('cuju').origin))
    code = (
        'import json, os, cuju\n'
        'path = list(cuju.__path__)\n'
        'import cuju.render\n'
        "print(json.dumps([path, os.path.dirname(cuju.__file__), hasattr(cuju.render, 'STRAY')]))\n"
    )
    environment = dict(os.environ)
    environment.pop('CUJU_RENDER', None)
    environment['PYTHONPATH'] = os.pathsep.join([home, str(tmp_path / 'stray')])
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path / 'portion',
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    path, own, stray_loaded = json.loads(result.stdout)
    normal = [os.path.normcase(folder) for folder in path]
    assert normal[0] == os.path.normcase(own)
    assert os.path.normcase(str(portion)) in normal
    assert os.path.normcase(str(stray)) not in normal
    assert all(os.path.isabs(folder) for folder in path)
    assert not stray_loaded

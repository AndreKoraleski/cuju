# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks what cuju does without its renderer, in new interpreters that cannot find `cuju.render`.

Each interpreter loads a `sitecustomize` that hides `cuju.render` from Python's path finder, as an
installation without the renderer lacks it, so every other finder runs as it would there.
"""

import os
import subprocess
import sys
import textwrap

import pytest

INSTALL = "pip install 'cuju[render]'"

HIDE = """
import importlib.machinery
import sys


class _HidingPathFinder:
    @classmethod
    def find_spec(cls, name, path=None, target=None):
        if name == 'cuju.render' or name.startswith('cuju.render.'):
            return None
        return importlib.machinery.PathFinder.find_spec(name, path, target)

    @classmethod
    def invalidate_caches(cls):
        importlib.machinery.PathFinder.invalidate_caches()


sys.meta_path[:] = [
    _HidingPathFinder if finder is importlib.machinery.PathFinder else finder
    for finder in sys.meta_path
]
"""


@pytest.fixture
def without_renderer(tmp_path):
    """Returns a function that runs a command line in an interpreter without the renderer."""
    (tmp_path / 'sitecustomize.py').write_text(HIDE)
    environment = dict(os.environ)
    environment.pop('CUJU_RENDER', None)
    environment['PYTHONPATH'] = os.pathsep.join(
        [str(tmp_path), *filter(None, [os.environ.get('PYTHONPATH')])]
    )
    environment['CUJU_RENDER_HOME'] = str(tmp_path / 'render')

    def run(*arguments, **variables):
        return subprocess.run(
            [sys.executable, *arguments],
            capture_output=True,
            text=True,
            timeout=300,
            env={**environment, **variables},
        )

    return run


def test_the_command_names_the_package(without_renderer):
    for arguments in (['render'], ['render', 'serve'], ['help', 'render']):
        result = without_renderer('-m', 'cuju', *arguments)
        assert result.returncode == 1
        assert result.stdout == ''
        (line,) = result.stderr.splitlines()
        assert line.startswith('cuju render: ') and INSTALL in line
    # The other commands run as before.
    result = without_renderer('-m', 'cuju', '--version')
    assert result.returncode == 0 and result.stdout.startswith('cuju ')


def test_running_the_module_names_the_package(without_renderer):
    result = without_renderer('-m', 'cuju.render')
    assert result.returncode != 0
    assert INSTALL in result.stderr
    assert 'Traceback' not in result.stderr


def test_importing_the_module_names_the_package(without_renderer):
    code = """
        import sys
        try:
            import cuju.render
        except ModuleNotFoundError as error:
            print(error.name)
            print(error)
        try:
            import cuju.render._live
        except ModuleNotFoundError as error:
            print(error.name)
        try:
            import cuju.no_such_module
        except ModuleNotFoundError as error:
            print(error)
    """
    result = without_renderer('-c', textwrap.dedent(code))
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == 'cuju.render' and INSTALL in lines[1]
    assert lines[2] == 'cuju.render'
    assert lines[3] == "No module named 'cuju.no_such_module'"


def test_find_spec_raises_and_the_documented_import_serves(without_renderer):
    # The finder that names the package answers `find_spec` as it answers the import, so a program
    # imports the renderer in a `try` block, as docs/api.md shows.
    code = """
        import importlib.util
        import cuju

        try:
            importlib.util.find_spec('cuju.render')
        except ModuleNotFoundError as error:
            print(error.name)
        try:
            import cuju.render as render
        except ModuleNotFoundError as error:
            if error.name != 'cuju.render':
                raise
            render = None
        print(render)
    """
    result = without_renderer('-c', textwrap.dedent(code))
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['cuju.render', 'None']


def test_the_variable_warns_once_and_the_batches_play_as_before(without_renderer):
    code = """
        import logging
        import sys

        records = []


        class Keep(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())


        logging.getLogger('cuju').addHandler(Keep())
        import cuju

        digests = [cuju.digest('5_vs_5_l4_r0', steps=30) for _ in range(2)]
        print(digests[0] == digests[1])
        print(digests[0])
        print(len(records))
        print(records[0] if records else '')
        print(any(name.startswith('cuju.render') for name in sys.modules))
    """
    plain = without_renderer('-c', textwrap.dedent(code))
    assert plain.returncode == 0, plain.stderr
    watched = without_renderer('-c', textwrap.dedent(code), CUJU_RENDER='on')
    assert watched.returncode == 0, watched.stderr
    plain_lines, watched_lines = plain.stdout.splitlines(), watched.stdout.splitlines()
    assert plain_lines[:3] == ['True', plain_lines[1], '0']
    assert watched_lines[:3] == ['True', plain_lines[1], '1']
    assert 'CUJU_RENDER' in watched_lines[3] and INSTALL in watched_lines[3]
    assert watched_lines[4] == 'False'
    # An off value asks for nothing, so it warns of nothing.
    off = without_renderer('-c', textwrap.dedent(code), CUJU_RENDER='off')
    assert off.stdout.splitlines()[:3] == ['True', plain_lines[1], '0']

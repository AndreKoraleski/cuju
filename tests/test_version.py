# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that the version has one source, which the package, its metadata and CMake read."""

import importlib.metadata
import os
import re

import pytest

import cuju

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_version():
    assert re.fullmatch(r'\d+\.\d+\.\d+([a-z]+\d+)?', cuju.__version__)
    # An installed package reports the same version in its metadata.
    if not os.path.abspath(cuju.__file__).startswith(ROOT):
        try:
            assert importlib.metadata.version('cuju') == cuju.__version__
        except importlib.metadata.PackageNotFoundError:
            pass


def test_every_build_file_reads_the_one_source():
    pyproject = os.path.join(ROOT, 'pyproject.toml')
    if not os.path.isfile(pyproject):
        pytest.skip('outside the source tree')
    with open(pyproject, encoding='utf-8') as handle:
        text = handle.read()
    assert 'dynamic = ["version"]' in text and 'python/cuju/_version.py' in text
    assert not re.search(r'^version = ', text, re.MULTILINE)
    with open(os.path.join(ROOT, 'CMakeLists.txt'), encoding='utf-8') as handle:
        assert 'python/cuju/_version.py' in handle.read()

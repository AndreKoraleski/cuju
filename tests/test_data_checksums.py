# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that `data/SHA256SUMS` lists the SHA-256 of every shipped bundle and of the animations."""

import hashlib
import os

import pytest

import cuju
from cuju import _checksums


def test_the_list_covers_the_animations_and_every_bundle():
    listed = _checksums.read_checksums()
    expected = ['animations.bin'] + [f'assets/{name}.cjb' for name in cuju.bundled_assets()]
    assert sorted(listed) == sorted(expected)


def test_every_file_has_its_listed_digest():
    listed = _checksums.read_checksums()
    for name, digest in listed.items():
        with open(os.path.join(_checksums.DATA_DIRECTORY, name), 'rb') as handle:
            assert hashlib.sha256(handle.read()).hexdigest() == digest, name


def test_the_list_is_what_the_generator_writes():
    with open(_checksums.CHECKSUMS, encoding='ascii') as handle:
        text = handle.read()
    # `ci/data_checksums.py` writes this text, so run it after a change to the package data.
    assert text == _checksums.format_checksums(_checksums.data_checksums())


def test_a_malformed_line_is_refused(tmp_path):
    path = tmp_path / 'SHA256SUMS'
    path.write_text('0123 animations.bin\n', encoding='ascii')
    with pytest.raises(ValueError, match='line 1'):
        _checksums.read_checksums(str(path))

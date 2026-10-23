# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that this build reproduces the expected digests of every shipped scenario."""

import json
import os

import pytest

import cuju
from conftest import CUDA, play_digest

# The package ships the digests, for `python -m cuju.check` as well.
DIGESTS = os.path.join(os.path.dirname(os.path.abspath(cuju.__file__)), 'data', 'digests.json')
with open(DIGESTS, encoding='utf-8') as handle:
    EXPECTED = json.load(handle)

# A small batch steps slowly on a GPU, so the GPU plays a few representative cases; the backends
# give the same bits, which `test_backends.py` checks as well.
GPU_CASES = ['5_vs_5/simple115v2', '5_vs_5/raw', '5_vs_5/stacked', 'mixed']


def digest(case, device):
    entry = EXPECTED['cases'][case]
    return play_digest(
        entry['assets'],
        device,
        EXPECTED['steps'],
        seed=EXPECTED['seed'],
        environments=EXPECTED['environments'],
        **entry['options'],
    )


def test_every_shipped_scenario_has_a_case():
    cased = {
        entry['assets'] for entry in EXPECTED['cases'].values() if isinstance(entry['assets'], str)
    }
    missing = [
        scenario for scenario in cuju.bundled_scenarios() if f'{scenario}_l1_r0' not in cased
    ]
    assert not missing


@pytest.mark.parametrize('case', sorted(EXPECTED['cases']))
def test_expected_digest_on_the_cpu(case):
    assert digest(case, 'cpu') == EXPECTED['cases'][case]['digest']


@pytest.mark.skipif(not CUDA, reason='no CUDA device or library')
@pytest.mark.parametrize('case', GPU_CASES)
def test_expected_digest_on_the_gpu(case):
    assert digest(case, 'cuda') == EXPECTED['cases'][case]['digest']

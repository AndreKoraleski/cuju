# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""The long self-check of full matches and many episode ends, opt-in with `CUJU_LONG=1`.

    CUJU_LONG=1 pytest tests/test_long.py

Each case plays thousands of steps, so it reaches what the short digests can't, such as the seeds
of later episodes, Kaggle's second half, the end of a match and the checkpoint reward across
episode ends. It takes a few minutes on the CPU and as long again on a GPU.
"""

import json
import os

import pytest
import torch

import cuju
from conftest import CUDA, play_digest

DIGESTS = os.path.join(os.path.dirname(os.path.abspath(cuju.__file__)), 'data', 'digests.json')
with open(DIGESTS, encoding='utf-8') as handle:
    EXPECTED = json.load(handle)
LONG = EXPECTED['long_cases']

opted_in = pytest.mark.skipif(
    os.environ.get('CUJU_LONG', '') in ('', '0'), reason='set CUJU_LONG=1 for the long cases'
)


def digest(case, device):
    entry = LONG[case]
    if entry['digest'] is None:
        pytest.fail(f'the long case {case} has no expected digest yet')
    return play_digest(
        entry['assets'],
        device,
        entry['steps'],
        seed=EXPECTED['seed'],
        environments=EXPECTED['environments'],
        **entry['options'],
    )


def test_the_long_cases_play_beyond_the_short_ones():
    # A full match of 3,000 steps ends within each case of a match scenario.
    assert all(entry['steps'] > EXPECTED['steps'] for entry in LONG.values())
    assert all(entry['steps'] > 3000 for name, entry in LONG.items() if 'full_match' in name)
    for entry in LONG.values():
        # The assets and options are valid, at the cost of one environment and one step.
        batch = cuju.BatchEnvironment(
            entry['assets'], environment_count=1, device='cpu', **entry['options']
        )
        batch.reset()
        batch.step(torch.zeros((1, batch.agents), dtype=torch.int32))
        batch.close()


@opted_in
@pytest.mark.parametrize('case', sorted(LONG))
def test_long_digest_on_the_cpu(case):
    assert digest(case, 'cpu') == LONG[case]['digest']


@opted_in
@pytest.mark.skipif(not CUDA, reason='no CUDA device or library')
@pytest.mark.parametrize('case', sorted(LONG))
def test_long_digest_on_the_gpu(case):
    assert digest(case, 'cuda') == LONG[case]['digest']

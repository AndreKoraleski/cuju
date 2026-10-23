# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that the outputs depend on neither the backend nor the number of threads."""

import pytest

from conftest import CUDA, play_digest


def test_threads_give_the_same_outputs():
    one = play_digest('5_vs_5_l4_r4', 'cpu', 60, environments=8, threads=1)
    several = play_digest('5_vs_5_l4_r4', 'cpu', 60, environments=8, threads=4)
    assert one == several


@pytest.mark.skipif(not CUDA, reason='no CUDA device or library')
@pytest.mark.parametrize('assets', ['5_vs_5_l4_r4', '11_vs_11_stochastic_l1_r1'])
def test_gpu_equals_cpu(assets):
    assert play_digest(assets, 'cuda', 60) == play_digest(assets, 'cpu', 60)

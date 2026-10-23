# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that a GPU without the memory for a batch says what it needs, and stays usable."""

import pytest

import cuju
from conftest import CUDA

pytestmark = pytest.mark.skipif(not CUDA, reason='no CUDA device or library')


def test_out_of_memory_says_what_it_needs():
    # Two million environments take about 750 GB of worlds, more than any GPU holds.
    with pytest.raises(RuntimeError, match='the GPU is out of memory') as raised:
        cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=2_000_000, device='cuda')
    message = str(raised.value)
    assert ' free of ' in message and 'torch.cuda.empty_cache()' in message
    batch = cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=2, device='cuda')
    assert batch.reset().shape == (2, 1, 115)
    batch.close()

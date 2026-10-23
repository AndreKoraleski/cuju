# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks a GPU batch whose buffers come from PyTorch's caching allocator."""

import pytest
import torch

import cuju
from conftest import CUDA, play_digest
from test_digests import EXPECTED

pytestmark = pytest.mark.skipif(not CUDA, reason='no CUDA device or library')


def test_torch_counts_the_batch_and_gets_it_back():
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=4, allocator='torch')
    held = torch.cuda.memory_allocated() - before
    # Four worlds alone take more than a megabyte.
    assert held > 1 << 20
    batch.reset()
    batch.step(torch.zeros((4, 4), dtype=torch.int32, device='cuda'))
    batch.close()
    assert torch.cuda.memory_allocated() == before


def test_the_default_allocator_stays_outside_torch():
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=4)
    assert torch.cuda.memory_allocated() == before
    batch.close()


def test_the_torch_allocator_gives_the_expected_digest():
    case = '5_vs_5/simple115v2'
    entry = EXPECTED['cases'][case]
    found = play_digest(
        entry['assets'],
        'cuda',
        EXPECTED['steps'],
        seed=EXPECTED['seed'],
        environments=EXPECTED['environments'],
        allocator='torch',
        **entry['options'],
    )
    assert found == entry['digest']


def test_torch_limits_apply_to_the_batch():
    torch.cuda.synchronize()
    total = torch.cuda.get_device_properties(0).total_memory
    # A fraction that leaves room for what torch holds and not for 512 worlds.
    fraction = (torch.cuda.memory_reserved() + (64 << 20)) / total
    torch.cuda.set_per_process_memory_fraction(fraction)
    try:
        with pytest.raises(torch.cuda.OutOfMemoryError):
            cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=512, allocator='torch')
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
    torch.cuda.empty_cache()


def test_an_unknown_allocator_raises():
    with pytest.raises(ValueError, match='allocator'):
        cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=1, allocator='malloc')

# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that `cuju.memory_estimate` counts what a batch requests at its creation."""

import os
import subprocess
import sys
import warnings

import pytest
import torch

import cuju
from conftest import CUDA
from cuju._batch import memory

CASES = [
    ('5_vs_5_l4_r0', {}),
    ('5_vs_5_l1_r1', {'representation': 'raw'}),
    ('5_vs_5_l2_r0', {'representation': 'extracted', 'stacked': True}),
    ('11_vs_11_stochastic_l1_r0', {'representation': 'simple115', 'agent_counts': [(3, 0)]}),
    (
        ['5_vs_5_l4_r4', '11_vs_11_stochastic_l1_r0'],
        {'assignment': [0, 1, 0, 1], 'difficulty': [(None, None), (None, 0.5)], 'with_raw': True},
    ),
]

needs_cuda = pytest.mark.skipif(not CUDA, reason='no CUDA device or library')


# A child process creates a CPU batch and prints the resident memory that the creation added. A
# fresh process holds no freed memory that the batch could reuse without new pages.
RESIDENT = """
import cuju

# The package loads the engine and PyTorch at the first use of a batch, so the memory they take
# stays out of the measurement.
cuju.BatchEnvironment


def resident():
    with open('/proc/self/status', encoding='ascii') as status:
        for line in status:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024


before = resident()
batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=1024, device='cpu', threads=4)
print(resident() - before)
"""

# A child process creates the first batches of its GPU and prints the local memory that the
# estimate gives, the memory that the first creation took beyond the second, and the local memory
# that an estimate gives once a batch exists.
LOCAL_MEMORY = """
import torch
import cuju


def free():
    torch.cuda.synchronize()
    return torch.cuda.mem_get_info()[0]


estimate = cuju.memory_estimate('5_vs_5_l4_r0', environment_count=64, device='cuda')
before = free()
first = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=64, device='cuda')
between = free()
second = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=64, device='cuda')
after = free()
later = cuju.memory_estimate('5_vs_5_l4_r0', environment_count=64, device='cuda')
print(estimate.local_memory_bytes, (before - between) - (between - after), later.local_memory_bytes)
"""

# A child process creates two batches of its GPU with a free memory that fits the batch alone, and
# prints the warnings of each creation.
FIRST_BATCH_WARNING = """
import warnings
import torch
import cuju

estimate = cuju.memory_estimate('5_vs_5_l1_r0', environment_count=2, device='cuda')
free = estimate + estimate.local_memory_bytes // 2
torch.cuda.mem_get_info = lambda device=None: (free, 2**40)
for _ in range(2):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=2, device='cuda').close()
    print(len(caught), *(str(warning.message) for warning in caught))
"""


def run_child(script):
    """Runs `script` in a fresh process and returns its standard output."""
    child = subprocess.run(
        [sys.executable, '-c', script], capture_output=True, text=True, timeout=300
    )
    assert child.returncode == 0, child.stderr[-2000:]
    return child.stdout


@pytest.mark.parametrize('assets, options', CASES)
def test_the_estimate_is_the_batch(device, assets, options):
    estimate = cuju.memory_estimate(assets, environment_count=4, device=device, **options)
    with cuju.BatchEnvironment(assets, environment_count=4, device=device, **options) as batch:
        assert batch.memory_bytes == estimate


def test_a_cpu_estimate_holds_no_local_memory():
    estimate = cuju.memory_estimate('5_vs_5_l4_r0', environment_count=2, device='cpu')
    assert isinstance(estimate, int) and estimate.local_memory_bytes == 0


def test_the_estimate_grows_by_one_world_and_its_buffers_per_environment(device):
    one, two, three = (
        cuju.memory_estimate('5_vs_5_l4_r0', environment_count=count, device=device)
        for count in (1, 2, 3)
    )
    # Each environment holds a world of a few hundred kilobytes, and the tables stay shared.
    assert three - two == two - one > 300_000


def test_the_estimate_raises_as_the_batch():
    with pytest.raises(ValueError, match='first index'):
        cuju.memory_estimate('5_vs_5_l1_r0', environment_count=2, first_index=-1, device='cpu')
    with pytest.raises(RuntimeError, match='agents'):
        cuju.memory_estimate('5_vs_5', agent_counts=[(9, 0)], environment_count=1, device='cpu')


@pytest.mark.skipif(not os.path.exists('/proc/self/status'), reason='no /proc')
def test_a_cpu_batch_takes_its_estimate_in_resident_memory():
    estimate = cuju.memory_estimate('5_vs_5_l4_r0', environment_count=1024, device='cpu', threads=4)
    taken = int(run_child(RESIDENT).split()[-1])
    # The CPU zeroes every buffer at its creation, so all of it is resident. The host copies of the
    # bundles and animations, the library and the alignment of each buffer add the rest. A fixed
    # thread count keeps the stacks and allocator arenas of the pool the same on every machine, and
    # the rest reaches about 90 megabytes on Arm.
    assert 0.98 * estimate <= taken <= estimate * 1.05 + 128 * 2**20


@needs_cuda
def test_torch_counts_the_estimate_of_a_gpu_batch():
    estimate = cuju.memory_estimate('5_vs_5_l4_r0', environment_count=64, device='cuda')
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    batch = cuju.BatchEnvironment(
        '5_vs_5_l4_r0', environment_count=64, device='cuda', allocator='torch'
    )
    taken = torch.cuda.memory_allocated() - before
    batch.close()
    # PyTorch rounds each request up to a multiple of 512 bytes, and the batch makes fewer than 200.
    assert estimate <= taken <= estimate + 200 * 512


@needs_cuda
def test_a_gpu_batch_takes_its_estimate_of_the_free_memory():
    # A first batch makes the driver reserve the local memory of the kernels, once per process.
    cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=1, device='cuda').close()
    estimate = cuju.memory_estimate('5_vs_5_l4_r0', environment_count=1024, device='cuda')
    torch.cuda.synchronize()
    before = torch.cuda.mem_get_info()[0]
    batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=1024, device='cuda')
    torch.cuda.synchronize()
    taken = before - torch.cuda.mem_get_info()[0]
    batch.close()
    # The driver rounds each allocation up to its pages, and other users of the GPU blur the count.
    assert abs(taken - estimate) <= 0.02 * estimate + 64 * 2**20


@needs_cuda
def test_a_first_gpu_batch_takes_the_local_memory_of_the_estimate():
    local, taken, later = (int(value) for value in run_child(LOCAL_MEMORY).split())
    # About 50 KB for each thread that the GPU runs at once, which the second batch finds held.
    assert local > 2**20 and later == 0
    # The driver rounds the reservation and each allocation up to its pages.
    assert abs(taken - local) <= 0.01 * local + 16 * 2**20


@needs_cuda
def test_the_warning_counts_the_local_memory_of_a_first_gpu_batch():
    first, second = run_child(FIRST_BATCH_WARNING).splitlines()
    assert first.startswith('1 ') and 'of it for the local memory of the kernels' in first
    assert second == '0'


def test_the_warning_names_both_amounts():
    with pytest.warns(UserWarning, match=r'requested 2\.0 MiB of cuda:0.*1\.0 MiB free.*WDDM'):
        memory._warn_memory(2 * 2**20, 2**20, torch.device('cuda', 0))
    with pytest.warns(UserWarning, match=r'3\.0 MiB of cuda:0, 2\.0 MiB of it for the local'):
        memory._warn_memory(3 * 2**20, 2**20, torch.device('cuda', 0), 2 * 2**20)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        memory._warn_memory(2**20, 2**20, torch.device('cuda', 0))


@needs_cuda
def test_a_gpu_batch_beyond_the_free_memory_warns(monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=2, device='cuda').close()
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device=None: (2**20, 2**30))
    with pytest.warns(UserWarning, match='1.0 MiB free before its creation'):
        cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=2, device='cuda').close()


def test_a_cpu_batch_never_warns(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device=None: (0, 2**30))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=2, device='cpu').close()

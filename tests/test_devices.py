# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the spellings of a device, which choose only where a batch runs."""

import pytest
import torch

import cuju
from conftest import CUDA, play_digest

needs_gpu = pytest.mark.skipif(not CUDA, reason='no CUDA device or library')


def test_torch_device_of_the_cpu():
    for device in ('cpu', torch.device('cpu'), 'cpu:0'):
        batch = cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=2, device=device)
        assert batch.device == 'cpu' and batch.torch_device == torch.device('cpu')
        assert batch.reset().device == torch.device('cpu')
        batch.close()


@needs_gpu
def test_an_indexed_gpu_gives_the_outputs_of_the_current_one():
    index = torch.cuda.current_device()
    for device in (f'cuda:{index}', torch.device('cuda', index)):
        batch = cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=2, device=device)
        assert batch.device == 'cuda' and batch.torch_device == torch.device('cuda', index)
        assert batch.reset().device == batch.torch_device
        batch.close()
    expected = play_digest('5_vs_5_l4_r0', 'cuda', 20)
    assert play_digest('5_vs_5_l4_r0', f'cuda:{index}', 20) == expected
    assert play_digest('5_vs_5_l4_r0', torch.device('cuda', index), 20) == expected


@needs_gpu
def test_a_missing_gpu_is_refused():
    missing = torch.cuda.device_count()
    with pytest.raises(cuju.DeviceError, match=f'from cuda:0 to cuda:{missing - 1}'):
        cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=1, device=f'cuda:{missing}')


@pytest.mark.skipif(torch.cuda.is_available(), reason='a CUDA device is present')
def test_a_gpu_without_cuda_is_refused():
    with pytest.raises(cuju.DeviceError, match="pass device='cpu'"):
        cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=1, device='cuda:0')


def test_other_devices_are_refused():
    for device in ('meta', 'tpu', torch.device('meta'), None):
        with pytest.raises(cuju.ChoiceError):
            cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=1, device=device)
    assert cuju.library_path(torch.device('cpu')) == cuju.library_path('cpu')
    assert cuju.library_path('cuda:3') == cuju.library_path('cuda')

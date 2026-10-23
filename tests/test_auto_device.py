# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that `device='auto'` picks the GPU where it can and the CPU elsewhere."""

import os

import torch

import cuju
from conftest import play_digest


def expected_backend():
    gpu_library = os.path.isfile(cuju.library_path('cuda')) or os.environ.get('CUJU_CUDA_LIBRARY')
    return 'cuda' if torch.cuda.is_available() and gpu_library else 'cpu'


def test_auto_picks_a_backend():
    batch = cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=2, device='auto')
    assert batch.device == expected_backend()
    assert batch.reset().device == batch.torch_device
    if batch.device == 'cuda':
        assert batch.torch_device.index == torch.cuda.current_device()
    batch.close()
    assert cuju.library_path('auto') == cuju.library_path(expected_backend())


def test_auto_without_a_gpu_takes_the_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    batch = cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=2, device='auto')
    assert batch.device == 'cpu' and batch.torch_device == torch.device('cpu')
    batch.close()


def test_auto_plays_the_same_matches():
    assert play_digest('5_vs_5_l4_r0', 'auto', 20) == play_digest('5_vs_5_l4_r0', 'cpu', 20)

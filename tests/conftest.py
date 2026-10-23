# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures of the self-check."""

import ctypes
import hashlib
import os
import platform

import pytest
import torch

import cuju

DEVICES = ['cpu', 'cuda']
# A CPU-only installation, such as the wheels for macOS and Linux on Arm, has no CUDA library.
CUDA_LIBRARY = os.path.isfile(cuju.library_path('cuda'))
CUDA = torch.cuda.is_available() and CUDA_LIBRARY


def pytest_report_header(config):
    lines = [f'cuju {cuju.__version__}, torch {torch.__version__}']
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        lines.append(
            f'CUDA {torch.version.cuda}, {torch.cuda.get_device_name()} '
            f'(compute capability {capability[0]}.{capability[1]})'
        )
    else:
        lines.append('no CUDA device: the CUDA cases are skipped, which is not a pass')
    if not CUDA_LIBRARY:
        lines.append('no CUDA library in this installation: the CUDA cases are skipped')
    flags = set()
    try:
        with open('/proc/cpuinfo', encoding='ascii', errors='replace') as handle:
            for line in handle:
                if line.startswith('flags'):
                    flags = set(line.partition(':')[2].split())
                    break
    except OSError:
        pass
    present = [name for name in ('avx2', 'bmi2', 'fma') if name in flags]
    lines.append(f'{platform.processor() or platform.machine()}, CPU flags: {" ".join(present)}')
    return lines


@pytest.fixture(params=DEVICES)
def device(request):
    if request.param == 'cuda' and not CUDA:
        pytest.skip('no CUDA device or library')
    return request.param


def tensor_bytes(tensor):
    """Returns the bytes of a tensor, read without numpy."""
    tensor = tensor.detach().to('cpu').contiguous()
    return ctypes.string_at(tensor.data_ptr(), tensor.numel() * tensor.element_size())


def actions_at(step, environments, agents, action_count):
    """Returns the actions of a step from integer arithmetic alone, the same on every machine."""
    values = [
        [(step * 7 + environment * 3 + agent * 5) % action_count for agent in range(agents)]
        for environment in range(environments)
    ]
    return torch.tensor(values, dtype=torch.int32)


def play_digest(assets, device, steps, seed=7, environments=4, **options):
    """Plays a batch with the actions of `actions_at` and returns the SHA-256 of its outputs.

    The digest covers every observation, reward and done flag of every step, then the state of
    every environment at the end (unless the batch stacks its observations).
    """
    batch = cuju.BatchEnvironment(
        assets, environment_count=environments, device=device, seed=seed, **options
    )
    digest = hashlib.sha256()

    def add(observation):
        if isinstance(observation, dict):
            for name in sorted(observation):
                digest.update(tensor_bytes(observation[name]))
        else:
            digest.update(tensor_bytes(observation))

    add(batch.reset())
    for step in range(steps):
        actions = actions_at(step, environments, batch.agents, batch.action_count)
        observation, reward, done, _ = batch.step(actions.to(batch.torch_device))
        add(observation)
        digest.update(tensor_bytes(reward))
        digest.update(tensor_bytes(done))
    if not batch.stacked:
        for index in range(environments):
            digest.update(batch.get_state(index))
    batch.close()
    return digest.hexdigest()

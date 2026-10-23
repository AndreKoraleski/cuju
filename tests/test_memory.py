# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that a batch dropped without `close` is freed, once no tensor of it remains."""

import gc
import os
import weakref

import pytest
import torch

import cuju


def resident_megabytes():
    """Returns the resident memory of the process, or `None` without `/proc`."""
    try:
        with open('/proc/self/status', encoding='ascii') as status:
            for line in status:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def free_gpu_megabytes(device):
    if device != 'cuda':
        return None
    torch.cuda.synchronize()
    return torch.cuda.mem_get_info()[0] // 2**20


@pytest.fixture
def without_collection():
    """Turns the cycle collector off, so that only reference counts free what goes."""
    gc.collect()
    gc.disable()
    yield
    gc.enable()


def play_and_drop(device):
    """Creates and steps a batch, drops it without `close`, and returns weak references."""
    batch = cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=2, device=device)
    batch.reset()
    for _ in range(3):
        batch.step(torch.zeros((2, 1), dtype=torch.int32, device=device))
    references = weakref.ref(batch), weakref.ref(batch._native)
    del batch
    return references


def test_dropped_batches_are_freed(device, without_collection):
    play_and_drop(device)
    resident, free = resident_megabytes(), free_gpu_megabytes(device)
    for _ in range(20):
        batch, native = play_and_drop(device)
        assert batch() is None and native() is None
    # Twenty batches that each held their tables would take gigabytes.
    if resident is not None:
        assert resident_megabytes() - resident < 300
    if free is not None:
        assert free - free_gpu_megabytes(device) < 100


def test_a_kept_tensor_outlives_its_batch(device, without_collection):
    batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=2, device=device)
    batch.reset()
    observation, reward, done, information = batch.step(
        torch.zeros((2, 4), dtype=torch.int32, device=device)
    )
    expected = observation.clone()
    references = weakref.ref(batch), weakref.ref(batch._native)
    del batch, reward, done, information
    # The batch is gone, but the memory that the tensor views stays.
    assert references[0]() is None and references[1]() is not None
    assert torch.equal(observation, expected)
    del observation
    assert references[1]() is None


DRILL = """from . import *


def build_scenario(builder):
    builder.config().game_duration = 50
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(0.2, 0.0, e_PlayerRole_CF)
"""


def test_close_frees_at_once_and_a_dropped_batch_removes_its_bundles(tmp_path, without_collection):
    drill = tmp_path / 'drill.py'
    drill.write_text(DRILL, encoding='utf-8')
    batch = cuju.BatchEnvironment(str(drill), environment_count=1, device='cpu')
    directory = batch._native.scenario_directory.name
    assert os.path.isdir(directory)
    closed = cuju.BatchEnvironment(str(drill), environment_count=1, device='cpu')
    closed_directory = closed._native.scenario_directory.name
    closed.close()
    # `close` frees at once.
    assert closed._native.environment is None and not os.path.exists(closed_directory)
    del batch
    assert not os.path.exists(directory)

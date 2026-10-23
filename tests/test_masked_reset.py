# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that `reset(mask=)` resets only the masked environments, each as GRF's `reset()`."""

import pytest
import torch

import cuju
from conftest import CUDA, actions_at, tensor_bytes

ASSETS = '5_vs_5_l2_r1'


def outputs(batch, observation, reward=None, done=None):
    """Returns the bytes of each environment's outputs."""
    rows = []
    for environment in range(batch.environment_count):
        row = [tensor_bytes(observation[environment])]
        if reward is not None:
            row += [tensor_bytes(reward[environment]), tensor_bytes(done[environment])]
        rows.append(row)
    return rows


def play(batch, indices, steps, start=0):
    """Steps `batch` with the actions of the environments `indices`, and returns their outputs."""
    played = [[] for _ in range(batch.environment_count)]
    for step in range(start, start + steps):
        actions = actions_at(step, max(indices) + 1, batch.agents, batch.action_count)[indices]
        observation, reward, done, _ = batch.step(actions.to(batch.device))
        for row, environment in zip(outputs(batch, observation, reward, done), played, strict=True):
            environment += row
    return played


def test_only_the_masked_environments_reset(device):
    batch = cuju.BatchEnvironment(ASSETS, environment_count=3, seed=4, device=device)
    batch.reset()
    before = play(batch, [0, 1, 2], 25)
    mask = torch.tensor([False, True, False], device=device)
    observation = batch.reset(mask=mask)
    reset_rows = outputs(batch, observation)
    after = play(batch, [0, 1, 2], 25, start=25)

    # The unmasked environments play on as if nothing happened.
    plain = cuju.BatchEnvironment(ASSETS, environment_count=3, seed=4, device=device)
    plain.reset()
    unbroken = play(plain, [0, 1, 2], 50)
    for environment in (0, 2):
        assert before[environment] + after[environment] == unbroken[environment]

    # The masked environment plays as a GRF environment with a second `reset()` after 25 steps.
    alone = cuju.BatchEnvironment(ASSETS, environment_count=1, seed=4, first_index=1, device=device)
    alone.reset()
    assert play(alone, [1], 25)[0] == before[1]
    assert outputs(alone, alone.reset())[0] == reset_rows[1]
    assert play(alone, [1], 25, start=25)[0] == after[1]


def test_masked_first_resets_equal_one_reset(device):
    whole = cuju.BatchEnvironment(ASSETS, environment_count=3, seed=8, device=device)
    expected = outputs(whole, whole.reset())
    parts = cuju.BatchEnvironment(ASSETS, environment_count=3, seed=8, device=device)
    parts.reset(mask=[True, False, True])
    with pytest.raises(RuntimeError, match='before its first step'):
        parts.step(torch.zeros((3, parts.agents), dtype=torch.int32))
    assert outputs(parts, parts.reset(mask=torch.tensor([0, 1, 0], device=device))) == expected
    assert play(parts, [0, 1, 2], 10) == play(whole, [0, 1, 2], 10)


def test_masked_reset_with_seeds(device):
    batch = cuju.BatchEnvironment(ASSETS, environment_count=2, device=device)
    batch.reset()
    play(batch, [0, 1], 5)
    batch.reset(mask=[True, True], seeds=[123, 456])
    seeded = cuju.BatchEnvironment(ASSETS, environment_count=2, device=device)
    seeded.reset()
    play(seeded, [0, 1], 5)
    seeded.reset(seeds=[123, 456])
    assert play(batch, [0, 1], 10) == play(seeded, [0, 1], 10)


def test_mask_errors():
    batch = cuju.BatchEnvironment(ASSETS, environment_count=2, device='cpu')
    with pytest.raises(ValueError, match='mask'):
        batch.reset(mask=[True, False, True])
    with pytest.raises(TypeError, match='mask'):
        batch.reset(mask=torch.ones(2))
    with pytest.raises(TypeError, match='mask'):
        batch.reset(mask=[1.0, 0.0])


@pytest.mark.skipif(not CUDA, reason='no CUDA device or library')
def test_a_cuda_graph_captures_a_masked_reset():
    def run(capture):
        batch = cuju.BatchEnvironment(ASSETS, environment_count=4, seed=2, device='cuda')
        batch.reset()
        mask = torch.tensor([True, False, False, True], device='cuda')
        actions = torch.zeros((4, batch.agents), dtype=torch.int32, device='cuda')
        # An eager step first makes every output view, which a capture must not create.
        batch.step(actions)
        batch.reset(mask=mask)
        torch.cuda.synchronize()
        if capture:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                batch.step(actions)
                batch.reset(mask=mask)
        rows = []
        for _ in range(3):
            if capture:
                graph.replay()
            else:
                batch.step(actions)
                batch.reset(mask=mask)
            rows.append(outputs(batch, batch.observation()))
        return rows

    assert run(True) == run(False)

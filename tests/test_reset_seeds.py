# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the seeds and flags of `reset`, where a seed of -1 draws exactly as `seeds=None` does."""

import pytest
import torch

import cuju
from conftest import actions_at, tensor_bytes

# Short episodes, so that the later resets draw from each environment's generator.
ASSETS = 'academy_empty_goal_close_l1_r0'


def play(batch, observation, steps):
    """Returns the bytes of each environment's outputs from `observation` on."""
    rows = [[tensor_bytes(row)] for row in observation]
    for step in range(steps):
        actions = actions_at(step, batch.environment_count, batch.agents, batch.action_count)
        observation, reward, done, _ = batch.step(actions.to(batch.device))
        for environment, row in enumerate(rows):
            row += [
                tensor_bytes(observation[environment]),
                tensor_bytes(reward[environment]),
                tensor_bytes(done[environment]),
            ]
    return rows


def run(device, **reset):
    batch = cuju.BatchEnvironment(ASSETS, environment_count=4, seed=6, device=device)
    return play(batch, batch.reset(**reset), 120)


def test_minus_one_draws_as_none(device):
    mixed = run(device, seeds=[11, -1, 13, -1])
    drawn = run(device)
    given = run(device, seeds=[11, 12, 13, 14])
    assert mixed[1] == drawn[1] and mixed[3] == drawn[3]
    assert mixed[0] == given[0] and mixed[2] == given[2]


def test_minus_one_everywhere_is_none(device):
    assert run(device, seeds=-1) == run(device)


def test_minus_one_in_a_masked_reset(device):
    def masked(seeds):
        batch = cuju.BatchEnvironment(ASSETS, environment_count=2, seed=6, device=device)
        batch.reset()
        play(batch, batch.observation(), 30)
        return play(batch, batch.reset(mask=[False, True], seeds=seeds), 30)

    assert masked([-1, -1]) == masked(None)
    assert masked([5, -1]) == masked(None)


def test_other_negative_seeds_are_refused():
    batch = cuju.BatchEnvironment(ASSETS, environment_count=2, device='cpu')
    with pytest.raises(ValueError, match='-1 to draw'):
        batch.reset(seeds=[-2, 3])


def test_floating_seeds_are_refused():
    batch = cuju.BatchEnvironment(ASSETS, environment_count=2, device='cpu')
    # GRF: the seed is a C++ `unsigned int`, which refuses every float, 5.0 included.
    for seeds in ([5.5, 3], 5.0, torch.tensor([5.0, 3.0])):
        with pytest.raises(TypeError, match='integer seeds'):
            batch.reset(seeds=seeds)
    with pytest.raises(TypeError, match='integer game_engine_random_seed'):
        cuju.BatchEnvironment(
            ASSETS, environment_count=2, device='cpu', game_engine_random_seed=5.0
        )
    batch.reset(seeds=torch.tensor([5, 3]))
    batch.close()


def test_reverse_flags_are_zero_or_one():
    batch = cuju.BatchEnvironment(ASSETS, environment_count=2, device='cpu')
    with pytest.raises(cuju.ArgumentError, match=r'reverse flags of 0 or 1, found \[2, 2\]'):
        batch.reset(seeds=[5, 6], reverse=2)
    with pytest.raises(cuju.ArgumentError, match='reverse flags'):
        batch.reset(seeds=[5, 6], reverse=[0, 2**32])
    with pytest.raises(TypeError, match='integer reverse'):
        batch.reset(seeds=[5, 6], reverse=0.5)
    batch.reset(seeds=[5, 6], reverse=[True, 0])
    batch.close()

# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that `first_index` plays the environments of a larger batch under their own indices."""

import pytest
import torch

import cuju
from conftest import actions_at, tensor_bytes

# Short episodes, so that the steps cross several resets and their seed draws.
ASSETS = 'academy_empty_goal_close_l1_r0'


def play(batch, indices, steps):
    """Returns the bytes of the outputs of each environment, with the actions of `indices`."""
    outputs = [[tensor_bytes(observation)] for observation in batch.reset()]
    for step in range(steps):
        actions = actions_at(step, max(indices) + 1, batch.agents, batch.action_count)[indices]
        observation, reward, done, information = batch.step(actions.to(batch.device))
        for environment in range(batch.environment_count):
            outputs[environment] += [
                tensor_bytes(observation[environment]),
                tensor_bytes(reward[environment]),
                tensor_bytes(done[environment]),
                tensor_bytes(information['final_observation'][environment]),
            ]
    return outputs


def test_a_slice_plays_the_environments_of_a_larger_batch(device):
    whole = cuju.BatchEnvironment(ASSETS, environment_count=7, seed=3, device=device)
    part = cuju.BatchEnvironment(ASSETS, environment_count=3, seed=3, first_index=4, device=device)
    expected = play(whole, list(range(7)), 120)
    assert play(part, [4, 5, 6], 120) == expected[4:]


def test_the_default_is_index_zero(device):
    plain = cuju.BatchEnvironment(ASSETS, environment_count=2, seed=9, device=device)
    explicit = cuju.BatchEnvironment(
        ASSETS, environment_count=2, seed=9, first_index=0, device=device
    )
    assert play(plain, [0, 1], 40) == play(explicit, [0, 1], 40)


def test_the_last_indices():
    batch = cuju.BatchEnvironment(ASSETS, environment_count=2, first_index=2**32 - 2, device='cpu')
    assert torch.isfinite(batch.reset()).all()


@pytest.mark.parametrize('first_index', [-1, 2**32 - 1, 2**40])
def test_indices_outside_the_seed_rule(first_index):
    with pytest.raises(ValueError, match='first index'):
        cuju.BatchEnvironment(ASSETS, environment_count=2, first_index=first_index, device='cpu')

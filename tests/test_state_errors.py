# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that a refused state names the field in which it differs from the environment."""

import pytest

import cuju
from conftest import actions_at


def state_of(assets, device):
    """Returns the state of one environment of `assets` after a few steps."""
    batch = cuju.BatchEnvironment(assets, environment_count=1, device=device)
    batch.reset()
    for step in range(5):
        batch.step(actions_at(step, 1, batch.agents, batch.action_count).to(device))
    state = batch.get_state(0)
    batch.close()
    return state


def started(assets, device):
    batch = cuju.BatchEnvironment(assets, environment_count=1, device=device)
    batch.reset()
    return batch


def test_another_agent_count(device):
    batch = started('5_vs_5_l4_r1', device)
    with pytest.raises(
        RuntimeError, match='right_agents is 0 in the state and 1 in the environment'
    ):
        batch.set_state(0, state_of('5_vs_5_l4_r0', device))


def test_another_scenario(device):
    batch = started('academy_empty_goal_close_l1_r0', device)
    with pytest.raises(RuntimeError, match=r'found another: ball_position is \(') as raised:
        batch.set_state(0, state_of('academy_empty_goal_l1_r0', device))
    # The hint of what must match follows the field.
    assert 'a state loads only into an environment of the same scenario' in str(raised.value)


def test_bytes_that_stop_short_or_run_on(device):
    state = state_of('5_vs_5_l4_r0', device)
    batch = started('5_vs_5_l4_r0', device)
    with pytest.raises(RuntimeError, match='do not load: state too short at'):
        batch.set_state(0, state[:-100])
    with pytest.raises(RuntimeError, match='do not load: 8 bytes follow the end of the state'):
        batch.set_state(0, state + bytes(8))
    batch.set_state(0, state)
    assert batch.get_state(0) == state

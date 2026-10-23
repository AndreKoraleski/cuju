# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that a state moved to another environment continues the same match."""

import cuju
from conftest import actions_at, tensor_bytes


def test_state_round_trip(device):
    batch = cuju.BatchEnvironment(
        '5_vs_5_l4_r0', environment_count=2, device=device, rewards='scoring,checkpoints'
    )
    batch.reset()
    for step in range(60):
        batch.step(actions_at(step, 2, 4, 19).to(device))
    batch.set_state(1, batch.get_state(0))
    # A state carries no generator, so the comparison stops where an episode ends.
    ended = False
    for step in range(60, 160):
        actions = actions_at(step, 1, 4, 19).repeat(2, 1).to(device)
        observation, reward, done, information = batch.step(actions)
        if bool(done[0]):
            ended = True
            observation = information['final_observation']
        assert tensor_bytes(observation[0]) == tensor_bytes(observation[1])
        assert tensor_bytes(reward[0]) == tensor_bytes(reward[1])
        if ended:
            break
    if not ended:
        assert batch.get_state(0) == batch.get_state(1)

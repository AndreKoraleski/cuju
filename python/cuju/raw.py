# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""GRF's raw observation dictionaries, from the raw tensors of a batch.

A batch returns raw observations as tensors `[environment_count, agents, ...]`, padded to the
largest team. These functions turn them, bit for bit, into GRF's list with one dictionary per agent
and numpy arrays sized to each team, so `cuju.raw.grf_observations(batch, 0)` returns GRF's list
for environment 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ['grf_observation', 'grf_observations', 'team_sizes']

# GRF: these values of a raw observation are Python integers.
_SCALARS = frozenset(
    {'active', 'ball_owned_player', 'ball_owned_team', 'designated', 'game_mode', 'steps_left'}
)


def grf_observation(
    raw: dict[str, Any], environment: int, agent: int, own_players: int, other_players: int
) -> dict[str, Any]:
    """Returns the raw observation of one agent as GRF's dictionary, in GRF's types.

    The dictionary equals the one in GRF's list of raw observations for that agent. Positions,
    directions, the ball and tiredness are `float64` arrays, which hold GRF's values exactly since
    GRF computes them in `float32`. The cards and the active flags are `bool` arrays, the roles
    `int64`, the sticky actions `uint8`, the single values Python integers and `score` a list of
    two.

    Args:
        raw: The raw tensors of a batch, `[environment_count, agents, ...]`, as `raw()` returns.
        environment: The index of the environment.
        agent: The index of the agent.
        own_players: The number of players of the agent's own team, which it sees as `left_team`.
        other_players: The number of players of the other team.
    """
    result = {}
    for name, tensor in raw.items():
        value = tensor[environment, agent].cpu().numpy()
        if name in _SCALARS:
            result[name] = int(value)
        elif name == 'score':
            result[name] = [int(goals) for goals in value]
        elif name == 'sticky_actions':
            result[name] = value.astype(np.uint8)
        else:
            if name.startswith('left_'):
                value = value[:own_players]
            elif name.startswith('right_'):
                value = value[:other_players]
            if value.dtype == np.float32:
                value = value.astype(np.float64)
            elif value.dtype == np.int32:
                value = value.astype(np.int64)
            else:
                value = value.copy()
            result[name] = value
    return result


def team_sizes(batch: Any, environment: int, agent: int) -> tuple[int, int]:
    """Returns the player counts of an agent's own team and of the other team."""
    left = int(batch.left_players[environment])
    right = int(batch.right_players[environment])
    # GRF: a right agent sees its own team as the left team.
    return (right, left) if agent >= int(batch.left_agents[environment]) else (left, right)


def grf_observations(
    batch: Any, environment: int, agents: list[int] | None = None, final: bool = False
) -> list[dict[str, Any]]:
    """Returns GRF's list of raw observations of an environment of a batch.

    Args:
        batch: A batch with the raw representation or `with_raw=True`.
        environment: The index of the environment.
        agents: The agents, in order, or `None` for every agent of the environment.
        final: Whether to read `final_raw()`, the step that ended the last episode, not `raw()`.
    """
    raw = batch.final_raw() if final else batch.raw()
    if agents is None:
        agents = range(int(batch.left_agents[environment] + batch.right_agents[environment]))
    return [
        grf_observation(raw, environment, agent, *team_sizes(batch, environment, agent))
        for agent in agents
    ]

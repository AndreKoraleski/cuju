# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""GRF's scripted players `bot` and `lazy`, for one agent of every environment of a batch.

`Bot` is an exact port of `gfootball/env/players/bot.py` and `Lazy` of `lazy.py`. Their actions are
indices into the batch's action set, and the bot needs the `full` set. The guide is in the
repository's `docs/migrating-from-grf.md`, under Scripted players and `control_all_players`.
"""

from __future__ import annotations

import math
from typing import Any

import torch

__all__ = ['Bot', 'Lazy']

# GRF: the indices of the bot's actions in the `full` set.
_IDLE, _LONG_PASS, _SHORT_PASS, _SHOT, _SPRINT = 0, 9, 11, 12, 13
_PRESSURE, _RELEASE_PRESSURE = 21, 30
# GRF: `_direction_action` scores the eight directions in the order top, top_left, left,
# bottom_left, bottom, bottom_right, right, top_right.
_DIRECTIONS = (3, 2, 1, 8, 7, 6, 5, 4)
# GRF: each direction vector divided by its norm, `np.linalg.norm`, which is `sqrt(2.0)` for a
# diagonal.
_DIAGONAL = 1 / math.sqrt(2.0)
_DIRECTION_VECTORS = (
    (0.0, -1.0),
    (-_DIAGONAL, -_DIAGONAL),
    (-1.0, 0.0),
    (-_DIAGONAL, _DIAGONAL),
    (0.0, 1.0),
    (_DIAGONAL, _DIAGONAL),
    (1.0, 0.0),
    (_DIAGONAL, -_DIAGONAL),
)
_FULL_ACTIONS = 33
_TARGET_X = 0.85
_SHOOT_DISTANCE = 0.15


def _norm(x, y):
    """Returns `np.linalg.norm` of the vectors `(x, y)`, the square root of their dot product."""
    # Exact: numpy's dot of two numbers adds `x * x` and `y * y` in this order, each rounded, with
    # no fused multiply-add, under the OpenBLAS kernels of x86-64 without AVX-512.
    return torch.sqrt(x * x + y * y)


def _dot(x, y, u, v):
    """Returns `np.dot` of the vectors `(x, y)` and `(u, v)`."""
    # Exact: numpy's dot starts from 0.0, which only turns a negative zero positive, and no
    # comparison of the bot tells the zeros apart.
    return x * u + y * v


def _first(values, maximum):
    """Returns the first index of the maximum or minimum on the last axis, as numpy's `argmax`.

    GRF: the bot's strict comparisons pick the first as well.
    """
    best = values.amax(-1, keepdim=True) if maximum else values.amin(-1, keepdim=True)
    positions = torch.arange(values.shape[-1], device=values.device).expand_as(values)
    return torch.where(values == best, positions, values.shape[-1]).amin(-1)


def _direction_action(x, y):
    """Returns, for each movement `(x, y)`, the direction action closest to it."""
    scores = torch.stack([_dot(x, y, u, v) for u, v in _DIRECTION_VECTORS], -1)
    return torch.tensor(_DIRECTIONS, device=x.device)[_first(scores, True)]


class Bot:
    """GRF's `bot` player for one agent of every environment of a batch.

    Each call of `act` reads the agent's current raw observation and returns GRF's bot's action.
    It reads the observation in `float64`, as GRF's bot reads numpy arrays, and decides in tensors
    over every environment at once on the batch's device.
    Like GRF's, the bot remembers its last action and whether it presses, and keeps both across
    episodes, since GRF's environment never resets its players' memory.

    Attributes:
        batch: The batch.
        agent: The agent that the bot plays.
    """

    def __init__(self, batch: Any, agent: int) -> None:
        """Binds a bot to an agent of a batch.

        Args:
            batch: A batch with the `full` action set and with `with_raw=True` or the raw
                representation.
            agent: The index of the agent, which must exist in every environment.

        Raises:
            ValueError: If the batch has another action set or no raw arrays, or if the agent is
                missing in an environment.
        """
        # GRF: the bot asserts the `full` action set.
        if batch.action_count != _FULL_ACTIONS:
            raise ValueError("expected a batch with action_set='full', which GRF's bot needs")
        if getattr(batch, '_raw_fields', None) is None:
            raise ValueError('expected a batch with the raw representation or with_raw=True')
        if not 0 <= agent < batch.agents or not bool(batch.agent_mask[:, agent].all()):
            raise ValueError(f'expected an agent of every environment, found {agent}')
        self.batch = batch
        self.agent = agent
        device = batch._torch_device
        right = agent >= batch.left_agents
        # A right agent sees its own team as the left team.
        self._own = torch.where(right, batch.right_players, batch.left_players).to(device)
        self._other = torch.where(right, batch.left_players, batch.right_players).to(device)
        self._last_action = torch.full((batch.environment_count,), _IDLE, device=device)
        self._pressure = torch.zeros(batch.environment_count, dtype=torch.bool, device=device)

    def reset(self, mask: Any = None) -> None:
        """Forgets the last action and the pressing, as a newly created GRF bot.

        GRF never does this between episodes, and it serves a bot that starts with a new
        environment.

        Args:
            mask: `[environment_count]` flags of the environments to reset, or `None` for all.
        """
        if mask is None:
            self._last_action.fill_(_IDLE)
            self._pressure.fill_(False)
        else:
            mask = torch.as_tensor(mask, dtype=torch.bool).to(self._pressure.device)
            self._last_action[mask] = _IDLE
            self._pressure[mask] = False

    def act(self, raw: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """Returns the bot's actions, `[environment_count]` `int32` indices into the `full` set.

        The bot remembers them for its next decision.

        Args:
            raw: The raw observation to act on, `batch.raw()` by default.
        """
        raw = self.batch.raw() if raw is None else raw
        action, pressure = self._decide(raw)
        self._last_action = action
        self._pressure = pressure
        return action.to(torch.int32)

    def _decide(self, raw):
        """Returns the action and the pressing of `bot.Player._get_action` for every environment."""
        agent = self.agent
        own = raw['left_team'][:, agent].double()
        other = raw['right_team'][:, agent].double()
        count = own.shape[1]
        slots = torch.arange(count, device=own.device)
        own_valid = slots < self._own[:, None]
        other_valid = slots < self._other[:, None]
        # GRF: `left_team[active]` with Python's indexing, where -1 is the last player.
        active = raw['active'][:, agent].long()
        active = torch.where(active < 0, active + self._own, active)
        rows = torch.arange(own.shape[0], device=own.device)
        position = own[rows, active]
        x, y = position[:, 0], position[:, 1]
        infinity = torch.tensor(math.inf, dtype=torch.float64, device=own.device)
        other_x, other_y = other[:, None, :, 0], other[:, None, :, 1]

        def distances(px, py):
            """`_object_distance` from points `[N, M]` to opponents, `[N, M, players]`, or inf."""
            distance = _norm(px[..., None] - other_x, py[..., None] - other_y)
            return torch.where(other_valid[:, None], distance, infinity)

        # The run to the goal, and the closest opponent in front, who may be missing.
        target_x, target_y = _TARGET_X - x, 0.0 - y
        move_action = _direction_action(target_x, target_y)
        ahead = _dot(
            target_x[:, None],
            target_y[:, None],
            other[..., 0] - x[:, None],
            other[..., 1] - y[:, None],
        )
        front = torch.where(ahead > 0, distances(x[:, None], y[:, None])[:, 0], infinity)
        front_index = _first(front, False)
        front_distance = front[rows, front_index]
        has_front = torch.isfinite(front_distance)
        front_distance = torch.where(has_front, front_distance, 2.0)
        # The best pass, where each teammate within 0.3 scores the closest approach of an
        # opponent to ten points along the pass.
        trajectory_x, trajectory_y = own[..., 0] - x[:, None], own[..., 1] - y[:, None]
        reach = _norm(trajectory_x, trajectory_y)
        closest = None
        for step in range(10):
            fraction = (step + 1) / 10.0
            px = x[:, None] + fraction * trajectory_x
            py = y[:, None] + fraction * trajectory_y
            distance = distances(px, py).amin(-1)
            closest = distance if closest is None else torch.minimum(closest, distance)
        score = torch.where(own_valid & ~(reach > 0.3), -closest, -infinity)
        best = own[rows, _first(score, True)]
        pass_x, pass_y = best[:, 0] - x, best[:, 1] - y
        pass_action = _direction_action(pass_x, pass_y)
        # Around the opponent in front, perpendicular to it and toward the goal.
        opponent = other[rows, front_index]
        delta_x, delta_y = opponent[:, 0] - x, opponent[:, 1] - y
        turn_x, turn_y = delta_y, -delta_x
        flip = target_x * turn_x < 0
        avoid_action = _direction_action(
            torch.where(flip, -turn_x, turn_x), torch.where(flip, -turn_y, turn_y)
        )
        alone = (best[:, 0] == x) & (best[:, 1] == y)
        blocked = front_distance < 0.08
        run = torch.where(
            blocked & alone,
            avoid_action,
            torch.where(
                blocked,
                torch.where(self._last_action == pass_action, _SHORT_PASS, pass_action),
                move_action,
            ),
        )
        ball = raw['ball'][:, agent].double()
        shoot = _SHOOT_DISTANCE > _norm(ball[:, 0] - _TARGET_X, ball[:, 1] - 0.0)
        action = torch.where(shoot, _SHOT, run)
        # The rules before the play, from the last to the first, as GRF checks them first to last.
        # Pressing ends wherever the play or its release is reached.
        action = torch.where(self._pressure, _RELEASE_PRESSURE, action)
        pressure = torch.zeros_like(self._pressure)
        defending = raw['ball_owned_team'][:, agent] == 1
        pressing = self._last_action == _PRESSURE
        action = torch.where(defending, torch.where(pressing, _SPRINT, _PRESSURE), action)
        pressure = torch.where(defending, pressing.logical_not() | self._pressure, pressure)
        set_piece = raw['game_mode'][:, agent] != 0
        action = torch.where(set_piece, _LONG_PASS, action)
        pressure = torch.where(set_piece, self._pressure, pressure)
        return action, pressure


class Lazy:
    """GRF's `lazy` player, which stands still with `idle` for its agents in every environment.

    Attributes:
        batch: The batch.
        agents: The agents that the player plays.
    """

    def __init__(self, batch: Any, agents: int | list[int]) -> None:
        """Binds a lazy player to agents of a batch.

        Args:
            batch: Any batch, since `idle` is index 0 of every action set.
            agents: An agent index or a list of them.
        """
        self.batch = batch
        self.agents = [agents] if isinstance(agents, int) else list(agents)

    def act(self, raw: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """Returns `idle`, `[environment_count, agents]`, or `[environment_count]` for one agent."""

        shape = (self.batch.environment_count, len(self.agents))
        actions = torch.full(shape, _IDLE, dtype=torch.int32, device=self.batch._torch_device)
        return actions[:, 0] if len(self.agents) == 1 else actions

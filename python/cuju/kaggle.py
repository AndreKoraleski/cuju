# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Kaggle's GRF competition protocol, with `players_raw` from a batch and Kaggle agents in a batch.

`KaggleAgent` runs a Kaggle agent function, such as a `submission.py` agent or a public rule-based
bot, as one agent of every environment of a batch, and `evaluate` plays whole matches with it. The
guide is in the repository's `docs/recipes.md`, under Train for the Kaggle competition.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import torch

from . import BatchEnvironment
from .raw import grf_observation, team_sizes

__all__ = ['KaggleAgent', 'evaluate', 'observation', 'players_raw']

# Kaggle: the time an agent may overrun across a match, in seconds, which every observation holds.
_OVERAGE_TIME = 60


class _Struct(dict):
    """A dictionary whose keys are also attributes, as Kaggle's `Struct` that agents receive."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def _lists(entry):
    """Returns a GRF raw dictionary with its numpy arrays as lists, as Kaggle's JSON makes them."""
    return {
        name: value.tolist() if isinstance(value, np.ndarray) else value
        for name, value in entry.items()
    }


def players_raw(
    batch: Any, environment: int, agent: int, final: bool = False
) -> list[dict[str, Any]]:
    """Returns `obs['players_raw']` of a Kaggle agent that plays an agent of a batch.

    The list holds one dictionary, GRF's raw observation of the agent's player, with lists for
    GRF's numpy arrays, Python floats for its `float64` values, as Kaggle's agents receive it after
    JSON. A right agent sees the match turned, with its own team as `left_team`, as in Kaggle.

    Args:
        batch: A batch with the raw representation or `with_raw=True`.
        environment: The index of the environment.
        agent: The index of the agent.
        final: Whether to read the step that ended the last episode, `final_raw()`.
    """
    raw = batch.final_raw() if final else batch.raw()
    return [
        _lists(grf_observation(raw, environment, agent, *team_sizes(batch, environment, agent)))
    ]


def observation(
    batch: Any, environment: int, agent: int, step: int = 0, final: bool = False
) -> _Struct:
    """Returns a Kaggle agent's observation, a dictionary whose keys are also attributes.

    It holds `players_raw`, `controlled_players`, and Kaggle's `step` and `remainingOverageTime`.

    Args:
        batch: A batch with the raw representation or `with_raw=True`.
        environment: The index of the environment.
        agent: The index of the agent.
        step: The steps of the episode so far, which Kaggle counts from 0.
        final: Whether to read the step that ended the last episode.
    """
    return _Struct(
        players_raw=players_raw(batch, environment, agent, final),
        controlled_players=1,
        step=step,
        remainingOverageTime=_OVERAGE_TIME,
    )


def _accepts_configuration(function):
    """Returns whether an agent takes a configuration, which Kaggle decides by parameter count."""
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2 or any(
        parameter.kind == parameter.VAR_POSITIONAL for parameter in parameters
    )


def _action(value, count):
    """Returns the action index of a Kaggle agent's answer, `[action]`, an integer or an enum."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f'expected one action for the one controlled player, found {value!r}')
        value = value[0]
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    # Kaggle: the `Action` enum of the football helpers holds the index as its value.
    if not isinstance(value, (int, np.integer)) and isinstance(getattr(value, 'value', None), int):
        value = value.value
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f'expected an integer action from the agent, found {value!r}')
    # GRF: an action indexes a Python list, so it may count from the end.
    if not -count <= int(value) < count:
        raise IndexError(f'expected an action from {-count} to {count - 1}, found {value}')
    return int(value)


class KaggleAgent:
    """Runs a Kaggle agent function as one agent of every environment of a batch.

    Each call of `act` builds the Kaggle observation of the agent in every environment from the
    batch's current raw observation, calls the function once per environment, and returns the
    actions. It counts Kaggle's `step` per environment from the first observation of each episode.

    Attributes:
        function: The agent function.
        batch: The batch.
        agent: The agent that the function plays.
        configuration: What a function that takes two arguments receives second.
    """

    def __init__(
        self, function: Any, batch: Any, agent: int, configuration: dict[str, Any] | None = None
    ) -> None:
        """Binds a Kaggle agent function to an agent of a batch.

        Args:
            function: `agent(obs)` or `agent(obs, configuration)`, which reads
                `obs['players_raw'][0]` and returns `[action]`, an index of the action set.
            batch: A batch with the raw representation or `with_raw=True`.
            agent: The index of the agent, which must exist in every environment.
            configuration: The configuration of Kaggle's environment that the function reads, if
                any, as a dictionary, and `None` passes an empty one.

        Raises:
            ValueError: If the batch has no raw arrays or the agent is missing somewhere.
        """
        if getattr(batch, '_raw_fields', None) is None:
            raise ValueError('expected a batch with the raw representation or with_raw=True')
        if not 0 <= agent < batch.agents or not bool(batch.agent_mask[:, agent].all()):
            raise ValueError(f'expected an agent of every environment, found {agent}')
        self.function = function
        self.batch = batch
        self.agent = agent
        self.configuration = _Struct(configuration or {})
        self._two_arguments = _accepts_configuration(function)
        self._steps = [0] * batch.environment_count

    def reset(self, mask: Any = None) -> None:
        """Counts Kaggle's `step` from 0 again, in the environments of `mask` or in all."""
        for environment in range(self.batch.environment_count):
            if mask is None or bool(mask[environment]):
                self._steps[environment] = 0

    def act(self, done: Any = None, skip: Any = None) -> torch.Tensor:
        """Returns the agent's actions, `[environment_count]` `int32` indices on the batch's device.

        Args:
            done: The done flags of the step before, where an automatic reset started a new
                episode, whose `step` counts from 0 again, or `None` if no episode ended.
            skip: Flags of the environments where the function is not called and the action is
                `idle`, such as those whose match is over.

        Raises:
            TypeError, ValueError, IndexError: If the function's answer is not one action of the
                batch's action set.
        """
        if done is not None:
            self.reset(torch.as_tensor(done).bool().cpu())
        skip = None if skip is None else torch.as_tensor(skip).bool().cpu()
        # One copy of the raw arrays to the host serves every environment.
        raw = {name: tensor.cpu() for name, tensor in self.batch.raw().items()}
        actions = []
        for environment in range(self.batch.environment_count):
            if skip is not None and bool(skip[environment]):
                actions.append(0)
                continue
            entry = grf_observation(
                raw, environment, self.agent, *team_sizes(self.batch, environment, self.agent)
            )
            obs = _Struct(
                players_raw=[_lists(entry)],
                controlled_players=1,
                step=self._steps[environment],
                remainingOverageTime=_OVERAGE_TIME,
            )
            answer = (
                self.function(obs, self.configuration)
                if self._two_arguments
                else self.function(obs)
            )
            actions.append(_action(answer, self.batch.action_count))
            self._steps[environment] += 1
        return torch.tensor(actions, dtype=torch.int32, device=self.batch._torch_device)


def evaluate(
    left: Any,
    right: Any = None,
    episodes: int = 1,
    scenario: str = '11_vs_11_kaggle',
    seed: int = 0,
    device: str = 'cpu',
    threads: int = 0,
    configuration: dict[str, Any] | None = None,
) -> list[tuple[int, int]]:
    """Plays matches of Kaggle agents, one per environment, and returns their final scores.

    Each environment plays its first episode only, as Kaggle plays every match in a newly created
    GRF environment, so with the even form of the scenario and its own team order.

    Args:
        left: The Kaggle agent function of the left team.
        right: The function of the right team, or `None` for GRF's built-in AI.
        episodes: The number of matches, one per environment.
        scenario: A shipped scenario or a GRF scenario file.
        seed: The seed of the batch, and match `k` is environment `k` of the seed rule.

        device: `cpu` or `cuda`.
        threads: The CPU threads.
        configuration: What a function that takes two arguments receives second.

    Returns:
        The goals of the left and of the right team in each match.
    """
    batch = BatchEnvironment(
        scenario,
        environment_count=episodes,
        representation='raw',
        seed=seed,
        auto_reset=False,
        device=device,
        threads=threads,
        agent_counts=[(1, 0 if right is None else 1)],
    )
    try:
        players = [KaggleAgent(left, batch, 0, configuration)]
        if right is not None:
            players.append(KaggleAgent(right, batch, 1, configuration))
        batch.reset()
        finished = torch.zeros(episodes, dtype=torch.bool)
        while not bool(finished.all()):
            actions = torch.stack([player.act(skip=finished) for player in players], 1)
            _, _, done, _ = batch.step(actions)
            finished |= done.bool().cpu()
        # The left agent sees its own goals first.
        scores = batch.raw()['score'][:, 0].cpu().tolist()
    finally:
        batch.close()
    return [(int(own), int(other)) for own, other in scores]

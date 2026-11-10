# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Kaggle's GRF competition protocol, with `players_raw` from a batch and Kaggle agents in a batch.

`KaggleAgent` runs a Kaggle agent function, such as a `submission.py` agent or a public rule-based
bot, as one agent of every environment of a batch, and `evaluate` plays whole matches with it.
`AgentRunner` calls such a function as Kaggle does and checks its answer, for a program that builds
the observations itself. The guide is in the repository's `docs/recipes.md`, under Train for the
Kaggle competition.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import torch

from . import BatchEnvironment
from .raw import grf_observation, team_sizes

__all__ = [
    'AgentRunner',
    'KaggleAgent',
    'accepts_configuration',
    'action_index',
    'action_indices',
    'evaluate',
    'observation',
    'players_raw',
]

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
    batch: Any,
    environment: int,
    agent: int | list[int],
    step: int = 0,
    final: bool = False,
    raw: dict[str, Any] | None = None,
) -> _Struct:
    """Returns a Kaggle agent's observation, a dictionary whose keys are also attributes.

    It holds `players_raw`, `controlled_players`, and Kaggle's `step` and `remainingOverageTime`.
    A list of agents gives Kaggle's multi-agent form, one entry of `players_raw` per agent in the
    order of the list, and `controlled_players` their count.

    Args:
        batch: A batch with the raw representation or `with_raw=True`.
        environment: The index of the environment.
        agent: The index of the agent, or a list of the agents that one function plays.
        step: The steps of the episode so far, which Kaggle counts from 0.
        final: Whether to read the step that ended the last episode.
        raw: The raw arrays to read in place of the batch's, such as one copy of `batch.raw()` on
            the host that the observations of several agents share. `final` then has no effect.
    """
    several = isinstance(agent, (list, tuple))
    agents = [int(index) for index in agent] if several else [int(agent)]
    if raw is None:
        raw = batch.final_raw() if final else batch.raw()
    return _Struct(
        players_raw=[
            _lists(grf_observation(raw, environment, index, *team_sizes(batch, environment, index)))
            for index in agents
        ],
        controlled_players=len(agents),
        step=step,
        remainingOverageTime=_OVERAGE_TIME,
    )


def accepts_configuration(function: Any) -> bool:
    """Returns whether an agent function takes a configuration, as Kaggle decides it.

    Kaggle passes the configuration as a second argument to a function with at least two
    positional parameters, or with `*args`, and calls any other function with the observation
    alone.
    """
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


def action_index(value: Any, count: int) -> int:
    """Returns the action index of a Kaggle agent's answer for one controlled player.

    Args:
        value: The answer, `[action]` as Kaggle's agents return it, an integer, a one-element
            array or an enum whose value is the index, such as `Action` of Kaggle's helpers.
        count: The actions of the action set, and an index from `-count` to `count - 1` counts
            from the end for a negative one, as in GRF.

    Raises:
        ValueError: If a list holds other than one action.
        TypeError: If the action is not an integer.
        IndexError: If the action lies outside the action set.
    """
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f'expected one action for the one controlled player, found {value!r}')
        value = value[0]
    return _index(value, count)


def action_indices(value: Any, count: int, players: int) -> list[int]:
    """Returns the action indices of an answer for `players` controlled players, one each.

    Args:
        value: The answer, a list, tuple or array with one action per controlled player, each in
            a form that `action_index` takes.
        count: The actions of the action set.
        players: The controlled players.

    Raises:
        ValueError: If the answer holds other than one action per player.
        TypeError, IndexError: As `action_index` raises them for an action.
    """
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if not isinstance(value, (list, tuple)) or len(value) != players:
        raise ValueError(
            f'expected one action for each of the {players} controlled players, found {value!r}'
        )
    return [_index(item, count) for item in value]


def _index(value, count):
    """Returns one action index, from an integer, a one-element array or an enum."""
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


# The names before 0.2.1, which the renderer of cuju 0.2.0 imports.
_accepts_configuration = accepts_configuration
_action = action_index
_actions = action_indices


class AgentRunner:
    """Calls a Kaggle agent function as Kaggle does, and checks its answer against an action set.

    The runner suits a program that builds the observations itself, such as one that plays a
    match step by step, and `KaggleAgent` covers the agents of a batch.

    Attributes:
        function: The agent function.
        configuration: What a function that takes a configuration receives second.
        accepts_configuration: Whether the function takes a configuration.
    """

    def __init__(self, function: Any, configuration: dict[str, Any] | None = None) -> None:
        """Binds a Kaggle agent function.

        Args:
            function: `agent(obs)` or `agent(obs, configuration)`, as `accepts_configuration`
                tells them apart.
            configuration: The configuration of Kaggle's environment that the function reads, as
                a dictionary, and `None` passes an empty one.
        """
        self.function = function
        self.configuration = _Struct(configuration or {})
        self.accepts_configuration = accepts_configuration(function)

    def __call__(self, obs: Any) -> Any:
        """Calls the function with an observation, and the configuration if it takes one, and
        returns its answer as it is."""
        if self.accepts_configuration:
            return self.function(obs, self.configuration)
        return self.function(obs)

    def act(self, obs: Any, count: int) -> list[int]:
        """Calls the function and returns one action index per controlled player.

        One controlled player may answer in Kaggle's single form, as `action_index` takes it, and
        several answer with one action each, in the order of `obs['players_raw']`.

        Args:
            obs: The observation, as `observation` returns it.
            count: The actions of the action set.

        Raises:
            TypeError, ValueError, IndexError: As `action_index` and `action_indices` raise them,
                and whatever the function raises.
        """
        players = len(obs['players_raw'])
        answer = self(obs)
        if players == 1:
            return [action_index(answer, count)]
        return action_indices(answer, count, players)


class KaggleAgent:
    """Runs a Kaggle agent function as one agent, or several, of every environment of a batch.

    Each call of `act` builds the Kaggle observation of the agent in every environment from the
    batch's current raw observation, calls the function once per environment, and returns the
    actions. It counts Kaggle's `step` per environment from the first observation of each episode.

    A function that controls several players receives Kaggle's multi-agent form, one raw
    observation per controlled player in `obs['players_raw']`, in the order of `agents`, with
    `controlled_players` their count, and returns one action per player in the same order.

    Attributes:
        function: The agent function.
        batch: The batch.
        agent: The first agent that the function plays.
        agents: Every agent that the function plays, `[agent]` for one.
        configuration: What a function that takes two arguments receives second.
    """

    def __init__(
        self,
        function: Any,
        batch: Any,
        agent: int,
        configuration: dict[str, Any] | None = None,
        agents: list[int] | None = None,
    ) -> None:
        """Binds a Kaggle agent function to an agent of a batch.

        Args:
            function: `agent(obs)` or `agent(obs, configuration)`, which reads
                `obs['players_raw'][0]` and returns `[action]`, an index of the action set.
            batch: A batch with the raw representation or `with_raw=True`.
            agent: The index of the agent, which must exist in every environment.
            configuration: The configuration of Kaggle's environment that the function reads, if
                any, as a dictionary, and `None` passes an empty one.
            agents: The agents of a function that controls several players, in the order of its
                observations and actions, or `None` for `agent` alone. With a list, `act` returns
                one column per agent.

        Raises:
            ValueError: If the batch has no raw arrays or an agent is missing somewhere.
        """
        if getattr(batch, '_raw_fields', None) is None:
            raise ValueError('expected a batch with the raw representation or with_raw=True')
        self.agents = [agent] if agents is None else [int(index) for index in agents]
        if not self.agents or len(set(self.agents)) != len(self.agents):
            raise ValueError(f'expected distinct agents, found {agents!r}')
        for index in self.agents:
            if not 0 <= index < batch.agents or not bool(batch.agent_mask[:, index].all()):
                raise ValueError(f'expected an agent of every environment, found {index}')
        self._several = agents is not None
        self.function = function
        self.batch = batch
        self.agent = self.agents[0]
        self.configuration = _Struct(configuration or {})
        self._two_arguments = accepts_configuration(function)
        self._steps = [0] * batch.environment_count

    def reset(self, mask: Any = None) -> None:
        """Counts Kaggle's `step` from 0 again, in the environments of `mask` or in all."""
        for environment in range(self.batch.environment_count):
            if mask is None or bool(mask[environment]):
                self._steps[environment] = 0

    def act(self, done: Any = None, skip: Any = None) -> torch.Tensor:
        """Returns the agent's actions, `[environment_count]` `int32` indices on the batch's device.

        A function given a list of `agents` returns `[environment_count, len(agents)]`.

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
        players = len(self.agents)
        for environment in range(self.batch.environment_count):
            if skip is not None and bool(skip[environment]):
                actions.append([0] * players if self._several else 0)
                continue
            entries = [
                _lists(
                    grf_observation(
                        raw, environment, agent, *team_sizes(self.batch, environment, agent)
                    )
                )
                for agent in self.agents
            ]
            obs = _Struct(
                players_raw=entries,
                controlled_players=players,
                step=self._steps[environment],
                remainingOverageTime=_OVERAGE_TIME,
            )
            answer = (
                self.function(obs, self.configuration)
                if self._two_arguments
                else self.function(obs)
            )
            if self._several:
                actions.append(action_indices(answer, self.batch.action_count, players))
            else:
                actions.append(action_index(answer, self.batch.action_count))
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

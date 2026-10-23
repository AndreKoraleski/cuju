# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""GRF's `gfootball.env.wrappers`, each wrapper over an environment with GRF's gym interface.

The wrappers run GRF's code over the environment that `create_environment` or `FootballEnv`
returns, so a stack built by hand gives GRF's outputs for the same arguments. `create_environment`
applies GRF's wrappers itself, from its arguments. `get_state` and `set_state` carry the counts of
`CheckpointRewardWrapper` and the frames of `FrameStack` in the state, as GRF's do, so a stack built
by hand and `create_environment` with the same arguments write equal bytes and load each other's
states. `PixelsStateWrapper` reads rendered frames, which cuju lacks, so it raises on construction.
`MultiAgentToSingleAgent` over a `BatchEnvironment` is `cuju.wrappers.MultiAgentToSingleAgent`, the
batched port.
"""

from __future__ import annotations

import collections
from typing import Any

import numpy as np

from ... import BatchEnvironment
from ... import wrappers as _batched
from ...env import FootballEnvironment, _gym
from . import football_action_set, observation_preprocessing

__all__ = [
    'CheckpointRewardWrapper',
    'FrameStack',
    'GetStateWrapper',
    'MultiAgentToSingleAgent',
    'PeriodicDumpWriter',
    'PixelsStateWrapper',
    'SMMWrapper',
    'Simple115StateWrapper',
    'SingleAgentObservationWrapper',
    'SingleAgentRewardWrapper',
]

# GRF: the action set `v2` adds `builtin_ai` to the 19 actions of `default`.
_BUILTIN_AI_ACTIONS = len(football_action_set.action_set_v2)


class _Wrapper:
    """GRF's `gym.Wrapper` over an environment with GRF's gym interface."""

    def __init__(self, env: Any) -> None:
        self.env = env

    def __getattr__(self, name: str) -> Any:
        # Only the names that the wrapper lacks reach here, as in gym's wrappers.
        if name == 'env':
            raise AttributeError(name)
        return getattr(self.env, name)

    def __str__(self) -> str:
        return f'<{type(self).__name__}{self.env}>'

    __repr__ = __str__

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def reset(self) -> Any:
        return self.env.reset()

    def step(self, action: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        return self.env.step(action)

    def seed(self, seed: int | None = None) -> Any:
        return self.env.seed(seed)

    def close(self) -> None:
        self.env.close()


class _ObservationWrapper(_Wrapper):
    """GRF's `gym.ObservationWrapper`, which converts each observation."""

    def reset(self) -> Any:
        return self.observation(self.env.reset())

    def step(self, action: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation: Any) -> Any:
        raise NotImplementedError


class _RewardWrapper(_Wrapper):
    """GRF's `gym.RewardWrapper`, which converts each reward."""

    def step(self, action: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward), done, info

    def reward(self, reward: Any) -> Any:
        raise NotImplementedError


def _agents(env):
    """Returns the agent axis of GRF's representation wrappers, from the action space."""
    action_shape = np.shape(env.action_space)
    return action_shape[0] if len(action_shape) else 1


class GetStateWrapper(_Wrapper):
    """GRF's wrapper that passes `get_state` and `set_state` through the wrappers that keep states.

    It refuses the states of any other wrapper between it and the environment, as GRF's does.
    """

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self._wrappers_with_support = {
            'CheckpointRewardWrapper',
            'FrameStack',
            'GetStateWrapper',
            'SingleAgentRewardWrapper',
            'SingleAgentObservationWrapper',
            'SMMWrapper',
            'PeriodicDumpWriter',
            'Simple115StateWrapper',
            'PixelsStateWrapper',
        }

    def _check_state_supported(self) -> None:
        """Raises `ValueError` for a wrapper without states, where GRF fails an assertion."""
        wrapper = self
        while not isinstance(wrapper, FootballEnvironment):
            name = type(wrapper).__name__
            if name not in self._wrappers_with_support:
                raise ValueError(f'get/set state not supported by {name} wrapper')
            wrapper = wrapper.env

    def get_state(self) -> bytes:
        self._check_state_supported()
        return self.env.get_state({})

    def set_state(self, state: bytes) -> None:
        self._check_state_supported()
        self.env.set_state(state)


class PeriodicDumpWriter(_Wrapper):
    """GRF's wrapper that keeps the dump options of the environment for some episodes only.

    Each `reset` sets the dump options of the configuration, which the environment reads for the
    episode that the reset starts. The first reset and then one in every `dump_frequency` keep the
    options, and the others turn the dumps off.
    """

    def __init__(self, env: Any, dump_frequency: int, render: bool = False) -> None:
        """Wraps an environment.

        Args:
            env: An environment of `create_environment` or `FootballEnv`, or a wrapper of one.
            dump_frequency: The resets in which one keeps the dump options, the first included,
                or 0 to turn the dumps off for every episode.
            render: Must stay `False`, since cuju renders nothing.

        Raises:
            ValueError: If `render` is set.
        """
        if render:
            raise ValueError('cuju renders nothing, so PeriodicDumpWriter supports no render')
        super().__init__(env)
        self._dump_frequency = dump_frequency
        self._render = render
        self._original_dump_config = {
            'write_video': env._config['write_video'],
            'dump_full_episodes': env._config['dump_full_episodes'],
            'dump_scores': env._config['dump_scores'],
        }
        self._current_episode_number = 0

    def reset(self) -> Any:
        if self._dump_frequency > 0 and (self._current_episode_number % self._dump_frequency == 0):
            self.env._config.update(self._original_dump_config)
        else:
            self.env._config.update(
                {'write_video': False, 'dump_full_episodes': False, 'dump_scores': False}
            )
        self._current_episode_number += 1
        return self.env.reset()


class Simple115StateWrapper(_ObservationWrapper):
    """GRF's wrapper that turns raw observations into 115 features per player."""

    def __init__(self, env: Any, fixed_positions: bool = False) -> None:
        """Wraps an environment.

        Args:
            env: An environment with raw observations.
            fixed_positions: Whether each team keeps its 44 values whatever the team sizes, GRF's
                simple115v2.
        """
        super().__init__(env)
        self.observation_space = _gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(_agents(self.env), 115), dtype=np.float32
        )
        self._fixed_positions = fixed_positions

    def observation(self, observation: list[dict[str, Any]]) -> np.ndarray:
        return Simple115StateWrapper.convert_observation(observation, self._fixed_positions)

    @staticmethod
    def convert_observation(observation: list[dict[str, Any]], fixed_positions: bool) -> np.ndarray:
        """Returns the simple115 or simple115v2 features, `[len(observation), 115]` of `float32`.

        The features are the positions and the directions of the left team, then of the right
        team, padded with -1 to 88 values, or each group to 22 with `fixed_positions`. The ball's
        position and direction, the owner of the ball one-hot, the active player one-hot and the
        game mode one-hot follow.
        """

        def do_flatten(obj):
            # GRF: a Python list or a numpy array.
            if type(obj) is list:
                return np.array(obj).flatten()
            return obj.flatten()

        final_obs = []
        for obs in observation:
            o = []
            if fixed_positions:
                for i, name in enumerate(
                    ['left_team', 'left_team_direction', 'right_team', 'right_team_direction']
                ):
                    o.extend(do_flatten(obs[name]))
                    # GRF: a team of fewer than 11 players is padded with -1.
                    if len(o) < (i + 1) * 22:
                        o.extend([-1] * ((i + 1) * 22 - len(o)))
            else:
                o.extend(do_flatten(obs['left_team']))
                o.extend(do_flatten(obs['left_team_direction']))
                o.extend(do_flatten(obs['right_team']))
                o.extend(do_flatten(obs['right_team_direction']))

            # GRF: 88 values are 11 players, 2 teams, positions and directions, x and y.
            if len(o) < 88:
                o.extend([-1] * (88 - len(o)))

            o.extend(obs['ball'])
            o.extend(obs['ball_direction'])
            if obs['ball_owned_team'] == -1:
                o.extend([1, 0, 0])
            if obs['ball_owned_team'] == 0:
                o.extend([0, 1, 0])
            if obs['ball_owned_team'] == 1:
                o.extend([0, 0, 1])

            active = [0] * 11
            if obs['active'] != -1:
                active[obs['active']] = 1
            o.extend(active)

            game_mode = [0] * 7
            game_mode[obs['game_mode']] = 1
            o.extend(game_mode)
            final_obs.append(o)
        return np.array(final_obs, dtype=np.float32)


class PixelsStateWrapper(_ObservationWrapper):
    """GRF's wrapper that scales rendered frames, which cuju lacks, so it raises on construction."""

    def __init__(
        self,
        env: Any,
        grayscale: bool = True,
        channel_dimensions: tuple[int, int] = (
            observation_preprocessing.SMM_WIDTH,
            observation_preprocessing.SMM_HEIGHT,
        ),
    ) -> None:
        """Raises, since the wrapper reads the frames of GRF's renderer.

        Raises:
            NotImplementedError: Always, since cuju renders nothing. The SMM of the `extracted`
                representation is the observation that cuju computes in its place.
        """
        raise NotImplementedError(
            'cuju renders nothing, so PixelsStateWrapper has no frames to read; use the SMM, '
            "representation='extracted' or SMMWrapper, in its place"
        )


class SMMWrapper(_ObservationWrapper):
    """GRF's wrapper that turns raw observations into the SMM."""

    def __init__(
        self,
        env: Any,
        channel_dimensions: tuple[int, int] = (
            observation_preprocessing.SMM_WIDTH,
            observation_preprocessing.SMM_HEIGHT,
        ),
    ) -> None:
        """Wraps an environment with raw observations, with the SMM's width and height."""
        super().__init__(env)
        self._channel_dimensions = channel_dimensions
        layers = observation_preprocessing.get_smm_layers(self.env.unwrapped._config)
        shape = (_agents(self.env), channel_dimensions[1], channel_dimensions[0], len(layers))
        self.observation_space = _gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

    def observation(self, observation: list[dict[str, Any]]) -> np.ndarray:
        return observation_preprocessing.generate_smm(
            observation,
            channel_dimensions=self._channel_dimensions,
            config=self.env.unwrapped._config,
        )


class SingleAgentObservationWrapper(_ObservationWrapper):
    """GRF's wrapper that keeps the observation of the first agent."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self.observation_space = _gym.spaces.Box(
            low=env.observation_space.low[0],
            high=env.observation_space.high[0],
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation: Any) -> Any:
        return observation[0]


class SingleAgentRewardWrapper(_RewardWrapper):
    """GRF's wrapper that keeps the reward of the first agent."""

    def reward(self, reward: Any) -> Any:
        return reward[0]


class CheckpointRewardWrapper(_RewardWrapper):
    """GRF's wrapper that adds the checkpoint reward, 0.1 for each of 10 distances to the goal.

    It reads the raw observations of the unwrapped environment after each step and adds to the
    `float32` reward of each agent in place, with the arithmetic of the installed numpy, as GRF's
    does. Its counts travel in the state under its name, as those of
    `rewards='scoring,checkpoints'` do.
    """

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self._collected_checkpoints = {}
        self._num_checkpoints = 10
        self._checkpoint_reward = 0.1

    def reset(self) -> Any:
        self._collected_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle: dict[str, Any]) -> bytes:
        """Returns the state of the environment with the counts in `to_pickle`, as GRF's does."""
        to_pickle['CheckpointRewardWrapper'] = self._collected_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state: bytes) -> dict[str, Any]:
        """Loads a state and its counts, and returns its pickled entries, as GRF's does."""
        from_pickle = self.env.set_state(state)
        self._collected_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward: np.ndarray) -> np.ndarray:
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if reward[rew_index] == 1:
                reward[rew_index] += self._checkpoint_reward * (
                    self._num_checkpoints - self._collected_checkpoints.get(rew_index, 0)
                )
                self._collected_checkpoints[rew_index] = self._num_checkpoints
                continue

            # GRF: only an active player who owns the ball collects.
            if (
                'ball_owned_team' not in o
                or o['ball_owned_team'] != 0
                or 'ball_owned_player' not in o
                or o['ball_owned_player'] != o['active']
            ):
                continue

            d = ((o['ball'][0] - 1) ** 2 + o['ball'][1] ** 2) ** 0.5

            # GRF: the checkpoints lie at distances from 0.99 down to 0.19.
            while self._collected_checkpoints.get(rew_index, 0) < self._num_checkpoints:
                if self._num_checkpoints == 1:
                    threshold = 0.99 - 0.8
                else:
                    threshold = 0.99 - 0.8 / (
                        self._num_checkpoints - 1
                    ) * self._collected_checkpoints.get(rew_index, 0)
                if d > threshold:
                    break
                reward[rew_index] += self._checkpoint_reward
                self._collected_checkpoints[rew_index] = (
                    self._collected_checkpoints.get(rew_index, 0) + 1
                )
        return reward


class FrameStack(_Wrapper):
    """GRF's wrapper that stacks the last `k` observations along the last axis.

    Its frames travel in the state under its name, as those of `stacked=True` do.
    """

    def __init__(self, env: Any, k: int) -> None:
        super().__init__(env)
        self.obs = collections.deque(maxlen=k)
        low = env.observation_space.low
        high = env.observation_space.high
        low = np.concatenate([low] * k, axis=-1)
        high = np.concatenate([high] * k, axis=-1)
        self.observation_space = _gym.spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )

    def reset(self) -> np.ndarray:
        observation = self.env.reset()
        self.obs.extend([observation] * self.obs.maxlen)
        return self._get_observation()

    def get_state(self, to_pickle: dict[str, Any]) -> bytes:
        """Returns the state of the environment with the frames in `to_pickle`, as GRF's does."""
        to_pickle['FrameStack'] = self.obs
        return self.env.get_state(to_pickle)

    def set_state(self, state: bytes) -> dict[str, Any]:
        """Loads a state and its frames, and returns its pickled entries, as GRF's does."""
        from_pickle = self.env.set_state(state)
        self.obs = from_pickle['FrameStack']
        return from_pickle

    def step(self, action: Any) -> tuple[np.ndarray, Any, Any, dict[str, Any]]:
        observation, reward, done, info = self.env.step(action)
        self.obs.append(observation)
        return self._get_observation(), reward, done, info

    def _get_observation(self) -> np.ndarray:
        return np.concatenate(list(self.obs), axis=-1)


class MultiAgentToSingleAgent(_Wrapper):
    """GRF's wrapper that gives the policy the designated player of each side it plays.

    The environment's agent controls every controllable player of a scenario that sets
    `control_all_players`, and each step moves the designated player with the policy's action and
    every other with `builtin_ai`. `create_environment` applies it itself for one player a side.
    Over a `BatchEnvironment` the class gives `cuju.wrappers.MultiAgentToSingleAgent`, the batched
    port, with that class's arguments.
    """

    def __new__(cls, env: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(env, BatchEnvironment):
            return _batched.MultiAgentToSingleAgent(env, *args, **kwargs)
        return super().__new__(cls)

    def __init__(self, env: Any, left_players: int, right_players: int) -> None:
        """Wraps an environment with raw observations.

        Args:
            env: An environment whose agent controls the players of a scenario with
                `control_all_players`, with the action set `v2` or `full`, which hold
                `builtin_ai`.
            left_players: 1 if the policy plays the left side, else 0.
            right_players: 1 if the policy plays the right side, else 0.

        Raises:
            ValueError: If a side has more than one player, or the action set of a cuju
                environment lacks `builtin_ai`, which GRF's `default` set lacks.
        """
        if left_players >= 2 or right_players >= 2:
            raise ValueError(
                f'expected 0 or 1 left and right players, found {left_players} and {right_players}'
            )
        unwrapped = env.unwrapped
        # GRF: the engine plays any core action, while a batch plays only its action set.
        if isinstance(unwrapped, FootballEnvironment) and unwrapped._num_actions < (
            _BUILTIN_AI_ACTIONS
        ):
            raise ValueError(
                'expected an action set that holds builtin_ai, which the wrapper gives the players '
                "that the policy does not move; pass other_config_options={'action_set': 'v2'}, "
                "which plays GRF's default set with the same indices"
            )
        players = left_players + right_players
        super().__init__(env)
        self._observation = None
        if players > 1:
            self.action_space = _gym.spaces.MultiDiscrete([env._num_actions] * players)
        else:
            self.action_space = _gym.spaces.Discrete(env._num_actions)

    def reset(self) -> list[dict[str, Any]]:
        self._observation = self.env.reset()
        return self._get_observation()

    def step(self, action: Any) -> tuple[list[dict[str, Any]], Any, Any, dict[str, Any]]:
        assert self._observation, 'Reset must be called before step'
        action = MultiAgentToSingleAgent.get_action(action, self._observation)
        self._observation, reward, done, info = self.env.step(action)
        return self._get_observation(), reward, done, info

    def _get_observation(self) -> list[dict[str, Any]]:
        return MultiAgentToSingleAgent.get_observation(self._observation)

    get_observation = staticmethod(_batched.MultiAgentToSingleAgent.get_observation)
    get_action = staticmethod(_batched.MultiAgentToSingleAgent.get_action)

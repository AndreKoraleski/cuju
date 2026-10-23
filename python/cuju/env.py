# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""GRF's `create_environment` and gymnasium ids, each environment a batch of one.

`create_environment` takes GRF's arguments with GRF's defaults and returns GRF's gym interface,
or gymnasium's with `api='gymnasium'`. It draws the seed of every episode from Python's `random` as
GRF does, so after the same `random.seed(...)` it plays GRF's matches. Importing `cuju` registers
the ids `GFootball-<scenario>-<representation>-v0`, which `gymnasium.make` and `gym.make` take.
"""

from __future__ import annotations

import collections
import importlib
import importlib.metadata
import importlib.util
import io
import os
import pickle
import platform
import random
import re
import struct
import sys
import warnings
from typing import Any

import numpy as np

try:
    import gymnasium as _gym
except ImportError:
    try:
        import gym as _gym
    except ImportError:
        raise ImportError(
            "cuju.env needs gymnasium, from pip install 'cuju[spaces]', or GRF's gym"
        ) from None

from . import BatchEnvironment, _replays, bundled_scenarios
from ._batch import frame_stack as _frame_stack
from ._batch.state import _pickle_episode, _StateUnpickler
from .errors import StateError
from .players import Bot, Lazy
from .raw import grf_observation, grf_observations
from .wrappers import MultiAgentToSingleAgent

__all__ = ['FootballEnvironment', 'create_environment', 'grf_observation', 'register_environments']

# The interfaces of `create_environment`, GRF's gym one first.
_APIS = ('gym', 'gymnasium')

# GRF: `observation_preprocessing.SMM_WIDTH` and `SMM_HEIGHT`.
_SMM_SIZE = (96, 72)
# GRF: every scenario that `scenario_builder` builds draws `random.randint(0, 2000000000)`.
_SEED_LIMIT = 2000000000
# GRF: the core actions in the order of the `full` set. The `default` and `v2` sets take the
# first 19 and 20.
_ACTION_NAMES = (
    'idle',
    'left',
    'top_left',
    'top',
    'top_right',
    'right',
    'bottom_right',
    'bottom',
    'bottom_left',
    'long_pass',
    'high_pass',
    'short_pass',
    'shot',
    'sprint',
    'release_direction',
    'release_sprint',
    'sliding',
    'dribble',
    'release_dribble',
    'builtin_ai',
    'keeper_rush',
    'pressure',
    'team_pressure',
    'switch',
    'release_long_pass',
    'release_high_pass',
    'release_short_pass',
    'release_shot',
    'release_keeper_rush',
    'release_sliding',
    'release_pressure',
    'release_team_pressure',
    'release_switch',
)
_ACTION_COUNTS = {'default': 19, 'v2': 20, 'full': 33}
# GRF: options of `Config` that only rendering reads, so without it they change nothing.
_DISPLAY_OPTIONS = frozenset(
    {
        'custom_display_stats',
        'display_game_stats',
        'real_time',
        'render_resolution_x',
        'render_resolution_y',
        'video_format',
        'video_quality_level',
    }
)
_OPTIONS = _DISPLAY_OPTIONS | {
    'action_set',
    'dump_full_episodes',
    'dump_scores',
    'episode_number',
    'game_engine_random_seed',
    'level',
    'physics_steps_per_frame',
    'players',
    'reverse_team_processing',
    'tracesdir',
    'write_video',
}
# The players of GRF's `gfootball/env/players` that a batch can play.
_PLAYERS = ('agent', 'bot', 'lazy')
# GRF: any other player is a module of `gfootball.env.players`, here one so named or a module path.
_PLAYER_MODULE = re.compile(r'\w+(\.\w+)*')


def _numpy(tensor):
    """Returns a host copy of a tensor as a numpy array."""
    return tensor.detach().cpu().numpy().copy()


def _parse_player(definition):
    """Returns the name, left players, right players and parameters of a player definition.

    GRF: `config.parse_player_definition`, which gives a player without counts one left player.
    """
    name, parameters = definition, {}
    if ':' in definition:
        # GRF: on Windows the path of a replay may hold a colon.
        if (
            platform.system() == 'Windows'
            and definition.startswith('replay:')
            and len(definition.split(':')) > 2
        ):
            name, text = 'replay', definition.split('replay:')[-1]
        else:
            name, text = definition.split(':')
        for parameter in text.split(','):
            key, value = parameter.split('=')
            parameters[key] = value
    counted = 'left_players' in parameters or 'right_players' in parameters
    left = int(parameters.pop('left_players', 0))
    right = int(parameters.pop('right_players', 0))
    return name, left if counted else 1, right, parameters


def _scenario_facts(asset):
    """Returns whether a scenario is deterministic and the controllable players of each side.

    The first is `None` where only the bundle knows, and the second is `None` unless the scenario
    sets `control_all_players`.
    """
    if isinstance(asset, str) and asset.endswith('.py') and os.path.isfile(asset):
        from . import scenarios

        # GRF: `create_environment` reads the scenario of episode 1, the odd form.
        odd = scenarios.describe(asset)[1]
        control = None
        if odd['values']['control_all_players']:
            control = tuple(
                sum(1 for player in odd[side] if player[4]) for side in ('left', 'right')
            )
        return bool(odd['values']['deterministic']), control
    if isinstance(asset, str) and not asset.endswith('.cjb') and os.sep not in asset:
        # GRF: no shipped scenario is deterministic or sets `control_all_players`.
        return False, None
    return None, None


def _numpy_values(value):
    """Returns a value of GRF's observation of the match with each array stand-in as numpy's."""
    from . import dumps

    if isinstance(value, dumps._Array):
        numpy_type, _ = value.kind
        return np.array(value.values, dtype=np.dtype(numpy_type.order + numpy_type.code)).reshape(
            value.shape
        )
    if isinstance(value, dict):
        return {key: _numpy_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_numpy_values(item) for item in value]
    return value


def _space_shape(base, agents, single, stacked):
    """Returns the shape of GRF's observation space for `agents` agents of the player."""
    # GRF: the representation wrappers size the agent axis from the action space, which is a
    # `Discrete` that counts as one agent for fewer than two agents.
    shape = (agents if agents > 1 else 1, *base)
    if single:
        shape = shape[1:]
    if stacked:
        shape = (*shape[:-1], shape[-1] * 4)
    return shape


class _Dumps:
    """GRF's dump options, which `cuju.dumps.DumpWriter` writes after each step, outside `step`.

    Each reset takes the dump options of the configuration for the episode that it starts. With a
    `frequency` above 1, only the first reset and then one in every `frequency` keep them, as GRF's
    `PeriodicDumpWriter` counts the resets of the environment.
    """

    def __init__(self, batch, full_episodes, goals, frequency, directory):
        try:
            from . import dumps
        except ImportError:
            raise NotImplementedError(
                "GRF's dump options write through cuju.dumps, which this installation lacks"
            ) from None
        self._frequency = frequency
        self._resets = 0
        # GRF: an empty `logdir` writes the dumps in the working directory.
        self._writer = dumps.DumpWriter(
            batch,
            directory or os.curdir,
            environments=[0],
            write_full_episode_dumps=bool(full_episodes),
            write_goal_dumps=bool(goals),
        )
        # GRF: one environment names its dumps by their kind and the time they start.
        self._writer._grf_names = True

    def after_step(self, actions):
        """Records the step that the batch's `actions` played and returns the dumps it wrote.

        GRF: `info['dumps']` lists a dictionary for each dump that the step finished, with the
        path of the dump and the name of its kind, as `ObservationProcessor` gives it.
        """
        return [{'dump': path, 'name': kind} for kind, path in self._writer._record(actions)]

    def follow(self, config):
        """Takes the dump options of GRF's `Config` for the episode that a reset starts."""
        # GRF: `create_environment` applies `PeriodicDumpWriter` only for a frequency above 1.
        kept = self._frequency <= 1 or self._resets % self._frequency == 0
        self._resets += 1
        self._writer.write_full_episode_dumps = kept and bool(config['dump_full_episodes'])
        self._writer.write_goal_dumps = kept and bool(config['dump_scores'])

    def close(self):
        self._writer.close()


# GRF: the keys of a state's pickle, of the episode logic and of the wrappers that keep states.
_CORE, _CHECKPOINTS, _FRAMES = 'FootballEnvCore', 'CheckpointRewardWrapper', 'FrameStack'
# GRF: the attributes of `EnvState`, the episode logic.
_LOGIC = ('previous_score_diff', 'previous_game_mode', 'prev_ball_owned_team')


def _split_state(state):
    """Returns the pickled entries of a state in GRF's byte format and its match bytes.

    The entries hold the episode logic under `FootballEnvCore` and those of GRF's wrappers, such
    as the counts of `CheckpointRewardWrapper` and the frames of `FrameStack`.

    Raises:
        StateError: If the bytes hold no pickle of the episode logic or it does not load.
    """
    if len(state) < 4 or not 0 < struct.unpack_from('<i', state)[0] <= len(state) - 4:
        raise StateError(
            'expected the bytes of a state: a pickle length, the pickle and the match, '
            f'found {len(state)} bytes that hold none'
        )
    (length,) = struct.unpack_from('<i', state)
    unpickler = _StateUnpickler(io.BytesIO(state[4 : 4 + length]))
    # The frames of `FrameStack` are numpy arrays in a deque.
    unpickler.stacked = True
    try:
        entries = unpickler.load()
        [vars(entries[_CORE])[name] for name in _LOGIC]
    except (pickle.UnpicklingError, KeyError, TypeError, EOFError) as error:
        raise StateError(
            f'expected the pickle of the episode logic in the state, found {error!r}'
        ) from error
    return entries, state[4 + length :]


def _pickled(value):
    """Returns the pickle of `value` without its protocol and stop opcodes."""
    return pickle.dumps(value, protocol=2)[2:-1]


def _own_frame(frame):
    """Returns a numeric frame as an array of its own with numpy's own dtype, else the frame.

    A batch's stack holds such arrays, which pickle writes in full with one dtype for all, while
    the frames of a loaded state share the dtypes that the pickle built.
    """
    if not isinstance(frame, np.ndarray) or frame.dtype.kind not in 'biufc':
        return frame
    return frame.astype(np.dtype(frame.dtype.str))


def _join_state(entries, match):
    """Returns the state in GRF's byte format of pickled entries and match bytes.

    The episode logic, the checkpoint counts and the frames are written as a batch writes them, so
    equal entries give equal bytes, and every other entry follows as pickle writes it.
    """
    episode = [vars(entries[_CORE])[name] for name in _LOGIC]
    collected = None
    if _CHECKPOINTS in entries:
        counts = entries[_CHECKPOINTS]
        collected = [counts.get(agent, 0) for agent in range(max(counts, default=-1) + 1)]
    frames = None
    if _FRAMES in entries:
        stack = entries[_FRAMES]
        frames = _pickled(
            collections.deque(map(_own_frame, stack), maxlen=getattr(stack, 'maxlen', None))
        )
    data = _pickle_episode(episode, collected, frames)
    others = b''.join(
        _pickled(key) + _pickled(value)
        for key, value in entries.items()
        if key not in (_CORE, _CHECKPOINTS, _FRAMES)
    )
    # The other entries join the dictionary before its last `SETITEMS` and `STOP`.
    data = data[:-2] + others + data[-2:]
    return struct.pack('<i', len(data)) + data + match


class FootballEnvironment(_gym.Env):
    """One GRF environment, from `create_environment`.

    The observation, the reward, the termination and the information of each step equal those of
    the environment that GRF's `create_environment` returns for the same arguments, bit for bit,
    after the same `random.seed(...)`. With GRF's gym interface, the default, `reset` returns the
    observation and `step` returns `(observation, reward, done, info)`. With gymnasium's, `reset`
    returns `(observation, info)` and `step` returns `(observation, reward, terminated, truncated,
    info)`, with `truncated` always `False`, as GRF's episodes end only by termination.

    Attributes:
        batch: The batch of one environment that plays the match.
        observation_space, action_space: GRF's spaces for the same arguments. The raw
            representation has no observation space, as in GRF, so its `observation_space` is
            `None`.
        _config: GRF's `Config` of the environment, `cuju.gfootball.env.config.Config`, which
            counts the episodes and holds the seed of each as GRF's does.
        _num_actions: The number of actions of the action set, as GRF's attribute.
        reward_range: The bounds of the reward, unbounded, as gym's `Env` gives GRF's.
    """

    # GRF: gym's `Env`, whose rewards are unbounded.
    reward_range = (-float('inf'), float('inf'))

    def __init__(self, batch, settings):
        self.batch = batch
        self._settings = settings
        self._config = settings['config']
        self._slots = settings['slots']
        # GRF: a configuration without an `agent` player steps with `step([])`, and its observation
        # is the match's own.
        self._agent = settings.get('agent', True)
        # GRF: the single-agent wrappers follow the agent counts of the arguments.
        self._single = settings['single']
        self._representation = settings['representation']
        self._fixed = settings['fixed_seed'] is not None
        self._done = False
        self._gymnasium = settings['api'] == 'gymnasium'
        self._dumps = settings['dumps']
        self._wrapper = settings['wrapper']
        # GRF: the scripted players of `extra_players`, which act on their own observations.
        self._scripted = [
            (
                Bot(batch, slots[0])
                if name == 'bot'
                else Lazy(batch, slots)
                if name == 'lazy'
                else name.bind(batch),
                slots,
            )
            for name, slots in settings['scripted']
        ]
        self._python_players = settings.get('python_players')
        # GRF: the steps of the episode that `write_dump` writes.
        self._episode_steps = _replays.EpisodeSteps()
        self._played = self._episode_steps.reset()
        count = settings['action_count']
        agents = len(self._slots) if self._wrapper is None else self._wrapper.players
        self.action_space = (
            _gym.spaces.MultiDiscrete([count] * agents)
            if agents > 1
            else _gym.spaces.Discrete(count)
        )
        width, height = settings['channel_dimensions']
        stacked = settings['stacked']
        if self._representation == 'extracted':
            shape = _space_shape((height, width, 4), agents, self._single, stacked)
            self.observation_space = _gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        elif self._representation != 'raw':
            shape = _space_shape((115,), agents, self._single, stacked)
            self.observation_space = _gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=shape, dtype=np.float32
            )
        else:
            # GRF: the raw representation has no space, and gymnasium's `Env` declares none.
            self.observation_space = None

    @property
    def _num_actions(self) -> int:
        """The number of actions of the action set, as GRF's `FootballEnv._num_actions`."""
        return self._settings['action_count']

    def _observe(self):
        """Returns the agent's observation of the current step, as GRF's wrappers give it."""
        if self._wrapper is not None:
            if self._representation == 'raw':
                return self._raw(self._wrapper._index[0].tolist())
            array = _numpy(self._wrapper.observation()[0])
        elif self._representation == 'raw':
            return self.observation()
        else:
            array = _numpy(self.batch.observation()[0, self._slots])
        return array[0] if self._single else array

    def _raw(self, slots):
        """Returns GRF's raw observation of each agent of `slots`."""
        return grf_observations(self.batch, 0, slots)

    def observation(self) -> list[dict[str, Any]] | dict[str, Any] | None:
        """Returns the raw observation of each of the agent's players, as GRF's unwrapped one.

        Without an `agent` player it is GRF's observation of the match, the dictionary that GRF's
        dumps hold, read from the players that the configuration holds. With no player at all the
        batch holds no raw observation to read, so it is `None`.
        """
        if not self._agent:
            return self._match_observation()
        return self._raw(self._slots)

    def _match_observation(self):
        """Returns GRF's observation of the match, `FootballEnvCore.observation()`, or `None`."""
        from . import dumps

        batch = self.batch
        if batch.agents == 0:
            return None
        raw = batch.raw()
        rows = [
            {name: tensor[0, agent].tolist() for name, tensor in raw.items()}
            for agent in range(batch.agents)
        ]
        values, _ = dumps._episode_values(batch, 0)
        # An ended episode keeps the designated players of its last step apart.
        designated = values[15:17] if self._done else values[13:15]
        episode = dumps._Episode(0, values, None, False, False)
        sticky = dumps._sticky_names(batch.action_count)
        return _numpy_values(dumps._match_observation(rows, episode, designated, sticky))

    def _actions(self, action):
        """Returns the agent's actions and those of the scripted players for the agent's `action`.

        The first are the actions of the batch's agents with those of the scripted players in
        their slots, or under `MultiAgentToSingleAgent` the policy's actions, one a side, which the
        wrapper places. The second lists each scripted player's slot with its action.
        """
        # GRF: `FootballEnv._action_to_list`, then one action per player of the agent.
        if isinstance(action, np.ndarray):
            action = action.tolist()
        if not isinstance(action, list):
            action = [action]
        count = len(self._slots) if self._wrapper is None else self._wrapper.players
        if len(action) != count:
            if not self._agent:
                # GRF: `FootballEnv.step` without an agent asserts that it gets no action.
                raise ValueError(f'step() received {len(action)} actions, but no agent is playing')
            raise ValueError(
                f'expected {count} actions, one for each player of the agent, found {len(action)}'
            )
        names = _ACTION_NAMES[: self._settings['action_count']]
        values = []
        for value in action:
            if hasattr(value, '_name') and hasattr(value, '_backend_action'):
                # GRF: a `CoreAction` stands for itself.
                if value._name not in names:
                    raise ValueError(
                        f'expected an action of the set {self._settings["action_set"]!r}, found '
                        f'{value._name!r}; create the environment with that set'
                    )
                value = names.index(value._name)
            elif isinstance(value, (bool, int, np.int32, np.int64)):
                value = int(value)
            else:
                raise TypeError(f'expected an integer action, found {value!r}')
            values.append(value)
        scripted = [
            (slot, value)
            for player, slots in self._scripted
            for slot, value in zip(slots, player.act().reshape(-1).tolist(), strict=True)
        ]
        if self._wrapper is not None:
            return values, scripted
        actions = [0] * self.batch.agents
        for slot, value in [*zip(self._slots, values, strict=True), *scripted]:
            actions[slot] = value
        return actions, scripted

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        """Starts the next episode, with a seed drawn from Python's `random` as GRF draws it.

        Args:
            seed: Seeds `np_random`, as the reset of GRF's gymnasium port does. The match does not
                depend on it.
            options: Ignored, as in GRF.

        Returns:
            The observation, and with gymnasium's interface an empty information dictionary after
            it.
        """
        if seed is not None:
            # GRF: gym 0.21 takes no seed in `reset`, so only gymnasium's base class seeds.
            super().reset(seed=seed)
        target = self.batch if self._wrapper is None else self._wrapper
        if self._dumps is not None:
            # GRF: the dump options of the configuration, which `PeriodicDumpWriter` sets before
            # the reset, hold for the episode that the reset starts.
            self._dumps.follow(self._config)
        episode_seed = None if self._fixed else random.randint(0, _SEED_LIMIT)
        if episode_seed is None:
            target.reset()
        else:
            target.reset(seeds=episode_seed)
        # GRF: each reset builds the scenario of the next episode number with the seed it drew.
        self._config._new_scenario(1, episode_seed)
        if self._python_players is not None:
            self._python_players.reset()
        self._played = self._episode_steps.reset()
        self._done = False
        if self._gymnasium:
            return self._observe(), {}
        return self._observe()

    def step(self, action: Any) -> tuple[Any, ...]:
        """Runs one step.

        Args:
            action: An action index, or a list or numpy array of one per player of the agent.

        Returns:
            The observation, the `float32` reward, a scalar for one player or without an agent,
            whether the episode ended and `{'score_reward': ...}`. With GRF's dump options, the
            information also holds `dumps`, as GRF's, a dictionary for each dump that the step
            finished, which maps `dump` to its path and `name` to its kind, such as
            `episode_done`. Gymnasium's interface puts `False` for truncation before the
            information.

        Raises:
            RuntimeError: If the episode ended, until `reset`.
            ValueError: If the number of actions is not the number of the agent's players.
        """
        if self._done:
            raise RuntimeError("can't call step() once the episode finished; call reset() first")
        actions, scripted = self._actions(action)
        if self._wrapper is None:
            _, reward, done, information = self.batch.step([actions])
            played = [actions]
        else:
            _, reward, done, information = self._wrapper.step([actions], _others=scripted)
            played = self._wrapper.last_actions
        # GRF: the reward of each of the agent's players, which `MultiAgentToSingleAgent` passes
        # on for every player that the agent controls.
        reward = _numpy(reward[0, self._slots])
        self._played.append(played)
        self._done = bool(done[0])
        score_reward = int(information['score_reward'][0])
        info = {'score_reward': score_reward}
        if self._dumps is not None:
            # GRF: the information lists the dumps that the step finished.
            written = self._dumps.after_step(played)
            if written:
                info['dumps'] = written
        observation = self._observe()
        if not self._agent:
            # GRF: without an agent the reward is the scoring reward itself.
            reward = np.array(score_reward, dtype=np.float32)
        elif self._single:
            reward = reward[0]
        if self._gymnasium:
            return observation, reward, self._done, False, info
        return observation, reward, self._done, info

    def seed(self, seed: int | None = None) -> None:
        """Does nothing and returns `None`, as gym's `Env.seed` under GRF's environment does.

        The match follows Python's `random`, or `game_engine_random_seed`, as GRF's does.
        """
        return None

    def disable_render(self) -> None:
        """Does nothing, since cuju renders nothing. GRF's turns its renderer off."""

    def _refuse_wrapper_state(self):
        """Raises `ValueError` under GRF's `MultiAgentToSingleAgent`, which refuses states too."""
        if self._wrapper is not None:
            raise ValueError(
                "GRF's MultiAgentToSingleAgent supports no states, so a scenario with "
                'control_all_players played by one player a side has none'
            )

    def get_state(self, to_pickle: dict[str, Any] | None = None) -> bytes:
        """Returns the state of the match in GRF's byte format, as GRF's `get_state(to_pickle)`.

        Args:
            to_pickle: The entries of GRF's wrappers over the environment, such as the counts of
                `CheckpointRewardWrapper` and the frames of `FrameStack`, which the state holds
                beside the match. The entries of `rewards='scoring,checkpoints'` and
                `stacked=True` replace those of the same keys, as GRF's innermost wrapper writes
                its entry last.

        Returns:
            The bytes, which `set_state` here or in GRF reads. Equal entries give equal bytes,
            whether the options of `create_environment` or GRF's wrappers wrote them.
        """
        self._refuse_wrapper_state()
        batch = self.batch
        entries, match = _split_state(batch.get_state(0, stacked=batch.stacked))
        own = {_CORE: entries[_CORE]}
        if _CHECKPOINTS in entries:
            # GRF: the counts go by the agent's players, which play the batch's slots.
            counts = entries[_CHECKPOINTS]
            own[_CHECKPOINTS] = {
                index: counts[slot] for index, slot in enumerate(self._slots) if slot in counts
            }
        if batch.stacked:
            # GRF: `FrameStack` holds the observations of the agent's players.
            stack = batch.observation()[0, self._slots].cpu()
            own[_FRAMES] = collections.deque(
                _frame_stack.frames_of(stack, len(self._slots)), maxlen=_frame_stack.FRAMES
            )
        return _join_state({**(to_pickle or {}), **own}, match)

    def set_state(self, state: bytes) -> dict[str, Any]:
        """Loads a state in GRF's byte format, as GRF's `set_state(state)`.

        Returns:
            The pickled entries of the state, as GRF's returns them, from which GRF's wrappers over
            the environment take theirs, such as the counts of `CheckpointRewardWrapper` and the
            frames of `FrameStack`.

        Raises:
            StateError: If the bytes hold no state that loads, or lack the counts of
                `rewards='scoring,checkpoints'` or the frames of `stacked=True`. It is a
                `ValueError` and a `RuntimeError`.
        """
        self._refuse_wrapper_state()
        batch = self.batch
        entries, match = _split_state(state)
        core = {_CORE: entries[_CORE]}
        own = dict(core)
        if batch._checkpoints and _CHECKPOINTS in entries:
            counts = entries[_CHECKPOINTS]
            own[_CHECKPOINTS] = {
                slot: counts[index] for index, slot in enumerate(self._slots) if index in counts
            }
        if batch.stacked and _FRAMES in entries:
            own[_FRAMES] = self._batch_frames(entries[_FRAMES])
        batch.set_state(0, _join_state(own, match), stacked=batch.stacked)
        # The replay of `write_dump` plays the match in a batch of raw observations.
        self._played = self._episode_steps.set_state(_join_state(core, match), self._done)
        self._done = False
        return entries

    def _batch_frames(self, frames):
        """Returns the frames of a state's `FrameStack` for every agent of the batch.

        GRF's `FrameStack` holds the observations of the agent's players, and the batch stacks
        those of the scripted players too. Their frames stay as the batch holds them, since only
        the agent sees a stack.

        Raises:
            StateError: If the frames are not four observations of the agent's players.
        """
        batch = self.batch
        agents = batch.agents
        if list(self._slots) == list(range(agents)):
            return frames
        own = list(frames) if isinstance(frames, (collections.deque, list, tuple)) else []
        current = _frame_stack.frames_of(batch.observation()[0, :agents].cpu(), agents)
        if len(own) != _frame_stack.FRAMES:
            raise StateError(
                f"expected the {_frame_stack.FRAMES} frames of GRF's FrameStack in the state"
            )
        shape = current[0].shape[1:]
        expected = shape if len(self._slots) == 1 else (len(self._slots), *shape)
        stack = []
        for frame, mine in zip(current, own, strict=True):
            mine = np.asarray(mine)
            if mine.shape != expected or mine.dtype != frame.dtype:
                raise StateError(
                    f'expected frames of the shape {expected} and the type {frame.dtype} in the '
                    f'state, found {mine.shape} and {mine.dtype}'
                )
            frame = frame.copy()
            frame[self._slots] = mine.reshape((len(self._slots), *shape))
            stack.append(frame)
        return collections.deque(stack, maxlen=_frame_stack.FRAMES)

    def write_dump(self, name: str) -> str | None:
        """Writes the last steps of the episode as a dump, as GRF's `write_dump(name)`.

        The dump `<logdir>/<name>_<date>-<time>.dump` holds the last 100 steps of the episode at
        most, the first with the configuration, as GRF writes it. Each name writes once an episode.
        The steps play again in a new batch of one environment, from the reset and each state
        loaded since, so a long episode takes a moment.

        Args:
            name: The name of the dump. GRF's options write `episode_done`, `score` and
                `lost_score`, so those names write nothing here.

        Returns:
            The path of the dump without `.dump`, as GRF returns it, or `None` if it writes none.

        Raises:
            RuntimeError: If the episode ended, until `reset`, where GRF's has no steps to dump.
        """
        return self._episode_steps.write_dump(self, name)

    def close(self) -> None:
        """Frees the batch."""
        if self._dumps is not None:
            self._dumps.close()
            self._dumps = None
        self.batch.close()


def create_environment(
    env_name: str = '',
    stacked: bool = False,
    representation: str = 'extracted',
    rewards: str = 'scoring',
    write_goal_dumps: bool = False,
    write_full_episode_dumps: bool = False,
    render: bool = False,
    write_video: bool = False,
    dump_frequency: int = 1,
    logdir: str = '',
    extra_players: list[str] | None = None,
    number_of_left_players_agent_controls: int = 1,
    number_of_right_players_agent_controls: int = 0,
    channel_dimensions: tuple[int, int] = _SMM_SIZE,
    other_config_options: dict[str, Any] | None = None,
    *,
    device: str = 'cpu',
    threads: int = 1,
    api: str = 'gym',
) -> FootballEnvironment:
    """Creates one GRF environment, as `gfootball.env.create_environment` with the same arguments.

    Args:
        env_name: A shipped scenario such as `11_vs_11_stochastic`, a GRF scenario file or a bundle.
        stacked: Whether the observation stacks the last 4, as GRF's `stacked`.
        representation: `extracted` for the SMM, `simple115`, `simple115v2` or `raw`.
        rewards: `scoring`, or `scoring,checkpoints`.
        write_goal_dumps: GRF's dumps of the frames before each goal, through `cuju.dumps`.
        write_full_episode_dumps: GRF's dumps of every episode, through `cuju.dumps`.
        render: Must stay `False`, since cuju renders nothing.
        write_video: Must stay `False`, for the same reason.
        dump_frequency: Keep the dump options for the first reset and then for one reset in
            every `dump_frequency`, as GRF's option.
        logdir: The directory of the dumps.
        extra_players: Further players, each a definition such as GRF's
            `'agent:left_players=1'`.
        number_of_left_players_agent_controls: The left players that the agent controls.
        number_of_right_players_agent_controls: The right players that the agent controls.
        channel_dimensions: The width and the height of the SMM.
        other_config_options: GRF's other `Config` values, which are `action_set`,
            `game_engine_random_seed`, `reverse_team_processing`, `episode_number`,
            `physics_steps_per_frame`, `players`, `level`, the dump settings, and the rendering
            settings, which change nothing here.
        device: Where the batch of one environment runs, `cpu` or `cuda`. Not a GRF argument.
        threads: The CPU threads of that batch. Not a GRF argument.
        api: `gym` for GRF's interface, where `reset` returns the observation and `step` four
            values, or `gymnasium` for gymnasium's, where `reset` returns two and `step` five. Not
            a GRF argument.

    Returns:
        The environment.

    Raises:
        ValueError: If `api` names neither interface, `env_name` is empty, `render` or
            `write_video` is set, a player definition names a player that cuju lacks or a second
            agent, or another argument is refused by `BatchEnvironment`, which raises `KeyError`
            for an unknown representation or action set.
        NotImplementedError: If a dump option is set and `cuju.dumps` is missing.
    """
    if api not in _APIS:
        raise ValueError(f'expected api {" or ".join(map(repr, _APIS))}, found {api!r}')
    if not env_name:
        raise ValueError('expected the name of a scenario in env_name')
    if render or write_video:
        raise ValueError('cuju renders nothing, so it supports neither render nor write_video')
    options = dict(other_config_options or {})
    unknown = sorted(set(options) - _OPTIONS)
    if unknown:
        warnings.warn(
            f'other_config_options holds {unknown}, which GRF keeps and ignores, as cuju does',
            stacklevel=2,
        )
    left_agents = number_of_left_players_agent_controls
    right_agents = number_of_right_players_agent_controls
    players = [f'agent:left_players={left_agents},right_players={right_agents}']
    facts = _scenario_facts(env_name)
    control = facts[1]
    # GRF: a scenario that sets `control_all_players` gives the agent every controllable player
    # of its sides, and `MultiAgentToSingleAgent` lets the policy play one a side.
    single_players = control is not None and left_agents in (0, 1) and right_agents in (0, 1)
    if single_players:
        if 'players' in options or not left_agents + right_agents:
            raise ValueError(
                'expected one player on one or both sides, and the players of extra_players '
                'rather than of other_config_options, for a scenario with control_all_players'
            )
        players = [
            f'agent:left_players={control[0] if left_agents else 0},'
            f'right_players={control[1] if right_agents else 0}'
        ]
    if extra_players is not None:
        players.extend(extra_players)
    # GRF: the values of the environment's `Config`, which `other_config_options` updates.
    values = {
        'dump_full_episodes': write_full_episode_dumps,
        'dump_scores': write_goal_dumps,
        'players': players,
        'level': env_name,
        'tracesdir': logdir,
        'write_video': write_video,
        **options,
    }
    batch, settings = _create(
        values,
        env_name,
        facts,
        config=None,
        representation=representation,
        rewards=rewards,
        stacked=stacked,
        channel_dimensions=channel_dimensions,
        dump_frequency=dump_frequency,
        sides=(left_agents, right_agents) if single_players else None,
        single=left_agents + right_agents == 1,
        device=device,
        threads=threads,
        api=api,
    )
    return FootballEnvironment(batch, settings)


def _create(
    values,
    env_name,
    facts,
    *,
    config,
    representation,
    rewards,
    stacked,
    channel_dimensions,
    dump_frequency,
    sides,
    single,
    device,
    threads,
    api,
):
    """Returns the batch and the settings of the environment of GRF's configuration values.

    Args:
        values: The values of GRF's `Config` before its scenario numbers the first episode, with
            the episode number of the options.
        env_name, facts: A scenario and its `_scenario_facts`, which spare reading them again when
            the level is that scenario.
        config: The `Config` that GRF's `FootballEnv` takes, which drew the seed of the first
            episode, or `None` to build the configuration as `create_environment` does.
        sides: The policy's sides for `MultiAgentToSingleAgent`, or `None` without it.
        single: Whether the agent controls one player, as GRF's single-agent wrappers check.
        representation, rewards, stacked, channel_dimensions, dump_frequency, device, threads, api:
            As `create_environment`.
    """
    single_players = sides is not None
    players = list(values['players'])
    level = values['level']
    action_set = values.get('action_set', 'default')
    if action_set not in _ACTION_COUNTS:
        raise KeyError(f'expected the action set {", ".join(map(repr, _ACTION_COUNTS))}')
    # GRF: each player controls the next positions of each side, in the order of the list.
    parsed, left, right, slots, scripted = [], 0, 0, None, []
    for definition in players:
        name, player_left, player_right, parameters = _parse_player(definition)
        # GRF: another name is a module of players, which loads with the players, below.
        if name not in _PLAYERS and not _PLAYER_MODULE.fullmatch(name):
            raise ValueError(
                f'expected the players {", ".join(_PLAYERS)}, or a module that defines the player, '
                f'in {definition!r}'
            )
        if name == 'bot' and player_left + player_right != 1:
            raise ValueError(f"GRF's bot controls one player, found {definition!r}")
        # GRF: the bot asserts the `full` action set.
        if name == 'bot' and action_set != 'full':
            raise ValueError("GRF's bot needs other_config_options={'action_set': 'full'}")
        parsed.append((name, left, player_left, right, player_right, parameters))
        left += player_left
        right += player_right
    for name, left_start, player_left, right_start, player_right, _ in parsed:
        positions = list(range(left_start, left_start + player_left)) + [
            left + position for position in range(right_start, right_start + player_right)
        ]
        if name == 'agent':
            if slots is not None:
                raise ValueError("expected one 'agent' player, as GRF allows")
            slots = positions
        elif positions or name not in _PLAYERS:
            scripted.append((name, positions))
    # GRF: `FootballEnv` plays without an agent too, stepping with `step([])`.
    agent = slots is not None
    slots = slots or []
    if config is None:
        # GRF: `create_environment` builds a configuration that reads the scenario, which draws a
        # seed whatever the options say.
        random.randint(0, _SEED_LIMIT)
    fixed_seed = values.get('game_engine_random_seed')
    if fixed_seed is not None and not 0 <= fixed_seed < 2**32:
        raise ValueError(
            f'expected a game_engine_random_seed from 0 to 2**32 - 1, found {fixed_seed}'
        )
    reverse = values.get('reverse_team_processing')
    episode_number = values.get('episode_number', 0)
    deterministic = (facts if level == env_name else _scenario_facts(level))[0]
    # GRF: without a fixed seed, the configuration of the environment draws the seed whose parity
    # sets the team order, and the reset inside the creation of the core draws the seed of the
    # episode that a step before the first `reset` plays. `FootballEnv` takes a configuration
    # that drew the first already.
    creation = None
    if fixed_seed is None:
        first = (
            random.randint(0, _SEED_LIMIT) if config is None else config['game_engine_random_seed']
        )
        creation = [first]
    python_players = None
    if any(name not in _PLAYERS for name, _ in scripted):
        from ._python_players import PythonPlayers

        # GRF: `FootballEnv` builds its players between the configuration and the core.
        config = _grf_config_values(values, config, deterministic, fixed_seed, creation)
        python_players = PythonPlayers(players, parsed, scripted, config, action_set)
        scripted = python_players.scripted
    if creation is not None:
        creation.append(random.randint(0, _SEED_LIMIT))
    numpy_legacy = int(np.__version__.split('.')[0]) < 2
    batch = BatchEnvironment(
        level,
        environment_count=1,
        representation=representation,
        rewards=rewards,
        # `v2` plays `default` with `builtin_ai` for the agents that the policy does not move.
        action_set='v2' if single_players and action_set == 'default' else action_set,
        numpy_legacy=numpy_legacy,
        auto_reset=False,
        with_raw=True,
        device=device,
        threads=threads,
        physics_steps_per_frame=values.get('physics_steps_per_frame', 10),
        channel_dimensions=tuple(channel_dimensions),
        stacked=stacked and not single_players,
        game_engine_random_seed=fixed_seed,
        agent_counts=[(left, right)],
        # Only the parity of the episode number reaches a bundle. The creation plays the episode
        # before the first, and one more reset sets the team order where only the bundle knows
        # whether the scenario is deterministic.
        episode_number=(episode_number + (0 if creation and deterministic is None else 1)) % 2,
    )
    try:
        if creation is None:
            batch.reset(reverse=None if reverse is None else int(bool(reverse)))
        elif deterministic is None:
            # The library sets the team order from the seed as GRF does, deterministic or not.
            batch.reset(seeds=creation[0], reverse=None if reverse is None else int(bool(reverse)))
            batch.reset(seeds=creation[1])
        else:
            if reverse is None:
                reverse = 0 if deterministic else creation[0] % 2
            batch.reset(seeds=creation[1], reverse=int(bool(reverse)))
        dumps = None
        dump_full = values.get('dump_full_episodes', False)
        dump_goals = values.get('dump_scores', False)
        if values.get('write_video', False):
            raise ValueError('cuju renders nothing, so it writes no video')
        if dump_full or dump_goals:
            dumps = _Dumps(
                batch, dump_full, dump_goals, dump_frequency, values.get('tracesdir', '')
            )
        wrapper = None
        if single_players:
            # GRF: the wrapper covers the agent's players, and the players of `extra_players`
            # play their own actions.
            wrapper = MultiAgentToSingleAgent(
                batch, *sides, action_set=action_set, stacked=stacked, _slots=slots
            )
        config = _grf_config(values, config, deterministic, fixed_seed, creation)
    except BaseException:
        batch.close()
        raise
    settings = {
        'config': config,
        'agent': agent,
        'slots': slots,
        'single': single,
        'wrapper': wrapper,
        'scripted': scripted,
        'representation': representation,
        'fixed_seed': fixed_seed,
        'action_set': action_set,
        'action_count': _ACTION_COUNTS[action_set],
        'channel_dimensions': tuple(channel_dimensions),
        'stacked': stacked,
        'dumps': dumps,
        'api': api,
        'python_players': python_players,
        'level': level,
    }
    return batch, settings


def _grf_config(values, config, deterministic, fixed_seed, creation):
    """Returns GRF's `Config` of an environment as its creation leaves it.

    Args:
        values, config: As `_create`.
        deterministic: Whether the scenario is deterministic, `None` where only its bundle knows.
        fixed_seed: The seed of the options, or `None`.
        creation: The seeds that the creation drew, or `None` with a fixed seed.
    """
    config = _grf_config_values(values, config, deterministic, fixed_seed, creation)
    # GRF: the creation of the core builds the scenario again with the same number and a new seed.
    config._new_scenario(0, None if creation is None else creation[1])
    return config


def _grf_config_values(values, config, deterministic, fixed_seed, creation):
    """Returns GRF's `Config` of an environment before the core's first reset, for `_grf_config`."""
    from .gfootball.env import config as grf_config

    if config is None:
        values = dict(values)
        # GRF: the `Config` numbers the scenario that it builds, and a scenario that is not
        # deterministic takes the team order from the parity of its seed unless the values hold one.
        values['episode_number'] = values.get('episode_number', 0) + 1
        if deterministic is False and 'reverse_team_processing' not in values:
            seed = fixed_seed if creation is None else creation[0]
            values['reverse_team_processing'] = bool(seed % 2)
        config = grf_config._without_draw(values)
        if creation is not None:
            # GRF: the seed that the configuration drew is a value of its scenario.
            config.set_scenario_value('game_engine_random_seed', creation[0])
    return config


# GRF: the ids of `gfootball/__init__.py`, by the suffix of each representation.
_GYM_REPRESENTATIONS = (
    ('SMM', 'extracted'),
    ('Pixels', 'pixels'),
    ('simple115', 'simple115'),
    ('simple115v2', 'simple115v2'),
)
_ENTRY_POINT = 'cuju.env:create_environment'


def _gym_before_026() -> bool:
    """Returns whether the installed gym precedes 0.26, which took gymnasium's interface."""
    try:
        version = importlib.metadata.version('gym')
    except importlib.metadata.PackageNotFoundError:
        return False
    match = re.match(r'(\d+)\.(\d+)', version)
    return match is not None and (int(match[1]), int(match[2])) < (0, 26)


def register_environments() -> None:
    """Registers GRF's ids `GFootball-<scenario>-<representation>-v0` for the shipped scenarios.

    It registers them as importing `gfootball` does, with gymnasium and with gym, each where
    installed. `import cuju` imports neither, and registers the ids with each as soon as it is
    imported. The gymnasium ids give gymnasium's interface and the gym ids GRF's. Gym 0.26 and
    later expect gymnasium's interface, so only an older gym, such as GRF's 0.21, gets the ids. The
    `Pixels` ids are registered as in GRF, and making one raises, since cuju renders nothing.
    """
    for module in ('gymnasium', 'gym'):
        if importlib.util.find_spec(module) is not None:
            _register_with(module)


def _register_with(module: str) -> None:
    """Registers GRF's ids with `gymnasium` or `gym`, as `register_environments` does."""
    if module == 'gym' and not _gym_before_026():
        return
    registration = importlib.import_module(f'{module}.envs.registration')
    registry = registration.registry
    # Gym 0.21 keeps its specs in `env_specs`, and later gyms and gymnasium in the registry.
    specs = registry if isinstance(registry, dict) else registry.env_specs
    for scenario in bundled_scenarios():
        for suffix, representation in _GYM_REPRESENTATIONS:
            name = f'GFootball-{scenario}-{suffix}-v0'
            spec = specs.get(name)
            # Older gym versions refuse an id twice, so a gym id that GRF holds stays GRF's.
            if spec is not None and (module == 'gym' or spec.entry_point == _ENTRY_POINT):
                continue
            registration.register(
                id=name,
                entry_point=_ENTRY_POINT,
                kwargs={'env_name': scenario, 'representation': representation, 'api': module},
            )


# A gymnasium or gym that this module's own imports brought in gets the ids here, since `cuju`
# cannot take them from a module that is still importing.
for _module in ('gymnasium', 'gym'):
    if sys.modules.get(_module) is not None:
        _register_with(_module)

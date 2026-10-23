# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""GRF's `gfootball.env.config`, the configuration of an environment and its player definitions.

`Config` holds GRF's values with the current episode's on top, and builds each episode's scenario
as GRF does, which draws its seed from Python's `random`. So after the same `random.seed(...)`,
`FootballEnv(Config(...))` plays GRF's match. See docs/migrating-from-grf.md.
"""

from __future__ import annotations

import copy
import os
import platform
import random
import re
import tempfile
from collections.abc import Callable
from typing import Any

from ... import scenarios
from ...env import _SEED_LIMIT, _scenario_facts

__all__ = [
    'Config',
    'count_left_players',
    'count_players',
    'count_right_players',
    'get_agent_number_of_players',
    'parse_player_definition',
]


def parse_player_definition(definition: str) -> tuple[str, dict[str, Any]]:
    """Returns the name of a player definition, such as `agent:left_players=2`, and its parameters.

    GRF: a definition that names no count gets one left player, and the counts that it names stay
    strings.
    """
    name = definition
    parameters = {'left_players': 0, 'right_players': 0}
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
    if parameters['left_players'] == 0 and parameters['right_players'] == 0:
        parameters['left_players'] = 1
    return name, parameters


def count_players(definition: str) -> int:
    """Returns the players of a player definition, on both sides."""
    _, parameters = parse_player_definition(definition)
    return int(parameters['left_players']) + int(parameters['right_players'])


def count_left_players(definition: str) -> int:
    """Returns the left players of a player definition."""
    return int(parse_player_definition(definition)[1]['left_players'])


def count_right_players(definition: str) -> int:
    """Returns the right players of a player definition."""
    return int(parse_player_definition(definition)[1]['right_players'])


def get_agent_number_of_players(players: list[str]) -> int:
    """Returns the players of the `agent` definitions among `players`."""
    return sum([count_players(player) for player in players if player.startswith('agent')])


# GRF's scenario files of the shipped scenarios, from which cuju's bundles were built.
_SCENARIO_FILES = os.path.join(os.path.dirname(scenarios.__file__), 'data', 'scenarios')
# The `build_scenario` of each scenario file loaded, by absolute path. GRF imports a scenario
# module once per process, so a module keeps its globals from one episode to the next.
_LOADED: dict[str, Callable[[Any], Any]] = {}


def _scenario_function(level: Any) -> Callable[[Any], Any]:
    """Returns the `build_scenario` of a level, a shipped scenario or a scenario file."""
    path = None
    if isinstance(level, str):
        if level.endswith('.py') and os.path.isfile(level):
            path = level
        elif re.fullmatch(r'\w+', level):
            shipped = os.path.join(_SCENARIO_FILES, level + '.py')
            path = shipped if os.path.isfile(shipped) else None
    if path is None:
        raise NotImplementedError(
            f'expected a shipped scenario or the path of a scenario file as the level, found '
            f'{level!r}, and a bundle holds no scenario file to build'
        )
    key = os.path.abspath(path)
    if key not in _LOADED:
        _LOADED[key] = scenarios.load_scenario(path)
    return _LOADED[key]


def _run_scenario(config: Config, builder: scenarios.Builder) -> scenarios.ScenarioConfig:
    """Runs the level's `build_scenario` on `builder` and sets the agents, as GRF's `Scenario`.

    It leaves out the seed and the team order, which `_build_scenario` and the `Scenario` of
    `scenario_builder` set each in its own way.
    """
    _scenario_function(config['level'])(builder)
    # GRF: `_FakePlayersForEmptyTeam` gives an empty team a lazy goalkeeper.
    for team in (scenarios.Team.e_Left, scenarios.Team.e_Right):
        builder.SetTeam(team)
        built = builder.config()
        if not (built.left_team if team == scenarios.Team.e_Left else built.right_team):
            builder.AddPlayer(-1.000000, 0.420000, scenarios.PlayerRole.e_PlayerRole_GK, True)
    built = builder.config()
    # GRF: `_BuildScenarioConfig`.
    built.real_time = config['real_time']
    built.left_agents = config.number_of_left_players()
    built.right_agents = config.number_of_right_players()
    return built


def _build_scenario(config: Config) -> scenarios.ScenarioConfig:
    """Returns the episode's scenario as GRF's `Scenario` builds it, after the seed is drawn."""
    built = _run_scenario(config, scenarios.Builder(config['episode_number']))
    if not built.deterministic and 'game_engine_random_seed' in config:
        built.game_engine_random_seed = config['game_engine_random_seed']
    if 'reverse_team_processing' in config:
        built.reverse_team_processing = config['reverse_team_processing']
    return built


def _defaults() -> dict[str, Any]:
    """Returns GRF's default values of a `Config`, in GRF's order."""
    values = {
        'action_set': 'default',
        'custom_display_stats': None,
        'display_game_stats': True,
        'dump_full_episodes': False,
        'dump_scores': False,
        'players': ['agent:left_players=1'],
        'level': '11_vs_11_stochastic',
        'physics_steps_per_frame': 10,
        'render_resolution_x': 1280,
        'real_time': False,
        'tracesdir': os.path.join(tempfile.gettempdir(), 'dumps'),
        'video_format': 'avi',
        'video_quality_level': 0,
        'write_video': False,
    }
    values['render_resolution_y'] = int(0.5625 * values['render_resolution_x'])
    return values


def _deterministic(level: Any, episode_number: int) -> bool | None:
    """Returns whether an episode's scenario is deterministic, `None` if only its bundle knows."""
    if isinstance(level, str) and level.endswith('.py') and os.path.isfile(level):
        from ... import scenarios

        # GRF: a scenario file may build another scenario for each parity of the episode number.
        return bool(scenarios.describe(level)[episode_number % 2]['values']['deterministic'])
    return _scenario_facts(level)[0]


class Config:
    """GRF's `Config`, the values of an environment with the values of the current episode on top.

    `config[key]` reads the episode's value first, `config[key] = value` and `update` set the
    environment's, and `get_dictionary` returns both merged, as the configuration of GRF's dumps.
    """

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        """Holds GRF's defaults updated with `values`, and builds the first episode's scenario.

        The scenario draws its seed from Python's `random` unless `values` holds
        `game_engine_random_seed`.
        """
        self._values = _defaults()
        if values:
            self._values.update(values)
        self.NewScenario()

    def number_of_left_players(self) -> int:
        return sum([count_left_players(player) for player in self._values['players']])

    def number_of_right_players(self) -> int:
        return sum([count_right_players(player) for player in self._values['players']])

    def number_of_players_agent_controls(self) -> int:
        return get_agent_number_of_players(self._values['players'])

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, self.__class__)
        return self._values == other._values and self._scenario_values == other._scenario_values

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __getitem__(self, key: str) -> Any:
        if key in self._scenario_values:
            return self._scenario_values[key]
        return self._values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._values[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._scenario_values or key in self._values

    def get_dictionary(self) -> dict[str, Any]:
        """Returns a copy of the values, with the episode's on top."""
        merged = copy.deepcopy(self._values)
        merged.update(self._scenario_values)
        return merged

    def set_scenario_value(self, key: str, value: Any) -> None:
        """Sets a value for the current episode only."""
        self._scenario_values[key] = value

    def serialize(self) -> dict[str, Any]:
        """Returns the values of the environment, without the episode's."""
        return self._values

    def update(self, config: dict[str, Any]) -> None:
        self._values.update(config)

    def ScenarioConfig(self) -> scenarios.ScenarioConfig:
        """Returns the current episode's scenario as GRF's `scenario_builder` builds it.

        The result is a `cuju.scenarios.ScenarioConfig` with GRF's fields, built at the first call
        of each episode from the values of that moment. Its floats
        keep the double values of the scenario, which GRF's engine rounds to `float32`.

        Raises:
            NotImplementedError: If the level is a bundle, which holds no scenario file.
        """
        if getattr(self, '_scenario_cfg', None) is None:
            self._scenario_cfg = _build_scenario(self)
        return self._scenario_cfg

    def NewScenario(self, inc: int = 1) -> None:
        """Builds the scenario of the episode number `inc` ahead, as GRF's method.

        GRF: the scenario draws the seed of the episode unless the values hold one, and a scenario
        that is not deterministic takes the team order from the parity of that seed unless the
        values hold an order.
        """
        seed = None
        if 'game_engine_random_seed' not in self._values:
            seed = random.randint(0, _SEED_LIMIT)
        self._new_scenario(inc, seed)
        deterministic = _deterministic(self['level'], self['episode_number'])
        if deterministic is False and 'reverse_team_processing' not in self:
            self['reverse_team_processing'] = bool(self['game_engine_random_seed'] % 2)

    def _new_scenario(self, inc: int, seed: int | None) -> None:
        """Moves the episode number `inc` ahead with `seed` or no seed, as `NewScenario` does."""
        if 'episode_number' not in self._values:
            self._values['episode_number'] = 0
        self._values['episode_number'] += inc
        self._scenario_values = {}
        self._scenario_cfg = None
        if seed is not None:
            self.set_scenario_value('game_engine_random_seed', seed)


def _without_draw(values: dict[str, Any]) -> Config:
    """Returns a `Config` of `values` without building a scenario, so nothing is drawn.

    The values already hold the episode number and the team order that the scenario set.
    """
    config = Config.__new__(Config)
    config._values = _defaults()
    config._values.update(values)
    config._scenario_values = {}
    return config

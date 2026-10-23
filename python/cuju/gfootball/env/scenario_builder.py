# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""GRF's `gfootball.env.scenario_builder`, the shipped scenarios and the builder of scenario files.

`all_scenarios` lists GRF's scenarios without its `tests` package. `Scenario(config)` builds the
scenario of a `Config` as GRF's does. `Player`, `Role` and `Team` are the classes of GRF's engine
that a scenario file uses.
"""

from __future__ import annotations

import random
from typing import Any

from ... import bundled_scenarios as all_scenarios
from ... import scenarios as _scenarios
from ...env import _SEED_LIMIT
from ...scenarios import FormationEntry as Player
from ...scenarios import PlayerRole as Role
from ...scenarios import Team
from . import config as _config

__all__ = ['Player', 'Role', 'Scenario', 'Team', 'all_scenarios']


class Scenario(_scenarios.Builder):
    """GRF's `Scenario(config)`, which builds the scenario of a configuration's current episode.

    It runs the `build_scenario` of the configuration's level, a shipped scenario or a scenario
    file, with itself as the builder, gives an empty team a lazy goalkeeper and sets the agents
    from the players, as GRF's does. Then, as GRF's `_BuildScenarioConfig`, it draws the seed of
    the episode from Python's `random` into the configuration's values of the episode unless the
    configuration holds one of its own, and a scenario that is not deterministic takes that seed,
    and the team order from its parity unless the configuration holds an order.
    `ScenarioConfig()` and `config()` return the scenario.
    """

    def __init__(self, config: _config.Config) -> None:
        """Builds the scenario of `config`.

        Args:
            config: A `cuju.gfootball.env.config.Config`, as GRF's `Scenario` takes.

        Raises:
            TypeError: If `config` is not a `Config`.
            NotImplementedError: If the level is a bundle, which holds no scenario file.
        """
        if not isinstance(config, _config.Config):
            raise TypeError(
                f"expected a cuju.gfootball.env.config.Config, as GRF's Scenario takes, found "
                f'{type(config).__name__}'
            )
        super().__init__(config['episode_number'])
        self._environment_config = config
        built = _config._run_scenario(config, self)
        # GRF: the seed is a value of the episode, so that the dump records it.
        if 'game_engine_random_seed' not in config._values:
            config.set_scenario_value('game_engine_random_seed', random.randint(0, _SEED_LIMIT))
        if not built.deterministic:
            built.game_engine_random_seed = config['game_engine_random_seed']
            if 'reverse_team_processing' not in config:
                config['reverse_team_processing'] = bool(config['game_engine_random_seed'] % 2)
        if 'reverse_team_processing' in config:
            built.reverse_team_processing = config['reverse_team_processing']

    def EpisodeNumber(self) -> Any:
        """Returns the configuration's episode number, as GRF's method reads it."""
        return self._environment_config['episode_number']

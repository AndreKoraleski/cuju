# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""GRF's `gfootball.env.football_env`, with `FootballEnv`, the environment under GRF's wrappers.

`FootballEnv(config)` plays the configuration of a `Config` with raw observations, as GRF's does.
"""

from __future__ import annotations

from ...env import _SMM_SIZE, FootballEnvironment, _create
from .config import Config

__all__ = ['FootballEnv']


class FootballEnv(FootballEnvironment):
    """GRF's `FootballEnv(config)`, with raw observations and GRF's gym interface.

    `reset` returns GRF's list of raw observations of the agent's players, and `step` returns that
    list, the `float32` array of their rewards, whether the episode ended and the information.
    After the same `random.seed(...)` before the `Config`, it plays GRF's match. The players of
    the configuration hold at most one `agent`, and may hold `bot`, `lazy` and other scripted
    players besides.

    Without an `agent`, as in GRF's `play_game.py`, the environment steps with `step([])`.
    `reset` and `step` then return GRF's observation of the match, the dictionary that GRF's
    dumps hold, and `step` returns the scoring reward as a `float32` scalar array. With no player
    at all the batch holds no raw observation to read, so the observation is `None`.
    """

    def __init__(self, config: Config, *, device: str = 'cpu', threads: int = 1) -> None:
        """Creates the environment of a configuration, as GRF's `FootballEnv(config)`.

        Args:
            config: A `Config`, whose episode number and seed move on with each reset, as GRF's.
            device: Where the batch of one environment runs, `cpu` or `cuda`. Not a GRF argument.
            threads: The CPU threads of that batch. Not a GRF argument.

        Raises:
            TypeError: If `config` is not a `Config`.
            ValueError: As `create_environment`, for the players of the configuration.
        """
        if not isinstance(config, Config):
            raise TypeError(
                f'expected a cuju.gfootball.env.config.Config, found {type(config).__name__}'
            )
        values = dict(config._values)
        # The creation builds the scenario of the configuration's episode number, which `_create`
        # takes as the number before it, as `create_environment` takes its option.
        values['episode_number'] = values.get('episode_number', 1) - 1
        batch, settings = _create(
            values,
            None,
            None,
            config=config,
            representation='raw',
            rewards='scoring',
            stacked=False,
            channel_dimensions=_SMM_SIZE,
            dump_frequency=1,
            sides=None,
            single=False,
            device=device,
            threads=threads,
            api='gym',
        )
        super().__init__(batch, settings)

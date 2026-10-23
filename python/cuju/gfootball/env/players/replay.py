# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""GRF's player `replay`, whose actions come from a dump, as `replay:path=<dump>,right_players=1`.

As in GRF, the player takes the actions of the dump from position `index`, the position of its
definition among the players, and ends the process once the dump has no step left.
"""

from __future__ import annotations

import sys
from typing import Any

from .... import dumps
from .. import player_base

__all__ = ['Player']


class Player(player_base.PlayerBase):
    """Player with actions coming from specific game replay."""

    def __init__(self, player_config: dict[str, Any], env_config: Any) -> None:
        player_base.PlayerBase.__init__(self, player_config)
        self._can_play_right = True
        self._replay = dumps.read_dump(player_config['path'])
        self._step = 0
        self._player = player_config['index']

    def take_action(self, observations: list[dict[str, Any]]) -> list[Any]:
        if self._step == len(self._replay):
            print('Replay finished.')
            sys.exit(0)
        names = self._replay[self._step]['debug']['action'][
            self._player : self.num_controlled_players() + self._player
        ]
        self._step += 1
        return [dumps._CORE_ACTIONS[name] for name in names]

# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""GRF's `gfootball.env.player_base`, the base class of a player of `extra_players`.

A module defines a player as GRF's do, with a class `Player(player_config, env_config)` whose
`take_action(observations)` returns one action index or `CoreAction` per observation. A player
that sets `_can_play_right` sees its right players' match as it is and acts in that frame, and the
others see it turned, as a right agent does.
"""

from __future__ import annotations

from typing import Any

__all__ = ['PlayerBase']


class PlayerBase:
    """Base player class."""

    def __init__(self, player_config: dict[str, Any] | None = None) -> None:
        self._num_left_controlled_players = 1
        self._num_right_controlled_players = 0
        self._can_play_right = False
        if player_config:
            self._num_left_controlled_players = int(player_config['left_players'])
            self._num_right_controlled_players = int(player_config['right_players'])

    def num_controlled_left_players(self) -> int:
        return self._num_left_controlled_players

    def num_controlled_right_players(self) -> int:
        return self._num_right_controlled_players

    def num_controlled_players(self) -> int:
        return self._num_left_controlled_players + self._num_right_controlled_players

    def reset(self) -> None:
        pass

    def can_play_right(self) -> bool:
        return self._can_play_right

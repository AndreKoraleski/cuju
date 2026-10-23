# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""GRF's `gfootball.env.players`, the modules of the players that `extra_players` names.

`bot` and `lazy` play in the batch. Any other `<name>` loads `cuju.gfootball.env.players.<name>`,
else the module `<name>` itself, and builds its `Player(player_config, env_config)` as GRF does.
`replay` plays the actions of a dump, as GRF's.
"""

# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""GRF's `gfootball.scenarios`, the names that GRF's scenario files import with `from . import *`.

`cuju.scenarios.load_scenario` gives a scenario file these names by itself, so a file runs
unchanged. This module serves code that imports them from GRF's package.
"""

from __future__ import annotations

from ..scenarios import PlayerRole, Team

__all__ = [
    'Team',
    'e_PlayerRole_AM',
    'e_PlayerRole_CB',
    'e_PlayerRole_CF',
    'e_PlayerRole_CM',
    'e_PlayerRole_DM',
    'e_PlayerRole_GK',
    'e_PlayerRole_LB',
    'e_PlayerRole_LM',
    'e_PlayerRole_RB',
    'e_PlayerRole_RM',
]

e_PlayerRole_GK = PlayerRole.e_PlayerRole_GK
e_PlayerRole_CB = PlayerRole.e_PlayerRole_CB
e_PlayerRole_LB = PlayerRole.e_PlayerRole_LB
e_PlayerRole_RB = PlayerRole.e_PlayerRole_RB
e_PlayerRole_DM = PlayerRole.e_PlayerRole_DM
e_PlayerRole_CM = PlayerRole.e_PlayerRole_CM
e_PlayerRole_LM = PlayerRole.e_PlayerRole_LM
e_PlayerRole_RM = PlayerRole.e_PlayerRole_RM
e_PlayerRole_AM = PlayerRole.e_PlayerRole_AM
e_PlayerRole_CF = PlayerRole.e_PlayerRole_CF

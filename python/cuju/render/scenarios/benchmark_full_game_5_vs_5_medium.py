# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from GRF and GRF_MARL by Light-MALib.
# https://github.com/jidiai/GRF_MARL
# ruff: noqa: F403, F405

from . import *


def build_scenario(builder):
    builder.config().game_duration = 3000
    builder.config().second_half = 1500
    builder.config().right_team_difficulty = 0.6
    builder.config().left_team_difficulty = 0.6
    builder.config().deterministic = False
    if builder.EpisodeNumber() % 2 == 0:
        first_team = Team.e_Left
        second_team = Team.e_Right
    else:
        first_team = Team.e_Right
        second_team = Team.e_Left
    builder.SetTeam(first_team)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, False, False)
    builder.AddPlayer(0.000000, 0.020000, e_PlayerRole_RM)
    builder.AddPlayer(0.000000, -0.020000, e_PlayerRole_CF)
    builder.AddPlayer(-0.1, -0.1, e_PlayerRole_LB)
    builder.AddPlayer(-0.1, 0.1, e_PlayerRole_CB)
    builder.SetTeam(second_team)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, False, False)
    builder.AddPlayer(-0.04, 0.040000, e_PlayerRole_RM)
    builder.AddPlayer(-0.04, -0.040000, e_PlayerRole_CF)
    builder.AddPlayer(-0.1, -0.1, e_PlayerRole_LB)
    builder.AddPlayer(-0.1, 0.1, e_PlayerRole_CB)

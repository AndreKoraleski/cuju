# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403, F405

"""GRF's scenario `11_vs_11_hard_stochastic`."""

from . import *


def build_scenario(builder):
    builder.config().game_duration = 3000
    builder.config().right_team_difficulty = 0.95
    builder.config().deterministic = False
    if builder.EpisodeNumber() % 2 == 0:
        first_team = Team.e_Left
        second_team = Team.e_Right
    else:
        first_team = Team.e_Right
        second_team = Team.e_Left
    builder.SetTeam(first_team)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK)
    builder.AddPlayer(0.000000, 0.020000, e_PlayerRole_RM)
    builder.AddPlayer(0.000000, -0.020000, e_PlayerRole_CF)
    builder.AddPlayer(-0.422000, -0.19576, e_PlayerRole_LB)
    builder.AddPlayer(-0.500000, -0.06356, e_PlayerRole_CB)
    builder.AddPlayer(-0.500000, 0.063559, e_PlayerRole_CB)
    builder.AddPlayer(-0.422000, 0.195760, e_PlayerRole_RB)
    builder.AddPlayer(-0.184212, -0.10568, e_PlayerRole_CM)
    builder.AddPlayer(-0.267574, 0.000000, e_PlayerRole_CM)
    builder.AddPlayer(-0.184212, 0.105680, e_PlayerRole_CM)
    builder.AddPlayer(-0.010000, -0.21610, e_PlayerRole_LM)
    builder.SetTeam(second_team)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK)
    builder.AddPlayer(-0.050000, 0.000000, e_PlayerRole_RM)
    builder.AddPlayer(-0.010000, 0.216102, e_PlayerRole_CF)
    builder.AddPlayer(-0.422000, -0.19576, e_PlayerRole_LB)
    builder.AddPlayer(-0.500000, -0.06356, e_PlayerRole_CB)
    builder.AddPlayer(-0.500000, 0.063559, e_PlayerRole_CB)
    builder.AddPlayer(-0.422000, 0.195760, e_PlayerRole_RB)
    builder.AddPlayer(-0.184212, -0.10568, e_PlayerRole_CM)
    builder.AddPlayer(-0.267574, 0.000000, e_PlayerRole_CM)
    builder.AddPlayer(-0.184212, 0.105680, e_PlayerRole_CM)
    builder.AddPlayer(-0.010000, -0.21610, e_PlayerRole_LM)

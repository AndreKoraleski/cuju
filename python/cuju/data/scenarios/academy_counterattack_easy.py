# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403, F405

"""GRF's scenario `academy_counterattack_easy`."""

from . import *


def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    builder.SetBallPosition(0.26, -0.11)

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(-0.672, -0.19576, e_PlayerRole_LB)
    builder.AddPlayer(-0.75, -0.06356, e_PlayerRole_CB)
    builder.AddPlayer(-0.75, 0.063559, e_PlayerRole_CB)
    builder.AddPlayer(-0.672, 0.19576, e_PlayerRole_RB)
    builder.AddPlayer(-0.434, -0.10568, e_PlayerRole_CM)
    builder.AddPlayer(-0.434, 0.10568, e_PlayerRole_CM)
    builder.AddPlayer(0.5, -0.3161, e_PlayerRole_CM)
    builder.AddPlayer(0.25, -0.1, e_PlayerRole_LM)
    builder.AddPlayer(0.25, 0.1, e_PlayerRole_RM)
    builder.AddPlayer(0.35, 0.316102, e_PlayerRole_CF)

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(0.128, -0.19576, e_PlayerRole_LB)
    builder.AddPlayer(0.4, -0.06356, e_PlayerRole_CB)
    builder.AddPlayer(-0.4, 0.063559, e_PlayerRole_CB)
    builder.AddPlayer(0.128, -0.19576, e_PlayerRole_RB)
    builder.AddPlayer(0.365, -0.10568, e_PlayerRole_CM)
    builder.AddPlayer(0.282, 0.0, e_PlayerRole_CM)
    builder.AddPlayer(0.365, 0.10568, e_PlayerRole_CM)
    builder.AddPlayer(0.54, -0.3161, e_PlayerRole_LM)
    builder.AddPlayer(0.51, 0.0, e_PlayerRole_RM)
    builder.AddPlayer(0.54, 0.316102, e_PlayerRole_CF)

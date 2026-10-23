# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403, F405

"""GRF's scenario `academy_run_to_score`."""

from . import *


def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True
    builder.SetBallPosition(0.02, 0.0)

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(0.0, 0.0, e_PlayerRole_CB)

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(0.12, 0.2, e_PlayerRole_LB)
    builder.AddPlayer(0.12, 0.1, e_PlayerRole_CB)
    builder.AddPlayer(0.12, 0.0, e_PlayerRole_CM)
    builder.AddPlayer(0.12, -0.1, e_PlayerRole_CB)
    builder.AddPlayer(0.12, -0.2, e_PlayerRole_RB)

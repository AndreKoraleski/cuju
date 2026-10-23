# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403, F405

"""GRF's scenario `academy_3_vs_1_with_keeper`."""

from . import *


def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True
    builder.SetBallPosition(0.62, 0.0)

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(0.6, 0.0, e_PlayerRole_CM)
    builder.AddPlayer(0.7, 0.2, e_PlayerRole_CM)
    builder.AddPlayer(0.7, -0.2, e_PlayerRole_CM)

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(-0.75, 0.0, e_PlayerRole_CB)

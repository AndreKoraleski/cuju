# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403, F405

"""GRF's scenario `1_vs_1_easy`."""

from . import *


def build_scenario(builder):
    builder.config().game_duration = 500
    builder.config().right_team_difficulty = 0.0
    builder.config().left_team_difficulty = 0.0
    builder.config().deterministic = False
    if builder.EpisodeNumber() % 2 == 0:
        first_team = Team.e_Left
        second_team = Team.e_Right
    else:
        first_team = Team.e_Right
        second_team = Team.e_Left
    builder.SetTeam(first_team)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK)
    builder.SetTeam(second_team)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK)

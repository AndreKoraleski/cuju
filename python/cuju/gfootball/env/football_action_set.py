# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""GRF's `gfootball.env.football_action_set`, the core actions, the action sets and their helpers.

The actions are the objects that `cuju.dumps` pickles as GRF's `CoreAction`, and the `step` of
`create_environment` takes them as GRF's does, as in `env.step(football_action_set.action_shot)`.
"""

from __future__ import annotations

from typing import Any

import numpy

from ...dumps import _CORE_ACTIONS
from ...dumps import _CoreAction as CoreAction

action_idle = _CORE_ACTIONS['idle']
action_builtin_ai = _CORE_ACTIONS['builtin_ai']
action_left = _CORE_ACTIONS['left']
action_top_left = _CORE_ACTIONS['top_left']
action_top = _CORE_ACTIONS['top']
action_top_right = _CORE_ACTIONS['top_right']
action_right = _CORE_ACTIONS['right']
action_bottom_right = _CORE_ACTIONS['bottom_right']
action_bottom = _CORE_ACTIONS['bottom']
action_bottom_left = _CORE_ACTIONS['bottom_left']
action_long_pass = _CORE_ACTIONS['long_pass']
action_high_pass = _CORE_ACTIONS['high_pass']
action_short_pass = _CORE_ACTIONS['short_pass']
action_shot = _CORE_ACTIONS['shot']
action_keeper_rush = _CORE_ACTIONS['keeper_rush']
action_sliding = _CORE_ACTIONS['sliding']
action_pressure = _CORE_ACTIONS['pressure']
action_team_pressure = _CORE_ACTIONS['team_pressure']
action_switch = _CORE_ACTIONS['switch']
action_sprint = _CORE_ACTIONS['sprint']
action_dribble = _CORE_ACTIONS['dribble']
action_release_direction = _CORE_ACTIONS['release_direction']
action_release_long_pass = _CORE_ACTIONS['release_long_pass']
action_release_high_pass = _CORE_ACTIONS['release_high_pass']
action_release_short_pass = _CORE_ACTIONS['release_short_pass']
action_release_shot = _CORE_ACTIONS['release_shot']
action_release_keeper_rush = _CORE_ACTIONS['release_keeper_rush']
action_release_sliding = _CORE_ACTIONS['release_sliding']
action_release_pressure = _CORE_ACTIONS['release_pressure']
action_release_team_pressure = _CORE_ACTIONS['release_team_pressure']
action_release_switch = _CORE_ACTIONS['release_switch']
action_release_sprint = _CORE_ACTIONS['release_sprint']
action_release_dribble = _CORE_ACTIONS['release_dribble']

action_set_v1 = [
    action_idle,
    action_left,
    action_top_left,
    action_top,
    action_top_right,
    action_right,
    action_bottom_right,
    action_bottom,
    action_bottom_left,
    action_long_pass,
    action_high_pass,
    action_short_pass,
    action_shot,
    action_sprint,
    action_release_direction,
    action_release_sprint,
    action_sliding,
    action_dribble,
    action_release_dribble,
]

action_set_v2 = [*action_set_v1, action_builtin_ai]

# GRF: every core action, which the `play_game` script needs.
full_action_set = [
    *action_set_v2,
    action_keeper_rush,
    action_pressure,
    action_team_pressure,
    action_switch,
    action_release_long_pass,
    action_release_high_pass,
    action_release_short_pass,
    action_release_shot,
    action_release_keeper_rush,
    action_release_sliding,
    action_release_pressure,
    action_release_team_pressure,
    action_release_switch,
]

action_set_dict = {
    'default': action_set_v1,
    'v2': action_set_v2,
    'full': full_action_set,
}

reverse_action_mapping = {
    action_long_pass: action_release_long_pass,
    action_high_pass: action_release_high_pass,
    action_short_pass: action_release_short_pass,
    action_shot: action_release_shot,
    action_keeper_rush: action_release_keeper_rush,
    action_sliding: action_release_sliding,
    action_pressure: action_release_pressure,
    action_team_pressure: action_release_team_pressure,
    action_switch: action_release_switch,
    action_sprint: action_release_sprint,
    action_dribble: action_release_dribble,
    action_release_long_pass: action_long_pass,
    action_release_high_pass: action_high_pass,
    action_release_short_pass: action_short_pass,
    action_release_shot: action_shot,
    action_release_keeper_rush: action_keeper_rush,
    action_release_sliding: action_sliding,
    action_release_pressure: action_pressure,
    action_release_team_pressure: action_team_pressure,
    action_release_switch: action_switch,
    action_release_sprint: action_sprint,
    action_release_dribble: action_dribble,
}


def get_action_set(config: Any) -> list[CoreAction]:
    """Returns the action set that a configuration names in `action_set`."""
    return action_set_dict[config['action_set']]


def get_sticky_actions(config: Any) -> list[CoreAction]:
    """Returns the sticky actions of the configuration's action set, in its order."""
    return [action for action in get_action_set(config) if action._sticky]


def named_action_from_action_set(action_set: list[CoreAction], action: Any) -> CoreAction:
    """Returns the action of an action set that `action` names, an action itself or its index."""
    if (
        hasattr(action, '__dict__')
        and action_set
        and set(action.__dict__) == set(action_set[0].__dict__)
    ):
        return action
    if isinstance(action, (numpy.int32, numpy.int64, int)):
        action = int(action)
        assert action < len(action_set), 'Action outside of action set'
        return action_set[action]
    raise AssertionError(f'Action {action} not found in action set')


def disable_action(action: CoreAction) -> CoreAction:
    """Returns the action that releases `action`, or the one that presses it again."""
    assert set(action.__dict__) == set(action_left.__dict__)
    if action._directional:
        return action_release_direction
    return reverse_action_mapping[action]

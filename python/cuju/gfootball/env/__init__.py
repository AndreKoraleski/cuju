# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""GRF's `gfootball.env`, with `create_environment` and GRF's modules of the package.

`create_environment` is `cuju.env.create_environment`, with GRF's arguments, defaults and gym
interface.

    from cuju.gfootball import env as football_env

    env = football_env.create_environment('academy_empty_goal_close', representation='simple115v2')
"""

from __future__ import annotations

from ...env import create_environment
from . import (
    config,
    football_action_set,
    football_env,
    observation_preprocessing,
    scenario_builder,
    script_helpers,
    wrappers,
)

__all__ = [
    'config',
    'create_environment',
    'football_action_set',
    'football_env',
    'observation_preprocessing',
    'scenario_builder',
    'script_helpers',
    'wrappers',
]

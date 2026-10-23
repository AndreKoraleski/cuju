# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""GRF's import paths over cuju, so GRF's code runs with `gfootball` replaced by `cuju.gfootball`.

    from cuju.gfootball import env as football_env

    env = football_env.create_environment('academy_empty_goal_close', representation='simple115v2')
    observation = env.reset()
    observation, reward, done, info = env.step(env.action_space.sample())

GRF's modules that need no rendering are here, as docs/migrating-from-grf.md lists. Importing
`cuju` registers GRF's ids with gym and gymnasium, as importing `gfootball` does, each once it is
imported itself.

As in GRF, `import cuju.gfootball` binds `cuju.gfootball.env`, which imports at its first use, so
that the import itself works without gymnasium or gym.
"""

from __future__ import annotations

import importlib
from typing import Any


def __getattr__(name: str) -> Any:
    # GRF: `import gfootball` imports `gfootball.env`, which needs gym, so here it waits for a use.
    if name == 'env':
        return importlib.import_module(f'{__name__}.env')
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

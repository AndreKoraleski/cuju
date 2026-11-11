# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""What cuju knows of its optional renderer, its module name `cuju.render` and one hook.

The renderer may be missing, so `cuju render`, `python -m cuju.render`, `import cuju.render` and
`CUJU_RENDER` each name the package that installs it. `import cuju` adds one finder at the end of
`sys.meta_path`, which Python asks only after every other finder has found nothing, and which
answers for `cuju.render` alone. Nothing imports the renderer until a command or `CUJU_RENDER` asks
for it, and every batch that cuju creates while `CUJU_RENDER` is set hands itself to
`cuju.render.on_batch_created`, which the renderer defines.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

# The module of the renderer, which cuju imports only when a command or the variable asks.
MODULE = 'cuju.render'
# The variable that asks every batch of a process for a live source of the renderer.
VARIABLE = 'CUJU_RENDER'
# The values of the variable that ask for nothing.
_OFF = ('', 'off', 'false', 'no')
MESSAGE = (
    "cuju's renderer, the package cuju-render, is not installed; install it with "
    "pip install 'cuju[render]'"
)
_LOGGER = logging.getLogger('cuju')
# Whether a batch has already warned that the variable found no renderer, which it does once.
_warned = False


class _Missing:
    """Names the package of the renderer when `cuju.render` is imported and no other finder has it.

    It sits last in `sys.meta_path`, so an installed renderer is found as any module is, and every
    other module that no finder has goes on to fail as it always does.
    """

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
        if name == MODULE:
            raise ModuleNotFoundError(MESSAGE, name=MODULE)
        return None


def install_finder() -> None:
    """Adds the finder of a missing renderer at the end of `sys.meta_path`, once."""
    if not any(isinstance(finder, _Missing) for finder in sys.meta_path):
        sys.meta_path.append(_Missing())


def is_missing(error: BaseException) -> bool:
    """Returns whether an import failed because the renderer itself is missing."""
    return isinstance(error, ModuleNotFoundError) and error.name == MODULE


def requested() -> bool:
    """Returns whether `CUJU_RENDER` asks for a live source."""
    return os.environ.get(VARIABLE, '').strip().lower() not in _OFF


def on_batch_created(batch: Any) -> None:
    """Hands a new batch to the renderer when `CUJU_RENDER` asks for it, and never fails the batch.

    A missing renderer gives one warning through the logger `cuju` per process, and a renderer that
    fails gives one warning for the batch, which then runs as it would without the variable.
    """
    global _warned
    if not requested():
        return
    try:
        import cuju.render as render
    except ModuleNotFoundError as error:
        if not is_missing(error):
            _warn_failure(error)
        elif not _warned:
            _warned = True
            _LOGGER.warning(
                'cuju: %s is set, but %s. Every batch runs without a live source.',
                VARIABLE,
                MESSAGE,
            )
        return
    except Exception as error:
        _warn_failure(error)
        return
    try:
        render.on_batch_created(batch)
    except Exception as error:
        _warn_failure(error)


def _warn_failure(error: BaseException) -> None:
    """Warns that the renderer gave a batch no live source, and why."""
    _LOGGER.warning(
        'cuju: %s gave the batch no live source, %s: %s', VARIABLE, type(error).__name__, error
    )

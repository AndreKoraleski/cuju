# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""The hidden folder of the renderer, which holds saved matches, teams and live sources."""

from __future__ import annotations

import os
import re
import secrets
import time
from pathlib import Path

# The variable that moves the folder, read on every call so that a test can set it.
HOME_VARIABLE = 'CUJU_RENDER_HOME'

_IDENTIFIER = re.compile(r'[a-z0-9][a-z0-9_-]{0,63}')


def home() -> Path:
    """Returns the folder of the renderer, `~/.cuju/render` or the one `CUJU_RENDER_HOME` names."""
    folder = os.environ.get(HOME_VARIABLE)
    return Path(folder).expanduser().resolve() if folder else Path.home() / '.cuju' / 'render'


def folder(kind: str) -> Path:
    """Returns the subfolder `matches`, `teams`, `live` or `requests`, created when missing."""
    path = home() / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_identifier(prefix: str) -> str:
    """Returns an identifier that sorts by creation time, such as `m-20261106-101500-3fa9`."""
    return f'{prefix}-{time.strftime("%Y%m%d-%H%M%S")}-{secrets.token_hex(2)}'


def valid_identifier(identifier: str) -> bool:
    """Returns whether `identifier` is safe as a file name, so no request can leave the folder."""
    return bool(_IDENTIFIER.fullmatch(identifier))

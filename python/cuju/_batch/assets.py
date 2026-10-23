# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""The asset bundles that the package ships, and the bundle that a name or path asks for."""

from __future__ import annotations

import os
import re

from ..errors import MissingFileError

_DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def bundled_assets() -> list[str]:
    """Returns the names of the asset bundles that the package ships, such as `5_vs_5_l4_r0`.

    `BatchEnvironment` takes such a name `<scenario>_l<left agents>_r<right agents>` in place of a
    path, with any number of agents of a shipped scenario.
    """
    directory = os.path.join(_DATA_DIRECTORY, 'assets')
    if not os.path.isdir(directory):
        return []
    return sorted(name[: -len('.cjb')] for name in os.listdir(directory) if name.endswith('.cjb'))


def _split_name(name):
    """Returns the scenario, left and right agents of `<scenario>_l<left>_r<right>`, or `None`."""
    match = re.fullmatch(r'(.+)_l(\d+)_r(\d+)', name)
    return None if match is None else (match.group(1), int(match.group(2)), int(match.group(3)))


def bundled_scenarios() -> list[str]:
    """Returns the scenarios of the asset bundles that the package ships, such as `5_vs_5`."""
    return sorted({split[0] for split in map(_split_name, bundled_assets()) if split})


def _resolve_asset(asset: str) -> tuple[str, tuple[int, int]]:
    """Returns the bundle path of a path or name, and the agents per side it asks for (-1 keeps)."""
    if callable(asset):
        raise TypeError(
            'expected the path of a scenario file or of a bundle, found a function; pass the path '
            'of its file, or build a bundle with cuju.scenarios.build_bundle'
        )
    # GRF: `create_environment` gives a scenario file or a bare name one left agent.
    if os.path.isfile(asset):
        return asset, (1, 0) if asset.endswith('.py') else (-1, -1)
    if asset.endswith(('.py', '.cjb')) or os.sep in asset or '/' in asset:
        raise MissingFileError(f'expected a scenario file or a bundle at {asset!r}, found no file')
    directory = os.path.join(_DATA_DIRECTORY, 'assets')
    shipped = os.path.join(directory, asset + '.cjb')
    if os.path.isfile(shipped):
        return shipped, (-1, -1)
    split = _split_name(asset)
    if split is None and asset in bundled_scenarios():
        split = (asset, 1, 0)
    if split:
        scenario, left, right = split
        # The bundle with the fewest agents serves any counts; the library sets up the agents anew.
        bases = [name for name in bundled_assets() if (_split_name(name) or ('',))[0] == scenario]
        if bases:
            base = min(bases, key=lambda name: sum(_split_name(name)[1:]))
            return os.path.join(directory, base + '.cjb'), (left, right)
    raise MissingFileError(
        f'expected the path of an asset bundle or `<scenario>_l<left>_r<right>` of one of '
        f'{bundled_scenarios()}, found {asset!r}'
    )

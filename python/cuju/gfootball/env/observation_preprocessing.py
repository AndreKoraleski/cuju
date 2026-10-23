# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""GRF's `gfootball.env.observation_preprocessing`, the super minimap (SMM) of raw observations.

`generate_smm` on cuju's raw observations gives GRF's SMM exactly, since they hold GRF's `float64`
values, and equals the observation of `representation='extracted'`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    'MINIMAP_NORM_X_MAX',
    'MINIMAP_NORM_X_MIN',
    'MINIMAP_NORM_Y_MAX',
    'MINIMAP_NORM_Y_MIN',
    'SMM_HEIGHT',
    'SMM_LAYERS',
    'SMM_WIDTH',
    'generate_smm',
    'get_smm_layers',
    'mark_points',
]

SMM_WIDTH = 96
SMM_HEIGHT = 72

SMM_LAYERS = ['left_team', 'right_team', 'ball', 'active']

# GRF: the minimap spans the pitch, normalized.
MINIMAP_NORM_X_MIN = -1.0
MINIMAP_NORM_X_MAX = 1.0
MINIMAP_NORM_Y_MIN = -1.0 / 2.25
MINIMAP_NORM_Y_MAX = 1.0 / 2.25

_MARKER_VALUE = 255


def get_smm_layers(config: Any) -> list[str]:
    """Returns the names of the SMM's layers, the same for every configuration."""
    return SMM_LAYERS


def mark_points(frame: np.ndarray, points: Any) -> None:
    """Marks the points `(x, y)` of a flat sequence on one SMM layer `frame[y, x]`."""
    for p in range(len(points) // 2):
        x = int(
            (points[p * 2] - MINIMAP_NORM_X_MIN)
            / (MINIMAP_NORM_X_MAX - MINIMAP_NORM_X_MIN)
            * frame.shape[1]
        )
        y = int(
            (points[p * 2 + 1] - MINIMAP_NORM_Y_MIN)
            / (MINIMAP_NORM_Y_MAX - MINIMAP_NORM_Y_MIN)
            * frame.shape[0]
        )
        x = max(0, min(frame.shape[1] - 1, x))
        y = max(0, min(frame.shape[0] - 1, y))
        frame[y, x] = _MARKER_VALUE


def generate_smm(
    observation: list[dict[str, Any]],
    config: Any = None,
    channel_dimensions: tuple[int, int] = (SMM_WIDTH, SMM_HEIGHT),
) -> np.ndarray:
    """Returns the SMM of each raw observation, `[len(observation), height, width, 4]` of `uint8`.

    Args:
        observation: GRF's list of raw observations, one per controlled player.
        config: The configuration of the environment, which changes nothing.
        channel_dimensions: The width and the height of the SMM.
    """
    frame = np.zeros(
        (
            len(observation),
            channel_dimensions[1],
            channel_dimensions[0],
            len(get_smm_layers(config)),
        ),
        dtype=np.uint8,
    )

    for o_i, o in enumerate(observation):
        for index, layer in enumerate(get_smm_layers(config)):
            assert layer in o
            if layer == 'active':
                if o[layer] == -1:
                    continue
                mark_points(frame[o_i, :, :, index], np.array(o['left_team'][o[layer]]).reshape(-1))
            else:
                mark_points(frame[o_i, :, :, index], np.array(o[layer]).reshape(-1))
    return frame

# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""The pickle of a state in GRF's format, written by opcode and read back without GRF."""

from __future__ import annotations

import pickle
import struct

from . import frame_stack as _frame_stack

# GRF's class of the episode logic, as the pickle of a state names it.
_EPISODE_CLASS = ('gfootball.env.football_env_core', 'EnvState')


class _EpisodeState:
    """Stands in for GRF's `EnvState` when the pickle of a state loads."""


class _StateUnpickler(pickle.Unpickler):
    """Unpickles a state, resolving only GRF's episode logic and the globals of a stacked state."""

    stacked = False

    def find_class(self, module, name):
        if (module, name) == _EPISODE_CLASS:
            return _EpisodeState
        found = _frame_stack.find_class(module, name) if self.stacked else None
        if found is not None:
            return found
        raise pickle.UnpicklingError(f'expected only the episode logic, found `{module}.{name}`')


def _pickle_integer(value):
    """Returns the pickle opcode of a signed 32-bit integer."""
    return b'J' + struct.pack('<i', value)


def _pickle_string(value):
    """Returns the pickle opcode of a string."""
    data = value.encode()
    return b'X' + struct.pack('<I', len(data)) + data


def _pickle_episode(episode, collected, frames=None):
    """Returns GRF's state pickle, written by opcode so that `EnvState` needs no import.

    `frames` holds the opcodes of the `FrameStack` deque, or `None` without a stack.
    """
    module, name = _EPISODE_CLASS
    logic = b''.join(
        _pickle_string(key) + _pickle_integer(value)
        for key, value in zip(
            ('previous_score_diff', 'previous_game_mode', 'prev_ball_owned_team'),
            episode,
            strict=True,
        )
    )
    data = (
        b'\x80\x02}('
        + _pickle_string('FootballEnvCore')
        + f'c{module}\n{name}\n'.encode()
        + b')\x81}('
        + logic
        + b'ub'
    )
    if collected is not None:
        checkpoints = b''.join(
            _pickle_integer(agent) + _pickle_integer(count)
            for agent, count in enumerate(collected)
            if count
        )
        data += _pickle_string('CheckpointRewardWrapper') + b'}(' + checkpoints + b'u'
    if frames is not None:
        data += _pickle_string('FrameStack') + frames
    return data + b'u.'

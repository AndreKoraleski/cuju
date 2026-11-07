# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Turns GRF states into the frames of a recording, and writes and reads recording files.

A frame holds what a viewer draws for one step, the world transform of every joint of every player,
the ball, the score, the clock and the game mode, all in the frame of the pitch. The format is in
the repository's `docs/rendering.md`, under Recording format.
"""

from __future__ import annotations

import gzip
import json
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .. import state as _state
from ..errors import StateError

MAGIC = b'CUJREC1\n'
VERSION = 1
NODES = (
    'player',
    'body',
    'middle',
    'neck',
    'left_shoulder',
    'left_elbow',
    'right_shoulder',
    'right_elbow',
    'left_thigh',
    'left_knee',
    'left_ankle',
    'right_thigh',
    'right_knee',
    'right_ankle',
)
# GRF: the skeleton of `player.object`, in pre-order, so every parent comes before its children.
PARENTS = (-1, 0, 1, 2, 2, 4, 2, 6, 1, 8, 9, 1, 11, 12)
# GRF: a node holds a position, a rotation, a scale and a local mode, 44 bytes in all.
_NODE_BYTES = 44
_NODE_SPAN = _NODE_BYTES * len(NODES)
_LOCAL_MODE_RELATIVE = 0
# GRF: `e_PlayerRole_GK`, the first role.
_GOALKEEPER = 0
# The head of a frame, eight integers, the ball's seven floats and four integers of the referee.
_FRAME_HEAD = struct.Struct('<8i7f4i')
_DEFAULT_TEAMS = (
    {'name': 'Left', 'color': [214, 48, 49], 'secondary': [255, 255, 255]},
    {'name': 'Right', 'color': [9, 132, 227], 'secondary': [255, 255, 255]},
)


# A joint's position is stored in steps of 2 mm, which spans 65 m either way of the center spot,
# and its rotation in steps of 1 / 32767, so each joint takes seven 16-bit integers.
POSITION_SCALE = 500
ROTATION_SCALE = 32767


def frame_bytes(players: int) -> int:
    """Returns the size of one frame of a recording of `players` players."""
    return _FRAME_HEAD.size + players * len(NODES) * 14 + 3 * (-(-players // 4) * 4)


def _quantize(nodes: np.ndarray) -> np.ndarray:
    """Returns joints `[..., 7]` as the 16-bit integers that a recording stores."""
    scale = np.array([POSITION_SCALE] * 3 + [ROTATION_SCALE] * 4, np.float64)
    return np.clip(np.rint(nodes * scale), -32767, 32767).astype('<i2')


def _field(path: str) -> _state.Field:
    """Returns the `Field` of a path of the layout."""
    return _state._fields()[path]


def _dtype(field: _state.Field) -> tuple[str, tuple[int, ...]]:
    """Returns the numpy type and shape of a field of the layout."""
    dtype, shape = _state._DTYPES.get(field.type, (None, ()))
    # GRF: an enumeration is a C++ `enum`, four bytes on every platform it runs on.
    if dtype is None and field.size == 4 and field.type[:1].isdigit():
        dtype = '<i4'
    if dtype is None or np.dtype(dtype).itemsize * int(np.prod(shape)) != field.size:
        return 'u1', (field.size,)
    return dtype, shape


class StateReader:
    """Reads the frames of a recording out of the states of one scenario's matches.

    The reader gathers every field it needs from a state in one call into the library, and derives
    the world transform of each joint from the local ones, as GRF's renderer does.

    Attributes:
        players: The number of players of the matches.
    """

    _HEAD = (
        ('reverse', 'environment.scenario.reverseTeamProcessing'),
        ('first', 'match.firstTeam'),
        ('left', 'environment.scenario.leftTeamSize'),
        ('right', 'environment.scenario.rightTeamSize'),
        ('step', 'environment.step'),
        ('goals_first', 'match.goalCount[0]'),
        ('goals_second', 'match.goalCount[1]'),
        ('in_set_piece', 'match.inSetPiece'),
        ('set_piece', 'match.referee.buffer.desiredSetPiece'),
        ('touch_team', 'match.lastTouchTeamID'),
        ('touch_first', 'match.teams[0].lastTouchPlayer'),
        ('touch_second', 'match.teams[1].lastTouchPlayer'),
        ('clock', 'match.matchTimeMilliseconds'),
        ('actual', 'match.actualTimeMilliseconds'),
        ('ball_position', 'match.ball.positionBuffer'),
        ('ball_rotation', 'match.ball.orientationBuffer'),
        ('foul_player', 'match.referee.foul.foulPlayer'),
        ('foul_victim', 'match.referee.foul.foulVictim'),
        ('foul_type', 'match.referee.foul.foulType'),
        ('offside_count', 'match.referee.offsidePlayers.count'),
        ('offside_first', 'match.referee.offsidePlayers[0]'),
        ('buffer_active', 'match.referee.buffer.active'),
        ('scorer', 'match.lastGoalScorer'),
    )
    _PLAYER = (
        ('active', 'base.isActive'),
        ('cards', 'cards'),
        ('possession', 'hasPossession'),
        ('controller', 'base.externalController'),
        ('action', 'base.humanoid.currentAnimation.functionType'),
        ('touch_time', 'base.lastTouchTimeMilliseconds'),
        ('touch_type', 'base.lastTouchType'),
    )

    def __init__(self, players: int) -> None:
        """Prepares the fields of matches with `players` players in all."""
        self.players = players
        fields, layout = [], []
        for name, path in self._HEAD:
            fields.append(_field(path))
            layout.append((name, *_dtype(fields[-1])))
        for player in range(players):
            prefix = f'match.players[{player}]'
            first = _field(f'{prefix}.base.humanoid.nodes[0].position')
            # The nodes lie one after another in memory, so one field covers all fourteen.
            fields.append(
                _state.Field(f'{prefix}.base.humanoid.nodes', first.offset, _NODE_SPAN, 'nodes')
            )
            layout.append((f'nodes{player}', 'u1', (_NODE_SPAN,)))
            for name, path in self._PLAYER:
                fields.append(_field(f'{prefix}.{path}'))
                layout.append((f'{name}{player}', *_dtype(fields[-1])))
        self._fields = fields
        self._dtype = np.dtype(layout)
        known = _state._fields()
        self._roles = {
            (team, index): known[path]
            for team, side in enumerate(('left', 'right'))
            for index in range(players)
            if (path := f'environment.scenario.{side}Team[{index}].role') in known
        }

    def _values(self, state: bytes, frame: str) -> np.void:
        """Returns the gathered fields of a state as one structured record."""
        return np.frombuffer(_state.gather(state, self._fields, frame), self._dtype, 1)[0]

    def _order(self, values: np.void) -> list[tuple[int, int]]:
        """Returns the team and the index in its team of each player of the state.

        GRF: the state lists the team processed first before the other, and its ID flips with the
        reverse team processing.
        """
        first = int(values['first']) ^ int(bool(values['reverse']))
        sizes = (int(values['left']), int(values['right']))
        if sum(sizes) != self.players:
            raise StateError(f'expected a match of {self.players} players, found {sizes}')
        return [(first, index) for index in range(sizes[first])] + [
            (1 - first, index) for index in range(sizes[1 - first])
        ]

    def lineup(self, state: bytes) -> list[dict[str, Any]]:
        """Returns the header entry of each player, its team, index, role and control.

        Args:
            state: A state of the match, as `get_state` returns it.
        """
        values = self._values(state, 'rest')
        order = self._order(values)
        roles = _state.read(state, [self._roles[entry] for entry in order])
        return [
            {
                'team': team,
                'index': index,
                'goalkeeper': int(role) == _GOALKEEPER,
                'controlled': int(values[f'controller{player}']) >= 0,
            }
            for player, ((team, index), role) in enumerate(zip(order, roles, strict=True))
        ]

    def frame(self, state: bytes, flags: int = 0) -> bytes:
        """Returns the frame of a state, as a recording stores it.

        Args:
            state: A state of the match, as `get_state` returns it.
            flags: The frame's flags, 1 when the episode ends on this step and 2 when an episode
                ended just before it, as an automatic reset shows.
        """
        values = self._values(state, 'pitch')
        order = self._order(values)
        first = order[0][0] if order else 0
        score = [0, 0]
        score[first], score[1 - first] = int(values['goals_first']), int(values['goals_second'])
        # GRF: `teams` holds the team processed first at index 0, as the goal counts do.
        touch = {first: int(values['touch_first']), 1 - first: int(values['touch_second'])}
        touch_team = int(values['touch_team'])
        nodes = _world_transforms(
            np.stack([values[f'nodes{player}'] for player in range(self.players)])
        )
        padded = -(-self.players // 4) * 4
        states = np.zeros(padded, np.uint8)
        actions = np.zeros(padded, np.uint8)
        touches = np.zeros(padded, np.uint8)
        # GRF: a step runs ten ticks of 10 ms, so a touch of this step lies within its last 100 ms.
        step_start = int(values['actual']) - 100
        owner = -1
        for player in range(self.players):
            cards = int(values[f'cards{player}'])
            # GRF: the engine moves a gamer's controller to the designated player as play moves,
            # so 8 marks the player that an agent or a person controls on this step.
            states[player] = (
                (1 if values[f'active{player}'] else 0)
                | (2 if cards > 0 else 0)
                | (4 if cards > 1 else 0)
                | (8 if int(values[f'controller{player}']) >= 0 else 0)
            )
            actions[player] = int(values[f'action{player}']) & 0xFF
            if int(values[f'touch_time{player}']) > step_start:
                touches[player] = int(values[f'touch_type{player}']) + 1
            team = order[player][0]
            # GRF: the raw observation's owner, a player in possession whose team touched last.
            if values[f'possession{player}'] and touch_team == team and touch[team] == player:
                owner = player
        game_mode = int(values['set_piece']) if values['in_set_piece'] else 0
        called = (
            bool(values['buffer_active'])
            and int(values['set_piece']) == 3
            and int(values['foul_type']) == 0
            and touch_team in touch
        )
        offside = (
            touch[touch_team]
            if called
            else (int(values['offside_first']) if int(values['offside_count']) > 0 else -1)
        )
        flags |= 4 if called else 0
        flags |= (int(values['scorer']) + 1) << 8
        head = _FRAME_HEAD.pack(
            # GRF: the engine counts -1 after a reset and adds one per step.
            int(values['step']) + 1,
            game_mode,
            score[0],
            score[1],
            touch_team if owner >= 0 else -1,
            owner,
            int(values['clock']),
            int(flags),
            *values['ball_position'].tolist(),
            *values['ball_rotation'].tolist(),
            # GRF: a foul type of 1 is a foul, 2 a yellow card and 3 a red card, and both players
            # index the match's players, as the header lists them.
            int(values['foul_player']) if int(values['foul_type']) else -1,
            int(values['foul_victim']) if int(values['foul_type']) else -1,
            int(values['foul_type']),
            offside,
        )
        return head + b''.join(
            (_quantize(nodes).tobytes(), states.tobytes(), actions.tobytes(), touches.tobytes())
        )


def _multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Returns the Hamilton products of quaternions stored x, y, z, w, as GRF multiplies them."""
    x1, y1, z1, w1 = np.moveaxis(first, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(second, -1, 0)
    return np.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
            w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        -1,
    )


def _rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Returns each `vector` rotated by its `quaternion`, with GRF's two cross products."""
    part = quaternion[..., :3]
    first = np.cross(part, vector)
    second = np.cross(part, first)
    return vector + first * (2.0 * quaternion[..., 3:]) + second * 2.0


def _world_transforms(blocks: np.ndarray) -> np.ndarray:
    """Returns `[players, nodes, 7]`, the world position and rotation of every node of each player.

    Speed: the nodes come in pre-order, so each node needs its parent alone, and every player goes
    at once.

    Args:
        blocks: `[players, nodes * 44]` bytes, the fourteen nodes of each player as the state holds
            them, each a position, a rotation, a scale and a local mode.
    """
    count = blocks.shape[0]
    records = np.ascontiguousarray(blocks, np.uint8).reshape(count, len(NODES), _NODE_BYTES)
    floats = records[..., :40].copy().view('<f4').astype(np.float64)
    modes = records[..., 40:44].copy().view('<i4')[..., 0]
    positions, rotations, scales = floats[..., 0:3], floats[..., 3:7], floats[..., 7:10]
    world = np.empty((count, len(NODES), 7))
    world_scale = np.empty((count, len(NODES), 3))
    for node, parent in enumerate(PARENTS):
        world[:, node, 3:] = rotations[:, node]
        world_scale[:, node] = scales[:, node]
        world[:, node, :3] = positions[:, node]
        if parent < 0:
            continue
        relative = modes[:, node] == _LOCAL_MODE_RELATIVE
        if not relative.any():
            continue
        rotation = _multiply(world[:, parent, 3:], rotations[:, node])
        rotation /= np.linalg.norm(rotation, axis=-1, keepdims=True)
        scale = world_scale[:, parent] * scales[:, node]
        position = (
            _rotate(world[:, parent, 3:], world_scale[:, parent] * positions[:, node])
            + world[:, parent, :3]
        )
        world[relative, node, 3:] = rotation[relative]
        world_scale[relative, node] = scale[relative]
        world[relative, node, :3] = position[relative]
    return world.astype(np.float32)


def header(
    title: str,
    scenario: str,
    lineup: list[dict[str, Any]],
    teams: list[dict[str, Any]] | None = None,
    frame_count: int | None = None,
    timing: dict[str, Any] | None = None,
) -> bytes:
    """Returns the magic, the header length and the header of a recording.

    Args:
        title: The title that the viewer shows.
        scenario: The name of the scenario.
        lineup: The entry of each player, as `StateReader.lineup` returns it.
        teams: The name, `color` and `secondary` color of the left and right team, each optional.
        frame_count: The number of frames, or None for a live stream.
        timing: The optional scenario timing fields.
    """
    given = [*(teams or ()), {}, {}]
    document = {
        'version': VERSION,
        'engine_scorer': True,
        'offside_called': True,
        **(timing or {}),
        'title': title,
        'scenario': scenario,
        'steps_per_second': 10,
        'nodes': list(NODES),
        'parents': list(PARENTS),
        'players': lineup,
        'teams': [{**_DEFAULT_TEAMS[side], **given[side]} for side in range(2)],
        'frame_bytes': frame_bytes(len(lineup)),
        'frame_count': frame_count,
    }
    text = json.dumps(document, separators=(',', ':')).encode()
    # Spaces after the JSON align the frames to four bytes, so a reader views their floats in place.
    text += b' ' * (-(len(MAGIC) + 4 + len(text)) % 4)
    return MAGIC + struct.pack('<I', len(text)) + text


def write(path: Path, head: bytes, frames: list[bytes]) -> None:
    """Writes a recording, its header and frames gzip-compressed, through a temporary file."""
    path = Path(path)
    temporary = path.with_name(path.name + '.part')
    with gzip.open(temporary, 'wb', compresslevel=6) as stream:
        stream.write(head)
        for frame in frames:
            stream.write(frame)
    temporary.replace(path)


def _parse_header(stream: Any) -> dict[str, Any]:
    """Returns the header at the start of `stream`, after checking the magic."""
    start = stream.read(len(MAGIC) + 4)
    if start[: len(MAGIC)] != MAGIC:
        raise StateError('expected a recording of cuju, found other bytes')
    (length,) = struct.unpack_from('<I', start, len(MAGIC))
    return json.loads(stream.read(length))


def read_header(path: Path) -> dict[str, Any]:
    """Returns the header of a recording file, reading only its start."""
    with gzip.open(path, 'rb') as stream:
        return _parse_header(stream)


def frames(path: Path) -> Iterator[bytes]:
    """Yields the frames of a recording file one after another."""
    with gzip.open(path, 'rb') as stream:
        size = _parse_header(stream)['frame_bytes']
        while chunk := stream.read(size):
            yield chunk


def scenario_timing(state: bytes) -> dict[str, Any]:
    """Returns the scenario timing without changing the state."""
    duration, second, score, possession, out = _state.read(
        state,
        [
            'environment.scenario.gameDuration',
            'environment.scenario.secondHalf',
            'environment.scenario.endEpisodeOnScore',
            'environment.scenario.endEpisodeOnPossessionChange',
            'environment.scenario.endEpisodeOnOutOfPlay',
        ],
    )
    duration, second = int(duration), int(second)
    half_ms = (1800 * second) & 0xFFFFFFFF
    return {
        'duration': duration,
        'second_half': second if 0 < second < duration and half_ms < 0x80000000 else None,
        'full_match': duration >= 3000 and not any((score, possession, out)),
    }

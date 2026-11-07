# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Reads single fields of a GRF state, such as the joints of a player or the spin of the ball.

`layout` names every field of the match in memory, with its offset, size and type, and `gather`
loads a state, as `BatchEnvironment.get_state` or GRF's `get_state` return it, and copies chosen
fields out of it. The guide is in the repository's `docs/states.md`, under Reading fields.
"""

from __future__ import annotations

import ctypes
import functools
import struct
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from ._batch import library as _libraries
from .errors import ArgumentError, ChoiceError, LibraryError, StateError

__all__ = ['FRAMES', 'Field', 'gather', 'layout', 'read']

# The frames of `gather`, as the C API numbers them.
FRAMES = {'rest': 0, 'pitch': 1}

# GRF: the state names each type as C++'s `typeid` does, and stores it little-endian.
_DTYPES = {
    'f': ('<f4', ()),
    'i': ('<i4', ()),
    'j': ('<u4', ()),
    'b': ('u1', ()),
    'c': ('i1', ()),
    'm': ('<u8', ()),
    'N7blunted7Vector3E': ('<f4', (3,)),
    'N7blunted10QuaternionE': ('<f4', (4,)),
}


class Field(NamedTuple):
    """One field of the match in memory.

    Attributes:
        path: Its name, such as `match.players[3].base.humanoid.nodes[0].rotation`.
        offset: Its offset in bytes from the start of the match in memory.
        size: Its size in bytes.
        type: The type that GRF's state format gives it, such as `f` for a float.
    """

    path: str
    offset: int
    size: int
    type: str


@functools.cache
def _library():
    """Returns a loaded library, the CPU one where it loads, else the CUDA one."""
    failures = []
    for device in ('cpu', 'cuda'):
        try:
            library = _libraries._load_library(None, device)
        except Exception as error:
            failures.append(f'{device}: {error}')
            continue
        library.cuju_state_layout.restype = ctypes.c_void_p
        library.cuju_state_layout.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)]
        library.cuju_state_gather.argtypes = [
            ctypes.c_char_p,
            ctypes.c_ulonglong,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        return library
    raise LibraryError('expected a library of cuju that loads, found ' + '; '.join(failures))


@functools.cache
def _fields() -> dict[str, Field]:
    """Returns the fields of `layout`, read once from the library."""
    library = _library()
    size = ctypes.c_ulonglong()
    pointer = library.cuju_state_layout(ctypes.byref(size))
    if not pointer:
        raise LibraryError(library.cuju_last_error().decode())
    fields = {}
    for line in ctypes.string_at(pointer, size.value).decode().splitlines():
        path, offset, length, kind = line.split('\t')
        fields[path] = Field(path, int(offset), int(length), kind)
    return fields


def layout() -> dict[str, Field]:
    """Returns every field of the match in memory, by path, in the order of GRF's state format.

    Each list covers its whole capacity, so a field such as the eleventh player exists even where a
    state holds fewer, and the offsets hold for every state of every scenario. A field past the
    count of its list holds no value of the match.

    Returns:
        A new dictionary from each path to its `Field`.
    """
    return dict(_fields())


def _match_bytes(state: bytes) -> bytes:
    """Returns the match of a state, the bytes after its pickle length and its pickle."""
    if len(state) < 4 or not 0 < struct.unpack_from('<i', state)[0] <= len(state) - 4:
        raise StateError(
            f'expected the bytes of a state: a pickle length, the pickle and the match, found '
            f'{len(state)} bytes that hold none'
        )
    (length,) = struct.unpack_from('<i', state)
    return state[4 + length :]


def _resolve(fields: Sequence[str | Field]) -> list[Field]:
    """Returns the `Field` of each path or field, refusing paths that the layout lacks."""
    known = _fields()
    resolved = []
    for field in fields:
        if isinstance(field, Field):
            resolved.append(field)
        elif field in known:
            resolved.append(known[field])
        else:
            raise ArgumentError(f'expected a path of cuju.state.layout(), found {field!r}')
    return resolved


def gather(state: bytes, fields: Sequence[str | Field], frame: str = 'rest') -> bytes:
    """Loads a state and returns the bytes of the chosen fields, one after another.

    Args:
        state: The bytes of a state, as `BatchEnvironment.get_state` or GRF's `get_state` return
            them.
        fields: Paths of `layout` or its `Field` values, in the order of the result.
        frame: `rest`, where the match is as the engine keeps it and the right team plays mirrored,
            or `pitch`, which turns the right team into the frame of the pitch as GRF's camera
            does, so both teams and the ball share one frame with the left goal at negative x.

    Returns:
        The bytes of every field in turn, each as long as its `size`.

    Raises:
        ChoiceError: If `frame` is neither `rest` nor `pitch`. It is a `KeyError` and a
            `ValueError`.
        ArgumentError: If a path is not in `layout`. It is a `ValueError`.
        StateError: If the bytes hold no state. It is a `ValueError` and a `RuntimeError`.
    """
    if frame not in FRAMES:
        raise ChoiceError(f'expected the frame rest or pitch, found {frame!r}')
    match = _match_bytes(state)
    resolved = _resolve(fields)
    offsets = np.array([field.offset for field in resolved], np.uint64)
    sizes = np.array([field.size for field in resolved], np.uint32)
    out = ctypes.create_string_buffer(int(sizes.sum()))
    library = _library()
    if library.cuju_state_gather(
        match,
        len(match),
        FRAMES[frame],
        offsets.ctypes.data,
        sizes.ctypes.data,
        len(resolved),
        ctypes.addressof(out),
    ):
        raise StateError(library.cuju_last_error().decode())
    return out.raw


def read(state: bytes, fields: Sequence[str | Field], frame: str = 'rest') -> list[np.ndarray]:
    """Loads a state and returns the value of each chosen field as a numpy array.

    A number or an enumeration becomes an array of no dimensions, a vector an array of three floats
    and a quaternion an array of four, x, y, z then w. A field of another type, such as a radian,
    which holds a float, a flag and padding, becomes its raw bytes as `uint8`.

    Args:
        state: The bytes of a state, as for `gather`.
        fields: Paths of `layout` or its `Field` values.
        frame: `rest` or `pitch`, as for `gather`.

    Returns:
        One array per field, in the order of `fields`.

    Raises:
        ChoiceError, ArgumentError, StateError: As `gather` raises them.
    """
    resolved = _resolve(fields)
    data = gather(state, resolved, frame)
    values, position = [], 0
    for field in resolved:
        chunk = data[position : position + field.size]
        position += field.size
        dtype, shape = _DTYPES.get(field.type, (None, ()))
        # GRF: an enumeration is a C++ `enum`, four bytes on every platform it runs on.
        if dtype is None and field.size == 4 and field.type[:1].isdigit():
            dtype = '<i4'
        if dtype is not None and np.dtype(dtype).itemsize * int(np.prod(shape)) == field.size:
            values.append(np.frombuffer(chunk, dtype).reshape(shape).copy())
        else:
            values.append(np.frombuffer(chunk, np.uint8).copy())
    return values

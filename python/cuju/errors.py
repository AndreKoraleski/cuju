# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""The exceptions that cuju raises, each also a subclass of the built-in type it stands for.

Every class derives from `Error` and from a built-in exception, so a clause such as
`except ValueError:` catches it as well. `ChoiceError` is also a `KeyError`, and
`EnvironmentIndexError` is both an `IndexError` and a `RuntimeError`.
"""

from __future__ import annotations

__all__ = [
    'ActionRangeError',
    'ArgumentError',
    'ChoiceError',
    'ClosedError',
    'ConfigurationError',
    'DeviceError',
    'EnvironmentIndexError',
    'Error',
    'LibraryError',
    'MissingFileError',
    'StateError',
    'UndefinedPathError',
]


class Error(Exception):
    """The base of every exception that cuju raises for a failure of its own."""

    # `KeyError` quotes its message, so every class here prints the message as it is.
    __str__ = Exception.__str__


class ConfigurationError(Error, ValueError):
    """An option of a batch or of a scenario that cuju refuses."""


class ChoiceError(ConfigurationError, KeyError):
    """A name outside the valid ones, such as a representation, an action set or a device."""


class ArgumentError(Error, ValueError):
    """An argument of a call that cuju refuses, such as actions of the wrong shape."""


class ActionRangeError(Error, IndexError):
    """An action index outside the action set, which GRF refuses as well."""


class EnvironmentIndexError(Error, IndexError, RuntimeError):
    """An environment index outside the batch."""


class StateError(Error, ValueError, RuntimeError):
    """A state that does not load into an environment, or a batch that has no GRF state."""


class ClosedError(Error, RuntimeError):
    """A call on a batch after `close`."""


class DeviceError(Error, RuntimeError):
    """A device that cannot run the batch, or a failure of CUDA.

    Such a device is a missing CUDA GPU, a GPU index past the last, or a processor without AVX2,
    BMI2 or FMA.
    """


class UndefinedPathError(Error, RuntimeError):
    """An environment that reached a path that GRF leaves undefined."""


class LibraryError(Error, RuntimeError):
    """A failure that the library reports, or a library of another backend."""


class MissingFileError(Error, FileNotFoundError):
    """A scenario file, a bundle, a library or another file that does not exist."""


def _from_library(message: str) -> Error:
    """Returns the exception for a failure message of the library."""
    if 'reached a path that GRF leaves undefined' in message:
        return UndefinedPathError(message)
    # A failed CUDA call starts the message with its name, such as `cudaMalloc`, and a GPU without
    # a kernel is a device fault.
    if message.startswith('cuda') or 'CUDA kernel' in message:
        return DeviceError(message)
    return LibraryError(message)

# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Batched Google Research Football (GRF) environments on the GPU or the CPU, bit for bit.

`BatchEnvironment` steps many environments per call, and its outputs are PyTorch tensors that share
its buffers. The guide and the reference are in the repository's `docs/` (`index.md`, `api.md`).
"""

from __future__ import annotations

import importlib.util
import sys
from typing import ClassVar

from ._batch.assets import bundled_assets, bundled_scenarios
from ._batch.environment import BatchEnvironment, memory_estimate
from ._batch.library import library_path
from ._batch.memory import _MemoryEstimate
from ._version import __version__
from .errors import (
    ActionRangeError,
    ArgumentError,
    ChoiceError,
    ClosedError,
    ConfigurationError,
    DeviceError,
    EnvironmentIndexError,
    Error,
    LibraryError,
    MissingFileError,
    StateError,
    UndefinedPathError,
)

__all__ = [
    'ActionRangeError',
    'ArgumentError',
    'BatchEnvironment',
    'ChoiceError',
    'ClosedError',
    'ConfigurationError',
    'DeviceError',
    'EnvironmentIndexError',
    'Error',
    'LibraryError',
    'MissingFileError',
    'MultiDeviceBatch',
    'StateError',
    'UndefinedPathError',
    '__version__',
    'bundled_assets',
    'bundled_scenarios',
    'library_path',
    'memory_estimate',
]

# The names keep `cuju` as their module, so that their reprs and pickles name the public path.
for _name in (
    BatchEnvironment,
    bundled_assets,
    bundled_scenarios,
    library_path,
    memory_estimate,
    _MemoryEstimate,
):
    _name.__module__ = __name__
del _name

# The digests and action logs build on `BatchEnvironment`.
from .action_log import ActionLog, Digest, digest, replay  # noqa: E402

__all__ += ['ActionLog', 'Digest', 'digest', 'replay']

# Last, since it builds on `BatchEnvironment`.
from .multi_device import MultiDeviceBatch  # noqa: E402


class _RegisteringLoader:
    """Runs the loader of gymnasium or gym, then registers GRF's ids with the module."""

    def __init__(self, loader, name):
        self._loader = loader
        self._name = name

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        _RegistrationFinder.waiting.discard(self._name)
        _register_ids(self._name)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _RegistrationFinder:
    """Hands gymnasium and gym, when imported, a loader that then registers GRF's ids."""

    waiting: ClassVar[set[str]] = set()
    # The names being looked up by the other finders, which this one then leaves to them.
    finding: ClassVar[set[str]] = set()

    def find_spec(self, name, path=None, target=None):
        if name not in self.waiting or name in self.finding:
            return None
        self.finding.add(name)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self.finding.discard(name)
        if spec is not None and spec.loader is not None:
            spec.loader = _RegisteringLoader(spec.loader, name)
        return spec


def _register_ids(module):
    # A `cuju.env` that is still importing registers the ids itself once it is done.
    try:
        from .env import _register_with
    except ImportError:
        return
    _register_with(module)


def _register_gymnasium_ids():
    """Registers GRF's ids with gymnasium and gym before 0.26, as importing `gfootball` does.

    Each gets them once it is imported, so that `import cuju` imports neither.
    """
    for module in ('gymnasium', 'gym'):
        if module in sys.modules:
            _register_ids(module)
        else:
            _RegistrationFinder.waiting.add(module)
    if not any(isinstance(finder, _RegistrationFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _RegistrationFinder())


_register_gymnasium_ids()

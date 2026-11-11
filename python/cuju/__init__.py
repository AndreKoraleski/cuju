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
    'ActionLog',
    'ActionRangeError',
    'ArgumentError',
    'BatchEnvironment',
    'ChoiceError',
    'ClosedError',
    'ConfigurationError',
    'DeviceError',
    'Digest',
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
    'digest',
    'library_path',
    'memory_estimate',
    'replay',
]

# The names keep `cuju` as their module, so that their reprs and pickles name the public path.
for _name in (bundled_assets, bundled_scenarios):
    _name.__module__ = __name__
del _name

# The names that build on PyTorch load on first use, so that `import cuju` and the commands of the
# renderer that step no batch start without importing PyTorch. Each maps to its module.
_LAZY = {
    'BatchEnvironment': '_batch.environment',
    'memory_estimate': '_batch.environment',
    'library_path': '_batch.library',
    '_MemoryEstimate': '_batch.memory',
    'ActionLog': 'action_log',
    'Digest': 'action_log',
    'digest': 'action_log',
    'replay': 'action_log',
    'MultiDeviceBatch': 'multi_device',
}


def _load_batch() -> None:
    """Imports the names of `BatchEnvironment` and its helpers, which then name `cuju` their module.

    The names keep `cuju` as their module, so that their reprs and pickles name the public path.
    """
    from ._batch.environment import BatchEnvironment, memory_estimate
    from ._batch.library import library_path
    from ._batch.memory import _MemoryEstimate

    loaded = {
        'BatchEnvironment': BatchEnvironment,
        'memory_estimate': memory_estimate,
        'library_path': library_path,
        '_MemoryEstimate': _MemoryEstimate,
    }
    for value in loaded.values():
        value.__module__ = __name__
    globals().update(loaded)


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    if module.startswith('_batch'):
        _load_batch()
    else:
        globals()[name] = getattr(importlib.import_module(f'.{module}', __name__), name)
    return globals()[name]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


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


def _with_renderer_portions(path):
    """Returns `path` and, after it, every folder of the renderer that `sys.path` holds elsewhere.

    The renderer, the package cuju-render, installs its files into this folder, and an editable
    install of it keeps them in its own checkout, whose `cuju` folder holds `render` and no
    `__init__.py`. Such a folder joins the package as an absolute path. A folder `cuju` with an
    `__init__.py` of its own is a whole cuju, such as an older installation or a checkout in the
    current directory, and stays out, so that its renderer never stands in for a missing one.
    """
    import os

    folders = list(path)
    seen = {os.path.normcase(os.path.abspath(folder)) for folder in folders}
    for entry in sys.path:
        if not isinstance(entry, str):
            continue
        folder = os.path.abspath(os.path.join(entry, __name__))
        key = os.path.normcase(folder)
        if key in seen or os.path.isfile(os.path.join(folder, '__init__.py')):
            continue
        if os.path.isdir(os.path.join(folder, 'render')):
            seen.add(key)
            folders.append(folder)
    return folders


__path__ = _with_renderer_portions(__path__)

# The renderer is optional, so importing it where it is missing names the package that installs it.
from . import _renderer  # noqa: E402

_renderer.install_finder()

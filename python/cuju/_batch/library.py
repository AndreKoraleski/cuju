# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""The library of each backend, the device a batch runs on, and the C signatures of the library."""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from typing import Any

import torch

from ..errors import ChoiceError, DeviceError, LibraryError, MissingFileError

# The directory of the package, where the libraries sit.
_PACKAGE_DIRECTORY = os.path.dirname(os.path.dirname(__file__))
# The library of each device, and the environment variable that overrides its path.
if os.name == 'nt':
    _LIBRARY_FILE = '{}.dll'
elif sys.platform == 'darwin':
    _LIBRARY_FILE = 'lib{}.dylib'
else:
    _LIBRARY_FILE = 'lib{}.so'
_LIBRARIES = {
    'cuda': (_LIBRARY_FILE.format('cuju_cuda'), 'CUJU_CUDA_LIBRARY'),
    'cpu': (_LIBRARY_FILE.format('cuju_cpu'), 'CUJU_CPU_LIBRARY'),
}


def _backend(device: Any) -> str:
    """Returns `cuda` or `cpu`, the backend of a device such as `cuda:1` or a `torch.device`."""
    if isinstance(device, str) and device in _LIBRARIES:
        return device
    try:
        kind = torch.device(device).type
    except (RuntimeError, TypeError):
        kind = None
    if kind not in _LIBRARIES:
        raise ChoiceError(
            f"expected the device 'cuda', 'cuda:N', 'cpu' or a torch.device, found {device!r}"
        )
    return kind


def _automatic(device: Any) -> Any:
    """Returns the CUDA device for `auto` if a GPU and the CUDA library exist, else `cpu`."""
    if not (isinstance(device, str) and device == 'auto'):
        return device
    name, variable = _LIBRARIES['cuda']
    library = os.environ.get(variable) or os.path.join(_PACKAGE_DIRECTORY, name)
    # A `CUJU_CUDA_LIBRARY` that names no file still picks the GPU, whose load then says so.
    present = bool(os.environ.get(variable)) or os.path.isfile(library)
    if present and torch.cuda.is_available():
        return f'cuda:{torch.cuda.current_device()}'
    return 'cpu'


def _placement(device: Any) -> torch.device:
    """Returns the torch device that a batch of `device` runs on, the current GPU for `cuda`."""
    if _backend(device) == 'cpu':
        return torch.device('cpu')
    if not torch.cuda.is_available():
        raise DeviceError(
            f'expected a CUDA GPU for device={str(device)!r}, found none that PyTorch can use; '
            "pass device='cpu' to run the batch on the CPU"
        )
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    count = torch.cuda.device_count()
    if not 0 <= index < count:
        raise DeviceError(
            f'expected a CUDA device from cuda:0 to cuda:{count - 1}, found cuda:{index}; '
            'PyTorch sees only these GPUs (CUDA_VISIBLE_DEVICES may hide others)'
        )
    return torch.device('cuda', index)


def library_path(device: Any = 'cuda') -> str:
    """Returns the path of the library that `BatchEnvironment` loads by default for a device.

    Args:
        device: `cuda`, `cuda:N` or a `torch.device` for `libcuju_cuda.so`, `cpu` for
            `libcuju_cpu.so`, or `auto` for the one that `device='auto'` picks (`cuju_cuda.dll`
            and `cuju_cpu.dll` on Windows, `libcuju_cpu.dylib` on macOS). The library sits inside
            the package unless `CUJU_CUDA_LIBRARY` or `CUJU_CPU_LIBRARY` names another path.

    Raises:
        ChoiceError: If `device` names neither. It is a `ValueError` and a `KeyError`.
    """
    name, variable = _LIBRARIES[_backend(_automatic(device))]
    return os.environ.get(variable) or os.path.join(_PACKAGE_DIRECTORY, name)


# The sysctl keys under which macOS reports each feature.
_MACOS_FEATURES = (
    ('AVX2', 'hw.optional.avx2_0'),
    ('BMI2', 'hw.optional.bmi2'),
    ('FMA', 'hw.optional.fma'),
)


def _macos_feature(key):
    """Returns whether macOS reports the processor feature `key`, true when it cannot tell."""
    try:
        sysctl = ctypes.CDLL(None).sysctlbyname
    except (OSError, AttributeError):
        return True
    value = ctypes.c_int(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    if sysctl(key.encode(), ctypes.byref(value), ctypes.byref(size), None, ctypes.c_size_t(0)):
        return True
    return value.value != 0


def _missing_processor_features():
    """Returns the features among AVX2, BMI2 and FMA that the processor lacks, as far as known."""
    if os.name == 'nt':
        # Windows answers for AVX2 (`PF_AVX2_INSTRUCTIONS_AVAILABLE`, 40), not for BMI2 or FMA.
        return [] if ctypes.windll.kernel32.IsProcessorFeaturePresent(40) else ['AVX2']
    if sys.platform == 'darwin':
        return [name for name, key in _MACOS_FEATURES if not _macos_feature(key)]
    try:
        with open('/proc/cpuinfo', encoding='ascii', errors='replace') as handle:
            for line in handle:
                if line.startswith('flags'):
                    flags = set(line.partition(':')[2].split())
                    return [name.upper() for name in ('avx2', 'bmi2', 'fma') if name not in flags]
    except OSError:
        pass
    return []


def _check_processor(path):
    """Raises `DeviceError` for the packaged CPU library on a CPU without AVX2, BMI2 or FMA."""
    # The packaged library targets x86-64-v3, so elsewhere it stops Python with an illegal
    # instruction; a library at another path is a build of its own. An aarch64 build targets the
    # base Armv8-A, which every aarch64 processor has.
    if platform.machine().lower() not in ('x86_64', 'amd64'):
        return
    if os.path.abspath(path) != os.path.join(
        os.path.abspath(_PACKAGE_DIRECTORY), _LIBRARIES['cpu'][0]
    ):
        return
    missing = _missing_processor_features()
    if missing:
        raise DeviceError(
            'the CPU library of cuju needs a processor with AVX2, BMI2 and FMA, and this one lacks '
            + ', '.join(missing)
        )


class _Configuration(ctypes.Structure):
    """`cuju_configuration` of the C API."""

    _fields_ = [
        ('version', ctypes.c_int),
        ('assets', ctypes.POINTER(ctypes.c_char_p)),
        ('asset_count', ctypes.c_int),
        ('assignment', ctypes.c_void_p),
        ('animations', ctypes.c_char_p),
        ('environment_count', ctypes.c_int),
        ('seed', ctypes.c_ulonglong),
        ('action_set', ctypes.c_int),
        ('observation', ctypes.c_int),
        ('checkpoints', ctypes.c_int),
        ('numpy_legacy', ctypes.c_int),
        ('auto_reset', ctypes.c_int),
        ('threads', ctypes.c_int),
        ('ticks_per_step', ctypes.c_int),
        ('smm_width', ctypes.c_int),
        ('smm_height', ctypes.c_int),
        ('stack', ctypes.c_int),
        ('fixed_seeds', ctypes.c_void_p),
        ('agent_counts', ctypes.c_void_p),
        ('episode_number', ctypes.c_int),
        ('first_index', ctypes.c_ulonglong),
        ('allocate', ctypes.c_void_p),
        ('release', ctypes.c_void_p),
        ('allocator_context', ctypes.c_void_p),
        ('pin_threads', ctypes.c_int),
        ('numa', ctypes.c_int),
        ('page_aligned', ctypes.c_int),
    ]


# `CUJU_CONFIGURATION_VERSION` of the C API.
_CONFIGURATION_VERSION = 2


def _load_library(path=None, device='cuda'):
    """Loads the library of `device`, by default at `library_path`, and declares its signatures."""
    variable = _LIBRARIES[device][1]
    if path is None and os.environ.get(variable) and not os.path.isfile(os.environ[variable]):
        raise MissingFileError(
            f'the environment variable {variable} names {os.environ[variable]!r}, which is no file'
        )
    path = path or library_path(device)
    if not os.path.isfile(path):
        # A CPU-only build, such as the wheels for macOS and Linux on Arm, has no CUDA library.
        advice = (
            "pass device='cpu', or build the package with a CUDA compiler"
            if device == 'cuda'
            else 'build the package with that backend or pick the other device'
        )
        raise MissingFileError(
            f'expected the library of the backend {device} at {path}, which this installation '
            f'lacks; {advice}'
        )
    if device == 'cpu':
        _check_processor(path)
    library = ctypes.CDLL(path)
    pointer, integer, string = ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p
    library.cuju_backend.restype = string
    backend = library.cuju_backend().decode()
    if backend != device:
        raise LibraryError(
            f'expected the library of the backend {device} at {path}, found {backend}'
        )
    library.cuju_last_error.restype = string
    library.cuju_environment_create.restype = pointer
    library.cuju_environment_create.argtypes = [ctypes.POINTER(_Configuration)]
    library.cuju_environment_destroy.argtypes = [pointer]
    library.cuju_memory_estimate.argtypes = [
        ctypes.POINTER(_Configuration),
        ctypes.POINTER(ctypes.c_ulonglong),
    ]
    library.cuju_local_memory_bytes.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)]
    library.cuju_environment_memory_bytes.restype = ctypes.c_ulonglong
    library.cuju_environment_memory_bytes.argtypes = [pointer]
    library.cuju_environment_count.argtypes = [pointer]
    library.cuju_environment_threads.argtypes = [pointer]
    library.cuju_environment_agents.argtypes = [pointer, pointer, pointer]
    library.cuju_environment_players.argtypes = [pointer, pointer, pointer]
    library.cuju_environment_set_difficulty.argtypes = [pointer, pointer]
    library.cuju_environment_difficulty.argtypes = [pointer, pointer]
    library.cuju_environment_difficulty.restype = None
    library.cuju_environment_reset.argtypes = [pointer, pointer, pointer, pointer]
    library.cuju_environment_reset_done.argtypes = [pointer, pointer]
    library.cuju_environment_reset_masked.argtypes = [
        pointer,
        pointer,
        integer,
        pointer,
        pointer,
        pointer,
        integer,
    ]
    library.cuju_environment_step.argtypes = [pointer, pointer, integer, pointer, integer]
    library.cuju_environment_step_async.argtypes = [pointer, pointer, integer, pointer]
    library.cuju_environment_step_wait.argtypes = [pointer]
    library.cuju_environment_synchronize.argtypes = [pointer, pointer]
    library.cuju_environment_get_state.restype = pointer
    library.cuju_environment_get_state.argtypes = [pointer, integer, pointer, pointer]
    library.cuju_environment_set_state.argtypes = [
        pointer,
        integer,
        ctypes.c_char_p,
        ctypes.c_ulonglong,
        pointer,
    ]
    library.cuju_environment_set_stack.argtypes = [
        pointer,
        integer,
        string,
        ctypes.c_char_p,
        ctypes.c_ulonglong,
    ]
    library.cuju_environment_buffer.restype = pointer
    library.cuju_environment_buffer.argtypes = [pointer, string]
    library.cuju_environment_raw_field.restype = string
    library.cuju_environment_raw_field.argtypes = [pointer, integer, pointer, pointer, pointer]
    return library

# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""The memory that a batch takes on its GPU, and the warning when it exceeds the free memory."""

from __future__ import annotations

import ctypes
import warnings

import torch

from ..errors import _from_library


def _free_device_memory(device, allocator):
    """Returns the bytes of a GPU that a new batch may take, or `None` where none are known."""
    try:
        free, _ = torch.cuda.mem_get_info(device)
    except RuntimeError:
        return None
    if allocator == 'torch':
        # PyTorch's cache holds memory that the driver counts as taken but the batch may use.
        free += torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return free


# The CUDA devices on which this process created a batch, so that the driver holds the local memory
# of the kernels there.
_LOCAL_MEMORY_HELD = set()


def _local_memory_bytes(library, device):
    """Returns the local memory that a batch on `device` adds for its kernels, 0 once one did."""
    if device.type != 'cuda' or device.index in _LOCAL_MEMORY_HELD:
        return 0
    local = ctypes.c_ulonglong()
    with torch.cuda.device(device):
        if library.cuju_local_memory_bytes(ctypes.byref(local)) != 0:
            raise _from_library(library.cuju_last_error().decode())
    return local.value


def _warn_memory(requested, free, device, local=0):
    """Warns if a batch requested more memory than its GPU had free before its creation.

    `requested` counts `local`, the local memory that the creation added for the kernels.
    """
    if requested <= free:
        return
    kernels = ''
    if local:
        kernels = f', {local / 2**20:.1f} MiB of it for the local memory of the kernels,'
    # The WDDM driver of Windows and WSL spills into shared system memory without an error.
    warnings.warn(
        f'the batch requested {requested / 2**20:.1f} MiB of {device}{kernels} and the GPU had '
        f'{free / 2**20:.1f} MiB free before its creation. On Windows and WSL the WDDM driver may '
        'then place the batch in shared system memory, which runs many times slower and raises '
        'no error. Use fewer environments, or check cuju.memory_estimate before the creation',
        stacklevel=3,
    )


class _MemoryEstimate(int):
    """The bytes of a batch, as `memory_estimate` returns them, with `local_memory_bytes`."""

    def __new__(cls, value, local_memory_bytes=0):
        estimate = super().__new__(cls, value)
        estimate.local_memory_bytes = local_memory_bytes
        return estimate

# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Page-locked output buffers of a CPU batch, through CUDA's `cudaHostRegister`."""

from __future__ import annotations

import torch

# The library gives each buffer of such a batch whole pages of its own.
_PAGE = 4096


def register(ranges: list[tuple[int, int]]) -> list[int]:
    """Page-locks each `(pointer, bytes)` range and returns the pointers for `unregister`.

    Raises:
        RuntimeError: If CUDA refuses a range, after the ranges locked before it are released.
    """
    runtime = torch.cuda.cudart()
    registered: list[int] = []
    try:
        for pointer, size in ranges:
            status = runtime.cudaHostRegister(pointer, (size + _PAGE - 1) // _PAGE * _PAGE, 0)
            if int(status) != 0:
                raise RuntimeError(
                    f'cudaHostRegister refused {size} bytes of the batch, with error {int(status)}'
                )
            registered.append(pointer)
    except BaseException:
        unregister(registered)
        raise
    return registered


def unregister(pointers: list[int]) -> None:
    """Releases what `register` locked. The batch must outlive the call."""
    runtime = torch.cuda.cudart()
    for pointer in pointers:
        runtime.cudaHostUnregister(pointer)

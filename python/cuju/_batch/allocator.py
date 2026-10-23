# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""The device memory of a batch through PyTorch's caching allocator."""

from __future__ import annotations

import ctypes

import torch

# The C signatures of `allocate` and `release` in `cuju_configuration`.
ALLOCATE = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
RELEASE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)


class TorchAllocator:
    """The `allocate` and `release` functions that the library calls, over torch's allocator.

    The memory comes from the default stream of the device, which the library's copies and fills
    use, and the library releases it only once the device is idle, so torch may hand it out again
    at once.

    Attributes:
        allocate, release: The C functions, which must outlive the batch.
        errors: The exceptions that torch raised in `allocate`, such as `torch.OutOfMemoryError`.
    """

    def __init__(self, device: torch.device) -> None:
        stream = torch.cuda.default_stream(device)
        errors: list[BaseException] = []

        def allocate(size, _context):
            try:
                return torch.cuda.caching_allocator_alloc(size, device, stream)
            # An exception cannot cross the library, so the batch raises it after the call.
            except Exception as error:
                errors.append(error)
                return None

        def release(pointer, _context):
            torch.cuda.caching_allocator_delete(pointer)

        self.allocate = ALLOCATE(allocate)
        self.release = RELEASE(release)
        self.errors = errors

    def pointers(self) -> tuple[int, int]:
        """Returns the addresses of `allocate` and `release` for `cuju_configuration`."""
        return (
            ctypes.cast(self.allocate, ctypes.c_void_p).value,
            ctypes.cast(self.release, ctypes.c_void_p).value,
        )

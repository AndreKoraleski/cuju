# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""The log that `CUJU_LOG=1` turns on, a line for each batch created, through the logger `cuju`.

The lines go to the handlers of the program's logging when it configured any, and otherwise to
standard error, each starting with `cuju:`.
"""

from __future__ import annotations

import logging
import os
import sys
import time

import torch

LOGGER = logging.getLogger('cuju')


class _StandardError(logging.StreamHandler):
    """Writes to the `sys.stderr` of each moment, which a test or a notebook may replace."""

    @property
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, value):
        pass


def enabled() -> bool:
    """Returns whether `CUJU_LOG` is set to a value other than an empty string or `0`."""
    return os.environ.get('CUJU_LOG', '') not in ('', '0')


def _handler():
    """Readies the logger, with a handler to standard error unless the program has handlers."""
    LOGGER.setLevel(logging.INFO)
    if LOGGER.handlers or logging.getLogger().handlers:
        return
    handler = _StandardError()
    handler.setFormatter(logging.Formatter('cuju: %(message)s'))
    LOGGER.addHandler(handler)


def info(message: str) -> None:
    """Logs a line if `CUJU_LOG` asks for it."""
    if enabled():
        _handler()
        LOGGER.info(message)


def start(device: str, torch_device: torch.device | None = None) -> tuple[float, int | None] | None:
    """Returns the time and the free device memory before a batch's creation, or `None` if off.

    Args:
        device: The backend of the batch, `cuda` or `cpu`.
        torch_device: The batch's own `torch.device`, whose free memory counts on the GPU, which
            need not be the current CUDA device.
    """
    if not enabled():
        return None
    _handler()
    free = None
    if device == 'cuda' and torch.cuda.is_available():
        index = None if torch_device is None else torch_device.index
        free = torch.cuda.mem_get_info(index)[0]
    return time.perf_counter(), free


def created(batch, assets, library, started) -> None:
    """Logs the creation of `batch` from `assets` through `library`, begun at `started`."""
    if started is None:
        return
    began, free = started
    seconds = time.perf_counter() - began
    if batch.device == 'cuda':
        index = batch._torch_device.index
        major, minor = torch.cuda.get_device_capability(index)
        place = (
            f'cuda:{index} ({torch.cuda.get_device_name(index)}, compute capability '
            f'{major}.{minor})'
        )
        if free is not None:
            # Other users of the GPU blur this difference of the free memory.
            taken = (free - torch.cuda.mem_get_info(index)[0]) / 2**20
            place += f', about {taken:.0f} MiB of device memory'
    else:
        place = f'the CPU with {batch.threads} threads'
    LOGGER.info(
        f'created {batch.environment_count} environments of {assets} on {place} through '
        f'{library} in {seconds:.2f} s: {batch.agents} agents, {batch.players} players a team, '
        f'{batch._representation}'
    )

# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""The `torch.library` operators through which `torch.compile` traces a step without a break.

`cuju::step` and `cuju::reset_done` name the batch by a key and mutate its output tensors, which
they take as arguments, so that the compiler orders every read of an output after the call and
never drops the call. Each runs the same library call as the eager method.
"""

import itertools
import weakref

import torch

# The batches by key, which drop a batch once it dies.
_BATCHES: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_KEYS = itertools.count(1)


def register(batch) -> int:
    """Returns a new key under which the operators find `batch`."""
    key = next(_KEYS)
    _BATCHES[key] = batch
    return key


def _batch(key):
    batch = _BATCHES.get(key)
    if batch is None:
        raise RuntimeError('the batch of this compiled step is gone')
    return batch


# PyTorch 2.4 brought `custom_op`; before it, `torch.compile` breaks the graph at a step instead.
AVAILABLE = hasattr(torch.library, 'custom_op')

if AVAILABLE:
    # `custom_op` reads the annotations as the schema, so they stay types, not strings.
    @torch.library.custom_op('cuju::step', mutates_args=('outputs',))
    def step(batch: int, actions: torch.Tensor, outputs: list[torch.Tensor]) -> None:
        """Steps the batch of key `batch` with `actions` and writes its output tensors `outputs`."""
        _batch(batch)._operator_step(actions, outputs)

    @step.register_fake
    def _step_fake(batch, actions, outputs):
        return None

    @torch.library.custom_op('cuju::reset_done', mutates_args=('outputs',))
    def reset_done(batch: int, outputs: list[torch.Tensor]) -> None:
        """Resets the ended environments of the batch of key `batch` and writes `outputs`."""
        _batch(batch)._operator_reset_done(outputs)

    @reset_done.register_fake
    def _reset_done_fake(batch, outputs):
        return None

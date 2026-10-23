# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that a CPU batch made before a fork steps in the child, or raises, and never hangs."""

import hashlib
import os
import signal
import threading
import time
import warnings

import pytest

import cuju
from conftest import actions_at, tensor_bytes

pytestmark = pytest.mark.skipif(not hasattr(os, 'fork'), reason='no os.fork')


def in_child(function, timeout=60):
    """Runs `function` in a forked child and returns the bytes it returns, failing on a hang."""
    read, write = os.pipe()
    with warnings.catch_warnings():
        # Python warns that a fork of a process with threads may deadlock, which is the point here.
        warnings.simplefilter('ignore', DeprecationWarning)
        pid = os.fork()
    if pid == 0:
        os.close(read)
        status = 1
        try:
            result = b'ok ' + function()
            status = 0
        except BaseException as error:
            result = f'error {type(error).__name__}: {error}'.encode()
        os.write(write, result[:4096])
        os._exit(status)
    os.close(write)
    deadline = time.monotonic() + timeout
    while os.waitpid(pid, os.WNOHANG) == (0, 0):
        if time.monotonic() > deadline:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            os.close(read)
            pytest.fail('the forked child hung')
        time.sleep(0.02)
    with os.fdopen(read, 'rb') as handle:
        return handle.read().decode(errors='replace')


def play(batch, steps):
    """Returns the SHA-256 of the observations and rewards of `steps` steps."""
    digest = hashlib.sha256()
    for step in range(steps):
        observation, reward, _, _ = batch.step(actions_at(step, batch.environment_count, 4, 19))
        digest.update(tensor_bytes(observation))
        digest.update(tensor_bytes(reward))
    return digest.hexdigest().encode()


def test_child_steps_a_batch_made_before_the_fork():
    batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=8, device='cpu', threads=4)
    batch.reset()
    child = in_child(lambda: play(batch, 20))
    assert child == 'ok ' + play(batch, 20).decode()


def test_child_closes_a_batch_made_before_the_fork():
    batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=8, device='cpu', threads=4)
    batch.reset()

    def close():
        batch.close()
        return b'closed'

    assert in_child(close) == 'ok closed'
    batch.close()


def test_fork_during_a_step_of_another_thread():
    batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=32, device='cpu', threads=4)
    batch.reset()
    stop = threading.Event()

    def keep_stepping():
        step = 0
        while not stop.is_set():
            batch.step(actions_at(step, 32, 4, 19))
            step += 1

    worker = threading.Thread(target=keep_stepping)
    worker.start()
    try:
        time.sleep(0.2)
        child = in_child(lambda: play(batch, 2))
    finally:
        stop.set()
        worker.join()
    # A fork in the middle of a step leaves the child an incomplete batch, which it refuses.
    assert child.startswith('ok ') or (
        child.startswith('error LibraryError') and 'forked while another thread' in child
    )

# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the opt-in check of the actions in GPU memory and the GPU debug mode."""

import os
import subprocess
import sys

import pytest
import torch

import cuju
from conftest import CUDA, actions_at, play_digest

ASSETS = '5_vs_5_l4_r0'
cuda = pytest.mark.skipif(not CUDA, reason='no CUDA device or library')

# A child process steps a batch on actions whose pointer names no device memory, so the step
# faults on the device, which ends the child's CUDA context.
FAULT = """
import os
import sys
sys.path.insert(0, sys.argv[1])
import torch
import cuju
from conftest import actions_at


class Unmapped(torch.Tensor):
    def data_ptr(self):
        return 8


batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=4, debug=sys.argv[2] == '1')
batch.reset()
actions = actions_at(0, 4, 4, 19).cuda().as_subclass(Unmapped)
try:
    batch.step(actions)
    print('stepped')
    torch.cuda.synchronize()
except Exception as error:
    print(type(error).__name__, error)
# The CUDA context is gone, so the child leaves without the usual teardown.
sys.stdout.flush()
os._exit(0)
"""


def actions(device, index=0, value=None):
    """Returns the actions of step `index`, with `value` for agent 2 of environment 1."""
    values = actions_at(index, 4, 4, 19)
    if value is not None:
        values[1, 2] = value
    return values.to(device)


@cuda
def test_device_actions_outside_the_set_raise_before_the_step():
    batch = cuju.BatchEnvironment(ASSETS, environment_count=4, device='cuda', check_actions=True)
    batch.reset()
    batch.step(actions('cuda', 0))
    observation = batch.observation().clone()
    with pytest.raises(cuju.ActionRangeError, match='found 25 on the device for agent 2 of '):
        batch.step(actions('cuda', 1, 25))
    with pytest.raises(IndexError, match='environment 1'):
        batch.step_async(actions('cuda', 1, -20))
    # No environment stepped, and the batch goes on.
    assert torch.equal(batch.observation(), observation)
    batch.step(actions('cuda', 1, -19))
    batch.close()


@cuda
def test_device_actions_go_unchecked_by_default_and_step_as_idle():
    batches = [cuju.BatchEnvironment(ASSETS, environment_count=4, device='cuda') for _ in '01']
    for batch in batches:
        batch.reset()
    batches[0].step(actions('cuda', 0, 25))
    batches[1].step(actions('cuda', 0, 0))
    assert batches[0].get_state(1) == batches[1].get_state(1)


@cuda
def test_padded_agents_go_unchecked():
    batch = cuju.BatchEnvironment(
        ['5_vs_5_l4_r4', '11_vs_11_stochastic_l1_r0'],
        environment_count=2,
        device='cuda',
        assignment=[0, 1],
        debug=True,
    )
    batch.reset()
    values = torch.zeros((2, 8), dtype=torch.int32)
    values[1, 1:] = 99
    batch.step(values.cuda())
    values[1, 0] = 99
    with pytest.raises(cuju.ActionRangeError, match='agent 0 of environment 1'):
        batch.step(values.cuda())


@cuda
def test_debug_waits_for_the_device_at_every_call():
    batch = cuju.BatchEnvironment(
        ASSETS, environment_count=64, device='cuda', auto_reset=False, debug=True
    )
    stream = torch.cuda.current_stream()
    batch.reset()
    for index in range(3):
        batch.step(actions_at(index, 64, 4, 19).cuda())
        assert stream.query()
    batch.reset_done()
    assert stream.query()
    batch.reset(mask=torch.ones(64, dtype=torch.bool, device='cuda'))
    assert stream.query()
    batch.step_async(actions_at(3, 64, 4, 19).cuda())
    assert stream.query()
    batch.step_wait()


@cuda
@pytest.mark.parametrize('debug', [True, False])
def test_debug_raises_a_device_fault_at_the_step_that_caused_it(debug):
    child = subprocess.run(
        [sys.executable, '-c', FAULT, os.path.dirname(__file__), str(int(debug))],
        capture_output=True,
        text=True,
        timeout=300,
    )
    last = child.stdout.strip().splitlines()[-1]
    if debug:
        # The step itself waits and raises the library's error, naming the CUDA call.
        assert last.startswith('DeviceError cudaStreamSynchronize'), child.stdout + child.stderr
    else:
        # The step returns, and the fault surfaces at a later wait.
        assert 'stepped' in child.stdout and not last.startswith('DeviceError'), child.stdout


def test_the_cpu_checks_every_action_before_its_step():
    batch = cuju.BatchEnvironment(ASSETS, environment_count=4, device='cpu', check_actions=True)
    batch.reset()
    with pytest.raises(cuju.ActionRangeError, match='found 25'):
        batch.step(actions('cpu', 0, 25))
    batch.close()


def test_the_options_change_no_output(device):
    expected = play_digest(ASSETS, 'cpu', 10)
    assert play_digest(ASSETS, device, 10, check_actions=True, debug=True) == expected

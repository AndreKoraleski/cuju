# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that a batch over several devices plays as one batch of its environments."""

import hashlib

import pytest
import torch

import cuju
from conftest import CUDA, actions_at, play_digest, tensor_bytes

ASSETS, STEPS, ENVIRONMENTS = '5_vs_5_l4_r4', 40, 5


def play_parts(devices, **options):
    """`play_digest` over a `MultiDeviceBatch`."""
    batch = cuju.MultiDeviceBatch(
        ASSETS, devices, environment_count=ENVIRONMENTS, seed=7, **options
    )
    digest = hashlib.sha256()
    digest.update(tensor_bytes(batch.reset()))
    for step in range(STEPS):
        actions = actions_at(step, ENVIRONMENTS, batch.agents, batch.action_count)
        observation, reward, done, _ = batch.step(actions.to(batch.output_device))
        for tensor in (observation, reward, done):
            digest.update(tensor_bytes(tensor))
    for index in range(ENVIRONMENTS):
        digest.update(batch.get_state(index))
    batch.close()
    return digest.hexdigest()


def expected(**options):
    return play_digest(ASSETS, 'cpu', STEPS, environments=ENVIRONMENTS, **options)


def test_one_device():
    assert play_parts(['cpu']) == expected()


def test_parts_on_the_cpu():
    assert play_parts(['cpu', 'cpu'], split=[2, 3]) == expected()
    assignment = [0, 1, 1, 0, 1]
    assets = [ASSETS, '5_vs_5_l1_r0']
    batch = cuju.MultiDeviceBatch(
        assets, ['cpu', 'cpu', 'cpu'], environment_count=5, assignment=assignment
    )
    assert batch.ranges == [(0, 2), (2, 4), (4, 5)]
    assert batch.agent_mask.sum(1).tolist() == [8, 1, 1, 8, 1]
    batch.close()


@pytest.mark.skipif(not CUDA, reason='no CUDA device or library')
def test_one_gpu():
    assert play_parts(['cuda']) == expected()


@pytest.mark.skipif(not CUDA, reason='no CUDA device or library')
def test_parts_on_one_gpu_and_the_cpu():
    # A machine with one GPU stands in for several by repeating it.
    assert play_parts(['cuda:0', 'cuda:0']) == expected()
    assert play_parts(['cuda', 'cpu'], output_device='cpu') == expected()


def test_a_bad_action_steps_no_part():
    batch = cuju.MultiDeviceBatch(ASSETS, ['cpu', 'cpu'], environment_count=4)
    batch.reset()
    before = [batch.get_state(index) for index in range(4)]
    actions = torch.zeros((4, 8), dtype=torch.int32)
    actions[3, 2] = 19
    with pytest.raises(cuju.ActionRangeError):
        batch.step(actions)
    assert [batch.get_state(index) for index in range(4)] == before
    batch.step(torch.zeros((4, 8), dtype=torch.int32))
    batch.close()


def test_a_bad_split_raises():
    with pytest.raises(cuju.ConfigurationError, match='split'):
        cuju.MultiDeviceBatch(ASSETS, ['cpu', 'cpu'], environment_count=4, split=[4, 0])
    with pytest.raises(cuju.ConfigurationError, match='device'):
        cuju.MultiDeviceBatch(ASSETS, ['cpu'], environment_count=4, device='cpu')
    with pytest.raises(cuju.ChoiceError, match='CUDA device or the CPU'):
        cuju.MultiDeviceBatch(ASSETS, ['meta'], environment_count=4)


def test_bad_arguments_raise_cuju_errors():
    batch = cuju.MultiDeviceBatch(ASSETS, ['cpu', 'cpu'], environment_count=4)
    with pytest.raises(cuju.ArgumentError, match='one for each of the 4'):
        batch.reset(seeds=[1, 2, 3])
    batch.reset()
    with pytest.raises(cuju.EnvironmentIndexError, match='from 0 to 3, found 4'):
        batch.get_state(4)
    batch.close()


def test_a_closed_batch_raises_closed_error():
    batch = cuju.MultiDeviceBatch(ASSETS, ['cpu', 'cpu'], environment_count=4)
    batch.reset()
    state = batch.get_state(0)
    batch.close()
    batch.close()
    actions = torch.zeros((4, 8), dtype=torch.int32)
    calls = [
        batch.reset,
        batch.reset_done,
        batch.observation,
        lambda: batch.step(actions),
        lambda: batch.get_state(0),
        lambda: batch.set_state(0, state),
    ]
    for call in calls:
        with pytest.raises(cuju.ClosedError, match='closed'):
            call()


@pytest.mark.skipif(not CUDA, reason='no CUDA device or library')
def test_a_gather_from_page_locked_outputs_lands_before_the_next_step():
    def play(busy):
        batch = cuju.MultiDeviceBatch(
            ASSETS, ['cpu'], environment_count=4, output_device='cuda', pin_memory=True, seed=3
        )
        kept = [batch.reset().clone()]
        for step in range(12):
            if busy:
                # The copies of the gather wait behind this work, while the CPU part goes on.
                torch.cuda._sleep(2**25)
            observation, reward, done, _ = batch.step(actions_at(step, 4, 8, batch.action_count))
            # A clone on the GPU runs after the gather, and must see this step's outputs.
            kept += [observation.clone(), reward.clone(), done.clone()]
        torch.cuda.synchronize()
        batch.close()
        return [tensor_bytes(tensor) for tensor in kept]

    assert play(True) == play(False)

# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks asynchronous steps, the placement of the CPU threads and page-locked outputs."""

import hashlib
import os

import pytest
import torch

import cuju
from conftest import CUDA, actions_at, play_digest, tensor_bytes

ASSETS, STEPS, ENVIRONMENTS = '5_vs_5_l4_r4', 60, 8


def play_async(device, **options):
    """`play_digest` through `step_async` and `step_wait`."""
    batch = cuju.BatchEnvironment(
        ASSETS, environment_count=ENVIRONMENTS, device=device, seed=7, **options
    )
    digest = hashlib.sha256()
    digest.update(tensor_bytes(batch.reset()))
    for step in range(STEPS):
        actions = actions_at(step, ENVIRONMENTS, batch.agents, batch.action_count).to(device)
        batch.step_async(actions)
        # The caller may reuse the actions at once.
        actions.fill_(0)
        observation, reward, done, _ = batch.step_wait()
        for tensor in (observation, reward, done):
            digest.update(tensor_bytes(tensor))
    for index in range(ENVIRONMENTS):
        digest.update(batch.get_state(index))
    batch.close()
    return digest.hexdigest()


def test_an_async_step_equals_a_step(device):
    expected = play_digest(ASSETS, device, STEPS, environments=ENVIRONMENTS)
    assert play_async(device) == expected


def test_a_call_during_an_async_step_waits_for_it():
    batch = cuju.BatchEnvironment(ASSETS, environment_count=4, device='cpu', threads=2)
    other = cuju.BatchEnvironment(ASSETS, environment_count=4, device='cpu', threads=2)
    batch.reset()
    other.reset()
    actions = actions_at(0, 4, batch.agents, batch.action_count)
    batch.step_async(actions)
    during = batch.get_state(1)
    other.step(actions)
    assert during == other.get_state(1)
    batch.step_wait()
    batch.close()
    other.close()


def test_async_steps_pair_with_waits():
    batch = cuju.BatchEnvironment(ASSETS, environment_count=2, device='cpu')
    batch.reset()
    with pytest.raises(RuntimeError, match='step_async first'):
        batch.step_wait()
    batch.step_async(torch.zeros((2, 8), dtype=torch.int32))
    with pytest.raises(RuntimeError, match='step_wait'):
        batch.step_async(torch.zeros((2, 8), dtype=torch.int32))
    batch.step_wait()
    with pytest.raises(IndexError):
        batch.step_async(torch.full((2, 8), 19, dtype=torch.int32))
    # No step started, so none is in flight.
    with pytest.raises(RuntimeError, match='step_async first'):
        batch.step_wait()
    batch.close()


@pytest.mark.parametrize(
    'placement', [{'pin_threads': True}, {'numa': True}, {'pin_threads': True, 'numa': True}]
)
def test_placed_threads_give_the_same_outputs(placement):
    expected = play_digest(ASSETS, 'cpu', STEPS, environments=ENVIRONMENTS, threads=3)
    assert (
        play_digest(ASSETS, 'cpu', STEPS, environments=ENVIRONMENTS, threads=3, **placement)
        == expected
    )
    assert play_async('cpu', threads=3, **placement) == play_async('cpu', threads=3)


@pytest.mark.skipif(not hasattr(os, 'sched_getaffinity'), reason='no affinity mask')
def test_numa_groups_give_the_same_outputs(monkeypatch):
    # Three nodes stand in for the machine's, so the groups take ranges of their own.
    processors = sorted(os.sched_getaffinity(0))
    nodes = [0] * (max(processors) + 1)
    for position, processor in enumerate(processors):
        nodes[processor] = position * 3 // len(processors)
    monkeypatch.setenv('CUJU_NUMA_NODES', ','.join(map(str, nodes)))
    expected = play_digest(ASSETS, 'cpu', STEPS, environments=ENVIRONMENTS, threads=5)
    for placement in ({'numa': True}, {'numa': True, 'pin_threads': True}):
        found = play_digest(ASSETS, 'cpu', STEPS, environments=ENVIRONMENTS, threads=5, **placement)
        assert found == expected
        assert play_async('cpu', threads=5, **placement) == expected


@pytest.mark.skipif(not CUDA, reason='no CUDA device or library')
def test_page_locked_outputs():
    batch = cuju.BatchEnvironment(
        ASSETS, environment_count=4, device='cpu', pin_memory=True, with_raw=True
    )
    observation = batch.reset()
    _, reward, done, information = batch.step(torch.zeros((4, 8), dtype=torch.int32))
    tensors = [observation, reward, done, information['final_observation']]
    assert all(tensor.is_pinned() for tensor in [*tensors, *batch.raw().values()])
    copied = observation.to('cuda', non_blocking=True)
    torch.cuda.synchronize()
    assert tensor_bytes(copied) == tensor_bytes(observation)
    batch.close()
    expected = play_digest(ASSETS, 'cpu', STEPS, environments=ENVIRONMENTS)
    assert play_digest(ASSETS, 'cpu', STEPS, environments=ENVIRONMENTS, pin_memory=True) == expected


@pytest.mark.skipif(not CUDA, reason='no CUDA device or library')
def test_pinned_host_inputs_may_change_once_the_call_returns():
    pinned = torch.zeros(4, dtype=torch.int64).pin_memory()
    # A GPU batch copies a pinned input to pageable memory, which the copy stages at once.
    assert not pinned.clone().is_pinned()

    def play(changed):
        batch = cuju.BatchEnvironment(ASSETS, environment_count=4, device='cuda', seed=3)
        batch.reset()
        actions = torch.zeros((4, 8), dtype=torch.int32).pin_memory()
        kept = []
        for step in range(12):
            actions.copy_(actions_at(step, 4, 8, batch.action_count))
            if changed:
                # The device reaches the copy of the actions only after this work.
                torch.cuda._sleep(2**25)
            if step % 2:
                batch.step_async(actions)
            else:
                batch.step(actions)
            if changed:
                actions.copy_(actions_at(step + 5, 4, 8, batch.action_count))
            if step % 2:
                batch.step_wait()
            kept.append(batch.observation().clone())
        mask = torch.tensor([1, 0, 1, 0], dtype=torch.uint8).pin_memory()
        seeds = torch.tensor([5, 6, 7, 8], dtype=torch.int64).pin_memory()
        if changed:
            torch.cuda._sleep(2**25)
        batch.reset(mask=mask, seeds=seeds)
        if changed:
            mask.copy_(torch.tensor([0, 1, 0, 1]))
            seeds.fill_(9)
        kept.append(batch.observation().clone())
        torch.cuda.synchronize()
        batch.close()
        return [tensor_bytes(tensor) for tensor in kept]

    assert play(True) == play(False)

# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that `torch.compile` traces a step whole and gives the outputs of an eager step."""

import importlib.util

import pytest
import torch

import cuju
from conftest import CUDA, actions_at, tensor_bytes

STEPS = 40

pytestmark = pytest.mark.skipif(
    not hasattr(torch.library, 'custom_op'), reason='torch.library.custom_op needs PyTorch 2.4'
)


def outputs_bytes(observation, reward, done, information):
    """Returns the bytes of every output of a step."""
    final = information['final_observation']
    return b''.join(
        tensor_bytes(tensor)
        for tensor in (observation, reward, done, final, information['score_reward'])
    )


def step(batch, actions):
    return batch.step(actions)


def reset_done(batch):
    batch.reset_done()


def play(device, step, reset_done, **options):
    """Plays `STEPS` steps through `step` and `reset_done`, and returns the bytes of each step."""
    batch = cuju.BatchEnvironment('5_vs_5_l4_r4', environment_count=3, device=device, **options)
    batch.reset()
    found = []
    for index in range(STEPS):
        actions = actions_at(index, 3, batch.agents, batch.action_count).to(device)
        found.append(outputs_bytes(*step(batch, actions)))
        if not batch.auto_reset:
            reset_done(batch)
    batch.close()
    return found


@pytest.mark.parametrize('auto_reset', [True, False])
def test_a_compiled_step_equals_an_eager_one(device, auto_reset):
    torch._dynamo.reset()
    compiled = (
        torch.compile(step, fullgraph=True, backend='aot_eager'),
        torch.compile(reset_done, fullgraph=True, backend='aot_eager'),
    )
    options = {'auto_reset': auto_reset, 'seed': 11}
    assert play(device, *compiled, **options) == play(device, step, reset_done, **options)


def test_a_squeezed_batch_compiles(device):
    torch._dynamo.reset()

    def run(step):
        batch = cuju.BatchEnvironment(
            'academy_empty_goal_close', environment_count=2, device=device, squeeze=True
        )
        batch.reset()
        found = [
            outputs_bytes(*step(batch, torch.full((2,), index % 19, device=device)))
            for index in range(10)
        ]
        batch.close()
        return found

    assert run(torch.compile(step, fullgraph=True, backend='aot_eager')) == run(step)


@pytest.mark.skipif(not CUDA, reason='no CUDA device or library')
# PyTorch's Windows wheels come without Triton, which inductor's GPU code needs.
@pytest.mark.skipif(importlib.util.find_spec('triton') is None, reason='no Triton')
def test_inductor_captures_the_step_and_a_policy_in_a_cuda_graph():
    torch._dynamo.reset()

    def act(batch, actions):
        observation, reward, done, information = batch.step(actions)
        # A policy's work in the same graph, the next actions from integer arithmetic.
        return observation, reward, done, information, (actions + 1) % batch.action_count

    def run(function):
        batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=4, device='cuda', seed=3)
        batch.reset()
        actions = torch.zeros((4, 4), dtype=torch.int32, device='cuda')
        found = []
        for _ in range(STEPS):
            *outputs, actions = function(batch, actions)
            found.append(outputs_bytes(*outputs))
            actions = actions.clone()
        batch.close()
        return found

    compiled = torch.compile(act, mode='reduce-overhead', fullgraph=True)
    assert run(compiled) == run(act)

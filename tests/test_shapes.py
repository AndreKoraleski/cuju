# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the shapes and types of the outputs for each representation and option."""

import torch

import cuju


def test_simple115v2(device):
    batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=6, device=device)
    observation = batch.reset()
    assert observation.shape == (6, 4, 115) and observation.dtype == torch.float32
    observation, reward, done, information = batch.step(
        torch.zeros((6, 4), dtype=torch.int32, device=device)
    )
    assert reward.shape == (6, 4) and done.shape == (6,)
    assert information['final_observation'].shape == (6, 4, 115)
    assert information['score_reward'].shape == (6,)


def test_extracted_and_stack(device):
    batch = cuju.BatchEnvironment(
        '5_vs_5_l1_r0', environment_count=2, device=device, representation='extracted', stacked=True
    )
    assert batch.reset().shape == (2, 1, 72, 96, 16)


def test_raw(device):
    batch = cuju.BatchEnvironment(
        '5_vs_5_l1_r1', environment_count=3, device=device, representation='raw'
    )
    raw = batch.reset()
    assert raw['left_team'].shape == (3, 2, 5, 2)
    assert raw['score'].dtype == torch.int32


def test_squeeze(device):
    batch = cuju.BatchEnvironment(
        'academy_empty_goal_close', environment_count=5, device=device, squeeze=True
    )
    assert batch.reset().shape == (5, 115)
    _, reward, _, _ = batch.step(torch.zeros(5, dtype=torch.int32, device=device))
    assert reward.shape == (5,)


def test_mixed_batch(device):
    batch = cuju.BatchEnvironment(
        ['5_vs_5_l4_r4', '11_vs_11_stochastic_l1_r0'],
        environment_count=4,
        device=device,
        assignment=torch.tensor([0, 0, 1, 1]),
    )
    assert batch.agents == 8
    assert batch.agent_mask.sum(1).tolist() == [8, 8, 1, 1]

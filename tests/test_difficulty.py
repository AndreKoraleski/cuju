# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that `batch.difficulty` reports the difficulty every environment plays."""

import torch

import cuju


def test_the_scenario_difficulties(device):
    # GRF: the scenario files set these, and a field they leave keeps GRF's 1.0 and 0.6.
    batch = cuju.BatchEnvironment(
        ['5_vs_5_l1_r0', '11_vs_11_easy_stochastic_l1_r0', '1_vs_1_easy_l1_r0'],
        environment_count=3,
        assignment=[0, 1, 2],
        device=device,
    )
    expected = torch.tensor([[0.05, 0.05], [1.0, 0.05], [0.0, 0.0]], dtype=torch.float32)
    assert batch.difficulty.dtype == torch.float32
    assert torch.equal(batch.difficulty, expected)


def test_a_set_difficulty_replaces_only_its_side(device):
    batch = cuju.BatchEnvironment(
        ['11_vs_11_stochastic_l1_r0', '5_vs_5_l1_r0'],
        environment_count=4,
        assignment=[0, 1, 0, 1],
        difficulty=[(None, 0.25), (0.75, None)],
        device=device,
    )
    expected = torch.tensor(
        [[1.0, 0.25], [0.75, 0.05], [1.0, 0.25], [0.75, 0.05]], dtype=torch.float32
    )
    assert torch.equal(batch.difficulty, expected)

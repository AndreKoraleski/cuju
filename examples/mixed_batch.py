# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Runs self-play, play against the built-in AI and an 11-a-side match in one batch.

python examples/mixed_batch.py [--device cuda]
"""

import argparse

import torch

import cuju

BUNDLES = ['5_vs_5_l4_r4', '5_vs_5_l4_r0', '11_vs_11_stochastic_l1_r0']


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--per-bundle', type=int, default=16)
    arguments = parser.parse_args()

    count = arguments.per_bundle * len(BUNDLES)
    assignment = torch.arange(len(BUNDLES)).repeat_interleave(arguments.per_bundle)
    batch = cuju.BatchEnvironment(
        BUNDLES, environment_count=count, device=arguments.device, assignment=assignment
    )
    observation = batch.reset()
    print(f'observation {tuple(observation.shape)}: room for {batch.agents} agents')
    for index, name in enumerate(BUNDLES):
        first = index * arguments.per_bundle
        print(
            f'{name}: {int(batch.left_agents[first])} left and {int(batch.right_agents[first])} '
            f'right agents, {int(batch.left_players[first])} v {int(batch.right_players[first])} '
            'players'
        )

    mask = batch.agent_mask.to(arguments.device)
    total = torch.zeros(count, device=arguments.device)
    for _ in range(200):
        actions = torch.randint(
            0,
            batch.action_count,
            (count, batch.agents),
            dtype=torch.int32,
            device=arguments.device,
        )
        _, reward, _, _ = batch.step(actions)
        # The agents that a bundle lacks have a reward of 0, and the mask keeps them out anyway.
        total += (reward * mask).sum(1)
    for index, name in enumerate(BUNDLES):
        part = total[index * arguments.per_bundle : (index + 1) * arguments.per_bundle]
        print(f'{name}: summed reward {float(part.sum()):+.1f}')


if __name__ == '__main__':
    main()

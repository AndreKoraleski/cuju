# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Plays random actions in a batch and reports the episodes it finished and its speed.

    python examples/random_rollout.py [--device cuda] [--environments 4096] [--steps 1000]

Every agent draws a uniform action from a seeded generator, so a run repeats exactly.
"""

import argparse
import time

import torch

import cuju


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scenario', default='5_vs_5_l4_r0')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--environments', type=int, default=64)
    parser.add_argument('--steps', type=int, default=1000)
    arguments = parser.parse_args()

    batch = cuju.BatchEnvironment(
        arguments.scenario, environment_count=arguments.environments, device=arguments.device
    )
    batch.reset()
    generator = torch.Generator(device=arguments.device).manual_seed(0)
    shape = (batch.environment_count, batch.agents)
    # Counted on the device, so that the loop never waits for it.
    episodes = torch.zeros((), dtype=torch.int64, device=arguments.device)
    goals = torch.zeros((), dtype=torch.int64, device=arguments.device)
    start = time.perf_counter()
    for _ in range(arguments.steps):
        actions = torch.randint(
            0,
            batch.action_count,
            shape,
            dtype=torch.int32,
            device=arguments.device,
            generator=generator,
        )
        _, _, done, information = batch.step(actions)
        episodes += done.sum()
        goals += (information['score_reward'] != 0).sum()
    if arguments.device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    steps = arguments.steps * batch.environment_count
    print(f'{steps} environment steps in {elapsed:.1f} s, {steps / elapsed:.0f} steps/s')
    print(f'{int(episodes)} episodes finished, {int(goals)} goals')


if __name__ == '__main__':
    main()

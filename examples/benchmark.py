# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Measures the environment steps per second of a batch, as docs/performance.md reports them.

    python examples/benchmark.py --scenario 5_vs_5_l4_r4 --device cuda --environments 16384
    python examples/benchmark.py --scenario 5_vs_5_l4_r4 --device cpu --environments 1 --threads 1
    python examples/benchmark.py --grf --scenario 5_vs_5 --left 4 --right 4

Each round drops the first steps, then times the rest; the median of the rounds is reported with
the spread. With `--grf`, it times one GRF environment in this process instead, which needs
`gfootball` installed.
"""

import argparse
import statistics
import time

import torch


def time_cuju(arguments):
    import cuju

    batch = cuju.BatchEnvironment(
        arguments.scenario,
        environment_count=arguments.environments,
        device=arguments.device,
        threads=arguments.threads,
    )
    batch.reset()
    generator = torch.Generator(device=arguments.device).manual_seed(0)
    shape = (batch.environment_count, batch.agents)

    def actions():
        return torch.randint(
            0,
            batch.action_count,
            shape,
            dtype=torch.int32,
            device=arguments.device,
            generator=generator,
        )

    for _ in range(arguments.warmup):
        batch.step(actions())
    if arguments.device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(arguments.steps):
        batch.step(actions())
    if arguments.device == 'cuda':
        torch.cuda.synchronize()
    return arguments.steps * batch.environment_count / (time.perf_counter() - start)


def time_grf(arguments):
    import random

    import gfootball.env as football_env

    random.seed(0)
    env = football_env.create_environment(
        env_name=arguments.scenario,
        representation='simple115v2',
        number_of_left_players_agent_controls=arguments.left,
        number_of_right_players_agent_controls=arguments.right,
    )
    env.reset()
    agents = arguments.left + arguments.right
    rng = random.Random(1)

    def step():
        action = [rng.randrange(19) for _ in range(agents)] if agents > 1 else rng.randrange(19)
        if env.step(action)[2]:
            env.reset()

    for _ in range(arguments.warmup):
        step()
    start = time.perf_counter()
    for _ in range(arguments.steps):
        step()
    return arguments.steps / (time.perf_counter() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scenario', default='5_vs_5_l4_r4')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--environments', type=int, default=64)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--grf', action='store_true', help='time GRF instead of cuju')
    parser.add_argument('--left', type=int, default=1, help='GRF agents on the left')
    parser.add_argument('--right', type=int, default=0, help='GRF agents on the right')
    arguments = parser.parse_args()

    measure = time_grf if arguments.grf else time_cuju
    rates = [measure(arguments) for _ in range(arguments.rounds)]
    median = statistics.median(rates)
    spread = (max(rates) - min(rates)) / median
    print(
        f'{median:.0f} environment steps per second, spread {100 * spread:.1f}% over '
        f'{arguments.rounds} rounds'
    )


if __name__ == '__main__':
    main()

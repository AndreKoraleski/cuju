# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Writes GRF dumps of chosen environments of a batch, reads one back and summarises the match.

    python examples/dumps.py [--device cuda] [--follow 0 1] [--logdir dumps]

The batch plays one full match in every environment with random actions from a seeded generator,
and `cuju.dumps.DumpWriter` follows the environments of `--follow`, writing each finished episode
and each goal as GRF's `write_full_episode_dumps` and `write_goal_dumps` would. The batch writes
the raw arrays with `with_raw=True`, so every step of a dump holds GRF's observation, which GRF's
`dump_to_video` needs to render it. `cuju.dumps.read_dump` then reads the first full episode back,
without GRF, and the script prints its steps, its score, the steps of its goals, the distance the
ball travelled and each team's share of possession.
"""

import argparse
import os

import torch

import cuju
import cuju.dumps


def summarise(path):
    """Prints a summary of the full-episode dump at `path`."""
    steps = cuju.dumps.read_dump(path)
    print(f'{os.path.basename(path)}, {len(steps)} steps')
    final = steps[-1]['observation']
    print(f'  score {final["score"][0]}:{final["score"][1]}')
    goals = [step['debug']['frame_cnt'] for step in steps if step['reward'] != 0]
    print(f'  goals at steps {goals}')
    ball = torch.stack([step['observation']['ball'] for step in steps])
    moves = (ball[1:, :2] - ball[:-1, :2]).norm(dim=1)
    # The ball jumps to the centre spot for the kick-off after a goal, which is no travel.
    travel = moves[moves < 0.2].sum()
    print(f'  the ball travelled {float(travel):.1f} along the ground, on a pitch 2 long')
    owners = torch.tensor([step['observation']['ball_owned_team'] for step in steps])
    owned = (owners >= 0).sum().clamp(min=1)
    print(
        f'  possession {float((owners == 0).sum() / owned):.0%} left, '
        f'{float((owners == 1).sum() / owned):.0%} right'
    )
    print(f'  actions of the first step {steps[0]["debug"]["action"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scenario', default='5_vs_5_l4_r0')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--environments', type=int, default=8)
    parser.add_argument('--follow', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--steps', type=int, default=3001, help='one full match by default')
    parser.add_argument('--logdir', default='dumps')
    parser.add_argument('--seed', type=int, default=0)
    arguments = parser.parse_args()

    generator = torch.Generator(device=arguments.device).manual_seed(arguments.seed)
    with cuju.BatchEnvironment(
        arguments.scenario,
        environment_count=arguments.environments,
        device=arguments.device,
        seed=arguments.seed,
        with_raw=True,
    ) as batch:
        writer = cuju.dumps.DumpWriter(
            batch, arguments.logdir, environments=arguments.follow, write_goal_dumps=True
        )
        batch.reset()
        shape = (batch.environment_count, batch.agents)
        for _ in range(arguments.steps):
            actions = torch.randint(
                0,
                batch.action_count,
                shape,
                dtype=torch.int32,
                device=arguments.device,
                generator=generator,
            )
            batch.step(actions)
            writer.record(actions)
        # An episode still running is written as it stands, as GRF does when it closes.
        writer.close()

    print(f'{len(writer.paths)} dumps in {arguments.logdir}')
    full = [path for path in writer.paths if os.path.basename(path).startswith('episode_done')]
    if full:
        summarise(full[0])


if __name__ == '__main__':
    main()

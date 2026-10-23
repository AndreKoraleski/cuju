# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Plays a Kaggle agent function against the built-in AI, or against itself, in a batch.

    python examples/kaggle_agent.py [--device cuda] [--matches 8] [--self]

`agent` is written as the `agent(obs)` of a Kaggle `submission.py` for the GRF competition. It reads
`obs['players_raw'][0]`, GRF's raw observation of its player, and returns a list of one action, so
the same function runs unchanged on Kaggle. It runs at the ball, carries it toward the right goal
and shoots from close range. `cuju.kaggle.evaluate` plays `--matches` matches of
`11_vs_11_kaggle`, one per environment, each the first episode of its environment as Kaggle plays
it, and the script prints the scores and the totals. The function runs once per agent and step in
Python, as on Kaggle, so a match takes far longer than a step of a policy on the device.
"""

import argparse
import math
import time

import cuju.kaggle

# The directions of GRF's default action set, from the right in steps of 45 degrees toward the
# bottom, since y grows toward the bottom of the pitch.
DIRECTIONS = [5, 6, 7, 8, 1, 2, 3, 4]
RIGHT, SHOT = 5, 12


def direction(dx, dy):
    """Returns the action that runs closest to the direction `(dx, dy)`."""
    return DIRECTIONS[round(math.atan2(dy, dx) / (math.pi / 4)) % 8]


def agent(obs):
    """Returns the action of the player of `obs`, as a Kaggle agent does."""
    raw = obs['players_raw'][0]
    x, y = raw['left_team'][raw['active']]
    ball = raw['ball']
    if raw['ball_owned_team'] == 0 and raw['ball_owned_player'] == raw['active']:
        if x > 0.7 and abs(y) < 0.25:
            return [SHOT]
        return [direction(1.0 - x, -y)]
    return [direction(ball[0] - x, ball[1] - y)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--matches', type=int, default=4)
    parser.add_argument('--self', action='store_true', help='play the agent on both sides')
    parser.add_argument('--seed', type=int, default=0)
    arguments = parser.parse_args()

    start = time.perf_counter()
    scores = cuju.kaggle.evaluate(
        agent,
        agent if arguments.self else None,
        episodes=arguments.matches,
        seed=arguments.seed,
        device=arguments.device,
    )
    elapsed = time.perf_counter() - start
    opponent = 'itself' if arguments.self else 'the built-in AI'
    print(f'the agent against {opponent} in {arguments.matches} matches, {elapsed:.0f} s')
    for match, (own, other) in enumerate(scores):
        print(f'  match {match}: {own}:{other}')
    wins = sum(own > other for own, other in scores)
    draws = sum(own == other for own, other in scores)
    print(f'wins {wins}, draws {draws}, losses {len(scores) - wins - draws}')
    print(f'goals for {sum(own for own, _ in scores)}, against {sum(other for _, other in scores)}')


if __name__ == '__main__':
    main()

# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Plays fixed matches of a policy against the built-in AI and reports its results.

    python examples/evaluate.py --weights ppo.pt [--device cuda] [--matches 1000]
    python examples/evaluate.py --random [--scenario academy_empty_goal_close]

The policy is the network of `examples/ppo.py`, loaded from its saved weights with
`torch.load(path, weights_only=True)`, and `--random` replaces it with uniform actions as a
baseline. Every environment plays exactly one episode, and `--seed` fixes the matches, so two
policies evaluated with the same seed meet the same opponents in the same situations. Each agent
samples from the policy with a seeded generator, and `--greedy` takes the most likely action
instead.

A match counts as a win when the left team ends it ahead, a draw when level and a loss when behind.
The win rate comes with the Wilson score interval at 95%, which stays between 0 and 1 for any
number of matches.
"""

import argparse
import math
import time

import torch
from ppo import Agent

import cuju


def wilson(successes, trials, z=1.96):
    """Returns the Wilson score interval of a proportion."""
    if trials == 0:
        return 0.0, 1.0
    rate = successes / trials
    scale = 1 + z * z / trials
    centre = (rate + z * z / (2 * trials)) / scale
    half = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / scale
    return centre - half, centre + half


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scenario', default='academy_empty_goal_close')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--weights', default='ppo.pt')
    parser.add_argument('--random', action='store_true', help='play uniform random actions')
    parser.add_argument('--greedy', action='store_true', help='take the most likely action')
    parser.add_argument('--matches', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=12345)
    arguments = parser.parse_args()

    device = torch.device(arguments.device)
    generator = torch.Generator(device=device).manual_seed(arguments.seed)
    with cuju.BatchEnvironment(
        arguments.scenario,
        environment_count=arguments.matches,
        device=arguments.device,
        seed=arguments.seed,
        auto_reset=False,
    ) as batch:
        count, agents = batch.environment_count, batch.agents
        if arguments.random:
            model = None
        else:
            model = Agent(action_count=batch.action_count).to(device)
            weights = torch.load(arguments.weights, map_location=device, weights_only=True)
            model.load_state_dict(weights)
            model.eval()

        actions = torch.zeros((count, agents), dtype=torch.int32, device=device)
        finished = torch.zeros(count, dtype=torch.bool, device=device)
        goals_for = torch.zeros(count, dtype=torch.int64, device=device)
        goals_against = torch.zeros(count, dtype=torch.int64, device=device)
        length = torch.zeros(count, dtype=torch.int64, device=device)
        observation = batch.reset()
        start = time.perf_counter()
        while not bool(finished.all()):
            if model is None:
                actions.random_(0, batch.action_count, generator=generator)
            else:
                with torch.no_grad():
                    logits = model.actor(observation)
                if arguments.greedy:
                    actions.copy_(logits.argmax(-1))
                else:
                    probabilities = logits.softmax(-1).reshape(count * agents, -1)
                    sample = torch.multinomial(probabilities, 1, generator=generator)
                    actions.copy_(sample.reshape(count, agents))
            observation, _, done, information = batch.step(actions)
            # An environment that has ended idles until the last one ends, and is left out.
            live = ~finished
            score = information['score_reward']
            goals_for += (score == 1) & live
            goals_against += (score == -1) & live
            length += live
            finished |= done.bool()
        elapsed = time.perf_counter() - start

    wins = int((goals_for > goals_against).sum())
    draws = int((goals_for == goals_against).sum())
    losses = int((goals_for < goals_against).sum())
    low, high = wilson(wins, count)
    policy = 'random actions' if model is None else arguments.weights
    print(f'{policy} in {count} matches of {arguments.scenario}, seed {arguments.seed}')
    print(f'wins {wins}, draws {draws}, losses {losses}')
    print(
        f'goals for {int(goals_for.sum())}, against {int(goals_against.sum())}, '
        f'{float(goals_for.sum()) / count:.3f} and {float(goals_against.sum()) / count:.3f} '
        'per match'
    )
    print(f'win rate {wins / count:.3f}, 95% interval {low:.3f} to {high:.3f}')
    print(f'mean length {float(length.sum()) / count:.1f} steps, played in {elapsed:.1f} s')


if __name__ == '__main__':
    main()

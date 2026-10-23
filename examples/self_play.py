# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Trains a policy against a frozen copy of itself, with agents on both sides of every match.

    python examples/self_play.py [--device cuda] [--scenario 5_vs_5_l4_r4] [--refresh 10]

The left agents act from the learning policy, the network of `examples/ppo.py`, and the right
agents from a frozen copy that takes the learner's weights every `--refresh` updates. Each side
sees the pitch from its own end, so one network plays either side. Only the learner's steps enter
the update, which is the clipped objective of `examples/ppo.py`. The checkpoint rewards are on by
default, since a full match yields few goals early in training. `--weights` starts both policies
from weights that `examples/ppo.py` or an earlier run saved.

Every update prints the steps per second and the goals of the learner and of its frozen opponent.
"""

import argparse
import copy
import time

import torch
from ppo import Agent, add_arguments, advantages, flatten, update

import cuju


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scenario', default='5_vs_5_l4_r4')
    parser.add_argument('--rewards', default='scoring,checkpoints')
    parser.add_argument('--total-steps', type=int, default=5_000_000, help='environment steps')
    parser.add_argument('--refresh', type=int, default=10, help='updates between snapshots')
    parser.add_argument('--weights', help='weights to start from')
    parser.add_argument('--save', default='self_play.pt', help='where the weights go')
    add_arguments(parser)
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    device = torch.device(arguments.device)
    with cuju.BatchEnvironment(
        arguments.scenario,
        environment_count=arguments.environments,
        device=arguments.device,
        rewards=arguments.rewards,
        seed=arguments.seed,
    ) as batch:
        count, agents, steps = batch.environment_count, batch.agents, arguments.rollout
        model = Agent(action_count=batch.action_count).to(device)
        if arguments.weights:
            weights = torch.load(arguments.weights, map_location=device, weights_only=True)
            model.load_state_dict(weights)
        frozen = copy.deepcopy(model).requires_grad_(False)
        optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate, eps=1e-5)

        # The first right agent sits at another slot in each bundle of a mixed batch.
        slot = torch.arange(agents, device=device)
        mask = batch.agent_mask.to(device)
        is_right = (slot >= batch.left_agents.to(device)[:, None]) & mask
        learner = mask & ~is_right
        if not bool(is_right.any()) or not bool(learner.any()):
            parser.error(f'{arguments.scenario} needs agents on both sides, such as 5_vs_5_l4_r4')

        observations = torch.zeros((steps, count, agents, 115), device=device)
        actions = torch.zeros((steps, count, agents), dtype=torch.int64, device=device)
        log_probs = torch.zeros((steps, count, agents), device=device)
        values = torch.zeros((steps, count, agents), device=device)
        rewards = torch.zeros((steps, count, agents), device=device)
        dones = torch.zeros((steps, count), dtype=torch.uint8, device=device)
        step_actions = torch.zeros((count, agents), dtype=torch.int32, device=device)
        goals_for = torch.zeros((), dtype=torch.int64, device=device)
        goals_against = torch.zeros((), dtype=torch.int64, device=device)

        updates = max(arguments.total_steps // (count * steps), 1)
        print(
            f'{arguments.scenario} in {count} environments, {updates} updates of '
            f'{count * steps:,} environment steps, a new opponent every {arguments.refresh}'
        )
        observation = batch.reset()
        start = time.perf_counter()
        for index in range(updates):
            if index and index % arguments.refresh == 0:
                frozen.load_state_dict(model.state_dict())
            for group in optimizer.param_groups:
                group['lr'] = arguments.learning_rate * (1 - index / updates)
            for t in range(steps):
                observations[t].copy_(observation)
                with torch.no_grad():
                    action, log_prob, _, value = model(observation)
                    opponent, _, _, _ = frozen(observation)
                actions[t], log_probs[t], values[t] = action, log_prob, value
                step_actions.copy_(torch.where(is_right, opponent, action))
                observation, reward, done, information = batch.step(step_actions)
                rewards[t].copy_(reward)
                dones[t].copy_(done)
                goals_for += (information['score_reward'] == 1).sum()
                goals_against += (information['score_reward'] == -1).sum()

            with torch.no_grad():
                next_value = model.value(observation)
            advantage, returns = advantages(
                rewards, values, dones, next_value, arguments.gamma, arguments.gae_lambda
            )
            rollout = flatten(
                {
                    'observations': observations,
                    'actions': actions,
                    'log_probs': log_probs,
                    'values': values,
                    'advantages': advantage,
                    'returns': returns,
                },
                learner,
            )
            _, value_loss, entropy = update(model, optimizer, rollout, arguments)

            done_steps = (index + 1) * count * steps
            rate = done_steps / (time.perf_counter() - start)
            print(
                f'update {index + 1}/{updates}  steps {done_steps:,}  {rate:,.0f} steps/s  '
                f'goals {int(goals_for)} for, {int(goals_against)} against  '
                f'entropy {entropy:.2f}  value loss {value_loss:.4f}'
            )
            goals_for.zero_()
            goals_against.zero_()

    torch.save(model.state_dict(), arguments.save)
    print(f'saved the weights to {arguments.save}')


if __name__ == '__main__':
    main()

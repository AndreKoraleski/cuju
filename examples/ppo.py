# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Trains one policy shared by every agent of a batch with proximal policy optimization.

    python examples/ppo.py [--device cuda] [--scenario academy_empty_goal_close] [--total-steps N]

The trainer follows the single-file style of CleanRL, in plain PyTorch. Every agent of the batch
acts from the same network, which tells the agents apart by the active player in their observation.
Each update plays a rollout of `--rollout` steps, estimates advantages with generalized advantage
estimation, and takes several epochs of the clipped objective and the clipped value loss over
shuffled minibatches. The observations of `simple115v2` already lie in small ranges, so they enter
the network as they are.

Every update prints the environment steps per second, the mean return of the episodes that ended
and their goals per episode. The weights are saved with `torch.save(model.state_dict(), path)`, and
`examples/evaluate.py` and `examples/self_play.py` load them.
"""

import argparse
import math
import time

import torch

import cuju

# The gain of the hidden layers in CleanRL's orthogonal initialization.
ROOT_TWO = math.sqrt(2)


class Agent(torch.nn.Module):
    """An actor and a critic, two separate perceptrons over one agent's observation."""

    def __init__(self, observation_size=115, action_count=19, hidden=256):
        super().__init__()

        def layer(inputs, outputs, gain=ROOT_TWO):
            linear = torch.nn.Linear(inputs, outputs)
            torch.nn.init.orthogonal_(linear.weight, gain)
            torch.nn.init.zeros_(linear.bias)
            return linear

        self.critic = torch.nn.Sequential(
            layer(observation_size, hidden),
            torch.nn.Tanh(),
            layer(hidden, hidden),
            torch.nn.Tanh(),
            layer(hidden, 1, gain=1.0),
        )
        # A small last layer starts the policy close to uniform.
        self.actor = torch.nn.Sequential(
            layer(observation_size, hidden),
            torch.nn.Tanh(),
            layer(hidden, hidden),
            torch.nn.Tanh(),
            layer(hidden, action_count, gain=0.01),
        )

    def value(self, observation):
        return self.critic(observation).squeeze(-1)

    def forward(self, observation, action=None):
        """Returns an action, its log probability, the entropy and the value, for any batch shape.

        Args:
            observation: Observations `[..., observation_size]`.
            action: The actions to score, or `None` to sample them.
        """
        distribution = torch.distributions.Categorical(logits=self.actor(observation))
        if action is None:
            action = distribution.sample()
        return (
            action,
            distribution.log_prob(action),
            distribution.entropy(),
            self.value(observation),
        )


def advantages(rewards, values, dones, next_value, gamma, lambda_):
    """Returns the generalized advantage estimates of a rollout and the returns they imply.

    Args:
        rewards: `[steps, environments, agents]`.
        values: The values of the observations the steps acted on, as `rewards`.
        dones: `[steps, environments]`, the episodes that each step ended, whose values stop there.
        next_value: The value of the observation after the last step, `[environments, agents]`.
        gamma: The discount.
        lambda_: The weight of longer estimates.
    """
    # The end of an episode counts as terminal, as most work on GRF treats it.
    carry = 1.0 - dones.float().unsqueeze(-1)
    advantage = torch.zeros_like(rewards)
    last = torch.zeros_like(next_value)
    for t in reversed(range(len(rewards))):
        following = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * carry[t] * following - values[t]
        last = delta + gamma * lambda_ * carry[t] * last
        advantage[t] = last
    return advantage, advantage + values


def update(model, optimizer, rollout, arguments):
    """Takes the epochs of the clipped objective over a flattened rollout.

    Args:
        model: The `Agent`.
        optimizer: Its optimizer.
        rollout: A dictionary of `observations`, `actions`, `log_probs`, `values`, `advantages`
            and `returns`, one row per agent step.
        arguments: The parsed arguments with the hyperparameters.

    Returns:
        The policy loss, the value loss and the entropy of the last minibatch.
    """
    size = len(rollout['observations'])
    minibatch = size // arguments.minibatches
    clip = arguments.clip
    for _ in range(arguments.epochs):
        order = torch.randperm(size, device=rollout['observations'].device)
        for start in range(0, minibatch * arguments.minibatches, minibatch):
            index = order[start : start + minibatch]
            _, log_prob, entropy, value = model(
                rollout['observations'][index], rollout['actions'][index]
            )
            ratio = (log_prob - rollout['log_probs'][index]).exp()
            advantage = rollout['advantages'][index]
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            policy_loss = torch.max(
                -advantage * ratio, -advantage * ratio.clamp(1 - clip, 1 + clip)
            ).mean()
            old_value, target = rollout['values'][index], rollout['returns'][index]
            clipped = old_value + (value - old_value).clamp(-clip, clip)
            value_loss = 0.5 * torch.max((value - target).pow(2), (clipped - target).pow(2)).mean()
            loss = (
                policy_loss
                + arguments.value_coefficient * value_loss
                - arguments.entropy_coefficient * entropy.mean()
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), arguments.max_gradient_norm)
            optimizer.step()
    return float(policy_loss.detach()), float(value_loss.detach()), float(entropy.detach().mean())


def add_arguments(parser):
    """Adds the hyperparameters of PPO, which `examples/self_play.py` shares."""
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--environments', type=int, default=512)
    parser.add_argument('--rollout', type=int, default=128, help='steps per update')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--minibatches', type=int, default=4)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--entropy-coefficient', type=float, default=0.01)
    parser.add_argument('--value-coefficient', type=float, default=0.5)
    parser.add_argument('--max-gradient-norm', type=float, default=0.5)


def flatten(rollout, valid):
    """Returns the tensors `[steps, environments, agents, ...]` of a rollout, a row per agent step.

    Args:
        rollout: A dictionary of tensors.
        valid: `[environments, agents]`, the agents whose rows to keep.
    """
    steps = len(next(iter(rollout.values())))
    count, agents = valid.shape
    keep = valid.reshape(-1)
    return {
        name: tensor.reshape(steps, count * agents, *tensor.shape[3:])[:, keep].flatten(0, 1)
        for name, tensor in rollout.items()
    }


def train(batch, arguments):
    """Trains an `Agent` on every agent of `batch` and saves its weights."""
    device = batch.torch_device
    count, agents, steps = batch.environment_count, batch.agents, arguments.rollout
    model = Agent(action_count=batch.action_count).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate, eps=1e-5)
    mask = batch.agent_mask.to(device)
    team_size = mask.sum(1)

    observations = torch.zeros((steps, count, agents, 115), device=device)
    actions = torch.zeros((steps, count, agents), dtype=torch.int64, device=device)
    log_probs = torch.zeros((steps, count, agents), device=device)
    values = torch.zeros((steps, count, agents), device=device)
    rewards = torch.zeros((steps, count, agents), device=device)
    dones = torch.zeros((steps, count), dtype=torch.uint8, device=device)
    step_actions = torch.zeros((count, agents), dtype=torch.int32, device=device)
    # Counted on the device, so that the rollout never waits for them.
    episode_return = torch.zeros(count, device=device)
    returns_sum = torch.zeros((), device=device)
    episodes = torch.zeros((), dtype=torch.int64, device=device)
    goals = torch.zeros((), dtype=torch.int64, device=device)

    updates = max(arguments.total_steps // (count * steps), 1)
    print(
        f'{count} environments of {arguments.scenario}, {agents} agent{"s" * (agents > 1)} each, '
        f'{updates} updates of {count * steps:,} environment steps'
    )
    observation = batch.reset()
    start = time.perf_counter()
    for index in range(updates):
        for group in optimizer.param_groups:
            group['lr'] = arguments.learning_rate * (1 - index / updates)
        for t in range(steps):
            observations[t].copy_(observation)
            with torch.no_grad():
                action, log_prob, _, value = model(observation)
            actions[t], log_probs[t], values[t] = action, log_prob, value
            step_actions.copy_(action)
            observation, reward, done, information = batch.step(step_actions)
            rewards[t].copy_(reward)
            dones[t].copy_(done)
            episode_return += (reward * mask).sum(1) / team_size
            returns_sum += (episode_return * done).sum()
            episodes += done.sum()
            goals += (information['score_reward'] == 1).sum()
            episode_return *= 1 - done

        with torch.no_grad():
            next_value = model.value(observation)
        advantage, returns = advantages(
            rewards, values, dones, next_value, arguments.gamma, arguments.gae_lambda
        )
        # Rows of absent agents, in a batch that mixes bundles, are left out.
        rollout = flatten(
            {
                'observations': observations,
                'actions': actions,
                'log_probs': log_probs,
                'values': values,
                'advantages': advantage,
                'returns': returns,
            },
            mask,
        )
        _, value_loss, entropy = update(model, optimizer, rollout, arguments)

        done_steps = (index + 1) * count * steps
        rate = done_steps / (time.perf_counter() - start)
        line = f'update {index + 1}/{updates}  steps {done_steps:,}  {rate:,.0f} steps/s'
        finished = int(episodes)
        if finished:
            line += (
                f'  return {float(returns_sum) / finished:+.3f}  '
                f'goals per episode {int(goals) / finished:.3f}  episodes {finished}'
            )
        print(f'{line}  entropy {entropy:.2f}  value loss {value_loss:.4f}')
        returns_sum.zero_()
        episodes.zero_()
        goals.zero_()

    torch.save(model.state_dict(), arguments.save)
    print(f'saved the weights to {arguments.save}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scenario', default='academy_empty_goal_close')
    parser.add_argument('--rewards', default='scoring', choices=['scoring', 'scoring,checkpoints'])
    parser.add_argument('--total-steps', type=int, default=2_000_000, help='environment steps')
    parser.add_argument('--save', default='ppo.pt', help='where the weights go')
    add_arguments(parser)
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    with cuju.BatchEnvironment(
        arguments.scenario,
        environment_count=arguments.environments,
        device=arguments.device,
        rewards=arguments.rewards,
        seed=arguments.seed,
    ) as batch:
        train(batch, arguments)


if __name__ == '__main__':
    main()

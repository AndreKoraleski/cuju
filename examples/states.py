# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Saves a match at a chance to shoot and branches it into many environments to compare choices.

    python examples/states.py [--device cuda] [--line 0.75] [--branches 32] [--horizon 50]

A search batch runs every agent toward the right goal, in random directions among top right, right
and bottom right, so that whoever has the ball carries it without a kick, until in one of its
matches a left agent holds the ball past `--line`, a position along the pitch from -1 at the left
goal to 1 at the right. `get_state` saves
that match, and `set_state` loads it into every environment of a second batch. In each group of
`--branches` environments the agent on the ball holds one opening action, such as a shot or a pass,
for `--opening` steps, and all groups otherwise play the same random actions, one sequence per
branch, so the openings alone differ between groups. The script reports what each opening led to
over `--horizon` steps, the goals for and against, the ball's mean position along the pitch, and
the share of branches with the ball still in the left team's possession.

One extra environment repeats the actions of the first branch, and the two end in the same
observation, bit for bit, since a state carries the whole match. The same pattern serves
counterfactual evaluation, where only the decision under study changes, and curriculum starts,
where training begins from chosen moments.
"""

import argparse

import torch

import cuju

# The indices of these actions in GRF's default action set.
OPENINGS = {'idle': 0, 'run': 5, 'long_pass': 9, 'high_pass': 10, 'short_pass': 11, 'shot': 12}
FORWARD = (4, 5, 6)


def find_chance(batch, line, search, generator):
    """Runs the agents forward until a left agent holds the ball past `line`.

    Returns:
        The environment, the number of steps played and the `[agents]` mask of the agent on the
        ball.
    """
    device = batch.torch_device
    shape = (batch.environment_count, batch.agents)
    left = torch.arange(batch.agents, device=device) < batch.left_agents.to(device)[:, None]
    forward = torch.tensor(FORWARD, dtype=torch.int32, device=device)
    for step in range(search):
        choice = torch.randint(0, len(FORWARD), shape, device=device, generator=generator)
        batch.step(forward[choice])
        raw = batch.raw()
        holder = left & (raw['active'] == raw['ball_owned_player'][:, :1])
        chance = (
            (raw['game_mode'][:, 0] == 0)
            & (raw['ball_owned_team'][:, 0] == 0)
            & (raw['ball'][:, 0, 0] > line)
            & holder.any(1)
        )
        found = chance.nonzero()
        if len(found):
            index = int(found[0])
            return index, step + 1, holder[index]
    raise RuntimeError(f'no chance in {search} steps, try a larger --search or a smaller --line')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scenario', default='5_vs_5_l4_r0')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--line', type=float, default=0.75)
    parser.add_argument('--search', type=int, default=3000, help='steps to look for a chance')
    parser.add_argument('--branches', type=int, default=32, help='environments per opening')
    parser.add_argument('--opening', type=int, default=3, help='steps to hold the opening')
    parser.add_argument('--horizon', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    arguments = parser.parse_args()

    generator = torch.Generator(device=arguments.device).manual_seed(arguments.seed)
    options = {'device': arguments.device, 'with_raw': True, 'seed': arguments.seed}
    with cuju.BatchEnvironment(arguments.scenario, environment_count=64, **options) as search:
        search.reset()
        index, step, holder = find_chance(search, arguments.line, arguments.search, generator)
        ball = search.raw()['ball'][index, 0].tolist()
        state = search.get_state(index)
    print(
        f'environment {index} has the ball at x {ball[0]:+.2f}, y {ball[1]:+.2f} after {step} '
        f'steps, a state of {len(state):,} bytes'
    )

    groups, branches = len(OPENINGS), arguments.branches
    count = groups * branches + 1
    with cuju.BatchEnvironment(arguments.scenario, environment_count=count, **options) as batch:
        for environment in range(count):
            batch.set_state(environment, state)
        device, agents = batch.torch_device, batch.agents
        opening = torch.tensor(list(OPENINGS.values()), dtype=torch.int32, device=device)
        opening = torch.cat([opening.repeat_interleave(branches), opening[:1]])[:, None]
        live = torch.ones(count, dtype=torch.bool, device=device)
        goals_for = torch.zeros(count, dtype=torch.int64, device=device)
        goals_against = torch.zeros(count, dtype=torch.int64, device=device)
        ball_x = torch.zeros(count, device=device)
        owned = torch.zeros(count, dtype=torch.bool, device=device)
        for t in range(arguments.horizon):
            follow = torch.randint(
                0,
                batch.action_count,
                (branches, agents),
                dtype=torch.int32,
                device=device,
                generator=generator,
            )
            # Every group, and the extra environment after them, takes the same draws.
            actions = torch.cat([follow.repeat(groups, 1), follow[:1]])
            if t < arguments.opening:
                actions = torch.where(holder, opening, actions)
            observation, _, done, information = batch.step(actions)
            raw = batch.raw()
            score = information['score_reward']
            goals_for += (score == 1) & live
            goals_against += (score == -1) & live
            # An ended episode restarts at a kick-off, so each branch keeps its last moment.
            live &= ~done.bool()
            ball_x = torch.where(live, raw['ball'][:, 0, 0], ball_x)
            owned = torch.where(live, raw['ball_owned_team'][:, 0] == 0, owned)

        same = torch.equal(observation[0], observation[-1])
        print(f'branch 0 and its repeat end in the same observation: {same}')
        print(f'{"opening":<12}{"goals for":>10}{"against":>9}{"ball x":>8}{"possession":>12}')
        for group, opening_name in enumerate(OPENINGS):
            part = slice(group * branches, (group + 1) * branches)
            print(
                f'{opening_name:<12}{float(goals_for[part].float().mean()):>10.3f}'
                f'{float(goals_against[part].float().mean()):>9.3f}'
                f'{float(ball_x[part].mean()):>+8.3f}'
                f'{float(owned[part].float().mean()):>12.2f}'
            )


if __name__ == '__main__':
    main()

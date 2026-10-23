# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Plays the same matches in cuju and in your own GRF and reports the first step where they differ.

    python examples/compare_with_grf.py --scenario 5_vs_5 --left 4 --right 0 --steps 3000
    python examples/compare_with_grf.py --scenario 11_vs_11_kaggle --left 1 --right 1 \\
        --representation raw --agent submission.py:agent

It needs `gfootball` installed. Two routes are compared:

- by seed: environment `index` of a cuju batch with seed `seed` plays the episodes of a GRF
    environment created after `random.seed(seed * 2**32 + index)`;
- by state: a state that GRF saves is loaded into cuju, and both continue with the same actions
    (in this direction GRF's loader cannot spoil the comparison; see docs/states.md).

At every step it compares, bit for bit, the observation of every agent in `--representation`, the
rewards, the done flag and the match within the state, which is the bytes of `get_state` after
GRF's pickle, since the pickle differs legitimately between the two (docs/states.md). The match
leaves out the three bytes of padding of each angle, which GRF never sets and saves as they are,
so two GRF environments of the same match differ there too. A difference is expected only with a
GRF built differently from the reference platform (docs/exactness.md).

The actions are random, drawn from `--seed`, unless `--agent module:function` names a policy, such
as a Kaggle submission. The policy is called once per agent and step with that agent's observation
in GRF's types, which for `raw` is Kaggle's form `{'players_raw': [raw], 'controlled_players': 1}`
and otherwise the agent's array. It returns an action index, or a list whose first entry is one,
as a Kaggle agent does. Each agent of each engine loads its own copy of the module, which draws
from Python's and NumPy's global generators with states of its own, so the copies stay in step and
GRF's seeds, which come from Python's `random`, stay the same. The script compares the actions
that the two copies of each agent pick.
"""

import argparse
import ctypes
import functools
import importlib.util
import os
import random
import sys

import numpy as np
import torch

import cuju


def grf_environment(arguments):
    import gfootball.env as football_env

    return football_env.create_environment(
        env_name=arguments.scenario,
        representation=arguments.representation,
        rewards=arguments.rewards,
        number_of_left_players_agent_controls=arguments.left,
        number_of_right_players_agent_controls=arguments.right,
    )


def first_observation(reset):
    # GRF on gym returns the observation, and a GRF ported to gymnasium returns it with a dict.
    return reset[0] if isinstance(reset, tuple) else reset


def match_bytes(state):
    """Returns the serialized match of a state, its bytes after the length and the pickle."""
    return state[4 + int.from_bytes(state[:4], 'little', signed=True) :]


@functools.cache
def padding_function():
    """Returns the CPU library's `cuju_state_padding`, which marks match bytes GRF never sets."""
    function = ctypes.CDLL(cuju.library_path('cpu')).cuju_state_padding
    function.restype = ctypes.c_int
    function.argtypes = [ctypes.c_char_p, ctypes.c_ulonglong, ctypes.c_void_p]
    return function


def state_difference(ours, theirs, padding=None):
    """Returns the first match byte where two states differ outside GRF's padding, or `None`."""
    ours, theirs = match_bytes(ours), match_bytes(theirs)
    if len(ours) != len(theirs):
        return min(len(ours), len(theirs))
    differ = np.frombuffer(ours, np.uint8) != np.frombuffer(theirs, np.uint8)
    if differ.any():
        mask = np.zeros(len(theirs), np.uint8)
        padding = padding or padding_function()
        if padding(theirs, len(theirs), mask.ctypes.data) != 0:
            raise ValueError('expected the bytes of a match in the state of GRF')
        differ &= mask == 0
    found = np.flatnonzero(differ)
    return int(found[0]) if found.size else None


def same(ours, theirs):
    """Returns whether two values hold the same numbers bit for bit, as GRF's types hold them.

    A `float32` of cuju widens exactly to the `float64` that GRF keeps of the engine's `float`.
    """
    ours, theirs = np.asarray(ours), np.asarray(theirs)
    if ours.shape != theirs.shape:
        return False
    floating = ours.dtype.kind == 'f', theirs.dtype.kind == 'f'
    if any(floating):
        return all(floating) and np.array_equal(
            ours.astype(np.float64).view(np.uint64), theirs.astype(np.float64).view(np.uint64)
        )
    return np.array_equal(ours, theirs)


def raw_of_agent(batch, raw, index, agent, left):
    """Returns the raw observation of one agent of environment `index` as GRF's dictionary."""
    own, other = int(batch.left_players[index]), int(batch.right_players[index])
    if agent >= left:
        # A right agent sees its own team as the left team.
        own, other = other, own
    result = {}
    for name, tensor in raw.items():
        value = tensor[index, agent].numpy()
        if name.startswith('left_team'):
            value = value[:own]
        elif name.startswith('right_team'):
            value = value[:other]
        if value.dtype == np.float32:
            value = value.astype(np.float64)
        elif name == 'score':
            value = [int(goals) for goals in value]
        elif name.endswith('_roles'):
            value = value.astype(np.int64)
        elif value.ndim == 0:
            value = int(value)
        result[name] = value
    return result


def cuju_agents(batch, observation, index, left, agents):
    """Returns the observation of each agent of environment `index`, as GRF gives it."""
    if isinstance(observation, dict):
        return [raw_of_agent(batch, observation, index, agent, left) for agent in range(agents)]
    return [observation[index, agent].numpy() for agent in range(agents)]


def grf_agents(observation, agents):
    """Returns the observation of each agent of a GRF environment."""
    if isinstance(observation, list):
        return observation
    observation = np.asarray(observation)
    if agents == 1:
        return [observation]
    return list(observation)


def difference(ours, theirs):
    """Returns what differs between the observations of each agent, or `None`."""
    for agent, (mine, grf) in enumerate(zip(ours, theirs, strict=True)):
        if isinstance(grf, dict):
            if sorted(mine) != sorted(grf):
                return f'the raw arrays of agent {agent}: {sorted(set(mine) ^ set(grf))}'
            for name in sorted(grf):
                if not same(mine[name], grf[name]):
                    return f'the observation of agent {agent} (raw `{name}`)'
        elif not same(mine, grf):
            return f'the observation of agent {agent}'
    return None


class Policy:
    """One copy of the policy of `--agent`, with generator states of its own."""

    def __init__(self, specification, copy, seed):
        location, _, name = specification.rpartition(':')
        if not location or not name:
            raise SystemExit(f'expected --agent module:function, found {specification!r}')
        if location.endswith('.py') or os.sep in location or '/' in location:
            path = location
        else:
            found = importlib.util.find_spec(location)
            if found is None or found.origin is None:
                raise SystemExit(f'found no module {location!r} for --agent')
            path = found.origin
        directory = os.path.dirname(os.path.abspath(path))
        if directory not in sys.path:
            # A submission imports its own modules from beside it.
            sys.path.insert(0, directory)
        spec = importlib.util.spec_from_file_location(f'compared_policy_{copy}', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.function = getattr(module, name)
        self.python = random.Random(seed).getstate()
        self.numpy = np.random.RandomState(seed % 2**32).get_state()

    def __call__(self, observation):
        python, numpy_state = random.getstate(), np.random.get_state()
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        try:
            action = self.function(observation)
        finally:
            self.python, self.numpy = random.getstate(), np.random.get_state()
            random.setstate(python)
            np.random.set_state(numpy_state)
        action = getattr(action, 'value', action)
        if isinstance(action, (list, tuple, np.ndarray)):
            action = getattr(action[0], 'value', action[0])
        return int(action)


class Actions:
    """Picks the actions of each step, random ones from the seed or those of the policies."""

    def __init__(self, arguments, agents, action_count):
        self.arguments = arguments
        self.agents = agents
        choices = random.Random(arguments.seed + 1)
        self.random = [
            [choices.randrange(action_count) for _ in range(agents)] for _ in range(arguments.steps)
        ]
        self.policies = None

    def start(self):
        """Gives every agent of both engines a fresh copy of the policy."""
        if self.arguments.agent:
            self.policies = [
                [
                    Policy(self.arguments.agent, f'{side}_{agent}', self.arguments.seed + agent)
                    for agent in range(self.agents)
                ]
                for side in ('grf', 'cuju')
            ]

    def pick(self, step, theirs, ours=None):
        """Returns GRF's and cuju's actions of a step, from each engine's observations."""
        if self.policies is None:
            return self.random[step], self.random[step]
        grf_actions = zip(self.policies[0], theirs, strict=True)
        grf = [policy(_agent_view(view)) for policy, view in grf_actions]
        if ours is None:
            return grf, None
        cuju_actions = zip(self.policies[1], ours, strict=True)
        return grf, [policy(_agent_view(view)) for policy, view in cuju_actions]


def _agent_view(observation):
    """Returns what a policy sees of one agent's observation, Kaggle's form for `raw`."""
    if isinstance(observation, dict):
        return {'players_raw': [observation], 'controlled_players': 1}
    return observation


def grf_step(env, actions, agents):
    """Steps GRF and returns its observation, reward and done flag."""
    result = env.step(actions if agents > 1 else actions[0])
    # Gymnasium splits GRF's done flag into terminated and truncated.
    done = bool(result[2]) or (len(result) == 5 and bool(result[3]))
    return result[0], np.asarray(result[1], dtype=np.float32).reshape(agents), done


def play(arguments, batch, index, env, actions, grf_observation, start_step=0, one_episode=False):
    """Steps environment `index` of `batch` and `env` with the same actions and compares them.

    Returns the number of steps compared, or -1 at the first difference.
    """
    left, agents = arguments.left, arguments.left + arguments.right
    rows = torch.zeros((batch.environment_count, batch.agents), dtype=torch.int32)
    ours = batch.observation()
    for step in range(start_step, arguments.steps):
        theirs = grf_agents(grf_observation, agents)
        grf_actions, cuju_actions = actions.pick(
            step, theirs, cuju_agents(batch, ours, index, left, agents)
        )
        if grf_actions != cuju_actions:
            print(f'the actions differ at step {step}: GRF {grf_actions}, cuju {cuju_actions}')
            return -1
        rows[index, :agents] = torch.tensor(cuju_actions, dtype=torch.int32)
        ours, reward, done, information = batch.step(rows)
        grf_observation, grf_reward, grf_done = grf_step(env, grf_actions, agents)
        ended = bool(done[index])
        # At the end of an episode the batch has started the next one already.
        final = information['final_observation'] if ended else ours
        found = difference(
            cuju_agents(batch, final, index, left, agents), grf_agents(grf_observation, agents)
        )
        if found is None and not same(reward[index, :agents].numpy(), grf_reward):
            found = 'the reward'
        if found is None and ended != grf_done:
            found = 'the done flag'
        if found is None and ended:
            if one_episode:
                return step + 1 - start_step
            grf_observation = first_observation(env.reset())
            found = difference(
                cuju_agents(batch, ours, index, left, agents), grf_agents(grf_observation, agents)
            )
            found = found and f'{found} after the reset'
        compare_state = arguments.state_interval and (step + 1) % arguments.state_interval == 0
        if found is None and compare_state:
            offset = state_difference(batch.get_state(index), env.get_state())
            if offset is not None:
                after = ' after the reset' if ended else ''
                found = f'the state (byte {offset} of the match){after}'
        if found is not None:
            print(f'{found} differs at step {step}')
            return -1
    return arguments.steps - start_step


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scenario', default='5_vs_5')
    parser.add_argument('--left', type=int, default=1)
    parser.add_argument('--right', type=int, default=0)
    parser.add_argument(
        '--representation',
        default='simple115v2',
        choices=['simple115', 'simple115v2', 'extracted', 'raw'],
    )
    parser.add_argument('--rewards', default='scoring')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument(
        '--agent',
        help='module:function of a policy, a path such as submission.py:agent or a module',
    )
    parser.add_argument(
        '--state-interval',
        type=int,
        default=1,
        help='compare the states every this many steps; 0 compares none',
    )
    arguments = parser.parse_args(argv)
    agents = arguments.left + arguments.right
    name = f'{arguments.scenario}_l{arguments.left}_r{arguments.right}'
    # The checkpoint reward adds in the precision of the numpy that GRF runs with here.
    legacy = int(np.__version__.split('.')[0]) < 2
    options = {
        'device': 'cpu',
        'rewards': arguments.rewards,
        'numpy_legacy': legacy,
        'representation': arguments.representation,
    }

    # By seed: the GRF environment is created after the seed of environment `index`.
    batch = cuju.BatchEnvironment(
        name, environment_count=arguments.index + 1, seed=arguments.seed, **options
    )
    actions = Actions(arguments, agents, batch.action_count)
    random.seed(arguments.seed * 2**32 + arguments.index)
    env = grf_environment(arguments)
    theirs = first_observation(env.reset())
    found = difference(
        cuju_agents(batch, batch.reset(), arguments.index, arguments.left, agents),
        grf_agents(theirs, agents),
    )
    if found is None and arguments.state_interval:
        offset = state_difference(batch.get_state(arguments.index), env.get_state())
        if offset is not None:
            found = f'the state (byte {offset} of the match)'
    if found is not None:
        print(f'{found} differs after the first reset')
        return 1
    actions.start()
    steps = play(arguments, batch, arguments.index, env, actions, theirs)
    if steps < 0:
        return 1
    print(f'by seed: {steps} steps equal')

    # By state: GRF saves a state midway, cuju loads it, and both play on to the end of that
    # episode. A state carries no generator, so the episodes after it draw other seeds.
    middle = arguments.steps // 2
    env = grf_environment(arguments)
    theirs = first_observation(env.reset())
    actions.start()
    for step in range(middle):
        grf_actions, _ = actions.pick(step, grf_agents(theirs, agents))
        theirs, _, done = grf_step(env, grf_actions, agents)
        if done:
            theirs = first_observation(env.reset())
    single = cuju.BatchEnvironment(name, environment_count=1, **options)
    single.set_state(0, env.get_state())
    # Both engines' policies start afresh from the state.
    actions.start()
    steps = play(arguments, single, 0, env, actions, theirs, start_step=middle, one_episode=True)
    if steps < 0:
        return 1
    print(f'by state: {steps} steps equal after the state of step {middle}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

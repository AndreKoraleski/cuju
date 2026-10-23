# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks `examples/compare_with_grf.py` against a stand-in for GRF that a cuju batch plays."""

import importlib.util
import os

import numpy as np
import pytest

import cuju

EXAMPLE = os.path.join(os.path.dirname(__file__), '..', 'examples', 'compare_with_grf.py')

pytestmark = pytest.mark.skipif(not os.path.isfile(EXAMPLE), reason='no examples directory')


@pytest.fixture
def example():
    spec = importlib.util.spec_from_file_location('compare_with_grf', EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StandIn:
    """A GRF environment that environment `index` of a cuju batch with the same seed plays."""

    def __init__(self, example, arguments, flip=None):
        self.example = example
        self.left, self.agents = arguments.left, arguments.left + arguments.right
        self.index = arguments.index
        self.batch = cuju.BatchEnvironment(
            f'{arguments.scenario}_l{arguments.left}_r{arguments.right}',
            environment_count=arguments.index + 1,
            device='cpu',
            seed=arguments.seed,
            representation=arguments.representation,
            rewards=arguments.rewards,
            numpy_legacy=int(np.__version__.split('.')[0]) < 2,
        )
        self.started = False
        self.steps = 0
        # `(what, step)`: the observation or the state to spoil at that step.
        self.flip = flip

    def grf(self, observation):
        agents = self.example.cuju_agents(
            self.batch, observation, self.index, self.left, self.agents
        )
        agents = [
            agent if isinstance(agent, dict) else np.array(agent, copy=True) for agent in agents
        ]
        if self.flip == ('observation', self.steps) and not isinstance(agents[0], dict):
            agents[0].reshape(-1).view(np.uint8)[0] ^= 1
        if isinstance(agents[0], dict):
            return agents
        return agents[0] if self.agents == 1 else np.stack(agents)

    def reset(self):
        # The batch starts the next episode at the end of the last, and GRF does it here.
        observation = self.batch.observation() if self.started else self.batch.reset()
        self.started = True
        return self.grf(observation), {}

    def step(self, actions):
        actions = np.reshape(actions, -1)
        rows = np.zeros((self.batch.environment_count, self.batch.agents), dtype=np.int32)
        rows[self.index, : self.agents] = actions
        observation, reward, done, information = self.batch.step(rows)
        self.steps += 1
        ended = bool(done[self.index])
        shown = information['final_observation'] if ended else observation
        reward = reward[self.index, : self.agents].numpy().copy()
        return self.grf(shown), reward if self.agents > 1 else reward[0], ended, False, {}

    def get_state(self):
        state = self.batch.get_state(self.index)
        if self.flip == ('state', self.steps):
            state = state[:-1] + bytes([state[-1] ^ 1])
        if self.flip == ('padding', self.steps):
            # GRF's padding of every angle holds whatever its memory held.
            match = len(state) - len(self.example.match_bytes(state))
            mask = np.zeros(len(state) - match, np.uint8)
            assert self.example.padding_function()(state[match:], len(mask), mask.ctypes.data) == 0
            noise = np.frombuffer(state, np.uint8).copy()
            noise[match:][mask == 1] ^= 0x5A
            state = noise.tobytes()
        return state


def run(example, monkeypatch, capsys, arguments, flip=None):
    monkeypatch.setattr(example, 'grf_environment', lambda parsed: StandIn(example, parsed, flip))
    status = example.main(arguments)
    return status, capsys.readouterr().out


@pytest.mark.parametrize(
    'representation, left, right',
    [('simple115v2', 1, 0), ('simple115', 2, 1), ('extracted', 1, 1), ('raw', 2, 1)],
)
def test_equal_engines_compare_equal(example, monkeypatch, capsys, representation, left, right):
    arguments = ['--scenario', '5_vs_5', '--representation', representation]
    arguments += ['--left', str(left), '--right', str(right), '--steps', '40', '--index', '1']
    status, out = run(example, monkeypatch, capsys, arguments)
    assert status == 0, out
    assert 'by seed: 40 steps equal' in out and 'by state:' in out


def test_a_differing_observation_is_found(example, monkeypatch, capsys):
    arguments = ['--scenario', 'academy_empty_goal_close', '--steps', '30']
    status, out = run(example, monkeypatch, capsys, arguments, flip=('observation', 7))
    assert status == 1
    assert 'the observation of agent 0 differs at step 6' in out


def test_a_differing_state_is_found(example, monkeypatch, capsys):
    arguments = ['--scenario', 'academy_empty_goal_close', '--steps', '30']
    status, out = run(example, monkeypatch, capsys, arguments, flip=('state', 5))
    assert status == 1
    assert 'the state (byte ' in out and 'differs at step 4' in out


def test_the_padding_of_angles_is_left_out(example, monkeypatch, capsys):
    arguments = ['--scenario', 'academy_empty_goal_close', '--steps', '30']
    status, out = run(example, monkeypatch, capsys, arguments, flip=('padding', 5))
    assert status == 0, out


def test_the_library_marks_the_padding_of_angles(example):
    batch = cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=1, device='cpu')
    batch.reset()
    match = example.match_bytes(batch.get_state(0))
    mask = np.zeros(len(match), np.uint8)
    assert example.padding_function()(match, len(match), mask.ctypes.data) == 0
    # Three bytes of each angle, a small part of the match.
    assert 0 < int(mask.sum()) < len(match) // 10 and int(mask.sum()) % 3 == 0
    assert example.padding_function()(b'not a match', 11, mask.ctypes.data) == -1
    batch.close()


def test_a_policy_hook_plays_both_engines(example, monkeypatch, capsys, tmp_path):
    # A Kaggle-style agent with a global counter and draws from Python's `random`.
    (tmp_path / 'submission.py').write_text(
        'import random\n'
        'calls = 0\n'
        '\n'
        '\n'
        'def agent(obs):\n'
        '    global calls\n'
        '    calls += 1\n'
        "    raw = obs['players_raw'][0]\n"
        "    if raw['ball_owned_team'] == 0 and calls % 3:\n"
        '        return [12]\n'
        '    return [random.randrange(19)]\n',
        encoding='utf-8',
    )
    arguments = ['--scenario', 'academy_empty_goal_close', '--representation', 'raw', '--steps']
    arguments += ['60', '--agent', f'{tmp_path / "submission.py"}:agent', '--right', '1']
    status, out = run(example, monkeypatch, capsys, arguments)
    assert status == 0, out
    assert 'by seed: 60 steps equal' in out


def test_raw_of_an_agent_has_the_types_of_grf(example):
    batch = cuju.BatchEnvironment(
        '5_vs_5_l1_r1', environment_count=1, device='cpu', representation='raw'
    )
    raw = batch.reset()
    left, right = (example.raw_of_agent(batch, raw, 0, agent, 1) for agent in (0, 1))
    assert left['left_team'].dtype == np.float64 and left['left_team'].shape == (5, 2)
    assert left['left_team_roles'].dtype == np.int64
    assert left['left_team_active'].dtype == np.bool_
    assert left['sticky_actions'].dtype == np.uint8
    assert isinstance(left['ball_owned_team'], int) and isinstance(left['score'], list)
    # A right agent sees its own team as the left team.
    assert np.array_equal(right['left_team'], -left['right_team'])
    batch.close()


def test_same_compares_bits(example):
    assert example.same(np.float32(0.1), np.float64(np.float32(0.1)))
    assert not example.same(np.float32(0.1), 0.1)
    assert not example.same(np.float64(0.0), np.float64(-0.0))
    assert not example.same(np.zeros(2), np.zeros(3))

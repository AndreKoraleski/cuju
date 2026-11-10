# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks Kaggle's `players_raw` and Kaggle agents against the raw tensors of a batch."""

import enum
import json

import pytest
import torch

import cuju
from conftest import tensor_bytes

np = pytest.importorskip('numpy')
kaggle = pytest.importorskip('cuju.kaggle')
cuju_raw = pytest.importorskip('cuju.raw')

# GRF: the 23 values of a raw observation of one agent.
KEYS = {
    'active',
    'ball',
    'ball_direction',
    'ball_owned_player',
    'ball_owned_team',
    'ball_rotation',
    'designated',
    'game_mode',
    'left_team',
    'left_team_active',
    'left_team_direction',
    'left_team_roles',
    'left_team_tired_factor',
    'left_team_yellow_card',
    'right_team',
    'right_team_active',
    'right_team_direction',
    'right_team_roles',
    'right_team_tired_factor',
    'right_team_yellow_card',
    'score',
    'steps_left',
    'sticky_actions',
}


@pytest.fixture(scope='module')
def batch():
    batch = cuju.BatchEnvironment(
        '5_vs_5_l1_r1', environment_count=2, device='cpu', representation='raw', seed=4
    )
    batch.reset()
    for step in range(20):
        batch.step(torch.tensor([[step % 19, (step * 3) % 19]] * 2, dtype=torch.int32))
    yield batch
    batch.close()


def test_players_raw_holds_grf_values_as_json(batch):
    raw = batch.raw()
    for agent in (0, 1):
        (entry,) = kaggle.players_raw(batch, 1, agent)
        assert set(entry) == KEYS
        # Kaggle's server sends it through JSON, which keeps every value.
        assert json.loads(json.dumps(entry)) == entry
        assert len(entry['left_team']) == 5 and len(entry['sticky_actions']) == 10
        assert isinstance(entry['ball'][0], float) and isinstance(entry['active'], int)
        theirs = raw['left_team'][1, agent, :5].double().numpy()
        assert np.array(entry['left_team']).tobytes() == theirs.tobytes()
    # The right agent sees the match turned by 180 degrees, its own goals first.
    left, right = (kaggle.players_raw(batch, 0, agent)[0] for agent in (0, 1))
    assert right['ball'][:2] == [-value for value in left['ball'][:2]]
    assert right['score'] == left['score'][::-1]


def test_grf_observations(batch):
    observations = cuju_raw.grf_observations(batch, 0)
    assert len(observations) == 2
    assert observations[0]['left_team'].dtype == np.float64
    assert observations[1]['right_team'].shape == (5, 2)
    assert kaggle.observation(batch, 0, 1, step=3).step == 3


class Action(enum.Enum):
    """Kaggle's `Action` enum of the football helpers, in part."""

    Idle = 0
    Shot = 12


def test_kaggle_agent_plays_as_its_actions():
    options = dict(environment_count=3, device='cpu', representation='raw', seed=1)
    played = cuju.BatchEnvironment('academy_3_vs_1_with_keeper_l1_r1', **options)
    direct = cuju.BatchEnvironment('academy_3_vs_1_with_keeper_l1_r1', **options)
    seen = []

    def agent(obs, configuration):
        seen.append((obs.step, configuration.get('name')))
        player = obs['players_raw'][0]
        # Shoot when the own team owns the ball, else stand still.
        return [Action.Shot if player['ball_owned_team'] == 0 else Action.Idle]

    opponent = kaggle.KaggleAgent(agent, played, 1, configuration={'name': 'test'})
    played.reset()
    direct.reset()
    for _ in range(40):
        expected = [
            12 if entry['ball_owned_team'] == 0 else 0
            for entry in (kaggle.players_raw(direct, environment, 1)[0] for environment in range(3))
        ]
        actions = opponent.act()
        assert actions.tolist() == expected
        _, _, done, _ = played.step(torch.stack([torch.zeros(3, dtype=torch.int32), actions], 1))
        theirs, _, _, _ = direct.step(torch.tensor([[0, action] for action in expected]))
        assert tensor_bytes(played.raw()['ball']) == tensor_bytes(theirs['ball'])
        opponent.reset(done.bool())
    assert seen[0] == (0, 'test') and seen[3] == (1, 'test')
    played.close()
    direct.close()


def test_refusals():
    plain = cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=1, device='cpu')
    with pytest.raises(ValueError, match='raw'):
        kaggle.KaggleAgent(lambda obs: [0], plain, 0)
    plain.close()
    batch = cuju.BatchEnvironment(
        '5_vs_5_l1_r0', environment_count=1, device='cpu', representation='raw'
    )
    batch.reset()
    with pytest.raises(IndexError):
        kaggle.KaggleAgent(lambda obs: [19], batch, 0).act()
    with pytest.raises(ValueError, match='one action'):
        kaggle.KaggleAgent(lambda obs: [1, 2], batch, 0).act()
    batch.close()


def test_evaluate_plays_one_match_per_environment():
    def shooter(obs):
        return [12]

    scores = kaggle.evaluate(shooter, episodes=2, scenario='academy_empty_goal_close')
    assert len(scores) == 2 and all(len(score) == 2 for score in scores)
    assert all(isinstance(goals, int) for score in scores for goals in score)


def test_agent_runner_calls_as_kaggle_and_checks_answers(batch):
    seen = []

    def one(obs):
        seen.append(('one', obs.step))
        return [Action.Shot]

    def two(obs, configuration):
        seen.append(('two', configuration.get('name')))
        return np.array([3, -1])

    def spread(*arguments):
        seen.append(('spread', len(arguments)))
        return 5

    assert kaggle.accepts_configuration(two) and kaggle.accepts_configuration(spread)
    assert not kaggle.accepts_configuration(one)
    single = kaggle.observation(batch, 0, 0, step=4)
    assert kaggle.AgentRunner(one).act(single, 19) == [12]
    assert kaggle.AgentRunner(spread).act(single, 19) == [5]
    host = {name: tensor.cpu() for name, tensor in batch.raw().items()}
    several = kaggle.observation(batch, 0, [0, 1], step=2, raw=host)
    assert several.controlled_players == 2
    assert several.players_raw == [
        kaggle.players_raw(batch, 0, 0)[0],
        kaggle.players_raw(batch, 0, 1)[0],
    ]
    runner = kaggle.AgentRunner(two, configuration={'name': 'test'})
    assert runner.accepts_configuration and runner.act(several, 19) == [3, -1]
    assert seen == [('one', 4), ('spread', 2), ('two', 'test')]
    with pytest.raises(ValueError, match='each of the 2'):
        kaggle.AgentRunner(lambda obs: [1]).act(several, 19)
    with pytest.raises(IndexError):
        kaggle.AgentRunner(lambda obs: [19]).act(single, 19)
    with pytest.raises(TypeError):
        kaggle.action_index(['shot'], 19)
    assert kaggle.action_index(np.array([7]), 19) == 7
    assert kaggle.action_indices((1, Action.Idle), 19, 2) == [1, 0]
    # The names before 0.2.1 stay as aliases.
    assert kaggle._action is kaggle.action_index and kaggle._actions is kaggle.action_indices
    assert kaggle._accepts_configuration is kaggle.accepts_configuration

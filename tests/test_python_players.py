# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks players of `extra_players` that Python modules define, as GRF loads them."""

import random
import sys
import types
import typing

import pytest

import cuju.dumps

np = pytest.importorskip('numpy')
pytest.importorskip('gymnasium')
cuju_env = pytest.importorskip('cuju.env')

from cuju.gfootball.env import player_base  # noqa: E402


def chase(observation):
    """Returns the direction from the active player to the ball, an index of the default set."""
    player = observation['left_team'][observation['active']]
    dx, dy = observation['ball'][0] - player[0], observation['ball'][1] - player[1]
    if abs(dx) > abs(dy):
        return 5 if dx > 0 else 1
    return 7 if dy > 0 else 3


class Chaser(player_base.PlayerBase):
    """Runs towards the ball, and keeps what GRF gives it."""

    built: typing.ClassVar[list] = []

    def __init__(self, player_config, env_config):
        player_base.PlayerBase.__init__(self, player_config)
        self.player_config = player_config
        self.env_config = env_config
        self.resets = 0
        self.seen = []
        Chaser.built.append(self)

    def reset(self):
        self.resets += 1

    def take_action(self, observations):
        self.seen.append(observations)
        return [chase(observation) for observation in observations]


class AsIs(Chaser):
    """Sees its right players' match as it is and moves in that frame."""

    def __init__(self, player_config, env_config):
        super().__init__(player_config, env_config)
        self._can_play_right = True

    def take_action(self, observations):
        self.seen.append(observations)
        return np.array([1] * len(observations))


@pytest.fixture
def modules(monkeypatch):
    """Defines `chaser` as GRF's players are defined, and `my_players.as_is` by its module path."""
    Chaser.built = []
    chaser = types.ModuleType('cuju.gfootball.env.players.chaser')
    chaser.Player = Chaser
    package = types.ModuleType('my_players')
    package.__path__ = []
    as_is = types.ModuleType('my_players.as_is')
    as_is.Player = AsIs
    monkeypatch.setitem(sys.modules, 'cuju.gfootball.env.players.chaser', chaser)
    monkeypatch.setitem(sys.modules, 'my_players', package)
    monkeypatch.setitem(sys.modules, 'my_players.as_is', as_is)


def play(env, steps, action=0):
    observations = [env.reset()]
    for _ in range(steps):
        observation, _, done, _ = env.step(action)
        observations.append(observation)
        if done:
            break
    return observations


def same_observation(first, second):
    assert sorted(first) == sorted(second)
    for key in first:
        a, b = first[key], second[key]
        if isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and np.array_equal(a, b), key
        else:
            assert a == b, key


def test_a_module_player_plays_as_an_agent_would(modules):
    random.seed(3)
    env = cuju_env.create_environment(
        'academy_3_vs_1_with_keeper', representation='raw', extra_players=['chaser:right_players=1']
    )
    played = play(env, 60)
    env.close()
    player = Chaser.built[0]
    # GRF: the player's configuration counts the definitions, and keeps their strings.
    assert player.player_config == {
        'index': 1,
        'player_agent': 0,
        'player_chaser': 0,
        'left_players': 0,
        'right_players': '1',
    }
    assert player.env_config['players'] == [
        'agent:left_players=1,right_players=0',
        'chaser:right_players=1',
    ]
    assert player.resets == 1 and player.env_config['episode_number'] == 2
    # The same match, with the right player as a second agent fed the same decisions.
    random.seed(3)
    env = cuju_env.create_environment(
        'academy_3_vs_1_with_keeper', representation='raw', number_of_right_players_agent_controls=1
    )
    observations = [env.reset()]
    for step in range(len(played) - 1):
        same_observation(observations[-1][1], player.seen[step][0])
        observation, *_ = env.step([0, chase(observations[-1][1])])
        observations.append(observation)
    env.close()
    for first, second in zip(played, observations, strict=True):
        same_observation(first[0], second[0])


def test_a_player_that_plays_the_right_as_it_is(modules):
    random.seed(5)
    env = cuju_env.create_environment(
        'academy_3_vs_1_with_keeper',
        representation='raw',
        extra_players=['my_players.as_is:right_players=1'],
    )
    played = play(env, 40)
    env.close()
    random.seed(5)
    env = cuju_env.create_environment(
        'academy_3_vs_1_with_keeper', representation='raw', number_of_right_players_agent_controls=1
    )
    # `left` in the frame of the match is `right` for the right agent.
    observations = play(env, len(played) - 1, [0, 5])
    env.close()
    seen = AsIs.built[0].seen
    for step, (first, second) in enumerate(zip(played, observations, strict=True)):
        same_observation(first[0], second[0])
        if step < len(seen):
            match = seen[step][0]
            # The match as it is, the left agent's view, and the right player's own values.
            for key in ('ball', 'left_team', 'right_team_roles', 'score', 'ball_owned_team'):
                assert np.array_equal(np.asarray(match[key]), np.asarray(second[0][key])), key
            assert match['active'] == second[1]['active']


def test_the_players_follow_the_seeds_of_grf(modules):
    random.seed(9)
    cuju_env.create_environment('academy_corner', extra_players=['chaser:right_players=1']).close()
    with_player = random.getstate()
    random.seed(9)
    cuju_env.create_environment('academy_corner').close()
    assert random.getstate() == with_player
    generator = random.Random(9)
    draws = [generator.randint(0, 2000000000) for _ in range(3)]
    env_config = Chaser.built[0].env_config
    # GRF: the configuration draws the second seed, and the core's first reset the third.
    assert env_config['episode_number'] == 1
    assert env_config['game_engine_random_seed'] == draws[2]
    assert env_config['reverse_team_processing'] == bool(draws[1] % 2)


def test_refusals(modules):
    with pytest.raises(ValueError, match='no_such_player'):
        cuju_env.create_environment('academy_corner', extra_players=['no_such_player'])
    with pytest.raises(ValueError, match='class Player'):
        cuju_env.create_environment('academy_corner', extra_players=['types:right_players=1'])

    class Silent(Chaser):
        def take_action(self, observations):
            return []

    sys.modules['cuju.gfootball.env.players.chaser'].Player = Silent
    env = cuju_env.create_environment('academy_corner', extra_players=['chaser:right_players=1'])
    env.reset()
    with pytest.raises(ValueError, match='Player provided 0 actions instead of 1'):
        env.step(0)
    env.close()


def test_the_replay_player_plays_a_dump(tmp_path):
    batch = cuju.BatchEnvironment(
        'academy_3_vs_1_with_keeper_l1_r1', environment_count=1, device='cpu', with_raw=True
    )
    writer = cuju.dumps.DumpWriter(batch, tmp_path)
    batch.reset()
    import torch

    rows = []
    while not writer.paths:
        actions = torch.tensor([[len(rows) % 19, (len(rows) * 7) % 19]], dtype=torch.int32)
        rows.append(actions[0].tolist())
        batch.step(actions)
        writer.record(actions)
    batch.close()
    steps = cuju.dumps.read_dump(writer.paths[0])
    config = steps[0]['debug']['config']
    env = cuju_env.create_environment(
        'academy_3_vs_1_with_keeper',
        representation='raw',
        extra_players=[f'replay:path={writer.paths[0]},right_players=1'],
        other_config_options={
            'game_engine_random_seed': config['game_engine_random_seed'],
            'reverse_team_processing': config['reverse_team_processing'],
        },
    )
    env.reset()
    for index, row in enumerate(rows):
        observation, _, done, _ = env.step(row[0])
        expected = steps[index]['observation']
        assert observation[0]['left_team'].tolist() == expected['left_team'].tolist()
        assert done == (index == len(rows) - 1)
    env.close()

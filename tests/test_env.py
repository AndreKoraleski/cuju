# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks GRF's `create_environment` and gymnasium ids against the batch that plays them."""

import os
import random
import re

import pytest

import cuju
from conftest import CUDA

gymnasium = pytest.importorskip('gymnasium')
np = pytest.importorskip('numpy')
cuju_env = pytest.importorskip('cuju.env')


def grf_draws(seed, count):
    """Returns the first `count` seeds that GRF draws after `random.seed(seed)`."""
    generator = random.Random(seed)
    return [generator.randint(0, 2000000000) for _ in range(count)]


def test_defaults_follow_grf():
    random.seed(0)
    env = cuju_env.create_environment('academy_empty_goal_close')
    assert env.observation_space == gymnasium.spaces.Box(0, 255, (72, 96, 4), np.uint8)
    assert env.action_space == gymnasium.spaces.Discrete(19)
    # GRF: gym's interface, `reset` returns the observation and `step` four values.
    observation = env.reset()
    assert observation.shape == (72, 96, 4) and observation.dtype == np.uint8
    assert observation in env.observation_space
    observation, reward, done, info = env.step(0)
    assert isinstance(reward, np.float32) and done is False
    assert info == {'score_reward': 0}
    env.close()


def test_gymnasium_interface_plays_the_same_match():
    outputs = []
    for api in ('gym', 'gymnasium'):
        random.seed(3)
        env = cuju_env.create_environment(
            'academy_empty_goal_close', representation='simple115v2', api=api
        )
        if api == 'gym':
            observations = [env.reset()]
            for step in range(20):
                observation, reward, done, info = env.step(step % 19)
                assert isinstance(done, bool) and isinstance(reward, np.float32)
                observations.append(observation)
        else:
            observation, info = env.reset()
            assert info == {}
            observations = [observation]
            for step in range(20):
                observation, reward, terminated, truncated, info = env.step(step % 19)
                assert truncated is False and isinstance(terminated, bool)
                assert isinstance(reward, np.float32)
                observations.append(observation)
        outputs.append(b''.join(observation.tobytes() for observation in observations))
        env.close()
    assert outputs[0] == outputs[1]
    with pytest.raises(ValueError, match='api'):
        cuju_env.create_environment('5_vs_5', api='gym26')


def test_episodes_follow_the_seed_rule():
    # Environment 0 of a batch with seed 5 plays GRF's environment after random.seed(5 * 2**32).
    random.seed(5 * 2**32)
    env = cuju_env.create_environment('academy_run_to_score', representation='simple115v2')
    batch = cuju.BatchEnvironment(
        'academy_run_to_score', environment_count=1, device='cpu', seed=5, auto_reset=False
    )
    for episode in range(2):
        ours = env.reset()
        theirs = batch.reset()
        assert ours.tobytes() == theirs[0, 0].numpy().tobytes()
        for step in range(40):
            action = (step * 5 + episode) % 19
            ours, reward, terminated, info = env.step(action)
            theirs, batch_reward, done, information = batch.step([[action]])
            assert ours.tobytes() == theirs[0, 0].numpy().tobytes()
            assert reward == batch_reward[0, 0].item() and terminated == bool(done[0])
            assert info['score_reward'] == int(information['score_reward'][0])
            if terminated:
                break
    env.close()
    batch.close()


def test_a_step_before_reset_plays_the_creation_episode():
    random.seed(9)
    env = cuju_env.create_environment(
        '5_vs_5', representation='simple115v2', number_of_left_players_agent_controls=2
    )
    draws = grf_draws(9, 3)
    # GRF: the creation plays episode 1, the odd form, from the third draw, and the second draw
    # sets the team order.
    batch = cuju.BatchEnvironment(
        '5_vs_5_l2_r0', environment_count=1, device='cpu', auto_reset=False, episode_number=1
    )
    batch.reset(seeds=draws[2], reverse=draws[1] % 2)
    for step in range(30):
        ours, *_ = env.step([step % 19, (step * 3) % 19])
        theirs, *_ = batch.step([[step % 19, (step * 3) % 19]])
        assert ours.tobytes() == theirs[0].numpy().tobytes()
    env.close()
    batch.close()


def test_fixed_seed_replays_every_episode():
    env = cuju_env.create_environment(
        'academy_counterattack_easy',
        representation='simple115',
        other_config_options={'game_engine_random_seed': 123},
    )
    first = env.reset()
    trajectory = [env.step(step % 19)[0] for step in range(20)]
    second = env.reset()
    assert first.tobytes() == second.tobytes()
    assert all(env.step(step % 19)[0].tobytes() == a.tobytes() for step, a in enumerate(trajectory))
    env.close()


def test_several_agents_and_stacks():
    env = cuju_env.create_environment(
        '5_vs_5',
        representation='simple115',
        stacked=True,
        number_of_left_players_agent_controls=2,
        number_of_right_players_agent_controls=1,
    )
    assert env.observation_space.shape == (3, 460)
    assert env.action_space == gymnasium.spaces.MultiDiscrete([19, 19, 19])
    observation = env.reset()
    assert observation.shape == (3, 460)
    _, reward, _, _ = env.step(np.array([0, 1, 2]))
    assert reward.shape == (3,) and reward.dtype == np.float32
    with pytest.raises(ValueError, match='expected 3 actions'):
        env.step([0, 1])
    env.close()


def test_raw_observations_have_grf_types():
    env = cuju_env.create_environment(
        '5_vs_5',
        representation='raw',
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=1,
    )
    # GRF: the raw representation has no space, so it is `None` on any base class.
    assert env.observation_space is None
    observation = env.reset()
    assert len(observation) == 2
    left, right = observation
    assert left['left_team'].dtype == np.float64 and left['left_team'].shape == (5, 2)
    assert left['left_team_roles'].dtype == np.int64
    assert left['sticky_actions'].dtype == np.uint8 and left['sticky_actions'].shape == (10,)
    assert isinstance(left['score'], list) and isinstance(left['game_mode'], int)
    # A right agent sees the match turned by 180 degrees.
    assert np.array_equal(right['ball'][:2], -left['ball'][:2])
    raw = env.batch.raw()
    assert np.array_equal(left['right_team'], raw['right_team'][0, 0, :5].double().numpy())
    env.close()


def test_gymnasium_ids():
    assert 'GFootball-11_vs_11_kaggle-simple115v2-v0' in gymnasium.envs.registry
    random.seed(4)
    env = gymnasium.make('GFootball-academy_pass_and_shoot_with_keeper-simple115v2-v0')
    made, _ = env.reset()
    env.close()
    random.seed(4)
    direct = cuju_env.create_environment(
        'academy_pass_and_shoot_with_keeper', representation='simple115v2', api='gymnasium'
    )
    observation, _ = direct.reset()
    assert made.tobytes() == observation.tobytes()
    direct.close()
    with pytest.raises(KeyError, match='renders nothing'):
        gymnasium.make('GFootball-5_vs_5-Pixels-v0')


def test_refusals():
    with pytest.raises(ValueError, match='renders nothing'):
        cuju_env.create_environment('5_vs_5', render=True)
    with pytest.raises(ValueError, match='players'):
        cuju_env.create_environment('5_vs_5', extra_players=['keyboard:right_players=1'])
    with pytest.raises(ValueError, match="one 'agent'"):
        cuju_env.create_environment('5_vs_5', extra_players=['agent:right_players=1'])
    with pytest.warns(UserWarning, match='ignores'):
        cuju_env.create_environment('5_vs_5', other_config_options={'colour': 1}).close()
    env = cuju_env.create_environment('academy_empty_goal_close', representation='simple115v2')
    env.reset()
    for _ in range(400):
        if env.step(12)[2]:
            break
    with pytest.raises(RuntimeError, match='reset'):
        env.step(0)
    env.close()


def test_dump_options(tmp_path):
    options = dict(
        representation='simple115v2', write_full_episode_dumps=True, logdir=str(tmp_path)
    )
    env = cuju_env.create_environment('academy_empty_goal_close', **options)
    env.reset()
    for _ in range(400):
        _, _, terminated, info = env.step(12)
        if terminated:
            break
    # GRF: a dictionary for each dump that the step finished, its path and its kind, and the
    # name `<logdir>/<kind>_<date>-<time>.dump` with microseconds.
    assert [list(dump) for dump in info['dumps']] == [['dump', 'name']]
    (dump,) = info['dumps']
    assert dump['name'] == 'episode_done' and os.path.isfile(dump['dump'])
    assert re.fullmatch(
        re.escape(str(tmp_path) + os.sep) + r'episode_done_\d{8}-\d{12}\.dump', dump['dump']
    )
    env.close()


def test_goal_dumps_list_grf_dictionaries(tmp_path):
    env = cuju_env.create_environment(
        'academy_empty_goal_close',
        representation='simple115v2',
        write_goal_dumps=True,
        write_full_episode_dumps=True,
        logdir=str(tmp_path),
    )
    env.reset()
    for _ in range(400):
        _, reward, terminated, info = env.step(12)
        if terminated:
            break
    assert reward == 1
    # GRF: `process_pending_dumps` lists the dump of the goal before that of the episode.
    assert [dump['name'] for dump in info['dumps']] == ['score', 'episode_done']
    for dump in info['dumps']:
        name = os.path.basename(dump['dump'])
        assert re.fullmatch(dump['name'] + r'_\d{8}-\d{12}\.dump', name)
    env.close()


@pytest.mark.parametrize('extra', ['bot:right_players=1', 'lazy:right_players=1'])
def test_a_stacked_state_with_a_scripted_player_loads(extra):
    random.seed(8)
    env = cuju_env.create_environment(
        '5_vs_5',
        representation='simple115v2',
        stacked=True,
        extra_players=[extra],
        other_config_options={'action_set': 'full'},
    )
    env.reset()
    for step in range(10):
        env.step(step % 19)
    state = env.get_state()
    stack = env._observe()
    later = [env.step(step % 19)[0] for step in range(10, 20)]
    assert later[-1].tobytes() != stack.tobytes()
    # GRF: the state's `FrameStack` holds the agent's frames, and the loaded stack is that one.
    env.set_state(state)
    assert env._observe().tobytes() == stack.tobytes()
    replayed = [env.step(step % 19)[0] for step in range(10, 20)]
    if extra.startswith('lazy'):
        # The lazy player keeps nothing between steps, so the match plays again as it did. The
        # bot keeps its last action, which no state holds, in GRF too.
        assert [frame.tobytes() for frame in replayed] == [frame.tobytes() for frame in later]
    env.close()


def test_gym_methods_of_grf():
    random.seed(6)
    env = cuju_env.create_environment('academy_empty_goal_close', representation='simple115v2')
    # GRF: gym's `Env.seed` seeds nothing, gym bounds no reward, and nothing renders here.
    assert env.seed(3) is None and env.disable_render() is None
    assert env.reward_range == (-float('inf'), float('inf'))
    random.seed(6)
    plain = cuju_env.create_environment('academy_empty_goal_close', representation='simple115v2')
    random.seed(7)
    observation = env.reset()
    random.seed(7)
    assert observation.tobytes() == plain.reset().tobytes()
    assert env.step(5)[0].tobytes() == plain.step(5)[0].tobytes()
    env.close()
    plain.close()


def dumped_episodes(env, episodes=3):
    """Returns, for each of `episodes` episodes from a reset, whether its steps wrote a dump."""
    found = []
    for episode in range(episodes):
        random.seed(40 + episode)
        env.reset()
        terminated, wrote = False, False
        while not terminated:
            _, _, terminated, info = env.step(5)
            wrote |= 'dumps' in info
        found.append(wrote)
    env.close()
    return found


def test_dump_frequency_counts_the_resets(tmp_path):
    env = cuju_env.create_environment(
        'academy_empty_goal_close',
        representation='simple115v2',
        write_full_episode_dumps=True,
        dump_frequency=2,
        logdir=str(tmp_path),
    )
    # GRF: `PeriodicDumpWriter` keeps the dumps of the first reset and of every second after it.
    assert dumped_episodes(env) == [True, False, True]


@pytest.mark.skipif(not CUDA, reason='no CUDA device or library')
def test_gpu_equals_cpu():
    outputs = []
    for device in ('cpu', 'cuda'):
        random.seed(2)
        env = cuju_env.create_environment(
            'academy_3_vs_1_with_keeper', representation='extracted', device=device
        )
        observations = [env.reset()] + [env.step(step % 19)[0] for step in range(30)]
        outputs.append(b''.join(observation.tobytes() for observation in observations))
        env.close()
    assert outputs[0] == outputs[1]

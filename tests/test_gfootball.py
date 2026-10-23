# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Runs GRF's own usage against `cuju.gfootball`, GRF's import paths over cuju."""

import glob
import os
import random

import pytest

gymnasium = pytest.importorskip('gymnasium')
np = pytest.importorskip('numpy')

from cuju import env as cuju_env  # noqa: E402
from cuju.gfootball import dump_to_txt  # noqa: E402
from cuju.gfootball import env as football_env  # noqa: E402
from cuju.gfootball import scenarios as grf_scenarios  # noqa: E402

football_action_set = football_env.football_action_set


def grf_draws(seed, count):
    """Returns the first `count` seeds that GRF draws after `random.seed(seed)`."""
    generator = random.Random(seed)
    return [generator.randint(0, 2000000000) for _ in range(count)]


def same_raw(first, second):
    """Asserts that two lists of raw observations hold the same values, bit for bit."""
    assert len(first) == len(second)
    for ours, theirs in zip(first, second, strict=True):
        assert list(ours) == list(theirs)
        for key, value in ours.items():
            if isinstance(value, np.ndarray):
                other = theirs[key]
                assert value.dtype == other.dtype and value.shape == other.shape, key
                assert value.tobytes() == other.tobytes(), key
            else:
                assert value == theirs[key], key


def test_grf_example_runs():
    # GRF's example of its README, with fewer steps.
    env = football_env.create_environment(
        env_name='academy_empty_goal_close',
        stacked=False,
        logdir='',
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=False,
    )
    env.reset()
    for _ in range(20):
        obs, _, done, info = env.step(env.action_space.sample())
        if done:
            break
    assert obs.shape == (72, 96, 4) and isinstance(done, bool) and 'score_reward' in info
    env.close()


def test_gym_ids_use_grf_interface():
    gym = pytest.importorskip('gym')
    if not cuju_env._gym_before_026():
        pytest.skip('gym 0.26 and later take the gymnasium ids')
    spec = gym.spec('GFootball-academy_empty_goal_close-simple115v2-v0')
    # Gym 0.21 keeps the arguments in `_kwargs`.
    kwargs = spec.kwargs if hasattr(spec, 'kwargs') else spec._kwargs
    assert spec.entry_point == 'cuju.env:create_environment'
    assert kwargs == {
        'env_name': 'academy_empty_goal_close',
        'representation': 'simple115v2',
        'api': 'gym',
    }
    env = football_env.create_environment(**kwargs)
    assert env.reset().shape == (115,)
    assert len(env.step(0)) == 4
    env.close()


def test_config_follows_grf():
    config = football_env.config
    random.seed(7)
    values = {
        'level': 'academy_empty_goal_close',
        'players': ['agent:left_players=2', 'bot:right_players=1'],
    }
    cfg = config.Config(values)
    draws = grf_draws(7, 2)
    # GRF: creating the configuration builds the scenario of episode 1, which draws once.
    assert cfg['game_engine_random_seed'] == draws[0]
    assert random.randint(0, 2000000000) == draws[1]
    assert cfg['episode_number'] == 1
    assert cfg['reverse_team_processing'] == bool(draws[0] % 2)
    assert cfg['action_set'] == 'default' and cfg['render_resolution_y'] == 720
    assert 'level' in cfg and 'game_engine_random_seed' in cfg and 'colour' not in cfg
    assert 'game_engine_random_seed' not in cfg.serialize()
    assert cfg.get_dictionary()['game_engine_random_seed'] == draws[0]
    assert cfg.number_of_left_players() == 2 and cfg.number_of_right_players() == 1
    assert cfg.number_of_players_agent_controls() == 2
    cfg['action_set'] = 'full'
    cfg.update({'physics_steps_per_frame': 5})
    assert cfg['action_set'] == 'full' and cfg['physics_steps_per_frame'] == 5
    # GRF: the counts that a definition names stay strings, and one without counts gets one left
    # player.
    assert config.parse_player_definition('agent:left_players=2,right_players=1') == (
        'agent',
        {'left_players': '2', 'right_players': '1'},
    )
    assert config.parse_player_definition('lazy') == (
        'lazy',
        {'left_players': 1, 'right_players': 0},
    )
    assert config.count_players('agent:right_players=3') == 3
    assert config.count_left_players('bot') == 1 and config.count_right_players('bot') == 0
    assert config.get_agent_number_of_players(['agent:left_players=2', 'lazy:right_players=1']) == 2
    # GRF: a seed of the values is not drawn, and its parity sets the team order.
    fixed = config.Config({'game_engine_random_seed': 5})
    assert fixed.serialize()['game_engine_random_seed'] == 5
    assert fixed['reverse_team_processing'] is True
    assert random.randint(0, 2000000000) == grf_draws(7, 3)[2]
    # GRF: the scenario of the episode, with the configuration's agents.
    assert cfg.ScenarioConfig().game_duration == 400
    assert (cfg.ScenarioConfig().left_agents, cfg.ScenarioConfig().right_agents) == (2, 1)


def test_unwrapped_config_and_actions():
    random.seed(9)
    env = football_env.create_environment('academy_empty_goal_close', representation='simple115v2')
    draws = grf_draws(9, 4)
    assert env.unwrapped._num_actions == 19
    cfg = env.unwrapped._config
    assert cfg['action_set'] == 'default' and cfg['level'] == 'academy_empty_goal_close'
    assert cfg['players'] == ['agent:left_players=1,right_players=0']
    assert cfg.number_of_players_agent_controls() == 1
    # GRF: the creation leaves episode 1 with the third draw, and the second sets the team order.
    assert cfg['episode_number'] == 1 and cfg['game_engine_random_seed'] == draws[2]
    assert cfg['reverse_team_processing'] == bool(draws[1] % 2)
    env.reset()
    assert cfg['episode_number'] == 2 and cfg['game_engine_random_seed'] == draws[3]
    env.close()
    full = football_env.create_environment(
        'academy_empty_goal_close', other_config_options={'action_set': 'full'}
    )
    assert full._num_actions == 33 and full._config['action_set'] == 'full'
    full.close()


def play(env, single):
    """Plays one episode start of 25 steps, and returns its observations, rewards and infos."""
    observations, rewards, infos = [env.reset()], [], []
    for step in range(25):
        action = (step * 5) % 19
        observation, reward, done, info = env.step(action if single else [action])
        observations.append(observation)
        rewards.append(reward)
        infos.append(info)
        if done:
            break
    return observations, rewards, infos


def test_football_env_plays_the_match_of_create_environment():
    scenario = 'academy_3_vs_1_with_keeper'
    random.seed(11)
    created = football_env.create_environment(scenario, representation='raw')
    created_config = list(created._config.get_dictionary().items())
    expected = play(created, single=True)
    random.seed(11)
    # GRF: `create_environment` first builds a configuration that only reads the scenario.
    random.randint(0, 2000000000)
    cfg = football_env.config.Config(
        {'level': scenario, 'players': ['agent:left_players=1,right_players=0'], 'tracesdir': ''}
    )
    env = football_env.football_env.FootballEnv(cfg)
    assert env._config is cfg and cfg['episode_number'] == 1
    assert list(cfg.get_dictionary().items()) == created_config
    assert env._num_actions == 19
    observations, rewards, infos = play(env, single=False)
    assert len(observations) == len(expected[0])
    for ours, theirs in zip(observations, expected[0], strict=True):
        same_raw(ours, theirs)
    for ours, theirs in zip(rewards, expected[1], strict=True):
        # GRF: `FootballEnv` returns a reward per player, `float32`.
        assert ours.dtype == np.float32 and ours.shape == (1,) and ours[0] == theirs
    assert infos == expected[2]
    env.close()
    created.close()
    with pytest.raises(TypeError, match='Config'):
        football_env.football_env.FootballEnv({'level': scenario})


def test_football_env_without_an_agent_steps_with_no_actions():
    from cuju.players import Bot

    scenario = 'academy_3_vs_1_with_keeper'
    values = {'level': scenario, 'action_set': 'full', 'tracesdir': ''}
    # GRF's `play_game.py` holds no agent, and its players play every step of `step([])`.
    random.seed(13)
    config = football_env.config.Config({**values, 'players': ['bot:left_players=1']})
    env = football_env.football_env.FootballEnv(config)
    assert env.action_space == gymnasium.spaces.Discrete(33)
    # The same match, with an agent that plays the bot's actions.
    random.seed(13)
    agent_config = football_env.config.Config({**values, 'players': ['agent:left_players=1']})
    agent = football_env.football_env.FootballEnv(agent_config)
    bot = Bot(agent.batch, 0)

    def check(match, seen):
        # GRF: without an agent, the observation is the match's, the core's dictionary.
        assert isinstance(match, dict)
        assert match['left_agent_controlled_player'] == [seen['active']]
        assert list(match['left_agent_sticky_actions'][0]) == list(seen['sticky_actions'])
        for key in ('ball', 'left_team', 'right_team', 'left_team_roles', 'right_team_active'):
            assert match[key].dtype == seen[key].dtype, key
            assert match[key].tobytes() == seen[key].tobytes(), key
        assert match['score'] == seen['score'] and match['steps_left'] == seen['steps_left']
        assert match['left_team_designated_player'] == seen['designated']

    random.seed(14)
    match = env.reset()
    random.seed(14)
    (seen,) = agent.reset()
    check(match, seen)
    for _ in range(60):
        match, reward, done, info = env.step([])
        (seen,), expected_reward, expected_done, expected_info = agent.step(
            int(bot.act().reshape(-1)[0])
        )
        # GRF: without an agent the reward is the scoring reward, a `float32` scalar array.
        assert reward.dtype == np.float32 and reward.shape == ()
        assert float(reward) == float(expected_reward[0])
        assert (done, info) == (expected_done, expected_info)
        check(match, seen)
        if done:
            break
    env.reset()
    with pytest.raises(ValueError, match='no agent is playing'):
        env.step([0])
    env.close()
    agent.close()


@pytest.mark.parametrize('scenario', ['academy_3_vs_1_with_keeper', '5_vs_5'])
def test_conversions_of_raw_observations_equal_the_representations(scenario):
    trajectories = {}
    for representation in ('raw', 'simple115', 'simple115v2', 'extracted'):
        random.seed(13)
        env = football_env.create_environment(scenario, representation=representation)
        observations = [env.reset()]
        for step in range(30):
            observation, _, done, _ = env.step((step * 7) % 19)
            observations.append(observation)
            if done:
                break
        trajectories[representation] = observations
        env.close()
    convert = football_env.wrappers.Simple115StateWrapper.convert_observation
    generate_smm = football_env.observation_preprocessing.generate_smm
    for index, raw in enumerate(trajectories['raw']):
        simple115 = convert(raw, False)
        assert simple115.dtype == np.float32 and simple115.shape == (1, 115)
        assert simple115[0].tobytes() == trajectories['simple115'][index].tobytes()
        assert convert(raw, True)[0].tobytes() == trajectories['simple115v2'][index].tobytes()
        smm = generate_smm(raw)
        assert smm.dtype == np.uint8 and smm.shape == (1, 72, 96, 4)
        assert smm[0].tobytes() == trajectories['extracted'][index].tobytes()


def test_wrappers_over_a_raw_environment():
    wrappers = football_env.wrappers
    random.seed(17)
    env = wrappers.Simple115StateWrapper(
        football_env.create_environment('academy_empty_goal', representation='raw'),
        fixed_positions=True,
    )
    assert env.observation_space.shape == (1, 115)
    assert env.unwrapped is env.env and env._num_actions == 19
    random.seed(17)
    plain = football_env.create_environment('academy_empty_goal', representation='simple115v2')
    random.seed(18)
    first = env.reset()
    random.seed(18)
    assert first[0].tobytes() == plain.reset().tobytes()
    observation = env.step(3)[0]
    assert observation[0].tobytes() == plain.step(3)[0].tobytes()
    env.close()
    plain.close()
    smm = wrappers.SMMWrapper(football_env.create_environment('5_vs_5', representation='raw'))
    assert smm.observation_space.shape == (1, 72, 96, 4)
    assert smm.reset().shape == (1, 72, 96, 4)
    smm.close()


def episodes(env, action, steps=60, seed=21):
    """Plays `steps` steps from a seeded reset, with a seeded reset after each end."""
    random.seed(seed)
    outputs = [env.reset()]
    for step in range(steps):
        observation, reward, done, info = env.step(action(step))
        outputs.append((observation, np.asarray(reward), done, info))
        if done:
            random.seed(seed + step)
            outputs.append(env.reset())
    return outputs


def fingerprint(outputs):
    """Returns the types and the bytes of the outputs of `episodes`, with the flags and infos."""
    found = []
    for output in outputs:
        observation, *rest = output if isinstance(output, tuple) else (output,)
        observation = np.asarray(observation)
        found.append((observation.dtype.str, observation.shape, observation.tobytes()))
        if rest:
            reward, done, info = rest
            found.append((reward.dtype.str, reward.shape, reward.tobytes(), done, info))
    return found


def advance(env, action, seed, steps):
    """Plays `steps` steps from a seeded reset, with a reset after each end."""
    random.seed(seed)
    env.reset()
    for step in range(steps):
        if env.step(action(step))[2]:
            env.reset()


def exchange_states(first, second, action):
    """Asserts that each environment takes the other's state and plays on as the other does."""
    for source, target in ((first, second), (second, first)):
        advance(source, action, 40, 12)
        state = source.get_state()
        # The target stands at another step of another episode when it takes the state.
        advance(target, lambda step: action(step + 3), 41, 7)
        target.set_state(state)
        assert target.get_state() == state
        for step in range(12, 40):
            ours, theirs = source.step(action(step)), target.step(action(step))
            assert fingerprint([ours]) == fingerprint([theirs])
            assert target.get_state() == source.get_state()
            if ours[2]:
                break


def test_grf_wrappers_built_by_hand_give_the_outputs_of_create_environment():
    wrappers = football_env.wrappers
    scenario, options = 'academy_3_vs_1_with_keeper', {'number_of_left_players_agent_controls': 2}

    def action(step):
        return [(5, 12, 5, 5)[step % 4], (step * 3) % 19]

    random.seed(30)
    created = football_env.create_environment(
        scenario,
        representation='simple115v2',
        rewards='scoring,checkpoints',
        stacked=True,
        **options,
    )
    expected = episodes(created, action)
    random.seed(30)
    raw = football_env.create_environment(scenario, representation='raw', **options)
    # GRF: `create_environment` stacks the same wrappers in this order.
    env = wrappers.GetStateWrapper(
        wrappers.FrameStack(
            wrappers.Simple115StateWrapper(wrappers.CheckpointRewardWrapper(raw), True), 4
        )
    )
    assert env.observation_space.shape == created.observation_space.shape == (2, 460)
    assert fingerprint(episodes(env, action)) == fingerprint(expected)
    # The checkpoints added tenths to the rewards.
    rewards = np.concatenate([output[1] for output in expected if isinstance(output, tuple)])
    assert bool((rewards % 1 != 0).any())
    # GRF: the wrappers carry their counts and frames in the state, as the options do.
    state = env.get_state()
    assert state == created.get_state()
    assert b'CheckpointRewardWrapper' in state and b'FrameStack' in state
    exchange_states(created, env, action)
    env.close()
    created.close()


def test_grf_single_agent_wrappers_over_football_env():
    wrappers = football_env.wrappers
    scenario = 'academy_empty_goal_close'
    random.seed(31)
    created = football_env.create_environment(
        scenario, representation='extracted', rewards='scoring,checkpoints', stacked=True
    )
    expected = episodes(created, lambda step: 5)
    random.seed(31)
    # GRF: `create_environment` first builds a configuration that only reads the scenario.
    random.randint(0, 2000000000)
    cfg = football_env.config.Config(
        {'level': scenario, 'players': ['agent:left_players=1,right_players=0'], 'tracesdir': ''}
    )
    single = wrappers.SingleAgentRewardWrapper(
        wrappers.SingleAgentObservationWrapper(
            wrappers.SMMWrapper(
                wrappers.CheckpointRewardWrapper(football_env.football_env.FootballEnv(cfg))
            )
        )
    )
    env = wrappers.GetStateWrapper(wrappers.FrameStack(single, 4))
    assert env.observation_space.shape == created.observation_space.shape == (72, 96, 16)
    assert fingerprint(episodes(env, lambda step: 5)) == fingerprint(expected)
    assert env.get_state() == created.get_state()
    exchange_states(created, env, lambda step: (5, 1, 3, 5)[step % 4])
    # Without `FrameStack`, the state of the stack equals that of the same options unstacked.
    random.seed(32)
    unstacked = football_env.create_environment(
        scenario, representation='extracted', rewards='scoring,checkpoints'
    )
    random.seed(32)
    random.randint(0, 2000000000)
    cfg = football_env.config.Config(
        {'level': scenario, 'players': ['agent:left_players=1,right_players=0'], 'tracesdir': ''}
    )
    checkpoint = wrappers.CheckpointRewardWrapper(football_env.football_env.FootballEnv(cfg))
    single = wrappers.GetStateWrapper(
        wrappers.SingleAgentRewardWrapper(
            wrappers.SingleAgentObservationWrapper(wrappers.SMMWrapper(checkpoint))
        )
    )
    exchange_states(unstacked, single, lambda step: 5)
    # The counts of the state reach the wrapper, as in GRF.
    assert checkpoint._collected_checkpoints
    env.close()
    created.close()
    single.close()
    unstacked.close()


def test_gym_methods_and_wrappers_that_need_rendering():
    env = football_env.create_environment('academy_empty_goal_close', representation='simple115v2')
    # GRF: gym's `Env` seeds nothing, bounds no reward, and `disable_render` turns rendering off.
    assert env.seed(3) is None and env.disable_render() is None
    assert env.reward_range == (-float('inf'), float('inf'))
    wrapper = football_env.wrappers.SingleAgentRewardWrapper(env)
    assert wrapper.seed(3) is None and wrapper.reward_range == env.reward_range
    with pytest.raises(NotImplementedError, match='renders nothing'):
        football_env.wrappers.PixelsStateWrapper(env)
    with pytest.raises(ValueError, match='renders nothing'):
        football_env.wrappers.PeriodicDumpWriter(env, 2, render=True)
    env.close()


def test_periodic_dump_writer_writes_the_dumps_of_dump_frequency(tmp_path):
    def dumped(env):
        found = []
        for episode in range(3):
            random.seed(40 + episode)
            env.reset()
            done, wrote = False, False
            while not done:
                _, _, done, info = env.step(5)
                wrote |= 'dumps' in info
            found.append(wrote)
        env.close()
        return found

    options = {'representation': 'simple115v2', 'write_full_episode_dumps': True}
    by_option = football_env.create_environment(
        'academy_empty_goal_close', dump_frequency=2, logdir=str(tmp_path / 'option'), **options
    )
    by_hand = football_env.wrappers.PeriodicDumpWriter(
        football_env.create_environment(
            'academy_empty_goal_close', logdir=str(tmp_path / 'hand'), **options
        ),
        2,
    )
    # GRF: the first reset and every second one after it keep the dump options.
    assert dumped(by_hand) == dumped(by_option) == [True, False, True]


def test_multi_agent_to_single_agent_statics():
    wrapper = football_env.wrappers.MultiAgentToSingleAgent
    observation = [
        {'designated': 3, 'active': 1},
        {'designated': 3, 'active': 3},
        {'designated': 0, 'active': 0},
    ]
    assert wrapper.get_observation(observation) == observation[1:]
    actions = wrapper.get_action([5, 7], observation)
    assert actions[0] is football_action_set.action_builtin_ai and actions[1:] == [5, 7]
    with pytest.raises(AssertionError, match='raw'):
        wrapper.get_observation([{'active': 0}])


def test_football_action_set():
    actions = football_action_set
    assert [len(actions.action_set_v1), len(actions.action_set_v2)] == [19, 20]
    assert len(actions.full_action_set) == 33
    assert repr(actions.action_right) == 'right' and actions.action_right._backend_action == 5
    assert actions.action_set_dict['default'] is actions.action_set_v1
    assert actions.get_action_set({'action_set': 'v2'})[19] is actions.action_builtin_ai
    sticky = [repr(action) for action in actions.get_sticky_actions({'action_set': 'default'})]
    assert sticky == [
        'left',
        'top_left',
        'top',
        'top_right',
        'right',
        'bottom_right',
        'bottom',
        'bottom_left',
        'sprint',
        'dribble',
    ]
    assert actions.reverse_action_mapping[actions.action_shot] is actions.action_release_shot
    assert actions.disable_action(actions.action_top) is actions.action_release_direction
    assert actions.disable_action(actions.action_sprint) is actions.action_release_sprint
    action_set = actions.action_set_v1
    assert actions.named_action_from_action_set(action_set, np.int64(5)) is actions.action_right
    assert actions.named_action_from_action_set(action_set, actions.action_shot) is (
        actions.action_shot
    )
    assert actions.action_shot == actions.action_shot and actions.action_idle < actions.action_left
    assert actions.action_shot.is_in_actionset({'action_set': 'default'})
    assert not actions.action_switch.is_in_actionset({'action_set': 'v2'})
    assert isinstance(actions.action_shot, actions.CoreAction)
    # GRF: `step` takes a core action as its index.
    outputs = []
    for action in (actions.action_shot, 12):
        random.seed(19)
        env = football_env.create_environment(
            'academy_empty_goal_close', representation='simple115v2'
        )
        env.reset()
        outputs.append(env.step(action)[0].tobytes())
        env.close()
    assert outputs[0] == outputs[1]


def test_scenario_builder_and_scenarios():
    builder_module = football_env.scenario_builder
    assert '11_vs_11_stochastic' in builder_module.all_scenarios()
    random.seed(15)
    config = football_env.config.Config(
        {'level': 'academy_empty_goal_close', 'players': ['agent:left_players=2']}
    )
    expected = config.ScenarioConfig()
    seed = config['game_engine_random_seed']
    # GRF: `Scenario(config)` builds the level of the configuration's episode, and draws a new
    # seed into the episode's values, since the configuration holds none of its own.
    random.seed(16)
    builder = builder_module.Scenario(config)
    assert builder.EpisodeNumber() == config['episode_number'] == 1
    assert builder.ScenarioConfig() is builder.config()
    built = builder.config()
    assert isinstance(built.left_team[0], builder_module.Player)
    assert [player.key() for player in built.left_team] == [
        player.key() for player in expected.left_team
    ]
    assert built.ball_position == expected.ball_position
    assert (built.left_agents, built.right_agents) == (2, 0)
    assert config['game_engine_random_seed'] == grf_draws(16, 1)[0] != seed
    assert built.game_engine_random_seed == config['game_engine_random_seed']
    assert built.reverse_team_processing == config['reverse_team_processing']
    # GRF: a seed of the configuration's own stays, and nothing is drawn.
    fixed = football_env.config.Config({'level': '5_vs_5', 'game_engine_random_seed': 7})
    state = random.getstate()
    assert builder_module.Scenario(fixed).config().game_engine_random_seed == 7
    assert random.getstate() == state
    builder.SetTeam(builder_module.Team.e_Right)
    builder.AddPlayer(1.0, 0.0, builder_module.Role.e_PlayerRole_GK)
    assert len(built.right_team) == len(expected.right_team) + 1
    with pytest.raises(TypeError, match='Config'):
        builder_module.Scenario(1)
    assert grf_scenarios.e_PlayerRole_CF is builder_module.Role.e_PlayerRole_CF
    assert grf_scenarios.Team is builder_module.Team


def test_dump_reads_and_prints_as_grf(tmp_path):
    random.seed(23)
    env = football_env.create_environment(
        'academy_empty_goal_close',
        representation='simple115v2',
        write_full_episode_dumps=True,
        logdir=str(tmp_path),
    )
    env.reset()
    expected = env.unwrapped._config.get_dictionary()
    for step in range(10):
        _, _, done, _ = env.step(football_action_set.action_right if step == 0 else 5)
        if done:
            break
    # The dump of an episode that the environment closes in is written at the close, as GRF's.
    env.close()
    paths = sorted(glob.glob(os.path.join(str(tmp_path), '*.dump')))
    assert len(paths) == 1
    helpers = football_env.script_helpers.ScriptHelpers()
    steps = helpers.load_dump(paths[0])
    # GRF: the dump holds the configuration of the episode.
    dumped = steps[0]['debug']['config']
    assert list(dumped) == list(expected)
    assert dumped['episode_number'] % 2 == expected['episode_number'] % 2
    del dumped['episode_number'], expected['episode_number']
    assert dumped == expected
    assert steps[0]['debug']['action'] == [football_action_set.action_right]
    assert all(
        isinstance(action, football_action_set.CoreAction)
        for step in steps
        for action in step['debug']['action']
    )
    left_team = steps[0]['observation']['left_team']
    assert isinstance(left_team, np.ndarray) and left_team.dtype == np.float64
    output = tmp_path / 'dump.txt'
    dump_to_txt.main(['--trace_file', paths[0], '--output', str(output)])
    text = output.read_text()
    assert text == str(helpers.load_dump(paths[0]))
    assert "'action': [right]" in text and 'array([' in text
    dump_to_txt.main([f'--trace_file={paths[0]}', f'--output={output}', '--noinclude_debug'])
    assert "'debug'" not in output.read_text()

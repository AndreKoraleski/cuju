# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Checks `control_all_players` and the `MultiAgentToSingleAgent` wrapper on GRF's test scenario."""

import random

import pytest
import torch

import cuju
from conftest import tensor_bytes

np = pytest.importorskip('numpy')
cuju_env = pytest.importorskip('cuju.env')
cuju_wrappers = pytest.importorskip('cuju.wrappers')

# GRF's `scenarios/tests/multiagent_wrapper.py`.
SCENARIO = """
from . import *


def build_scenario(builder):
  builder.config().control_all_players = True
  builder.config().game_duration = 3000
  builder.config().right_team_difficulty = 1.0
  builder.config().left_team_difficulty = 1.0
  builder.config().deterministic = False
  if builder.EpisodeNumber() % 2 == 0:
    first_team = Team.e_Left
    second_team = Team.e_Right
  else:
    first_team = Team.e_Right
    second_team = Team.e_Left
  builder.SetTeam(first_team)
  builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK)
  builder.AddPlayer(0.000000,  0.020000, e_PlayerRole_RM)
  builder.AddPlayer(0.000000, -0.020000, e_PlayerRole_CF)
  builder.AddPlayer(-0.422000, -0.19576, e_PlayerRole_LB)
  builder.AddPlayer(-0.500000, -0.06356, e_PlayerRole_CB)
  builder.AddPlayer(-0.500000, 0.063559, e_PlayerRole_CB)
  builder.AddPlayer(-0.422000, 0.195760, e_PlayerRole_RB)
  builder.AddPlayer(-0.184212, -0.10568, e_PlayerRole_CM)
  builder.AddPlayer(-0.267574, 0.000000, e_PlayerRole_CM)
  builder.AddPlayer(-0.184212, 0.105680, e_PlayerRole_CM)
  builder.AddPlayer(-0.010000, -0.21610, e_PlayerRole_LM)
  builder.SetTeam(second_team)
  builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK)
  builder.AddPlayer(-0.050000, 0.000000, e_PlayerRole_RM)
  builder.AddPlayer(-0.010000, 0.216102, e_PlayerRole_CF)
  builder.AddPlayer(-0.422000, -0.19576, e_PlayerRole_LB)
  builder.AddPlayer(-0.500000, -0.06356, e_PlayerRole_CB)
  builder.AddPlayer(-0.500000, 0.063559, e_PlayerRole_CB)
  builder.AddPlayer(-0.422000, 0.195760, e_PlayerRole_RB)
  builder.AddPlayer(-0.184212, -0.10568, e_PlayerRole_CM)
  builder.AddPlayer(-0.267574, 0.000000, e_PlayerRole_CM)
  builder.AddPlayer(-0.184212, 0.105680, e_PlayerRole_CM)
  builder.AddPlayer(-0.010000, -0.21610, e_PlayerRole_LM)
"""


@pytest.fixture(scope='module')
def scenario(tmp_path_factory):
    path = tmp_path_factory.mktemp('scenarios') / 'multiagent_wrapper.py'
    path.write_text(SCENARIO)
    return str(path)


@pytest.fixture(scope='module')
def bundle(scenario, tmp_path_factory):
    from cuju import scenarios

    path = tmp_path_factory.mktemp('bundles') / 'multiagent_wrapper_l0_r0.cjb'
    return scenarios.build_bundle(scenario, str(path))


def test_agents_must_hold_every_player(bundle):
    batch = cuju.BatchEnvironment(bundle, environment_count=1, device='cpu', agent_counts=[(11, 0)])
    batch.close()
    with pytest.raises(RuntimeError, match='control_all_players'):
        cuju.BatchEnvironment(bundle, environment_count=1, device='cpu', agent_counts=[(2, 0)])


def designated_actions(batch, actions, sides):
    """Returns the batch actions of GRF's `MultiAgentToSingleAgent.get_action`."""
    raw = batch.raw()
    inner = torch.full((batch.environment_count, batch.agents), 19, dtype=torch.int32)
    for environment in range(batch.environment_count):
        chosen = [
            agent
            for agent in range(batch.agents)
            if int(raw['active'][environment, agent]) == int(raw['designated'][environment, agent])
        ]
        assert len(chosen) == len(sides)
        for agent, action in zip(chosen, actions[environment].tolist(), strict=True):
            inner[environment, agent] = action
    return inner


@pytest.mark.parametrize('counts', [(11, 0), (11, 11)])
def test_wrapper_equals_grf_selection(bundle, counts):
    sides = [side for side, count in enumerate(counts) if count]
    options = dict(
        environment_count=3, device='cpu', action_set='v2', with_raw=True, agent_counts=[counts]
    )
    wrapped = cuju.BatchEnvironment(bundle, **options)
    plain = cuju.BatchEnvironment(bundle, **options)
    wrapper = cuju_wrappers.MultiAgentToSingleAgent(
        wrapped, *(int(bool(count)) for count in counts), action_set='default'
    )
    observation = wrapper.reset()
    plain.reset()
    assert observation.shape == (3, len(sides), 115)
    for step in range(80):
        actions = torch.tensor(
            [
                [(step * 7 + environment + side * 5) % 19 for side in sides]
                for environment in range(3)
            ]
        )
        inner = designated_actions(plain, actions, sides)
        observation, reward, done, _ = wrapper.step(actions)
        expected, expected_reward, expected_done, _ = plain.step(inner)
        assert tensor_bytes(reward) == tensor_bytes(expected_reward)
        assert tensor_bytes(done) == tensor_bytes(expected_done)
        index = wrapper._index
        rows = torch.arange(3)[:, None]
        assert tensor_bytes(observation) == tensor_bytes(expected[rows, index])
    wrapped.close()
    plain.close()


def test_stacked_wrapper_stacks_the_designated_observations(bundle):
    options = dict(
        environment_count=2, device='cpu', action_set='v2', with_raw=True, agent_counts=[(11, 0)]
    )
    stacked = cuju_wrappers.MultiAgentToSingleAgent(
        cuju.BatchEnvironment(bundle, **options), 1, 0, stacked=True, squeeze=True
    )
    single = cuju_wrappers.MultiAgentToSingleAgent(
        cuju.BatchEnvironment(bundle, **options), 1, 0, squeeze=True
    )
    frames = [single.reset()] * 4
    assert torch.equal(stacked.reset(), torch.cat(frames, -1))
    for step in range(30):
        actions = torch.full((2,), step % 19)
        frames = [*frames[1:], single.step(actions)[0]]
        observation, reward, _, _ = stacked.step(actions)
        assert torch.equal(observation, torch.cat(frames, -1))
        assert reward.shape == (2,)
    stacked.close()
    single.close()


def test_create_environment_with_control_all_players(scenario):
    random.seed(1)
    env = cuju_env.create_environment(scenario, representation='simple115v2')
    assert env.observation_space.shape == (115,)
    assert env.batch.agents == 11
    with pytest.raises(RuntimeError, match='reset'):
        env.step(0)
    observation = env.reset()
    assert observation.shape == (115,)
    observation, reward, _, _ = env.step([5])
    assert observation.shape == (115,) and isinstance(reward, np.float32)
    # GRF: `GetStateWrapper` refuses states through `MultiAgentToSingleAgent`.
    with pytest.raises(ValueError, match='no states'):
        env.get_state()
    env.close()
    env = cuju_env.create_environment(
        scenario,
        representation='raw',
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=1,
    )
    observation = env.reset()
    assert len(observation) == 2
    assert all(entry['active'] == entry['designated'] for entry in observation)
    _, reward, _, _ = env.step([0, 0])
    # GRF: the wrapper passes on the reward of all 22 agents.
    assert reward.shape == (22,)
    env.close()
    with pytest.raises(RuntimeError, match='control_all_players'):
        cuju_env.create_environment(scenario, number_of_left_players_agent_controls=2)


def test_grf_wrapper_over_one_environment_equals_create_environment(scenario):
    grf_wrappers = pytest.importorskip('cuju.gfootball.env.wrappers')
    random.seed(4)
    expected = cuju_env.create_environment(scenario, representation='raw')
    random.seed(4)
    # GRF: the agent controls all 11 players, and the wrapper gives the policy the designated one.
    inner = cuju_env.create_environment(
        scenario,
        representation='raw',
        number_of_left_players_agent_controls=11,
        other_config_options={'action_set': 'v2'},
    )
    wrapper = grf_wrappers.MultiAgentToSingleAgent(inner, 1, 0)
    assert type(wrapper) is grf_wrappers.MultiAgentToSingleAgent
    assert wrapper.action_space.n == 20
    wrapped = grf_wrappers.SingleAgentRewardWrapper(wrapper)
    # Each reset draws its seed from Python's `random`, as GRF's does.
    random.seed(5)
    ours = wrapped.reset()
    random.seed(5)
    observations = [(ours, expected.reset())]
    for step in range(60):
        action = [(step * 7) % 19]
        ours, reward, done, info = wrapped.step(action)
        theirs, expected_reward, expected_done, expected_info = expected.step(action)
        observations.append((ours, theirs))
        assert reward.tobytes() == expected_reward.tobytes()
        assert (done, info) == (expected_done, expected_info)
    for ours, theirs in observations:
        assert len(ours) == len(theirs) == 1
        for key, value in ours[0].items():
            assert np.asarray(value).tobytes() == np.asarray(theirs[0][key]).tobytes(), key
    with pytest.raises(ValueError, match='not supported by MultiAgentToSingleAgent'):
        grf_wrappers.GetStateWrapper(wrapped).get_state()
    wrapped.close()
    expected.close()
    random.seed(4)
    default = cuju_env.create_environment(
        scenario, representation='raw', number_of_left_players_agent_controls=11
    )
    with pytest.raises(ValueError, match='builtin_ai'):
        grf_wrappers.MultiAgentToSingleAgent(default, 1, 0)
    default.close()


def test_control_all_players_with_extra_players(scenario):
    grf_wrappers = pytest.importorskip('cuju.gfootball.env.wrappers')
    extra = ['lazy:right_players=11']
    random.seed(17)
    env = cuju_env.create_environment(scenario, representation='raw', extra_players=extra)
    assert env.batch.agents == 22
    random.seed(17)
    # GRF: `create_environment` wraps the agent's players and adds the extra players beside.
    inner = cuju_env.create_environment(
        scenario,
        representation='raw',
        number_of_left_players_agent_controls=11,
        extra_players=extra,
        other_config_options={'action_set': 'v2'},
    )
    expected = grf_wrappers.SingleAgentRewardWrapper(
        grf_wrappers.MultiAgentToSingleAgent(inner, 1, 0)
    )
    random.seed(18)
    observations = [(env.reset(), None)]
    random.seed(18)
    observations[0] = (observations[0][0], expected.reset())
    for step in range(60):
        action = [(step * 5) % 19]
        ours, reward, done, info = env.step(action)
        theirs, expected_reward, expected_done, expected_info = expected.step(action)
        observations.append((ours, theirs))
        assert np.asarray(reward).tobytes() == np.asarray(expected_reward).tobytes()
        assert (done, info) == (expected_done, expected_info)
        if done:
            break
    for ours, theirs in observations:
        assert len(ours) == len(theirs) == 1
        for key, value in ours[0].items():
            assert np.asarray(value).tobytes() == np.asarray(theirs[0][key]).tobytes(), key
    env.close()
    expected.close()


def test_grf_wrapper_over_a_batch_is_the_batched_port(bundle):
    grf_wrappers = pytest.importorskip('cuju.gfootball.env.wrappers')
    options = dict(
        environment_count=1, device='cpu', action_set='v2', with_raw=True, agent_counts=[(11, 0)]
    )
    wrapper = grf_wrappers.MultiAgentToSingleAgent(cuju.BatchEnvironment(bundle, **options), 1, 0)
    assert type(wrapper) is cuju_wrappers.MultiAgentToSingleAgent
    wrapper.close()

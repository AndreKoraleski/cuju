# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks GRF's `Config` and its scenario hooks in `cuju.gfootball.env.config`."""

import os
import random

import pytest

import cuju
from cuju import scenarios
from cuju.gfootball.env import config

ASSETS = os.path.join(os.path.dirname(cuju.__file__), 'data', 'assets')


def test_a_config_builds_the_scenario_and_draws_its_seed():
    random.seed(4)
    generator = random.Random(4)
    values = config.Config({'level': 'academy_empty_goal_close'})
    seed = generator.randint(0, 2000000000)
    # GRF: one draw, the seed of the episode, kept as a value of the scenario.
    assert random.getstate() == generator.getstate()
    assert values['episode_number'] == 1
    assert values['game_engine_random_seed'] == seed
    assert values['reverse_team_processing'] == bool(seed % 2)
    assert values.get_dictionary()['game_engine_random_seed'] == seed
    assert 'game_engine_random_seed' not in values.serialize()
    scenario = values.ScenarioConfig()
    assert scenario.game_duration == 400 and scenario.end_episode_on_score
    assert scenario.game_engine_random_seed == seed
    assert scenario.reverse_team_processing == bool(seed % 2)
    assert (scenario.left_agents, scenario.right_agents) == (1, 0)
    assert [player.key() for player in scenario.right_team] == [(1.0, 0.0, 0, False, True)]
    assert scenario.ball_position[:2] == [0.77, 0.0]


def test_new_scenario_follows_the_episode_number():
    values = config.Config({'level': '11_vs_11_kaggle', 'game_engine_random_seed': 7})
    state = random.getstate()
    first = values.ScenarioConfig()
    values.NewScenario()
    # GRF: a fixed seed draws nothing, and the scenario swaps its teams with the parity.
    assert random.getstate() == state
    assert values['episode_number'] == 2
    second = values.ScenarioConfig()
    assert first.game_engine_random_seed == second.game_engine_random_seed == 7
    assert [player.key() for player in first.left_team] == [
        player.key() for player in second.right_team
    ]
    values.NewScenario(inc=0)
    assert values['episode_number'] == 2


def test_set_scenario_value_lasts_one_episode():
    values = config.Config({'level': '5_vs_5', 'players': ['agent:left_players=2,right_players=1']})
    assert values.number_of_left_players() == 2 and values.number_of_right_players() == 1
    assert values.number_of_players_agent_controls() == 3
    assert (values.ScenarioConfig().left_agents, values.ScenarioConfig().right_agents) == (2, 1)
    values.set_scenario_value('action_set', 'full')
    assert values['action_set'] == 'full' and 'action_set' in values
    assert values.serialize()['action_set'] == 'default'
    values.NewScenario()
    assert values['action_set'] == 'default'


def test_a_scenario_file_and_missing_levels(tmp_path):
    path = tmp_path / 'one_team.py'
    path.write_text(
        'from . import *\n\n\n'
        'def build_scenario(builder):\n'
        '    builder.config().game_duration = 10\n'
        '    builder.config().deterministic = True\n'
        '    builder.SetTeam(Team.e_Left)\n'
        '    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)\n'
    )
    values = config.Config({'level': str(path), 'game_engine_random_seed': 3})
    scenario = values.ScenarioConfig()
    # GRF: an empty team gets a lazy goalkeeper, and a deterministic scenario keeps its seed.
    assert [player.key() for player in scenario.right_team] == [(-1.0, 0.42, 0, True, True)]
    assert scenario.game_engine_random_seed == 42
    assert 'reverse_team_processing' not in values
    # A bundle holds no scenario file.
    for level in ('no_such_scenario', os.path.join(ASSETS, '5_vs_5_l0_r0.cjb')):
        with pytest.raises(NotImplementedError, match='scenario file'):
            config.Config({'level': level, 'game_engine_random_seed': 1}).ScenarioConfig()


def test_player_definitions_follow_grf():
    assert config.parse_player_definition('bot') == (
        'bot',
        {'left_players': 1, 'right_players': 0},
    )
    # GRF: counts stay the strings of the definition.
    assert config.parse_player_definition('lazy:right_players=2,speed=3') == (
        'lazy',
        {'left_players': 0, 'right_players': '2', 'speed': '3'},
    )
    assert config.count_players('agent:left_players=2,right_players=1') == 3
    assert config.get_agent_number_of_players(['agent:left_players=2', 'bot:right_players=1']) == 2


@pytest.mark.parametrize('name', ['academy_corner', '11_vs_11_kaggle', 'academy_run_to_score'])
def test_shipped_scenario_files_build_the_shipped_bundles(name, tmp_path):
    source = os.path.join(os.path.dirname(cuju.__file__), 'data', 'scenarios', name + '.py')
    out = tmp_path / f'{name}.cjb'
    scenarios.build_bundle(source, str(out), name=name, device='cpu')
    with open(os.path.join(ASSETS, f'{name}_l0_r0.cjb'), 'rb') as handle:
        assert out.read_bytes() == handle.read()

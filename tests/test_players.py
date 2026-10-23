# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Checks the batched `bot` and `lazy` players against GRF's own code on the same observations."""

import random

import pytest
import torch

import cuju
import cuju.players
from conftest import CUDA

np = pytest.importorskip('numpy')
cuju_env = pytest.importorskip('cuju.env')

# The indices of GRF's core actions in the `full` set.
FULL = {
    'idle': 0,
    'left': 1,
    'top_left': 2,
    'top': 3,
    'top_right': 4,
    'right': 5,
    'bottom_right': 6,
    'bottom': 7,
    'bottom_left': 8,
    'long_pass': 9,
    'short_pass': 11,
    'shot': 12,
    'sprint': 13,
    'pressure': 21,
    'release_pressure': 30,
}


class GrfBot:
    """GRF's `gfootball/env/players/bot.py` as it is, with action names for its `CoreAction`s."""

    def __init__(self):
        self._observation = None
        self._last_action = 'idle'
        self._shoot_distance = 0.15
        self._pressure_enabled = False

    def _object_distance(self, object1, object2):
        return np.linalg.norm(np.array(object1) - np.array(object2))

    def _direction_action(self, delta):
        all_directions = [
            'top',
            'top_left',
            'left',
            'bottom_left',
            'bottom',
            'bottom_right',
            'right',
            'top_right',
        ]
        all_directions_vec = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]
        all_directions_vec = [np.array(v) / np.linalg.norm(np.array(v)) for v in all_directions_vec]
        best_direction = np.argmax([np.dot(delta, v) for v in all_directions_vec])
        return all_directions[best_direction]

    def _closest_opponent_to_object(self, o):
        min_d = None
        closest = None
        for p in self._observation['right_team']:
            d = self._object_distance(o, p)
            if min_d is None or d < min_d:
                min_d = d
                closest = p
        assert closest is not None
        return closest

    def _closest_front_opponent(self, o, target):
        delta = target - o
        min_d = None
        closest = None
        for p in self._observation['right_team']:
            delta_opp = p - o
            if np.dot(delta, delta_opp) <= 0:
                continue
            d = self._object_distance(o, p)
            if min_d is None or d < min_d:
                min_d = d
                closest = p
        return closest

    def _score_pass_target(self, active, player):
        trajectory = player - active
        dist_closest_traj = None
        for i in range(10):
            position = active + (i + 1) / 10.0 * trajectory
            opp_traj = self._closest_opponent_to_object(position)
            dist_traj = self._object_distance(position, opp_traj)
            if dist_closest_traj is None or dist_traj < dist_closest_traj:
                dist_closest_traj = dist_traj
        return -dist_closest_traj

    def _best_pass_target(self, active):
        best_score = None
        best_target = None
        for player in self._observation['left_team']:
            if self._object_distance(player, active) > 0.3:
                continue
            score = self._score_pass_target(active, player)
            if best_score is None or score > best_score:
                best_score = score
                best_target = player
        return best_target

    def _avoid_opponent(self, active, opponent, target):
        delta = opponent - active
        delta_t = target - active
        new_delta = [delta[1], -delta[0]]
        if delta_t[0] * new_delta[0] < 0:
            new_delta = [-new_delta[0], -new_delta[1]]
        return self._direction_action(new_delta)

    def _get_action(self):
        active = self._observation['left_team'][self._observation['active']]
        if self._observation['game_mode'] != 0:
            return 'long_pass'
        if self._observation['ball_owned_team'] == 1:
            if self._last_action == 'pressure':
                return 'sprint'
            self._pressure_enabled = True
            return 'pressure'
        if self._pressure_enabled:
            self._pressure_enabled = False
            return 'release_pressure'
        target_x = 0.85
        if self._shoot_distance > np.linalg.norm(self._observation['ball'][:2] - [target_x, 0]):
            return 'shot'
        move_target = [target_x, 0]
        move_action = self._direction_action(move_target - active)
        closest_front_opponent = self._closest_front_opponent(active, move_target)
        if closest_front_opponent is not None:
            dist_front_opp = self._object_distance(active, closest_front_opponent)
        else:
            dist_front_opp = 2.0
        if dist_front_opp < 0.08:
            best_pass_target = self._best_pass_target(active)
            if np.array_equal(best_pass_target, active):
                move_action = self._avoid_opponent(active, closest_front_opponent, move_target)
            else:
                delta = best_pass_target - active
                direction_action = self._direction_action(delta)
                if self._last_action == direction_action:
                    return 'short_pass'
                return direction_action
        return move_action

    def take_action(self, observation):
        self._observation = observation
        self._last_action = self._get_action()
        return FULL[self._last_action]


def views(batch, agent):
    """Returns the GRF observation of an agent in every environment of a batch."""
    raw = batch.raw()
    right = agent >= int(batch.left_agents[0])
    sizes = (int(batch.left_players[0]), int(batch.right_players[0]))
    own, other = sizes[::-1] if right else sizes
    return [
        cuju_env.grf_observation(raw, environment, agent, own, other)
        for environment in range(batch.environment_count)
    ]


@pytest.mark.parametrize('assets', ['11_vs_11_stochastic_l1_r1', '5_vs_5_l1_r1'])
def test_bot_equals_grf_bot(assets):
    environments = 2
    batch = cuju.BatchEnvironment(
        assets, environment_count=environments, device='cpu', action_set='full', with_raw=True
    )
    bots = [cuju.players.Bot(batch, agent) for agent in (0, 1)]
    references = [[GrfBot() for _ in range(environments)] for _ in bots]
    batch.reset()
    generator = torch.Generator().manual_seed(3)
    for step in range(100):
        actions = torch.zeros((environments, 2), dtype=torch.int32)
        for agent, bot in enumerate(bots):
            ours = bot.act()
            theirs = [
                reference.take_action(view)
                for reference, view in zip(references[agent], views(batch, agent), strict=True)
            ]
            assert ours.tolist() == theirs, f'step {step}, agent {agent}'
            actions[:, agent] = ours
        # Random moves now and then put the bots in other situations.
        noise = torch.rand((environments, 2), generator=generator) < 0.2
        random_actions = torch.randint(0, 19, (environments, 2), generator=generator)
        batch.step(torch.where(noise, random_actions, actions).to(torch.int32))
    batch.close()


def test_scripted_players_of_create_environment():
    # Environment 0 of a batch with seed 2 plays GRF's environment after random.seed(2 * 2**32).
    random.seed(2 * 2**32)
    environment = cuju_env.create_environment(
        'academy_3_vs_1_with_keeper',
        representation='raw',
        number_of_left_players_agent_controls=0,
        extra_players=['bot:left_players=1', 'lazy:right_players=1'],
        other_config_options={'action_set': 'full'},
    )
    batch = cuju.BatchEnvironment(
        'academy_3_vs_1_with_keeper',
        environment_count=1,
        device='cpu',
        seed=2,
        action_set='full',
        with_raw=True,
        auto_reset=False,
        agent_counts=[(1, 1)],
    )
    bot, lazy = cuju.players.Bot(batch, 0), cuju.players.Lazy(batch, 1)
    observation = environment.reset()
    batch.reset()
    assert observation == []
    for _ in range(80):
        _, _, terminated, _ = environment.step([])
        _, _, done, _ = batch.step(torch.stack([bot.act(), lazy.act()], 1))
        ours, theirs = environment.batch.raw(), batch.raw()
        assert all(torch.equal(ours[name], theirs[name]) for name in ours)
        assert terminated == bool(done[0])
        if terminated:
            break
    environment.close()
    batch.close()


def test_refusals_and_lazy():
    batch = cuju.BatchEnvironment('5_vs_5_l2_r0', environment_count=2, device='cpu', with_raw=True)
    with pytest.raises(ValueError, match='full'):
        cuju.players.Bot(batch, 0)
    lazy = cuju.players.Lazy(batch, [0, 1])
    assert lazy.act().tolist() == [[0, 0], [0, 0]]
    batch.close()
    with pytest.raises(ValueError, match='full'):
        cuju_env.create_environment('5_vs_5', extra_players=['bot:right_players=1'])
    with pytest.raises(ValueError, match='one player'):
        cuju_env.create_environment(
            '5_vs_5',
            extra_players=['bot:right_players=2'],
            other_config_options={'action_set': 'full'},
        )


@pytest.mark.skipif(not CUDA, reason='no CUDA device or library')
def test_bot_on_the_gpu_equals_the_cpu():
    actions = {}
    for device in ('cpu', 'cuda'):
        batch = cuju.BatchEnvironment(
            '11_vs_11_stochastic_l1_r1',
            environment_count=4,
            device=device,
            action_set='full',
            with_raw=True,
        )
        bots = [cuju.players.Bot(batch, agent) for agent in (0, 1)]
        batch.reset()
        history = []
        for _ in range(100):
            step = torch.stack([bot.act() for bot in bots], 1)
            history.append(step.cpu())
            batch.step(step)
        actions[device] = torch.stack(history)
        batch.close()
    assert torch.equal(actions['cpu'], actions['cuda'])

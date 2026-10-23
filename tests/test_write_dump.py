# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks GRF's `write_dump(name)` on the environment of `create_environment`."""

import os
import random
import re

import pytest

import cuju.dumps

pytest.importorskip('numpy')
pytest.importorskip('gymnasium')
cuju_env = pytest.importorskip('cuju.env')

from test_dumps import same  # noqa: E402


def without_config(steps):
    for step in steps:
        step['debug'].pop('config', None)
    return steps


def test_a_dump_holds_the_last_100_steps(tmp_path):
    random.seed(1)
    env = cuju_env.create_environment(
        '11_vs_11_easy_stochastic',
        representation='simple115v2',
        write_full_episode_dumps=True,
        logdir=str(tmp_path),
    )
    env.reset()
    for step in range(130):
        env.step(step % 19)
    path = env.write_dump('mine')
    assert re.fullmatch(r'mine_\d{8}-\d{12}', os.path.basename(path))
    assert os.path.dirname(path) == str(tmp_path)
    # GRF: a name dumps once an episode, and the options' own dumps start only by themselves.
    assert env.write_dump('mine') is None
    assert env.write_dump('episode_done') is None and env.write_dump('score') is None
    short = env.write_dump('shutdown')
    expected = env._config.get_dictionary()
    # Closing writes the unfinished episode, as GRF leaves it.
    env.close()
    (full_path,) = [name for name in os.listdir(tmp_path) if name.startswith('episode_done')]
    full = cuju.dumps.read_dump(os.path.join(tmp_path, full_path))
    steps = cuju.dumps.read_dump(path + '.dump')
    assert len(full) == 130 and len(steps) == 100 and steps[0]['debug']['frame_cnt'] == 30
    # GRF: the configuration of the environment, as `get_dictionary` gives it.
    assert steps[0]['debug']['config'] == expected
    assert expected['tracesdir'] == str(tmp_path) and expected['episode_number'] == 2
    assert all('config' not in step['debug'] for step in steps[1:])
    same(without_config(steps), without_config(full[30:130]))
    same(cuju.dumps.read_dump(short + '.dump'), cuju.dumps.read_dump(path + '.dump'))


def test_dumps_before_a_step_and_after_the_end(tmp_path):
    random.seed(2)
    env = cuju_env.create_environment('academy_empty_goal_close', logdir=str(tmp_path))
    env.reset()
    path = env.write_dump('empty')
    # GRF: the dump of no step is an empty file.
    assert os.path.getsize(path + '.dump') == 0
    for _ in range(5):
        env.step(5)
    steps = cuju.dumps.read_dump(env.write_dump('five') + '.dump')
    assert [step['debug']['frame_cnt'] for step in steps] == list(range(5))
    assert steps[0]['debug']['config']['level'] == 'academy_empty_goal_close'
    done = False
    while not done:
        _, _, done, _ = env.step(12)
    with pytest.raises(RuntimeError, match='reset'):
        env.write_dump('late')
    env.reset()
    env.step(0)
    assert len(cuju.dumps.read_dump(env.write_dump('five') + '.dump')) == 1
    env.close()


def test_a_dump_follows_a_loaded_state(tmp_path):
    random.seed(3)
    env = cuju_env.create_environment(
        'academy_3_vs_1_with_keeper', representation='raw', logdir=str(tmp_path)
    )
    env.reset()
    observations = []
    for step in range(20):
        if step == 8:
            state = env.get_state()
        observations.append(env.step(step % 19)[0])
    env.set_state(state)
    for step in range(8, 14):
        env.step(step % 19)
    steps = cuju.dumps.read_dump(env.write_dump('both') + '.dump')
    env.close()
    # GRF: the steps after the state join the steps of the episode before it.
    frames = [step['debug']['frame_cnt'] for step in steps]
    assert frames == list(range(20)) + list(range(8, 14))
    for index, step in enumerate(steps[20:]):
        left = observations[8 + index][0]['left_team']
        assert step['observation']['left_team'].numpy().tolist() == left.tolist()
    rewards = [step['reward'] for step in steps]
    assert steps[-1]['cumulative_reward'] == sum(rewards)

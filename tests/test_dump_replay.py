# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks `cuju.gfootball.replay` and GRF's `ScriptHelpers().replay(dump, render=False)`."""

import pytest

import cuju
import cuju.dumps
from conftest import CUDA, actions_at
from cuju import _replays
from cuju.gfootball import replay
from cuju.gfootball.env import script_helpers


def episode_dump(directory, scenario='academy_3_vs_1_with_keeper_l1_r1', **options):
    """Plays one episode of a batch and returns its full dump and the actions it played."""
    batch = cuju.BatchEnvironment(
        scenario, environment_count=1, device='cpu', with_raw=True, seed=11, **options
    )
    writer = cuju.dumps.DumpWriter(batch, directory, write_goal_dumps=True)
    batch.reset()
    rows = []
    while not any('episode_done' in path for path in writer.paths):
        actions = actions_at(len(rows), 1, batch.agents, batch.action_count)
        rows.append(actions[0].tolist())
        batch.step(actions)
        writer.record(actions)
    batch.close()
    return next(path for path in writer.paths if 'episode_done' in path), writer.paths, rows


def test_a_dump_replays_equal(tmp_path, capsys):
    path, _, rows = episode_dump(tmp_path)
    result = replay.check(path)
    assert result == (len(rows), True, None)
    script_helpers.ScriptHelpers().replay(path, render=False)
    assert replay.main(['--trace_file', path]) == 0
    assert f"equals the dump's {len(rows)} steps" in capsys.readouterr().out


def regenerated(path, rows):
    """Returns the steps of a dump played again, as `DumpWriter` builds them."""
    steps = cuju.dumps.read_dump(path)
    config = steps[0]['debug']['config']
    batch = _replays.batch_for(config)
    _replays.start(batch, config)
    player = _replays.Player(batch)
    player.begin(0)
    entries = []
    for row in rows:
        entry, _, _ = player.step(row)
        entries.append(entry)
    batch.close()
    entries[0]['debug']['config'] = config
    return entries


def test_a_difference_is_found(tmp_path, capsys):
    path, _, rows = episode_dump(tmp_path)
    changed = len(rows) // 2
    entries = regenerated(path, rows)
    assert replay.check(path) == (len(rows), True, None)
    entries[changed]['observation']['ball'].values[1] += 2**-30
    cuju.dumps._write_file(path, map(cuju.dumps._pickle, entries))
    result = replay.check(path)
    assert result.steps == changed and result.ended is False
    assert result.difference.startswith(f'step {changed}.observation.ball: the replay gives ')
    with pytest.raises(ValueError, match='differs from the dump at step'):
        script_helpers.ScriptHelpers().replay(path, render=False)
    assert replay.main([f'--trace_file={path}']) == 1
    assert 'differs from the dump at step' in capsys.readouterr().out


def test_refusals(tmp_path):
    path, _, rows = episode_dump(tmp_path, 'academy_empty_goal_close_l1_r0')
    helpers = script_helpers.ScriptHelpers()
    with pytest.raises(ValueError, match='render'):
        helpers.replay(path)
    with pytest.raises(ValueError, match='frames per second'):
        helpers.replay(path, fps=20, render=False)
    assert len(helpers.load_dump(path)) == len(rows)
    entries = regenerated(path, rows)[1:]
    entries[0]['debug']['config'] = cuju.dumps.read_dump(path)[0]['debug']['config']
    cuju.dumps._write_file(path, map(cuju.dumps._pickle, entries))
    with pytest.raises(ValueError, match='beginning of its episode'):
        replay.check(path)


def test_other_ticks_and_devices(tmp_path):
    path, _, rows = episode_dump(tmp_path, physics_steps_per_frame=5)
    assert replay.check(path) == (len(rows), True, None)
    assert replay.main(['--trace_file', path, '--fps', '10']) == 2
    if CUDA:
        assert replay.check(path, device='cuda') == (len(rows), True, None)

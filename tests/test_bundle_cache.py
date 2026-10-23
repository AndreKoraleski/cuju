# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the cache of built bundles and the warning for a scenario too long for 10 ticks a step."""

import warnings
from pathlib import Path

import pytest

import cuju
from conftest import CUDA, play_digest
from cuju import scenarios

SCENARIO = """from . import *


def build_scenario(builder):
    builder.config().game_duration = {duration}
    builder.config().end_episode_on_score = True
    builder.SetBallPosition(0.6, 0.0)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(0.6, 0.0, e_PlayerRole_CF)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(-0.8, 0.1, e_PlayerRole_CB)
"""


def scenario_file(directory, duration=400, name='drill'):
    path = directory / f'{name}.py'
    path.write_text(SCENARIO.format(duration=duration), encoding='utf-8')
    return str(path)


@pytest.fixture
def builds(monkeypatch):
    """Counts the bundles that `build_bundle` writes."""
    calls = []
    build = scenarios.build_bundle

    def counted(*arguments, **options):
        calls.append(arguments[0])
        return build(*arguments, **options)

    monkeypatch.setattr(scenarios, 'build_bundle', counted)
    return calls


def test_the_cache_is_off_by_default(tmp_path, monkeypatch, builds):
    monkeypatch.delenv('CUJU_BUNDLE_CACHE', raising=False)
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    assert scenarios.cache_directory() is None
    path = scenario_file(tmp_path)
    for _ in range(2):
        cuju.BatchEnvironment(path, environment_count=1, device='cpu').close()
    assert len(builds) == 2
    assert not (tmp_path / 'cache').exists()


def test_the_cache_directory_follows_the_variable(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path))
    monkeypatch.setenv('CUJU_BUNDLE_CACHE', '0')
    assert scenarios.cache_directory() is None
    monkeypatch.setenv('CUJU_BUNDLE_CACHE', '1')
    assert scenarios.cache_directory() == str(tmp_path / 'cuju' / 'bundles')
    monkeypatch.setenv('CUJU_BUNDLE_CACHE', str(tmp_path / 'mine'))
    assert scenarios.cache_directory() == str(tmp_path / 'mine')


def test_a_cached_file_builds_once_and_plays_the_same(tmp_path, monkeypatch, builds):
    path = scenario_file(tmp_path)
    monkeypatch.delenv('CUJU_BUNDLE_CACHE', raising=False)
    expected = play_digest(path, 'cpu', 30, environments=2)
    monkeypatch.setenv('CUJU_BUNDLE_CACHE', str(tmp_path / 'cache'))
    assert play_digest(path, 'cpu', 30, environments=2) == expected
    assert play_digest(path, 'cpu', 30, environments=2) == expected
    # One build without the cache, then one for both batches with it.
    assert len(builds) == 2
    (cached,) = (tmp_path / 'cache').iterdir()
    scenarios.build_bundle(path, str(tmp_path / 'direct.cjb'))
    assert cached.read_bytes() == (tmp_path / 'direct.cjb').read_bytes()
    assert scenarios.cached_bundle(path, str(tmp_path / 'cache')) == str(cached)
    assert len(builds) == 3


@pytest.mark.parametrize('cache', [False, True])
def test_a_scenario_file_builds_with_the_batch_animations(tmp_path, monkeypatch, cache):
    from cuju._batch.assets import _DATA_DIRECTORY

    path = scenario_file(tmp_path)
    # A copy of the shipped tables, which the batch names by a path of its own.
    tables = tmp_path / 'animations.bin'
    tables.write_bytes((Path(_DATA_DIRECTORY) / 'animations.bin').read_bytes())
    if cache:
        monkeypatch.setenv('CUJU_BUNDLE_CACHE', str(tmp_path / 'cache'))
    else:
        monkeypatch.delenv('CUJU_BUNDLE_CACHE', raising=False)
    seen = []
    build = scenarios.build_bundle

    def spied(*arguments, **options):
        seen.append(options.get('animations'))
        return build(*arguments, **options)

    monkeypatch.setattr(scenarios, 'build_bundle', spied)
    cuju.BatchEnvironment(path, animations=str(tables), environment_count=1, device='cpu').close()
    # The bundle and its key in the cache take the tables that the batch plays with.
    assert seen == [str(tables)]
    if cache:
        (cached,) = (tmp_path / 'cache').iterdir()
        found = scenarios.cached_bundle(path, str(tmp_path / 'cache'), animations=str(tables))
        assert found == str(cached) and len(seen) == 1


def test_an_edited_file_builds_anew(tmp_path, builds):
    path = scenario_file(tmp_path)
    first = scenarios.cached_bundle(path, str(tmp_path / 'cache'))
    scenario_file(tmp_path, duration=401)
    second = scenarios.cached_bundle(path, str(tmp_path / 'cache'))
    assert first != second and len(builds) == 2
    assert sorted(map(str, (tmp_path / 'cache').iterdir())) == sorted([first, second])


def test_an_unusable_cache_warns_and_builds_without_it(tmp_path, monkeypatch):
    (tmp_path / 'file').write_text('', encoding='utf-8')
    monkeypatch.setenv('CUJU_BUNDLE_CACHE', str(tmp_path / 'file' / 'cache'))
    with pytest.warns(UserWarning, match='builds the scenario file without it'):
        batch = cuju.BatchEnvironment(scenario_file(tmp_path), environment_count=1, device='cpu')
    assert batch.agents == 1
    batch.close()


@pytest.mark.skipif(not CUDA, reason='no CUDA device or library')
def test_both_backends_cache_the_same_bytes(tmp_path):
    path = scenario_file(tmp_path)
    on_cpu = scenarios.cached_bundle(path, str(tmp_path / 'cache'), device='cpu')
    on_gpu = scenarios.cached_bundle(path, str(tmp_path / 'cache'), device='cuda')
    with open(on_cpu, 'rb') as cpu, open(on_gpu, 'rb') as gpu:
        assert cpu.read() == gpu.read()


def test_a_scenario_too_long_for_10_ticks_warns(tmp_path):
    # A batch holds 512 seconds of play: 5,119 steps of 10 ticks, or 5,120 steps of 9.
    longest = scenario_file(tmp_path, duration=5119, name='longest')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        bundle = scenarios.build_bundle(longest, str(tmp_path / 'longest.cjb'))
    cuju.BatchEnvironment(bundle, environment_count=1, device='cpu').close()
    longer = scenario_file(tmp_path, duration=5120, name='longer')
    with pytest.warns(UserWarning, match=r'5120 steps.*5119 steps.*physics_steps_per_frame=9'):
        bundle = scenarios.build_bundle(longer, str(tmp_path / 'longer.cjb'))
    with pytest.raises(RuntimeError, match='seconds of play'):
        cuju.BatchEnvironment(bundle, environment_count=1, device='cpu')
    batch = cuju.BatchEnvironment(
        bundle, environment_count=1, device='cpu', physics_steps_per_frame=9
    )
    batch.close()

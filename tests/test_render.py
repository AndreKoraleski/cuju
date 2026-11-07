# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the state fields, the renderer's recordings, its live sources and its server."""

import gzip
import io
import json
import os
import struct
import threading
import time
import urllib.request

import numpy as np
import pytest
import torch

import cuju
import cuju.errors
import cuju.state
from conftest import actions_at, tensor_bytes
from cuju.render import _home, _library, _live, _recording, _server, _teams


@pytest.fixture
def render_home(tmp_path, monkeypatch):
    monkeypatch.setenv(_home.HOME_VARIABLE, str(tmp_path / 'render'))
    return tmp_path / 'render'


def test_layout_offsets():
    layout = cuju.state.layout()
    position = layout['match.players[0].base.humanoid.nodes[0].position']
    assert (position.size, position.type) == (12, 'N7blunted7Vector3E')
    # GRF: a node holds 44 bytes, so the fourteen nodes of a player lie one after another.
    last = layout['match.players[0].base.humanoid.nodes[13].position']
    assert last.offset - position.offset == 13 * 44
    assert layout['match.goalCount[1]'].offset == layout['match.goalCount[0]'].offset + 4
    assert all(field.offset >= 0 and field.size > 0 for field in layout.values())


def test_gather_refuses_bad_input():
    with pytest.raises(cuju.StateError):
        cuju.state.gather(b'abc', ['environment.step'])
    batch = cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=1, device='cpu')
    batch.reset()
    state = batch.get_state(0)
    with pytest.raises(cuju.ArgumentError):
        cuju.state.gather(state, ['no.such.field'])
    with pytest.raises(cuju.ChoiceError):
        cuju.state.gather(state, ['environment.step'], frame='mirror')


def test_pitch_frame_matches_raw(device):
    """The pitch frame turns every player into the raw observation's positions, bit for bit."""
    batch = cuju.BatchEnvironment(
        '5_vs_5_l4_r1', environment_count=2, device=device, with_raw=True, seed=3
    )
    batch.reset(reverse=[0, 1])
    reader = _recording.StateReader(10)
    for step in range(80):
        batch.step(actions_at(step, 2, 5, 19).to(device))
        raw = {name: tensor.cpu() for name, tensor in batch.raw().items()}
        for environment in range(2):
            state = batch.get_state(environment)
            order = reader._order(reader._values(state, 'pitch'))
            paths = [
                f'match.players[{player}].base.humanoid.spatialState.position'
                for player in range(10)
            ]
            positions = cuju.state.read(state, paths, frame='pitch')
            for player, (team, index) in enumerate(order):
                name = 'left_team' if team == 0 else 'right_team'
                x, y = positions[player][0], positions[player][1]
                expected = raw[name][environment, 0, index]
                # GRF: the raw observation divides by 54.4 and by -83.6 in double precision.
                assert np.float32(np.float64(x) / 54.4) == expected[0]
                assert np.float32(np.float64(y) / -83.6) == expected[1]
            frame = reader.frame(state)
            _, mode, left, right, owned_team, owned = struct.unpack_from('<6i', frame)
            assert [left, right] == raw['score'][environment, 0].tolist()
            assert mode == int(raw['game_mode'][environment, 0])
            assert owned_team == int(raw['ball_owned_team'][environment, 0])
            if owned >= 0:
                assert order[owned][1] == int(raw['ball_owned_player'][environment, 0])


def test_world_transforms_are_upright():
    batch = cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=1, device='cpu')
    batch.reset()
    reader = _recording.StateReader(10)
    for step in range(20):
        batch.step(actions_at(step, 1, 1, 19))
    frame = reader.frame(batch.get_state(0))
    head = _recording._FRAME_HEAD.size
    packed = np.frombuffer(frame, '<i2', 10 * 14 * 7, head).reshape(10, 14, 7).astype(np.float64)
    scale = [_recording.POSITION_SCALE] * 3 + [_recording.ROTATION_SCALE] * 4
    nodes = packed / np.array(scale)
    neck = _recording.NODES.index('neck')
    ankle = _recording.NODES.index('left_ankle')
    assert (nodes[:, neck, 2] > nodes[:, ankle, 2] + 1.0).all()
    assert np.allclose(np.linalg.norm(nodes[:, :, 3:], axis=2), 1, atol=2e-4)
    assert (np.abs(nodes[:, 0, 0]) < 60).all() and (np.abs(nodes[:, 0, 1]) < 45).all()


def test_watching_changes_no_output(device, render_home):
    """A batch that a viewer watches returns the same bits as one that nobody watches."""
    outputs = []
    for watched in (False, True):
        batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=2, device=device, seed=11)
        viewer = None
        if watched:
            source = cuju.render.watch(batch, environment=1)
            viewer = _live._Viewer()
            source._viewers.append(viewer)
        outputs_seen = [tensor_bytes(batch.reset())]
        for step in range(120):
            if viewer is not None:
                while not viewer.frames.empty():
                    viewer.frames.get_nowait()
            observation, reward, done, _ = batch.step(actions_at(step, 2, 4, 19).to(device))
            outputs_seen += [tensor_bytes(observation), tensor_bytes(reward), tensor_bytes(done)]
        outputs_seen += [batch.get_state(0), batch.get_state(1)]
        outputs.append(outputs_seen)
        if watched:
            assert viewer.header is not None
        batch.close()
    assert outputs[0] == outputs[1]


def test_render_variable_changes_no_digest(monkeypatch, render_home):
    plain = cuju.digest('5_vs_5_l4_r0', steps=60)
    monkeypatch.setenv(_live.LIVE_VARIABLE, 'on')
    assert cuju.digest('5_vs_5_l4_r0', steps=60) == plain
    assert not list((render_home / 'live').glob('*.json'))


def test_recorder_writes_episodes(render_home):
    batch = cuju.BatchEnvironment(
        'academy_empty_goal_close_l1_r0', environment_count=1, device='cpu'
    )
    recorder = cuju.render.Recorder(batch)
    batch.reset()
    recorder.capture()
    step = 0
    while len(recorder.saved) < 2:
        _, _, done, _ = batch.step(torch.tensor([[12 if step % 3 else 5]], dtype=torch.int32))
        recorder.capture(done)
        step += 1
    first = recorder.saved[0]
    document = _recording.read_header(first)
    frames = list(_recording.frames(first))
    assert document['frame_count'] == len(frames)
    assert document['second_half'] is None
    assert document['duration'] > 0
    assert document['engine_scorer']
    assert document['offside_called']
    assert all(len(frame) == document['frame_bytes'] for frame in frames)
    assert [struct.unpack_from('<i', frame)[0] for frame in frames] == list(range(len(frames)))
    summary = {entry['id']: entry for entry in _library.matches()}[first.stem]
    assert summary['steps'] == len(frames)
    assert summary['scenario'] == 'academy_empty_goal_close_l1_r0'
    # The shots of the episode show as touches during shot animations of the agent's player.
    players = len(document['players'])
    padded = -(-players // 4) * 4
    start = _recording._FRAME_HEAD.size + players * 14 * 14 + padded
    actions = [frame[start : start + players] for frame in frames]
    touches = [frame[start + padded : start + padded + players] for frame in frames]
    shots = [
        index
        for index in range(len(frames))
        if any(t and a == 8 for a, t in zip(actions[index], touches[index], strict=True))
    ]
    assert shots


def test_late_viewer_gets_history(render_home):
    batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=1, device='cpu')
    source = cuju.render.watch(batch, backlog=True)
    batch.reset()
    for step in range(40):
        batch.step(actions_at(step, 1, 4, 19))
    connection = _server.http.client.HTTPConnection('127.0.0.1', source.port, timeout=10)
    connection.request('GET', '/stream')
    response = connection.getresponse()
    start = response.read(12)
    (length,) = struct.unpack_from('<I', start, 8)
    document = json.loads(response.read(length))
    assert document['second_half'] == _recording.scenario_timing(batch.get_state(0))['second_half']
    size = document['frame_bytes']
    history = [response.read(size) for _ in range(40)]
    steps = [struct.unpack_from('<i', frame)[0] for frame in history]
    assert steps == list(range(1, 41))
    connection.close()
    batch.close()


def _get(address, path):
    with urllib.request.urlopen(address + path, timeout=10) as response:
        return response.status, response.read()


def _send(address, path, data, method='POST'):
    request = urllib.request.Request(address + path, data=data, method=method)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def test_server_api(render_home):
    batch = cuju.BatchEnvironment(
        'academy_empty_goal_close_l1_r0', environment_count=1, device='cpu'
    )
    recorder = cuju.render.Recorder(batch, title='a test match')
    batch.reset()
    recorder.capture()
    for _ in range(5):
        _, _, done, _ = batch.step(torch.zeros((1, 1), dtype=torch.int32))
        recorder.capture(done)
    path = recorder.save()
    server = _server.http.server.ThreadingHTTPServer(
        ('127.0.0.1', 0), _server.make_handler(_server._Requests())
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    address = f'http://127.0.0.1:{server.server_address[1]}'
    try:
        status, body = _get(address, '/api/matches')
        listed = json.loads(body)
        assert status == 200 and [entry['title'] for entry in listed] == ['a test match']
        moved = _send(
            address,
            f'/api/matches/{path.stem}',
            b'{"folder": "League / Week 1", "title": "renamed"}',
            'PATCH',
        )
        assert (moved['folder'], moved['title']) == ('League/Week 1', 'renamed')
        with pytest.raises(urllib.error.HTTPError):
            _send(address, f'/api/matches/{path.stem}', b'{"folder": "../out"}', 'PATCH')
        status, body = _get(address, f'/api/matches/{path.stem}')
        assert gzip.decompress(body).startswith(_recording.MAGIC)
        assert 'academy_empty_goal_close' in json.loads(_get(address, '/api/scenarios')[1])
        assert json.loads(_get(address, '/api/info')[1]) == {'version': cuju.__version__}
        folder = _teams.folder_bundle(
            [
                ('submission/main.py', b'def agent(obs):\n    return [0]\n'),
                ('submission/w.bin', b'1'),
            ]
        )
        team = _send(address, '/api/teams?name=idle&filename=submission&format=5v5', folder)
        assert (team['format'], len(team['style']['players'])) == ('5v5', 5)
        teams = json.loads(_get(address, '/api/teams')[1])
        assert [entry['id'] for entry in teams] == ['builtin', team['id']]
        changes = {
            'format': '11v11',
            'style': {'kit': {'pattern': 'hoops'}, 'players': [{'name': 'Ana', 'number': 9}]},
        }
        team = _send(address, f'/api/teams/{team["id"]}', json.dumps(changes).encode(), 'PATCH')
        assert len(team['style']['players']) == 11
        assert team['style']['players'][0] == {
            'name': 'Ana',
            'number': 9,
            'skin': 'grf',
            'hair': 'grf',
            'hair_color': 'grf',
            'model': 'team',
            'celebration': 'team',
        }
        # A player takes a model of its own, and a model's colors give way to the team's.
        changes = {
            'style': {
                'players': [{}, {'model': 'kaykit-knight'}],
                'colors': {'kaykit-knight': {'#A1B2C3': '#ff0000'}},
            }
        }
        team = _send(address, f'/api/teams/{team["id"]}', json.dumps(changes).encode(), 'PATCH')
        assert team['style']['players'][1]['model'] == 'kaykit-knight'
        assert team['style']['colors'] == {'kaykit-knight': {'#a1b2c3': '#ff0000'}}
        moves = {
            'style': {
                'celebrations': ['knee_slide', 'clip:Cheer'],
                'players': [{'celebration': 'robot_dance'}],
            }
        }
        team = _send(address, f'/api/teams/{team["id"]}', json.dumps(moves).encode(), 'PATCH')
        assert team['style']['celebrations'] == ['knee_slide', 'clip:Cheer']
        assert team['style']['players'][0]['celebration'] == 'robot_dance'
        fans = {'style': {'crowd': {'colors': [[10, 20, 30], [200, 200, 0]]}}}
        team = _send(address, f'/api/teams/{team["id"]}', json.dumps(fans).encode(), 'PATCH')
        assert team['style']['crowd'] == {'colors': [[10, 20, 30], [200, 200, 0]]}
        for wrong in (
            {'players': [{'model': 'custom'}]},
            {'colors': {'grf': {}}},
            {'crowd': {'colors': []}},
            {'celebrations': ['Bad Name!']},
        ):
            with pytest.raises(urllib.error.HTTPError):
                _send(
                    address,
                    f'/api/teams/{team["id"]}',
                    json.dumps({'style': wrong}).encode(),
                    'PATCH',
                )
        with pytest.raises(urllib.error.HTTPError):
            _send(
                address,
                f'/api/teams/{team["id"]}',
                b'{"style": {"kit": {"pattern": "x"}}}',
                'PATCH',
            )
        png = b'\x89PNG\r\n\x1a\n' + bytes(32)
        team = _send(address, f'/api/teams/{team["id"]}/images/shirt-front', png, 'PUT')
        assert team['images'] == ['shirt-front']
        assert _get(address, f'/api/teams/{team["id"]}/images/shirt-front')[1] == png
        sound = b'OggS' + bytes(60)
        team = _send(address, f'/api/teams/{team["id"]}/sounds/goal', sound, 'PUT')
        assert team['sounds'] == ['goal']
        assert _get(address, f'/api/teams/{team["id"]}/sounds/goal')[1] == sound
        with pytest.raises(urllib.error.HTTPError):
            _send(address, f'/api/teams/{team["id"]}/sounds/goal', b'not a sound', 'PUT')
        archive = _get(address, f'/api/teams/{team["id"]}/export')[1]
        copy = _send(address, '/api/teams?filename=export.zip', archive)
        assert copy['name'] == 'idle' and copy['format'] == '11v11'
        assert copy['style'] == team['style'] and copy['images'] == ['shirt-front']
        assert copy['sounds'] == ['goal']
        assert _get(address, f'/api/teams/{copy["id"]}/images/shirt-front')[1] == png
        models = json.loads(_get(address, '/api/models')[1])
        assert all(model['joints']['body'] for model in models)
        request = json.dumps({'left': team['id'], 'right': 'builtin', 'scenario': '5_vs_5'})
        with pytest.raises(urllib.error.HTTPError):
            _send(address, '/api/requests', request.encode())
        for ability in ([1.5, None], [None], 'high', [True, None]):
            request = json.dumps({'left': 'builtin', 'right': 'builtin', 'ability': ability})
            with pytest.raises(urllib.error.HTTPError) as refused:
                _send(address, '/api/requests', request.encode())
            assert refused.value.code == 400
        if (_server._WEB / 'index.html').is_file():
            assert _get(address, '/index.html')[0] == 200
        with pytest.raises(urllib.error.HTTPError):
            _get(address, '/../../secret')
        # A page of another site, or a host name of another site that resolves here, is refused.
        port = server.server_address[1]
        for headers in (
            {'Origin': 'https://example.com'},
            {'Origin': 'http://127.0.0.1:1'},
            {'Host': f'evil.example:{port}'},
        ):
            connection = _server.http.client.HTTPConnection('127.0.0.1', port, timeout=10)
            connection.request('DELETE', f'/api/teams/{team["id"]}', headers=headers)
            assert connection.getresponse().status == 403
            connection.close()
        for kind, identifier in (('matches', path.stem), ('teams', team['id'])):
            request = urllib.request.Request(f'{address}/api/{kind}/{identifier}', method='DELETE')
            with urllib.request.urlopen(request, timeout=10) as response:
                assert response.status == 204
        assert json.loads(_get(address, '/api/matches')[1]) == []
    finally:
        server.shutdown()
        server.server_close()


def test_every_field_reads_and_mirrored_arrays_agree():
    layout = cuju.state.layout()
    batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=2, device='cpu', seed=3)
    batch.reset(reverse=[0, 1])
    for step in range(30):
        batch.step(actions_at(step, 2, 4, 19))
    for environment in range(2):
        state = batch.get_state(environment)
        # Every field of the layout lies inside the match, so all of them read at once.
        assert len(cuju.state.gather(state, list(layout.values()))) == sum(
            field.size for field in layout.values()
        )
        position, prediction = cuju.state.read(
            state, ['match.ball.positionBuffer', 'match.ball.predictions[0]'], frame='pitch'
        )
        assert np.sign(position[:2]).tolist() == np.sign(prediction[:2]).tolist()


def test_stacked_batch_shows_and_a_failing_source_spares_the_step(render_home, monkeypatch):
    batch = cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=2, device='cpu', stacked=True)
    source = cuju.render.watch(batch, backlog=True)
    batch.reset()
    batch.step(torch.zeros((2, 1), dtype=torch.int32))
    assert len(source._history) == 1

    def fail(done):
        raise OSError('disk full')

    monkeypatch.setattr(source, 'after_step', fail)
    batch.step(torch.zeros((2, 1), dtype=torch.int32))
    assert batch._live is None
    batch.close()


@pytest.mark.parametrize(
    'name', ['D:evil.py', '../out.py', '/etc/x', 'a\\b.py', 'a/../../b.py', 'a\x00b']
)
def test_hostile_member_names_are_refused(name):
    with pytest.raises(cuju.ArgumentError):
        _teams._check_member(name)


def test_live_stream_through_the_server(render_home):
    batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=1, device='cpu')
    source = cuju.render.watch(batch, backlog=True)
    batch.reset()
    for step in range(5):
        batch.step(actions_at(step, 1, 4, 19))
    server = _server.http.server.ThreadingHTTPServer(
        ('127.0.0.1', 0), _server.make_handler(_server._Requests())
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        listed = json.loads(_get(f'http://127.0.0.1:{port}', '/api/live')[1])
        assert [entry['id'] for entry in listed] == [source.identifier]
        connection = _server.http.client.HTTPConnection('127.0.0.1', port, timeout=10)
        connection.request('GET', f'/api/live/{source.identifier}/stream')
        response = connection.getresponse()
        assert response.status == 200
        assert response.read(8) == _recording.MAGIC
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        source.close()
    # A closed source lets go of its batch.
    assert batch._live is None
    batch.close()


def test_relative_home_resolves(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(_home.HOME_VARIABLE, 'relative-home')
    assert _home.home() == tmp_path / 'relative-home'


def test_recorder_keeps_the_final_goal_with_automatic_resets(render_home):
    batch = cuju.BatchEnvironment(
        'academy_empty_goal_close_l1_r0', environment_count=1, device='cpu'
    )
    recorder = cuju.render.Recorder(batch)
    batch.reset()
    recorder.capture()
    step = 0
    while not recorder.saved:
        _, reward, done, _ = batch.step(torch.tensor([[12 if step % 3 else 5]], dtype=torch.int32))
        recorder.capture(done)
        step += 1
    scored = float(reward[0, 0]) > 0
    last = list(_recording.frames(recorder.saved[0]))[-1]
    assert struct.unpack_from('<2i', last, 8) == ((1, 0) if scored else (0, 0))
    assert struct.unpack_from('<i', last, 28)[0] & 1


def test_ability_sets_the_difficulty_that_plays(render_home):
    path = cuju.render.play(
        'builtin', 'builtin', 'academy_empty_goal_close', seed=3, ability=(0.2, None)
    )
    raw = gzip.decompress(path.read_bytes())
    (length,) = struct.unpack_from('<I', raw, 8)
    header = json.loads(raw[12 : 12 + length])
    # The left plays the ability given, the right the scenario's own.
    assert header['teams'][0]['ability'] == 0.2
    assert 0 <= header['teams'][1]['ability'] <= 1
    with pytest.raises(cuju.errors.ArgumentError):
        cuju.render.play('builtin', 'builtin', 'academy_empty_goal_close', ability=(-0.1, None))


def test_command_line_reports_errors(render_home, capsys):
    from cuju.render.__main__ import main

    assert main(['team', 'export', 'no-such-team']) == 1
    assert 'expected a team of the registry' in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(['team', 'remove'])


def test_game_controller_follows_grf():
    """The keys of a step become GRF's actions as GRF's own controller turns them."""
    from cuju.render._game import Controller, own_frame

    controller = Controller()
    none = frozenset()
    steps = [
        # A direction comes first, then stays held without a further action.
        ({'right'}, none),
        ({'right'}, none),
        # Up with right is the diagonal, and letting go releases the direction.
        ({'right', 'up'}, none),
        (set(), none),
        # A tap between two steps still shoots, and its release is idle in the default set.
        (set(), frozenset({'shot'})),
        (set(), none),
        (set(), none),
        # Sprint stays pressed until it is let go.
        ({'sprint'}, none),
        ({'sprint'}, none),
        (set(), none),
        # A key of two buttons gives the first in GRF's order, then the second on the next step.
        ({'high_pass', 'sliding'}, none),
        ({'high_pass', 'sliding'}, none),
        # A change of direction wins over a button in the same step.
        ({'left', 'dribble'}, none),
        # The high pass and the slide release, one a step, before the dribble presses.
        ({'left', 'dribble'}, none),
        ({'left', 'dribble'}, none),
        ({'left', 'dribble'}, none),
        ({'left'}, none),
    ]
    actions = [controller.act(frozenset(held), pressed) for held, pressed in steps]
    assert actions == [5, 0, 4, 14, 12, 0, 0, 13, 0, 15, 10, 16, 1, 0, 0, 17, 18]
    # The right team plays turned half a circle.
    assert own_frame(frozenset({'left', 'up', 'shot'}), 1) == {'right', 'down', 'shot'}


def test_game_actions_are_the_actions_of_a_batch(render_home):
    """A game plays exactly the match that a batch plays for the same actions."""
    from cuju.render._game import Game

    game = Game('5_vs_5', (True, True), ability=(0.3, None), seed=5)
    keys = [('right', 'sprint'), ('up',), (), ('shot',), ('left', 'down'), ('dribble',)]
    for step in range(60):
        held = frozenset(keys[(step // 5) % len(keys)])
        game.step({0: (held, frozenset()), 1: (held, frozenset({'short_pass'}))})
    batch = cuju.BatchEnvironment(
        '5_vs_5_l1_r1',
        environment_count=1,
        representation='raw',
        seed=5,
        auto_reset=False,
        device='cpu',
        difficulty=[(0.3, None)],
    )
    batch.reset()
    for row in game.actions:
        batch.step(torch.tensor([row], dtype=torch.int32))
    assert batch.get_state(0) == game.batch.get_state(0)
    assert len({tuple(row) for row in game.actions}) > 3
    # Each frame marks the players that the people control, one a side.
    start = _recording._FRAME_HEAD.size + 10 * 14 * 14
    for frame in game.frames[1:]:
        controlled = [k for k in range(10) if frame[start + k] & 8]
        assert len(controlled) == 2
        assert {game.lineup[k]['team'] for k in controlled} == {0, 1}
    batch.close()
    game.close()


def _until_ended(address, deadline):
    entry = None
    while time.monotonic() < deadline:
        entry = json.loads(_get(address, '/api/game')[1])
        if entry['state'] in ('ended', 'failed'):
            break
        time.sleep(0.2)
    return entry


def test_game_through_the_server(render_home, monkeypatch):
    monkeypatch.setattr(_server, '_GAME_PACE', 400)
    server = _server.http.server.ThreadingHTTPServer(
        ('127.0.0.1', 0), _server.make_handler(_server._Requests())
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    address = f'http://127.0.0.1:{port}'
    try:
        assert json.loads(_get(address, '/api/game')[1]) is None
        for wrong in (
            {'scenario': 'no_such_scenario'},
            {'humans': [False, False]},
            {'humans': [1, 0]},
            {'ability': [2, None]},
            {'record': 'yes'},
        ):
            with pytest.raises(urllib.error.HTTPError) as refused:
                _send(address, '/api/game', json.dumps(wrong).encode())
            assert refused.value.code == 400
        start = {'scenario': 'academy_empty_goal_close', 'humans': [True, False], 'seed': 4}
        game = _send(address, '/api/game', json.dumps(start).encode())
        assert game['state'] == 'starting'
        with pytest.raises(urllib.error.HTTPError) as refused:
            _send(address, '/api/game', json.dumps(start).encode())
        assert refused.value.code == 409
        identifier = game['id']
        connection = _server.http.client.HTTPConnection('127.0.0.1', port, timeout=120)
        connection.request('GET', f'/api/game/{identifier}/stream')
        response = connection.getresponse()
        assert response.status == 200
        head = response.read(12)
        assert head[:8] == _recording.MAGIC
        header = json.loads(response.read(struct.unpack_from('<I', head, 8)[0]))
        assert [team['person'] for team in header['teams']] == [True, False]
        first = response.read(header['frame_bytes'])
        assert struct.unpack_from('<i', first)[0] == 0
        # Bad input and input from another site are refused.
        path = f'/api/game/{identifier}/input'
        for wrong in (
            {'side': 1, 'held': []},
            {'side': 0, 'held': ['jump']},
            {'side': 0, 'pressed': ['left']},
            {'side': 0, 'held': 'right'},
            {'side': 0, 'extra': 1},
            {'side': True},
        ):
            with pytest.raises(urllib.error.HTTPError) as refused:
                _send(address, path, json.dumps(wrong).encode())
            assert refused.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as refused:
            _send(address, f'/api/game/{identifier}/pause', b'{"paused": "no"}')
        assert refused.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as refused:
            _send(address, '/api/game/g-other/input', b'{"side": 0}')
        assert refused.value.code == 404
        for headers in ({'Origin': 'https://example.com'}, {'Host': f'evil.example:{port}'}):
            other = _server.http.client.HTTPConnection('127.0.0.1', port, timeout=10)
            other.request('POST', path, body=b'{"side": 0, "held": ["right"]}', headers=headers)
            assert other.getresponse().status == 403
            other.close()
        _send(address, f'/api/game/{identifier}/pause', b'{"paused": false}')
        steps = []
        held = json.dumps({'side': 0, 'held': ['right'], 'pressed': ['shot']}).encode()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            entry = _send(address, path, held)
            if entry['state'] in ('ended', 'failed'):
                break
            frame = response.read(header['frame_bytes'])
            if len(frame) < header['frame_bytes']:
                break
            steps.append(struct.unpack_from('<i', frame)[0])
        entry = _until_ended(address, deadline)
        assert entry['state'] == 'ended', entry
        assert len(steps) > 5 and steps == list(range(1, len(steps) + 1))
        connection.close()
        recording = render_home / 'matches' / f'{entry["match"]}.cjr'
        document = _recording.read_header(recording)
        assert document['title'].startswith('You against ')
        assert document['frame_count'] == entry['step'] + 1
        # A second game starts once the first has ended, and quitting ends it.
        game = _send(address, '/api/game', json.dumps({**start, 'record': False}).encode())
        _send(address, f'/api/game/{game["id"]}/pause', b'{"paused": false}')
        request = urllib.request.Request(f'{address}/api/game/{game["id"]}', method='DELETE')
        with urllib.request.urlopen(request, timeout=10) as answer:
            assert answer.status == 200
        entry = _until_ended(address, time.monotonic() + 120)
        assert entry['state'] == 'ended' and entry['match'] is None, entry
    finally:
        server.shutdown()
        server.server_close()


# A toy submission that controls several players and sends each one a direction of its own, by
# the index of its player, so the test sees each action reach its player.
_SEVERAL = """
def agent(obs):
    actions = []
    for raw in obs['players_raw']:
        player = raw['active']
        assert raw['left_team_roles'][player] != 0, 'a goalkeeper reached the submission'
        actions.append(3 if player % 2 else 7)
    assert len(actions) == obs['controlled_players']
    return actions
"""
_SINGLE = """
def agent(obs):
    return [5 if obs['step'] % 7 else 12]
"""


def _short_matches(monkeypatch, steps):
    """Ends every match that `play` starts after `steps` steps."""
    from cuju.render import _play

    real = _play.BatchEnvironment

    def short(*arguments, **options):
        batch = real(*arguments, **options)
        step, count = batch.step, [0]

        def limited(actions):
            observation, reward, done, information = step(actions)
            count[0] += 1
            return observation, reward, done | (count[0] >= steps), information

        batch.step = limited
        return batch

    monkeypatch.setattr(_play, 'BatchEnvironment', short)


def _toy_team(tmp_path, name, source, format, control=None):
    folder = tmp_path / name
    folder.mkdir()
    (folder / 'main.py').write_text(source)
    team = _teams.add_folder(folder, name, format)
    return _teams.update(team['id'], control=control) if control else team


def _frames(path):
    document = _recording.read_header(path)
    return document, list(_recording.frames(path))


def test_team_control_is_checked_and_travels(render_home, tmp_path):
    team = _toy_team(tmp_path, 'several', _SEVERAL, '11v11')
    assert team['control'] == _teams.DEFAULT_CONTROL
    for wrong in (
        {'players': 12},
        {'players': 0},
        {'players': True},
        {'players': 9, 'goalkeeper': 'builtin'},
        {'players': 10, 'goalkeeper': 'nobody'},
        {'players': 2, 'extra': 1},
    ):
        with pytest.raises(cuju.ArgumentError):
            _teams.update(team['id'], control=wrong)
    keeper = {'players': 10, 'goalkeeper': 'builtin'}
    team = _teams.update(team['id'], control=keeper)
    assert _teams.agent_count(team) == 11
    # A change of format that the control does not fit returns it to the default.
    assert _teams.update(team['id'], format='5v5')['control']['players'] == 1
    team = _teams.update(team['id'], format='11v11', control=keeper)
    _, archive = _teams.export(team['id'])
    copy = _teams.add('copy', 'copy.zip', archive)
    assert copy['control'] == _teams.check_control(keeper, '11v11')
    with pytest.raises(cuju.ArgumentError):
        _teams.add('wrong', 'main.py', b'def agent(obs):\n    return [0]\n', '5v5', {'players': 6})


def _moved(document, frames, team, steps):
    """Returns the change of y of each player of `team` over `steps` steps, by index."""
    players = len(document['players'])
    start, end = frames[0], frames[steps]
    head = _recording._FRAME_HEAD.size
    moves = {}
    for k, entry in enumerate(document['players']):
        if entry['team'] != team:
            continue
        y0 = struct.unpack_from('<h', start, head + k * 196 + 2)[0]
        y1 = struct.unpack_from('<h', end, head + k * 196 + 2)[0]
        moves[entry['index']] = (y1 - y0) / _recording.POSITION_SCALE
    assert len(moves) == (5 if players == 10 else 11)
    return moves


@pytest.mark.parametrize(
    ('format', 'scenario', 'control', 'moving'),
    [
        ('5v5', '5_vs_5', {'players': 4, 'goalkeeper': 'agent'}, range(1, 5)),
        ('11v11', '11_vs_11_kaggle', {'players': 10, 'goalkeeper': 'builtin'}, range(1, 11)),
    ],
)
def test_team_controls_several_players(
    render_home, tmp_path, monkeypatch, format, scenario, control, moving
):
    """Each controlled player follows its own action, and the goalkeeper stays with the AI."""
    _short_matches(monkeypatch, 60)
    team = _toy_team(tmp_path, 'several', _SEVERAL, format, control)
    path = cuju.render.play(team['id'], 'builtin', scenario, seed=2)
    document, frames = _frames(path)
    moves = _moved(document, frames, 0, 60)
    # Odd players run up the pitch, towards positive y, and even players down.
    obeyed = [index for index in moving if (moves[index] > 1) == bool(index % 2)]
    assert len(obeyed) >= len(moving) - 1, moves
    assert all(abs(moves[index]) > 1 for index in obeyed)
    # Every frame marks the controlled players of the left team, and none of the right.
    head = _recording._FRAME_HEAD.size + len(document['players']) * 196
    marked = {
        k for frame in frames[1:] for k in range(len(document['players'])) if frame[head + k] & 8
    }
    assert {document['players'][k]['team'] for k in marked} == {0}


def test_single_player_team_plays_as_before(render_home, tmp_path, monkeypatch):
    """A team of one controlled player plays the match of one agent, frame for frame."""
    from cuju.kaggle import KaggleAgent

    _short_matches(monkeypatch, 80)
    team = _toy_team(tmp_path, 'single', _SINGLE, '5v5')
    path = cuju.render.play(team['id'], 'builtin', '5_vs_5', seed=9)
    _, frames = _frames(path)
    batch = cuju.BatchEnvironment(
        '5_vs_5_l1_r0',
        environment_count=1,
        representation='raw',
        seed=9,
        auto_reset=False,
        device='cpu',
    )
    player = KaggleAgent(_teams.load_agent(team['id']), batch, 0)
    reader = _recording.StateReader(10)
    batch.reset()
    expected = [reader.frame(batch.get_state(0))]
    for step in range(80):
        batch.step(player.act().reshape(1, 1))
        expected.append(reader.frame(batch.get_state(0), 1 if step == 79 else 0))
    assert frames == expected
    batch.close()


def _control(
    players, assignment='dynamic', selection=None, goalkeeper='scenario', action_set='default'
):
    return dict(
        players=players,
        assignment=assignment,
        selection=selection or [],
        goalkeeper=goalkeeper,
        action_set=action_set,
    )


@pytest.mark.parametrize(
    ('scenario', 'format', 'control', 'active', 'engine_count', 'side', 'opponent'),
    [
        ('5_vs_5', '5v5', _control(2), None, 2, 0, False),
        (
            '11_vs_11_kaggle',
            '11v11',
            _control(3, 'fixed', ['CF', 'CB', 'CB'], 'builtin'),
            [9, 2, 3],
            11,
            0,
            True,
        ),
        ('11_vs_11_kaggle', '11v11', _control(2, 'fixed', [0, 9], 'agent'), [0, 9], 11, 1, True),
        (
            'benchmark_full_game_11_vs_11_hard',
            '11v11',
            _control(10, 'fixed', list(range(1, 11))),
            list(range(1, 11)),
            10,
            0,
            False,
        ),
        ('5_vs_5', '5v5', _control(2, action_set='v2'), None, 2, 1, True),
        (
            'academy_3_vs_1_with_keeper',
            '11v11',
            _control(2, 'fixed', [3, 1], 'builtin'),
            [3, 1],
            4,
            0,
            True,
        ),
        (
            'academy_3_vs_1_with_keeper',
            '11v11',
            _control(1, 'fixed', ['CB'], 'builtin'),
            [1],
            2,
            1,
            False,
        ),
    ],
)
def test_team_assignments_match_engine(
    render_home,
    tmp_path,
    monkeypatch,
    scenario,
    format,
    control,
    active,
    engine_count,
    side,
    opponent,
):
    """Toy actions reach their selected players in either side of a mixed match."""
    from cuju.raw import grf_observation, team_sizes

    steps = 80
    _short_matches(monkeypatch, steps)
    source = f"""
expected = {active!r}
count = {control['players']}
hand = {control['action_set'] == 'v2'}

def agent(obs):
    players = [raw['active'] for raw in obs['players_raw']]
    assert obs['controlled_players'] == count
    assert expected is None or players == expected
    return [19 if hand and k == 0 else (3 if player % 2 else 7)
            for k, player in enumerate(players)]
"""
    team = _toy_team(tmp_path, 'selected', source, format, control)
    other = _toy_team(tmp_path, 'single', 'def agent(obs):\n    return [-1]\n', format)
    sides = [team['id'], other['id'] if opponent else 'builtin']
    if side:
        sides.reverse()
    path = cuju.render.play(*sides, scenario, seed=7, device='cpu')
    document, frames = _frames(path)
    counts = [engine_count, int(opponent)]
    if side:
        counts.reverse()
    extended = control['assignment'] == 'fixed' or control['action_set'] == 'v2'
    asset = str(_teams.scenario_path(scenario)) if scenario.startswith('benchmark_') else scenario
    batch = cuju.BatchEnvironment(
        asset,
        environment_count=1,
        representation='raw',
        seed=7,
        auto_reset=False,
        device='cpu',
        agent_counts=[tuple(counts)],
        action_set='v2' if extended else 'default',
    )
    try:
        batch.reset()
        reader = _recording.StateReader(len(document['players']))
        expected = [reader.frame(batch.get_state(0))]
        first = 0 if side == 0 else counts[0]
        for step in range(steps):
            raw = batch.raw()
            columns = list(range(first, first + engine_count))
            by_player = {
                int(
                    grf_observation(raw, 0, column, *team_sizes(batch, 0, column))['active']
                ): column
                for column in columns
            }
            # A step on which GRF hands the players out again leaves them to the built-in AI,
            # as the runner does.
            complete = len(by_player) == len(columns) and all(i in by_player for i in active or ())
            chosen = (
                columns
                if active is None
                else ([by_player[index] for index in active] if complete else [])
            )
            actions = torch.zeros((1, sum(counts)), dtype=torch.int32)
            if opponent:
                actions[0, counts[0] if side == 0 else 0] = 18
            for column in columns:
                actions[0, column] = 19
            for k, column in enumerate(chosen):
                player = int(
                    grf_observation(raw, 0, column, *team_sizes(batch, 0, column))['active']
                )
                actions[0, column] = (
                    19 if control['action_set'] == 'v2' and k == 0 else 3 if player % 2 else 7
                )
            _, _, done, _ = batch.step(actions)
            final = bool(done[0]) or step == steps - 1
            expected.append(reader.frame(batch.get_state(0), int(final)))
            if final:
                break
        assert frames == expected
    finally:
        batch.close()


@pytest.mark.parametrize('side', [0, 1])
def test_fixed_routing_survives_mixed_set_pieces(device, side):
    """Selected players keep their observation order as GRF changes agents at kickoff."""
    from cuju.render import _play

    counts = (11, 1) if side == 0 else (1, 11)
    batch = cuju.BatchEnvironment(
        '11_vs_11_kaggle',
        environment_count=1,
        device=device,
        representation='raw',
        agent_counts=[counts],
        action_set='v2',
        seed=2,
        auto_reset=False,
    )
    try:
        batch.reset()
        first = 0 if side == 0 else 1
        columns = list(range(first, first + 11))
        selected = [9, 3, 1]
        assignments = set()
        modes = set()
        cleared = 0
        for _ in range(150):
            raw = {name: tensor.cpu() for name, tensor in batch.raw().items()}
            active = raw['active'][0, columns].tolist()
            actions = torch.zeros((1, 12), dtype=torch.int32, device=batch.torch_device)
            actions[:, columns] = 19
            found = _play._player_columns(raw, columns, selected, {})
            if found is None:
                # GRF clears the side's selection at a set piece, so every agent reports no player
                # until it hands the players out again, and the built-in AI plays the step.
                assert -1 in active
                cleared += 1
            else:
                chosen, missing = found
                assert missing == []
                assert sorted(active) == list(range(11))
                assignments.add(tuple(active))
                assert raw['active'][0, chosen].tolist() == selected
                actions[:, chosen] = torch.tensor(
                    [3, 7, 5], dtype=torch.int32, device=batch.torch_device
                )
                assert all(
                    int(actions[0, column]) == 19 for column in columns if column not in chosen
                )
            modes.add(int(raw['game_mode'][0, first]))
            batch.step(actions)
        # The set pieces cleared the selection, and the players came back to the agents in the
        # same order or, on one side, in another, which the routing follows.
        assert cleared > 0 and assignments
        assert len(modes) > 1
    finally:
        batch.close()


def test_fixed_routing_keeps_a_player_off_the_pitch():
    """A selected player that no agent reports keeps its agent, while a cleared selection skips."""
    from cuju.render import _play

    def raw(*active):
        # The side's agents are columns 4 to 6, after four agents of the other side.
        return {'active': torch.tensor([[0, 0, 0, 0, *active]], dtype=torch.int32)}

    columns, selected, held = [4, 5, 6], [9, 3, 1], {}
    assert _play._player_columns(raw(1, 9, 3), columns, selected, held) == ([5, 6, 4], [])
    # Player 3 leaves the pitch, and its agent now holds player 4.
    assert _play._player_columns(raw(1, 9, 4), columns, selected, held) == ([5, 6, 4], [6])
    # Two agents report player 9, so player 3 takes its last agent while that one is free.
    assert _play._player_columns(raw(1, 9, 9), columns, selected, held) == ([5, 6, 4], [6])
    assert _play._player_columns(raw(1, -1, 4), columns, selected, held) is None


def test_earlier_team_control_migrates(render_home, tmp_path):
    """Earlier metadata selects the same players after loading and export."""
    team = _toy_team(tmp_path, 'earlier', _SEVERAL, '11v11')
    meta = _teams._meta_path(team['id'])
    saved = json.loads(meta.read_text())
    saved['control'] = {'players': 10, 'goalkeeper': 'builtin'}
    meta.write_text(json.dumps(saved))
    control = _teams.entry(team['id'])['control']
    assert control == _control(10, 'fixed', list(range(1, 11)), 'builtin')
    _, archive = _teams.export(team['id'])
    assert _teams.add('copy', 'copy.zip', archive)['control'] == control
    saved['control'] = {'players': 2, 'goalkeeper': 'agent'}
    meta.write_text(json.dumps(saved))
    assert _teams.entry(team['id'])['control'] == _control(2)


@pytest.mark.parametrize(
    ('scenario', 'control', 'message'),
    [
        ('5_vs_5', _control(5), 'only 4 controllable players'),
        ('academy_3_vs_1_with_keeper', _control(5), 'only 4 controllable players'),
        ('11_vs_11_kaggle', _control(1, 'fixed', ['AM'], 'builtin'), 'cannot select player AM'),
        (
            'benchmark_full_game_11_vs_11_easy',
            _control(1, 'fixed', ['GK'], 'agent'),
            'cannot select player GK',
        ),
        ('11_vs_11_kaggle', _control(1, 'fixed', ['CF']), 'controllable goalkeeper'),
    ],
)
def test_team_scenario_refusal(render_home, tmp_path, scenario, control, message):
    """A refused match identifies the scenario and the conflicting setting."""
    format = '5v5' if scenario == '5_vs_5' else '11v11'
    team = _toy_team(tmp_path, 'refused', _SEVERAL, format, control)
    with pytest.raises(cuju.ArgumentError, match=message) as caught:
        cuju.render.play(team['id'], 'builtin', scenario, seed=1)
    assert scenario in str(caught.value)


def test_default_submission_rejects_builtin_action(render_home, tmp_path, monkeypatch):
    """Each submission uses its own action range in a mixed action set match."""
    _short_matches(monkeypatch, 1)
    default = _toy_team(tmp_path, 'default', 'def agent(obs):\n    return [19]\n', '5v5')
    v2 = _toy_team(
        tmp_path, 'v2', 'def agent(obs):\n    return [19]\n', '5v5', _control(1, action_set='v2')
    )
    with pytest.raises(IndexError, match='to 18'):
        cuju.render.play(default['id'], v2['id'], '5_vs_5', seed=1)


@pytest.mark.parametrize('size', [5, 11])
@pytest.mark.parametrize(('level', 'difficulty'), [('easy', 0.05), ('medium', 0.6), ('hard', 1.0)])
def test_training_scenario_description(size, level, difficulty):
    """Training scenarios keep the upstream formation, timing and episode parity."""
    from cuju.scenarios import Builder, load_scenario

    name = f'benchmark_full_game_{size}_vs_{size}_{level}'
    assert name in _server._scenarios()
    build = load_scenario(_teams.scenario_path(name))
    configs = []
    for episode in (0, 1):
        builder = Builder(episode)
        build(builder)
        config = builder.config()
        assert config.game_duration == 3000
        assert config.second_half == 1500
        assert config.deterministic is False
        assert config.left_team_difficulty == difficulty
        assert config.right_team_difficulty == difficulty
        for formation in (config.left_team, config.right_team):
            assert len(formation) == size
            assert formation[0].role == 0
            assert formation[0].lazy is False
            assert formation[0].controllable is False
            assert all(player.controllable for player in formation[1:])
        configs.append(config)
    assert [p.key() for p in configs[0].left_team] == [p.key() for p in configs[1].right_team]
    assert [p.key() for p in configs[0].right_team] == [p.key() for p in configs[1].left_team]


@pytest.mark.parametrize('reverse', [0, 1])
def test_recorded_engine_scorer_and_called_offside(monkeypatch, reverse):
    batch = cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=1, device='cpu')
    try:
        batch.reset(reverse=[reverse])
        reader = _recording.StateReader(10)
        state = batch.get_state(0)
        values = reader._values(state, 'pitch')
        values = {name: values[name].copy() for name in values.dtype.names}
        values.update(
            buffer_active=True,
            set_piece=3,
            foul_type=0,
            offside_count=2,
            offside_first=1,
            touch_team=0,
            scorer=7,
        )
        first = reader._order(values)[0][0]
        values['touch_first' if first == 0 else 'touch_second'] = 3
        monkeypatch.setattr(reader, '_values', lambda state, frame: values)
        frame = reader.frame(state)
        flags = struct.unpack_from('<i', frame, 28)[0]
        assert flags & 4
        assert (flags >> 8) & 31 == 8
        assert struct.unpack_from('<i', frame, 72)[0] == 3
    finally:
        batch.close()


@pytest.mark.parametrize(
    'duration,second,expected',
    [(3000, 1500, 1500), (400, 999999999, None), (100, 1500, None)],
)
def test_recording_scenario_timing(monkeypatch, duration, second, expected):
    monkeypatch.setattr(
        _recording._state, 'read', lambda state, fields: [duration, second, False, False, False]
    )
    timing = _recording.scenario_timing(b'')
    document = _recording._parse_header(
        io.BytesIO(_recording.header('Match', 'custom', [], timing=timing))
    )
    assert document['second_half'] == expected
    assert document['duration'] == duration
    assert document['full_match'] == (duration >= 3000)


def test_live_close_with_full_viewer_queue(render_home):
    batch = cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=1, device='cpu')
    batch.reset()
    source = _live.LiveSource(batch, announce=False)
    viewer = _live._Viewer()
    for _ in range(_live._BUFFER):
        viewer.frames.put_nowait(b'frame')
    source._viewers.append(viewer)
    try:
        source.close()
        assert viewer.frames.get(timeout=1) is None
        assert viewer.frames.empty()
    finally:
        source.close()
        batch.close()


def test_cancel_during_child_start(monkeypatch, render_home):
    requests = object.__new__(_server._Requests)
    requests._lock = threading.Lock()
    requests._processes = {}
    entry = {
        'id': 'race',
        'state': 'queued',
        'left': 'builtin',
        'right': 'builtin',
        'scenario': '5_vs_5',
        'seed': None,
        'ability': [None, None],
        'progress': 0,
        'error': None,
    }
    requests._entries = [entry]
    started, release, killed = threading.Event(), threading.Event(), threading.Event()

    class Process:
        stderr = ()

        @property
        def stdout(self):
            assert killed.wait(5)
            return ()

        def kill(self):
            killed.set()

        def wait(self):
            return -1

    def start(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return Process()

    monkeypatch.setattr(_server.subprocess, 'Popen', start)
    worker = threading.Thread(target=requests._play_child, args=(entry,))
    worker.start()
    assert started.wait(5)
    canceller = threading.Thread(target=requests.cancel, args=('race',))
    canceller.start()
    release.set()
    worker.join(5)
    canceller.join(5)
    assert not worker.is_alive()
    assert not canceller.is_alive()
    assert killed.is_set()
    assert entry['error'] == 'cancelled'


def test_cli_team_archive_round_trip(render_home, tmp_path, capsys):
    """The command line imports its own export with the team settings intact."""
    from cuju.render.__main__ import main

    folder = tmp_path / 'submission'
    folder.mkdir()
    (folder / 'main.py').write_text('def agent(obs):\n    return [0]\n')
    assert main(['team', 'add', str(folder), '--name', 'Visitors', '--format', '5v5']) == 0
    team = _teams.teams()[-1]
    control = {'players': 2, 'assignment': 'fixed', 'selection': [1, 2]}
    team = _teams.update(team['id'], control=control)
    archive = tmp_path / 'team.zip'
    assert main(['team', 'export', team['id'], '--out', str(archive)]) == 0
    assert main(['team', 'add', str(archive)]) == 0
    copy = _teams.teams()[-1]
    assert copy['control'] == team['control']
    assert copy['style'] == team['style']
    assert copy['format'] == '5v5'
    assert copy['name'] == 'Visitors'
    assert main(['team', 'list']) == 0
    assert 'Visitors' in capsys.readouterr().out
    assert main(['team', 'remove', copy['id']]) == 0
    assert all(entry['id'] != copy['id'] for entry in _teams.teams())
    assert main(['team', 'add', str(tmp_path / 'missing.zip')]) == 1
    broken = tmp_path / 'broken.zip'
    broken.write_bytes(b'bad archive')
    assert main(['team', 'add', str(broken)]) == 1


def test_cli_dispatch_options(render_home, tmp_path, monkeypatch):
    """The command line forwards every server and match option."""
    from cuju.render import __main__, _play

    calls = []
    monkeypatch.setattr(_server, 'serve', lambda *args: calls.append(args))
    assert __main__.main(['serve', '--port', '8799', '--open']) == 0
    assert calls.pop() == (8799, True)
    monkeypatch.setattr(_play, 'play', lambda *args: calls.append(args) or tmp_path / 'game')
    assert (
        __main__.main(
            [
                '--home',
                str(render_home),
                'play',
                '--left',
                'home',
                '--right',
                'away',
                '--scenario',
                '5_vs_5',
                '--seed',
                '7',
                '--device',
                'cpu',
                '--left-ability',
                '0.2',
                '--right-ability',
                '0.8',
            ]
        )
        == 0
    )
    assert calls.pop() == ('home', 'away', '5_vs_5', 7, None, 'cpu', (0.2, 0.8))


@pytest.mark.parametrize(
    'arguments',
    [
        ['serve', '--port', 'bad'],
        ['serve', '--port', '-1'],
        ['serve', '--port', '65536'],
        ['team', 'export'],
        ['team', 'remove'],
        ['team', 'unknown'],
        ['team', 'add', '--format', 'bad'],
        ['team', 'add'],
        ['play', '--seed', 'bad'],
        ['play', '--left-ability', 'bad'],
        ['play', '--right-ability', 'bad'],
    ],
)
def test_cli_argument_refusals(arguments):
    """Invalid command arguments produce an error exit."""
    from cuju.render.__main__ import main

    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


def test_api_methods_and_request_boundaries(render_home):
    """Every route refuses foreign hosts and origins before handling a request."""
    routes = [
        '/api/info',
        '/api/scenarios',
        '/api/models',
        '/api/matches',
        '/api/matches/missing',
        '/api/live',
        '/api/live/missing/stream',
        '/api/teams',
        '/api/teams/missing',
        '/api/teams/missing/export',
        '/api/teams/missing/model',
        '/api/teams/missing/images/shirt-front',
        '/api/teams/missing/sounds/goal',
        '/api/requests',
        '/api/requests/missing',
        '/api/game',
        '/api/game/missing',
        '/api/game/missing/stream',
        '/api/game/missing/input',
        '/api/game/missing/pause',
    ]
    server = _server.http.server.ThreadingHTTPServer(
        ('127.0.0.1', 0), _server.make_handler(_server._Requests())
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        for method in ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']:
            for path in routes:
                for headers in [
                    {'Host': f'foreign.invalid:{port}'},
                    {'Origin': 'http://foreign.invalid'},
                    {'Origin': 'null'},
                    {'Origin': 'http://127.0.0.1:1'},
                ]:
                    if method == 'GET' and 'Origin' in headers:
                        continue
                    connection = _server.http.client.HTTPConnection('127.0.0.1', port, timeout=5)
                    connection.request(method, path, body=b'{}', headers=headers)
                    response = connection.getresponse()
                    assert response.status == 403, (method, path, headers, response.read())
                    response.read()
                    connection.close()
            connection = _server.http.client.HTTPConnection('127.0.0.1', port, timeout=5)
            connection.request(method, '/api/unknown', body=b'{}')
            response = connection.getresponse()
            assert response.status == 404
            response.read()
            connection.close()
        for origin in [None, f'http://127.0.0.1:{port}']:
            connection = _server.http.client.HTTPConnection('127.0.0.1', port, timeout=5)
            headers = {'Origin': origin} if origin else {}
            connection.request('GET', '/api/info', headers=headers)
            response = connection.getresponse()
            assert response.status == 200
            assert json.loads(response.read()) == {'version': cuju.__version__}
            connection.close()
    finally:
        server.shutdown()
        server.server_close()


def test_api_team_assets_and_missing_resources(render_home):
    """Team assets support upload, download and removal with clear refusals."""
    server = _server.http.server.ThreadingHTTPServer(
        ('127.0.0.1', 0), _server.make_handler(_server._Requests())
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    address = f'http://127.0.0.1:{server.server_address[1]}'
    try:
        team = _send(address, '/api/teams?filename=main.py', b'def agent(obs):\n    return 0\n')
        prefix = f'/api/teams/{team["id"]}'
        model = (_server._WEB / 'assets/player.glb').read_bytes()
        bones = _teams._skin_joints(model)
        joints = urllib.parse.quote(json.dumps({'body': sorted(bones)[0]}))
        for payload, mapping in [(b'bad', joints), (model, '%7B%7D')]:
            with pytest.raises(urllib.error.HTTPError) as error:
                _send(address, f'{prefix}/model?joints={mapping}', payload, 'PUT')
            assert error.value.code == 400
        entry = _send(address, f'{prefix}/model?joints={joints}', model, 'PUT')
        assert entry['model']['bytes'] == len(model)
        assert _get(address, f'{prefix}/model')[1] == model
        for slot, payload in [
            ('images/shirt-front', b'\x89PNG\r\n\x1a\n' + bytes(32)),
            ('sounds/goal', b'OggS' + bytes(60)),
        ]:
            _send(address, f'{prefix}/{slot}', payload, 'PUT')
            assert _get(address, f'{prefix}/{slot}')[1] == payload
        for resource in ['model', 'images/shirt-front', 'sounds/goal']:
            request = urllib.request.Request(f'{address}{prefix}/{resource}', method='DELETE')
            with urllib.request.urlopen(request, timeout=5) as response:
                assert response.status == 204
            for method in ['GET', 'DELETE']:
                request = urllib.request.Request(f'{address}{prefix}/{resource}', method=method)
                with pytest.raises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(request, timeout=5)
                assert error.value.code == 404
        for path in [
            '/api/matches/missing',
            '/api/live/missing/stream',
            '/api/teams/missing/export',
            '/api/game/missing/stream',
        ]:
            with pytest.raises(urllib.error.HTTPError) as error:
                _get(address, path)
            assert error.value.code == 404
        for path in ['/api/matches/missing', '/api/requests/missing', '/api/game/missing']:
            with pytest.raises(urllib.error.HTTPError) as error:
                _send(address, path, b'', 'DELETE')
            assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_responses_refuse_framing_and_bad_announcements_are_skipped(render_home):
    """Every response forbids framing and sniffing, and a bad live announcement spares the list."""
    live = _home.folder('live')
    (live / 'missing.json').write_text('{"id": "missing"}')
    (live / 'text.json').write_text('{"id": "text", "pid": "many"}')
    (live / 'null.json').write_text('{"id": "null", "pid": null}')
    (live / 'list.json').write_text('[1, 2]')
    running = {'id': 'running', 'title': 'a', 'scenario': 'b', 'environment': 0}
    (live / 'running.json').write_text(json.dumps({**running, 'pid': os.getpid()}))
    assert [source['id'] for source in _server._sources()] == ['running']
    server = _server.http.server.ThreadingHTTPServer(
        ('127.0.0.1', 0), _server.make_handler(_server._Requests())
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    address = f'http://127.0.0.1:{server.server_address[1]}'
    expected = {
        'X-Frame-Options': 'DENY',
        'Content-Security-Policy': "frame-ancestors 'none'",
        'X-Content-Type-Options': 'nosniff',
    }
    try:
        for path in ['/', '/api/live', '/api/matches/missing']:
            try:
                with urllib.request.urlopen(address + path, timeout=10) as response:
                    headers = response.headers
            except urllib.error.HTTPError as error:
                headers = error.headers
            assert {name: headers[name] for name in expected} == expected, path
        assert [source['id'] for source in json.loads(_get(address, '/api/live')[1])] == ['running']
    finally:
        server.shutdown()
        server.server_close()


_SAMPLES = ['sample-11v11', 'sample-5v5']


def _serve_once(monkeypatch):
    """Starts the server as `cuju render` does and stops it at once."""

    def stop(server):
        raise KeyboardInterrupt

    monkeypatch.setattr(_server.http.server.ThreadingHTTPServer, 'serve_forever', stop)
    _server.serve(0)


def _library_ids():
    return sorted(entry['id'] for entry in _library.matches())


def test_new_home_gets_the_samples_once(render_home, monkeypatch):
    _serve_once(monkeypatch)
    assert _library_ids() == _SAMPLES
    for entry in _library.matches():
        assert entry['title'].startswith('Sample match')
        assert entry['folder'] == ''
        assert (
            entry['steps'] == _recording.read_header(_library.path_of(entry['id']))['frame_count']
        )
    _serve_once(monkeypatch)
    assert _library_ids() == _SAMPLES
    assert sorted(path.name for path in (render_home / 'matches').iterdir()) == sorted(
        f'{name}.{suffix}' for name in _SAMPLES for suffix in ('cjr', 'json')
    )


def test_deleted_sample_stays_deleted_until_requested(render_home, monkeypatch, capsys):
    from cuju.render.__main__ import main

    _serve_once(monkeypatch)
    assert _library.remove('sample-5v5')
    _serve_once(monkeypatch)
    assert _library_ids() == ['sample-11v11']
    capsys.readouterr()
    assert main(['sample', '5v5']) == 0
    assert 'added sample-5v5' in capsys.readouterr().out
    assert _library_ids() == _SAMPLES
    for name in _SAMPLES:
        assert _library.remove(name)
    assert main(['sample']) == 0
    assert _library_ids() == _SAMPLES
    capsys.readouterr()
    assert main(['sample']) == 0
    assert 'already holds' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(['sample', '7v7'])


def test_samples_keep_the_users_files(render_home, monkeypatch):
    matches = _home.folder('matches')
    (matches / 'sample-11v11.cjr').write_bytes(b'mine')
    (matches / 'sample-5v5.json').write_text('{"id": "sample-5v5", "title": "mine"}')
    _serve_once(monkeypatch)
    assert (matches / 'sample-11v11.cjr').read_bytes() == b'mine'
    assert json.loads((matches / 'sample-5v5.json').read_text())['title'] == 'mine'
    assert not (matches / 'sample-5v5.cjr').exists()
    assert _library.add_samples() == []


@pytest.mark.parametrize('name,players', [('sample-11v11', 22), ('sample-5v5', 10)])
def test_bundled_samples_are_full_recordings(name, players):
    path = _library._SAMPLES / f'{name}.cjr'
    document = _recording.read_header(path)
    frames = list(_recording.frames(path))
    assert len(document['players']) == players
    assert len(frames) == document['frame_count'] == 3002
    assert all(len(frame) == document['frame_bytes'] for frame in frames)
    assert struct.unpack_from('<i', frames[-1], 28)[0] & 1
    assert document['title'].startswith('Sample match')
    left, right = document['teams']
    assert left['name'] != right['name'] and left['color'] != right['color']
    assert sum(struct.unpack_from('<2i', frames[-1], 8)) >= 2

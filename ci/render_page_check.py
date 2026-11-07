# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks renderer pages through Chrome and CDP against the local mock server."""

import base64
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


class CDP:
    """A local CDP connection using the standard library."""

    def __init__(self, url):
        from urllib.parse import urlsplit

        address = urlsplit(url)
        self.socket = socket.create_connection((address.hostname, address.port), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        request = (
            f'GET {address.path} HTTP/1.1\r\nHost: {address.netloc}\r\n'
            f'Upgrade: websocket\r\nConnection: Upgrade\r\n'
            f'Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n'
        )
        self.socket.sendall(request.encode())
        header = b''
        while not header.endswith(b'\r\n\r\n'):
            header += self.socket.recv(1)
        assert b'101' in header, header
        self.sequence = 0
        self.errors = []

    def read(self, size):
        data = b''
        while len(data) < size:
            chunk = self.socket.recv(size - len(data))
            if not chunk:
                raise RuntimeError('Chrome connection closed.')
            data += chunk
        return data

    def call(self, method, params=None):
        self.sequence += 1
        data = json.dumps({'id': self.sequence, 'method': method, 'params': params or {}}).encode()
        mask = os.urandom(4)
        size = len(data)
        header = (
            bytes([129, 128 | size]) if size < 126 else bytes([129, 254]) + struct.pack('!H', size)
        )
        self.socket.sendall(
            header + mask + bytes(value ^ mask[index % 4] for index, value in enumerate(data))
        )
        while True:
            first, second = self.read(2)
            size = second & 127
            if size == 126:
                size = struct.unpack('!H', self.read(2))[0]
            elif size == 127:
                size = struct.unpack('!Q', self.read(8))[0]
            payload = self.read(size)
            if first & 15 != 1:
                continue
            message = json.loads(payload)
            if message.get('method') == 'Runtime.exceptionThrown':
                self.errors.append(message['params'])
            if message.get('id') == self.sequence:
                if 'error' in message:
                    raise RuntimeError(message['error'])
                return message.get('result', {})

    def evaluate(self, expression):
        result = self.call(
            'Runtime.evaluate',
            {
                'expression': expression,
                'returnByValue': True,
                'awaitPromise': True,
            },
        )
        if 'exceptionDetails' in result:
            raise RuntimeError(result['exceptionDetails'])
        return result.get('result', {}).get('value')


def wait_json(url):
    for _ in range(150):
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                return json.load(response)
        except (OSError, ValueError):
            time.sleep(0.2)
    raise RuntimeError(f'Server startup failed at {url}')


def main():
    root = Path(__file__).resolve().parents[1]
    processes = []
    client = None
    checks = 0
    try:
        with tempfile.TemporaryDirectory(prefix='cuju-page-') as profile:
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(root / 'ci/render_mock_server.py'),
                        '--port',
                        '8790',
                        '--short',
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
            print('Starting mock server.', flush=True)
            wait_json('http://127.0.0.1:8790/api/scenarios')
            print('Starting Chrome.', flush=True)
            processes.append(
                subprocess.Popen(
                    [
                        'google-chrome',
                        '--headless=new',
                        '--mute-audio',
                        '--no-sandbox',
                        '--disable-background-networking',
                        '--proxy-server=http://127.0.0.1:9',
                        '--proxy-bypass-list=127.0.0.1',
                        '--disable-component-update',
                        '--disable-sync',
                        '--no-first-run',
                        '--disable-default-apps',
                        '--remote-debugging-port=8791',
                        f'--user-data-dir={profile}',
                        '--window-size=800,600',
                        'about:blank',
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env={**os.environ, 'XDG_CONFIG_HOME': profile, 'XDG_DATA_HOME': profile},
                )
            )
            targets = wait_json('http://127.0.0.1:8791/json')
            target = next(item for item in targets if item['type'] == 'page')
            print('Connecting to Chrome.', flush=True)
            client = CDP(target['webSocketDebuggerUrl'])
            client.call('Runtime.enable')
            client.call('Page.enable')
            client.call(
                'Page.addScriptToEvaluateOnNewDocument',
                {
                    'source': """
                localStorage.setItem('cuju.muted', 'true');
                localStorage.setItem('cuju.quality', '"low"');
                Element.prototype.requestPointerLock = function () {
                    throw new Error('Pointer capture refused by the page check.');
                };
            """
                },
            )
            client.call('Page.navigate', {'url': 'http://127.0.0.1:8790/'})
            for _ in range(150):
                if client.evaluate('!!(window.cuju && window.cuju.menus)'):
                    break
                time.sleep(0.2)
            else:
                raise RuntimeError('Renderer startup failed.')
            for page in [
                'home',
                'watch',
                'recordings',
                'matches',
                'teams',
                'field',
                'play',
                'help',
            ]:
                result = client.evaluate(f"""
                    document.querySelector('[data-page="{page}"]').click();
                    ({{
                        page: cuju.page,
                        visible: !document.getElementById('page-{page}').hidden,
                        hash: location.hash
                    }})
                """)
                assert result['page'] == page and result['visible'], result
                assert result['hash'] == f'#page={page}', result
                checks += 1
            for page in ['home', 'teams', 'field', 'play']:
                client.evaluate(f"cuju.showPage('{page}'); document.activeElement.blur()")
                client.call(
                    'Input.dispatchKeyEvent',
                    {
                        'type': 'keyDown',
                        'key': 'h',
                        'code': 'KeyH',
                    },
                )
                assert client.evaluate('cuju.page') == 'help'
                checks += 1
                client.call(
                    'Input.dispatchKeyEvent',
                    {
                        'type': 'keyDown',
                        'key': 'Escape',
                        'code': 'Escape',
                    },
                )
                assert client.evaluate('cuju.page') == page
                checks += 1
            result = client.evaluate("""
                (async () => {
                    const { api } = await import('./js/api.js');
                    const entries = await api.matches();
                    await cuju.openMatch(entries[0].id);
                    cuju.updateHash(true);
                    const sourceHash = location.hash;
                    cuju.showPage('teams');
                    const teamsHash = location.hash;
                    cuju.showPage('watch');
                    cuju.updateHash(true);
                    return sourceHash.startsWith('#match=' + entries[0].id)
                        && teamsHash === '#page=teams'
                        && location.hash.startsWith('#match=' + entries[0].id);
                })()
            """)
            assert result, 'Source addresses'
            checks += 1
            client.evaluate(
                "cuju.director.stop(true); cuju.timeline.pause(); cuju.showPage('watch')"
            )
            for key, code in [
                (' ', 'Space'),
                ('k', 'KeyK'),
                ('.', 'Period'),
                (',', 'Comma'),
                ('Home', 'Home'),
                ('End', 'End'),
                ('v', 'KeyV'),
                ('b', 'KeyB'),
                ('p', 'KeyP'),
                ('g', 'KeyG'),
            ]:
                client.evaluate('document.activeElement.blur()')
                client.call('Input.dispatchKeyEvent', {'type': 'keyDown', 'key': key, 'code': code})
                client.call('Input.dispatchKeyEvent', {'type': 'keyUp', 'key': key, 'code': code})
                assert client.evaluate("cuju.page === 'watch'")
                if code == 'Space':
                    assert client.evaluate('cuju.timeline.playing')
                elif code == 'KeyK':
                    assert not client.evaluate('cuju.timeline.playing')
                checks += 1
            result = client.evaluate("""
                (async () => {
                    const { api } = await import('./js/api.js');
                    cuju.showPage('teams');
                    await cuju.library.refresh();
                    const form = cuju.library.upload;
                    const fileInput = form.querySelector('input[type=file]:not([webkitdirectory])');
                    const name = form.querySelector('input[type=text]');
                    const original = api.uploadTeam;
                    let sent = null;
                    api.uploadTeam = async (...args) => { sent = args; return {}; };
                    try {
                        const choose = () => {
                            const data = new DataTransfer();
                            data.items.add(new File(['archive'], 'export.zip'));
                            fileInput.files = data.files;
                            fileInput.dispatchEvent(new Event('change'));
                            name.value = 'Imported';
                        };
                        choose();
                        form.dispatchEvent(new Event('submit', { cancelable: true }));
                        await new Promise((resolve) => setTimeout(resolve, 20));
                        const untouched = sent && sent[4] === null;
                        choose();
                        const count = form.querySelector('input[type=number]');
                        count.value = '2';
                        count.dispatchEvent(new Event('change', { bubbles: true }));
                        form.dispatchEvent(new Event('submit', { cancelable: true }));
                        await new Promise((resolve) => setTimeout(resolve, 20));
                        return untouched && sent[4].players === 2;
                    } finally {
                        api.uploadTeam = original;
                    }
                })()
            """)
            assert result, 'Import form preserves archive controls until edited'
            checks += 1
            result = client.evaluate("""
                (async () => {
                    const { controlForm } = await import('./js/api.js');
                    const form = controlForm('11v11', {
                        players: 1, assignment: 'fixed', selection: ['CB'],
                        goalkeeper: 'scenario', action_set: 'default',
                    });
                    const buttons = form.querySelectorAll('button');
                    if (form.querySelectorAll('[aria-pressed=true]').length !== 1) return false;
                    buttons[3].click();
                    let value = JSON.parse(form.value);
                    if (value.selection.join(',') !== '2,3' || value.players !== 2) return false;
                    buttons[2].click();
                    buttons[3].click();
                    value = JSON.parse(form.value);
                    return value.assignment === 'dynamic' && value.selection.length === 0;
                })()
            """)
            assert result, 'Lineup selection and repeated roles'
            checks += 1
            client.evaluate("""
                (async () => {
                    cuju.showPage('teams');
                    await cuju.library.refresh();
                    cuju.library.root.querySelector('li').click();
                    for (let i = 0; i < 100 && !cuju.editor.sectionLabel; i++) {
                        await new Promise((resolve) => setTimeout(resolve, 50));
                    }
                })()
            """)
            for section in [
                'identity',
                'kit',
                'roster',
                'model',
                'fans',
                'celebrations',
                'sounds',
                'sponsors',
                'export',
            ]:
                assert client.evaluate(f"""
                    cuju.editor.showSection('{section}');
                    cuju.editor.section === '{section}' && !!cuju.editor.formHost.children.length
                """)
                checks += 1
            client.evaluate("cuju.editor.close(true); cuju.showPage('help')")
            assert client.evaluate(
                "document.getElementById('page-help').contains(document.getElementById('help'))"
            )
            checks += 1

            result = client.evaluate("""
                (async () => {
                    const { api } = await import('./js/api.js');
                    const sources = await api.live();
                    const original = Object.fromEntries(
                        ['startGame', 'gameStream', 'pauseGame', 'gameInput', 'quitGame', 'game']
                            .map((name) => [name, api[name]])
                    );
                    let quitCount = 0;
                    let entry = { id: sources[0].id, state: 'paused', score: [0, 0] };
                    api.startGame = async () => entry;
                    api.gameStream = (id, signal) => api.liveStream(id, signal);
                    api.pauseGame = async (id, paused) => {
                        entry = { ...entry, state: paused ? 'paused' : 'playing' };
                        return entry;
                    };
                    const inputs = [];
                    const readPads = navigator.getGamepads;
                    api.gameInput = async (id, input) => { inputs.push(input); return entry; };
                    api.game = async () => entry;
                    api.quitGame = async () => {
                        quitCount++;
                        entry = { ...entry, state: 'ended', ending: 'quit' };
                        return entry;
                    };
                    try {
                        cuju.showPage('play');
                        await cuju.play.start();
                        for (let i = 0; i < 100 && !cuju.play.game.started; i++) {
                            await new Promise((resolve) => setTimeout(resolve, 100));
                        }
                        if (!cuju.play.game.started) throw new Error('Play stream startup');
                        const keys = [
                            ['ArrowLeft', 'ArrowLeft', ['left']],
                            ['ArrowRight', 'ArrowRight', ['right']],
                            ['ArrowUp', 'ArrowUp', ['up']],
                            ['ArrowDown', 'ArrowDown', ['down']],
                            ['w', 'KeyW', ['long_pass']], ['a', 'KeyA', ['high_pass', 'sliding']],
                            ['s', 'KeyS', ['short_pass']], ['d', 'KeyD', ['shot']],
                            ['e', 'KeyE', ['sprint']], ['c', 'KeyC', ['dribble']],
                        ];
                        for (const [key, code, actions] of keys) {
                            cuju.play.view.focus();
                            cuju.play.view.dispatchEvent(new KeyboardEvent('keydown', {
                                key, code, bubbles: true,
                            }));
                            await new Promise((resolve) => setTimeout(resolve, 20));
                            if (!actions.every((action) => inputs.at(-1).held.includes(action))) {
                                throw new Error('Play input ' + code);
                            }
                            cuju.play.view.dispatchEvent(new KeyboardEvent('keyup', {
                                key, code, bubbles: true,
                            }));
                            await new Promise((resolve) => setTimeout(resolve, 20));
                            if (inputs.at(-1).held.length) throw new Error('Play release ' + code);
                        }
                        const pad = {
                            index: 0, connected: true, mapping: 'standard', axes: [0, 0],
                            buttons: Array.from({ length: 16 }, () => ({ pressed: false })),
                        };
                        Object.defineProperty(navigator, 'getGamepads', {
                            configurable: true, value: () => [pad],
                        });
                        cuju.play.pollPads();
                        for (const [index, actions] of [
                            [0, ['short_pass']], [1, ['shot']],
                            [2, ['high_pass', 'sliding']], [3, ['long_pass']], [7, ['sprint']], [5, ['dribble']],
                            [12, ['up']], [13, ['down']], [14, ['left']], [15, ['right']],
                        ]) {
                            pad.buttons[index].pressed = true;
                            cuju.play.pollPads();
                            await new Promise((resolve) => setTimeout(resolve, 20));
                            if (!actions.every((action) => inputs.at(-1).held.includes(action))) {
                                throw new Error('Controller button ' + index);
                            }
                            pad.buttons[index].pressed = false;
                            cuju.play.pollPads();
                            await new Promise((resolve) => setTimeout(resolve, 20));
                        }
                        pad.buttons[9].pressed = true;
                        cuju.play.pollPads();
                        if (cuju.play.menu.hidden) throw new Error('Controller pause');
                        pad.buttons[9].pressed = false;
                        cuju.play.pollPads();
                        pad.buttons[9].pressed = true;
                        cuju.play.pollPads();
                        if (!cuju.play.menu.hidden) throw new Error('Controller resume');
                        pad.buttons[9].pressed = false;
                        cuju.play.pause(true);
                        const sameGame = cuju.play.game;
                        const pick = (text) => [...cuju.play.menu.querySelectorAll('button')]
                            .find((button) => button.textContent === text).click();
                        pick('Home');
                        await new Promise((resolve) => setTimeout(resolve, 50));
                        const home = cuju.page === 'home' && quitCount === 0
                            && cuju.play.game === sameGame && sameGame.paused;
                        cuju.showPage('play');
                        await new Promise((resolve) => setTimeout(resolve, 50));
                        const resumeAvailable = !cuju.play.menu.hidden;
                        pick('Resume');
                        await new Promise((resolve) => setTimeout(resolve, 50));
                        const resumed = !sameGame.paused && cuju.play.menu.hidden;
                        cuju.play.pause(true);
                        pick('Quit to Home');
                        for (let i = 0; i < 100 && cuju.page !== 'home'; i++) {
                            await new Promise((resolve) => setTimeout(resolve, 50));
                        }
                        return home && resumeAvailable && resumed && quitCount === 1
                            && cuju.page === 'home' && cuju.play.game === null;
                    } finally {
                        Object.defineProperty(navigator, 'getGamepads', {
                            configurable: true, value: readPads,
                        });
                        if (cuju.play.game?.source) cuju.play.game.source.live.stop();
                        Object.assign(api, original);
                    }
                })()
            """)
            assert result, 'Play Home, Resume and Quit to Home'
            checks += 1
            assert client.evaluate("localStorage.getItem('cuju.muted')") == 'true'
            assert not client.evaluate('!!document.pointerLockElement')
            assert not client.errors, client.errors
            print(
                f'PASS. {checks} page, source, key, import and editor checks. Chrome stayed muted.'
            )
    except Exception as error:
        print(f'FAIL. {checks} checks passed. {error}', file=sys.stderr)
        return 1
    finally:
        if client:
            client.socket.close()
        for process in reversed(processes):
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return 0


if __name__ == '__main__':
    sys.exit(main())

# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""The local server of the renderer, which serves the page and its API on the loopback interface.

It lists, sends and deletes recordings, lists and relays live sources, keeps the team registry and
plays requested matches, each in a process of its own. It uses only Python's standard library.
"""

from __future__ import annotations

import collections
import http.client
import http.server
import json
import mimetypes
import os
import subprocess
import sys
import threading
import urllib.parse
from pathlib import Path
from typing import Any

from .. import bundled_scenarios
from .._version import __version__
from ..errors import ArgumentError
from . import _home, _library, _teams

_WEB = Path(__file__).parent / 'web'
# A requested match ends after this many seconds, a full match plays in a few minutes.
_MATCH_SECONDS = 30 * 60
_TYPES = {
    '.js': 'text/javascript',
    '.mjs': 'text/javascript',
    '.css': 'text/css',
    '.html': 'text/html',
    '.json': 'application/json',
    '.glb': 'model/gltf-binary',
    '.webp': 'image/webp',
    '.png': 'image/png',
    '.svg': 'image/svg+xml',
}


class _Requests:
    """The queue of requested matches, played one at a time in child processes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[dict[str, Any]] = []
        self._processes: dict[str, subprocess.Popen] = {}
        self._wake = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def add(
        self,
        left: str,
        right: str,
        scenario: str,
        seed: int | None,
        ability: Any = (None, None),
    ) -> dict[str, Any]:
        """Queues a match and returns its entry."""
        if scenario not in _scenarios():
            raise ArgumentError(f'expected a shipped scenario, found {scenario!r}')
        _teams.check_scenario([_teams.entry(left), _teams.entry(right)], scenario)
        ability = list(_teams.check_ability(ability))
        entry = {
            'id': _home.new_identifier('r'),
            'left': left,
            'right': right,
            'scenario': scenario,
            'seed': seed,
            'ability': ability,
            'state': 'queued',
            'progress': 0.0,
            'match': None,
            'error': None,
        }
        with self._lock:
            self._entries.append(entry)
        self._wake.set()
        return dict(entry)

    def entries(self) -> list[dict[str, Any]]:
        """Returns a copy of every request, newest first."""
        with self._lock:
            return [dict(entry) for entry in reversed(self._entries)]

    def _run(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            while True:
                with self._lock:
                    queued = [entry for entry in self._entries if entry['state'] == 'queued']
                if not queued:
                    break
                self._play(queued[0])

    def cancel(self, identifier: str) -> bool:
        """Cancels a queued match or ends a running one, and returns whether it existed."""
        with self._lock:
            entry = next((item for item in self._entries if item['id'] == identifier), None)
            if entry is None or entry['state'] not in ('queued', 'running'):
                return False
            process = self._processes.get(identifier)
            entry['state'], entry['error'] = 'failed', 'cancelled'
        if process is not None:
            process.kill()
        return True

    def _play(self, entry: dict[str, Any]) -> None:
        """Plays one requested match in a child process, which nothing it does can stop for good."""
        try:
            self._play_child(entry)
        except Exception as error:
            with self._lock:
                entry['state'], entry['error'] = 'failed', f'{type(error).__name__}: {error}'

    def _play_child(self, entry: dict[str, Any]) -> None:
        command = [
            sys.executable,
            '-m',
            'cuju.render._play',
            '--left',
            entry['left'],
            '--right',
            entry['right'],
            '--scenario',
            entry['scenario'],
        ]
        if entry['seed'] is not None:
            command += ['--seed', str(entry['seed'])]
        for side, value in zip(('left', 'right'), entry['ability'], strict=True):
            if value is not None:
                command += [f'--{side}-ability', repr(value)]
        with self._lock:
            if entry['state'] != 'queued':
                return
            entry['state'] = 'running'
            # Starting and registering the child share the cancellation lock. The child runs in the
            # renderer's folder, so no module beside the server's working folder shadows cuju or
            # torch, and it reads nothing from the server's terminal.
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=_home.home(),
                env=os.environ.copy(),
            )
            self._processes[entry['id']] = process
        stderr: collections.deque[str] = collections.deque(maxlen=20)
        threading.Thread(target=lambda: stderr.extend(process.stderr), daemon=True).start()
        expired = threading.Event()

        def expire() -> None:
            expired.set()
            process.kill()

        timer = threading.Timer(_MATCH_SECONDS, expire)
        timer.daemon = True
        timer.start()
        try:
            for line in process.stdout:
                try:
                    report = json.loads(line)
                    progress = float(report.get('progress', entry['progress']))
                except (ValueError, TypeError, AttributeError):
                    continue
                with self._lock:
                    entry['progress'] = progress
                    if isinstance(report.get('match'), str):
                        entry['match'] = report['match']
                    if isinstance(report.get('error'), str):
                        entry['error'] = report['error']
            code = process.wait()
        finally:
            timer.cancel()
            with self._lock:
                self._processes.pop(entry['id'], None)
        with self._lock:
            if entry['state'] != 'running':
                return
            if code == 0 and entry['match']:
                entry['state'] = 'done'
            else:
                entry['state'] = 'failed'
                if expired.is_set():
                    entry['error'] = f'the match ran past {_MATCH_SECONDS // 60} minutes'
                elif not entry['error']:
                    tail = ''.join(stderr).strip()
                    entry['error'] = f'the match ended with exit code {code}' + (
                        f', {tail}' if tail else ''
                    )


class GameBusyError(Exception):
    """Raised when a game is asked for while another one plays."""


class NoGameError(Exception):
    """Raised for a game that is not the latest one."""


# A game plays this many steps a second, real time, and a test may play faster.
_GAME_PACE = 10
# The page's input of one step names at most this many inputs, the four directions and the buttons.
_GAME_INPUTS = 16
# A stream waits this long for the game's process to start and announce its port.
_GAME_START_SECONDS = 120


class _Games:
    """The game that a person plays on the page, one at a time, in a child process.

    The process reads its commands, the input of each person's side, pause and quit, as JSON lines
    on its standard input, and reports its state, its port and the recording it saved as JSON lines
    on its standard output.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entry: dict[str, Any] | None = None
        self._process: subprocess.Popen | None = None
        self._started = threading.Condition(self._lock)

    def start(
        self,
        scenario: str,
        humans: Any,
        ability: Any = (None, None),
        seed: int | None = None,
        record: bool = True,
    ) -> dict[str, Any]:
        """Starts a game and returns its entry.

        Raises:
            ArgumentError: If the scenario is not shipped, the sides name no person or an ability
                lies outside 0 to 1.
            GameBusyError: If another game still plays.
        """
        if scenario not in _scenarios():
            raise ArgumentError(f'expected a shipped scenario, found {scenario!r}')
        if (
            not isinstance(humans, (list, tuple))
            or len(humans) != 2
            or not all(isinstance(value, bool) for value in humans)
            or not any(humans)
        ):
            raise ArgumentError(
                f'expected two flags for a person on the left and right, one true, found {humans!r}'
            )
        ability = list(_teams.check_ability(ability))
        if not isinstance(record, bool):
            raise ArgumentError(f'expected record as true or false, found {record!r}')
        entry = {
            'id': _home.new_identifier('g'),
            'scenario': scenario,
            'humans': list(humans),
            'ability': ability,
            'seed': seed,
            'record': record,
            'state': 'starting',
            'reason': None,
            'ending': None,
            'step': 0,
            'score': [0, 0],
            'teams': None,
            'match': None,
            'error': None,
        }
        command = [
            sys.executable,
            '-m',
            'cuju.render._game',
            '--scenario',
            scenario,
            '--left',
            'person' if humans[0] else 'builtin',
            '--right',
            'person' if humans[1] else 'builtin',
            '--pace',
            repr(float(_GAME_PACE)),
        ]
        if seed is not None:
            command += ['--seed', str(seed)]
        if record:
            command.append('--record')
        for side, value in zip(('left', 'right'), ability, strict=True):
            if value is not None:
                command += [f'--{side}-ability', repr(value)]
        with self._lock:
            if self._entry is not None and self._entry['state'] not in ('ended', 'failed'):
                raise GameBusyError('a game already plays, quit it first')
            # The child runs in the renderer's folder and writes its errors to a pipe of its own.
            _home.home().mkdir(parents=True, exist_ok=True)
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=_home.home(),
                env=os.environ.copy(),
            )
            self._entry, self._process = entry, process
        threading.Thread(target=self._watch, args=(entry, process), daemon=True).start()
        return dict(entry)

    def current(self) -> dict[str, Any] | None:
        """Returns a copy of the latest game's entry, or `None` before the first."""
        with self._lock:
            return None if self._entry is None else self._public(self._entry)

    @staticmethod
    def _public(entry: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in entry.items() if key != 'port'}

    def _watch(self, entry: dict[str, Any], process: subprocess.Popen) -> None:
        """Follows the reports of a game's process until it ends."""
        stderr: collections.deque[str] = collections.deque(maxlen=20)
        threading.Thread(target=lambda: stderr.extend(process.stderr), daemon=True).start()
        for line in process.stdout:
            try:
                report = json.loads(line)
            except ValueError:
                continue
            if not isinstance(report, dict):
                continue
            with self._lock:
                for key in (
                    'state',
                    'reason',
                    'ending',
                    'step',
                    'score',
                    'match',
                    'error',
                    'seed',
                    'teams',
                ):
                    if key in report:
                        entry[key] = report[key]
                if isinstance(report.get('port'), int):
                    entry['port'] = report['port']
                    self._started.notify_all()
        code = process.wait()
        with self._lock:
            if code == 0 and entry['state'] == 'ended':
                pass
            elif entry['state'] != 'ended':
                entry['state'] = 'failed'
                if not entry['error']:
                    tail = ''.join(stderr).strip()
                    entry['error'] = f'the game ended with exit code {code}' + (
                        f', {tail}' if tail else ''
                    )
            self._started.notify_all()

    def port(self, identifier: str) -> int | None:
        """Waits for the game's stream to open and returns its port, or `None` if it never does."""
        with self._lock:
            entry = self._entry
            if entry is None or entry['id'] != identifier:
                return None
            self._started.wait_for(
                lambda: 'port' in entry or entry['state'] in ('ended', 'failed'),
                _GAME_START_SECONDS,
            )
            return entry.get('port') if entry['state'] not in ('ended', 'failed') else None

    def send(self, identifier: str, command: dict[str, Any]) -> dict[str, Any]:
        """Passes a command to the game's process and returns the game's entry.

        Raises:
            NoGameError: If the latest game has another identifier.
        """
        with self._lock:
            entry, process = self._entry, self._process
            if entry is None or entry['id'] != identifier or process is None:
                raise NoGameError('no such game')
            if entry['state'] in ('ended', 'failed'):
                return self._public(entry)
            try:
                process.stdin.write(json.dumps(command) + '\n')
                process.stdin.flush()
            except (OSError, ValueError):
                pass
            return self._public(entry)

    def input(self, identifier: str, document: dict[str, Any]) -> dict[str, Any]:
        """Checks the input of a person's side and passes it to the game.

        The input names the side, 0 for the left and 1 for the right, the inputs held now and the
        buttons pressed since the last input, each from `cuju.render._game.INPUTS`.

        Raises:
            ArgumentError: If the input holds anything else.
        """
        from ._game import BUTTONS, INPUTS

        unknown = set(document) - {'side', 'held', 'pressed'}
        if unknown:
            raise ArgumentError(f'expected side, held and pressed, found {sorted(unknown)!r}')
        side = document.get('side')
        with self._lock:
            humans = self._entry['humans'] if self._entry is not None else [False, False]
        if isinstance(side, bool) or side not in (0, 1) or not humans[side]:
            raise ArgumentError(f'expected the side of a person, 0 or 1, found {side!r}')
        command: dict[str, Any] = {'side': side}
        for key, allowed in (('held', INPUTS), ('pressed', BUTTONS)):
            value = document.get(key, [])
            if (
                not isinstance(value, list)
                or len(value) > _GAME_INPUTS
                or not all(isinstance(name, str) and name in allowed for name in value)
            ):
                raise ArgumentError(f'expected {key} as a list of {", ".join(allowed)}')
            command[key] = sorted(set(value))
        return self.send(identifier, command)

    def quit(self, identifier: str) -> dict[str, Any]:
        """Asks the game to end, saving it when it records, and ends its process if it hangs."""
        entry = self.send(identifier, {'quit': True})
        with self._lock:
            process = self._process
        if process is not None:

            def stop() -> None:
                # A game still starting reads the quit once it is ready, so the wait for a hang
                # starts when its stream opens.
                self.port(identifier)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()

            threading.Thread(target=stop, daemon=True).start()
        return entry


def _seed(value: Any) -> int | None:
    """Returns the seed of a request, an integer from 0 to 2**64 - 1, or `None` to draw one."""
    if value in (None, ''):
        return None
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**64:
        raise ArgumentError(f'expected a seed from 0 to 2**64 - 1, found {value!r}')
    return value


def _control(query: dict[str, str]) -> dict[str, Any] | None:
    """Returns uploaded team control, accepting earlier count and goalkeeper queries."""
    if 'control' in query:
        return json.loads(query['control'])
    if 'players' not in query and 'goalkeeper' not in query:
        return None
    players = query.get('players', '1')
    if not players.isdigit():
        raise ArgumentError(f'expected a number of controlled players, found {players!r}')
    return {'players': int(players), 'goalkeeper': query.get('goalkeeper', 'agent')}


def _scenarios() -> list[str]:
    """Returns the shipped scenarios, without agent counts."""
    return sorted([*bundled_scenarios(), *_teams.renderer_scenarios()])


def _sources() -> list[dict[str, Any]]:
    """Returns the live sources whose process runs, and withdraws the announcements of others."""
    sources = []
    for path in _home.folder('live').glob('*.json'):
        try:
            source = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        # An announcement without a whole number for its process is skipped, and the list goes on.
        pid = source.get('pid') if isinstance(source, dict) else None
        if not isinstance(pid, int) or isinstance(pid, bool):
            continue
        if not _alive(pid):
            path.unlink(missing_ok=True)
            continue
        sources.append(source)
    return sources


def _alive(pid: int) -> bool:
    """Returns whether a process runs, without signalling it."""
    if pid <= 0:
        return False
    if sys.platform == 'win32':
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        # Windows: 259 is `STILL_ACTIVE`.
        return code.value == 259
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# The names of the loopback interface, which a request's host must carry.
_LOOPBACK = ('127.0.0.1', 'localhost', '::1')
# The largest body a request may send, a submission of 100 MB with room for its index.
_MAXIMUM_BODY = 110 * 1024 * 1024


def make_handler(
    requests: _Requests, games: _Games | None = None
) -> type[http.server.BaseHTTPRequestHandler]:
    """Returns the request handler of the server, which serves the loopback interface alone."""
    allowed_hosts = set(_LOOPBACK)
    games = games if games is not None else _Games()

    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = 'cuju-render'
        timeout = 60

        def end_headers(self) -> None:
            """Adds to every response the headers that keep other sites from framing the page."""
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Content-Security-Policy', "frame-ancestors 'none'")
            self.send_header('X-Content-Type-Options', 'nosniff')
            super().end_headers()

        def _allowed(self) -> bool:
            """Refuses a request of another site, which could upload and run code.

            A page of another site that names this server, or a host name of another site that
            resolves here, sends another host or origin, so the server answers only requests that
            name it by its own host and come from its own page or from no page at all.
            """
            named = self.headers.get('Host') or ''
            if urllib.parse.urlsplit('//' + named).hostname not in allowed_hosts:
                self._error(403, 'expected a request to this server by its loopback address')
                return False
            # A page served by another port of this machine is another site, so the whole
            # origin, scheme, host and port, must be this server's own.
            origin = self.headers.get('Origin')
            if origin is not None and self.command != 'GET' and origin != f'http://{named}':
                self._error(403, 'expected a request from the page of this server')
                return False
            try:
                length = int(self.headers.get('Content-Length') or 0)
            except ValueError:
                self._error(400, 'expected a Content-Length of digits')
                return False
            if length > _MAXIMUM_BODY:
                self._error(413, 'expected a request body of at most 110 MB')
                return False
            return True

        def log_message(self, *arguments: Any) -> None:
            pass

        def _handle(self, method: Any) -> None:
            if not self._allowed():
                return
            try:
                method()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            except (ArgumentError, ValueError, TypeError, KeyError, AttributeError) as error:
                self._error(400, f'{type(error).__name__}: {error}')
            except Exception as error:
                self._error(500, f'{type(error).__name__}: {error}')

        def _json(self, value: Any, status: int = 200) -> None:
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._json({'error': message}, status)

        def _body(self) -> bytes:
            length = int(self.headers.get('Content-Length') or 0)
            return self.rfile.read(length) if length > 0 else b''

        def _document(self) -> dict[str, Any]:
            document = json.loads(self._body() or b'{}')
            if not isinstance(document, dict):
                raise ArgumentError('expected a JSON object')
            return document

        def _route(self) -> tuple[list[str], dict[str, str]]:
            parsed = urllib.parse.urlsplit(self.path)
            parts = [urllib.parse.unquote(part) for part in parsed.path.split('/') if part]
            query = dict(urllib.parse.parse_qsl(parsed.query))
            return parts, query

        def do_GET(self) -> None:
            self._handle(self._get)

        def _get(self) -> None:
            parts, _ = self._route()
            try:
                if parts[:1] != ['api']:
                    self._static(parts)
                elif parts == ['api', 'matches']:
                    self._json(_library.matches())
                elif len(parts) == 3 and parts[1] == 'matches':
                    self._recording(parts[2])
                elif parts == ['api', 'live']:
                    self._json(
                        [
                            {key: source[key] for key in ('id', 'title', 'scenario', 'environment')}
                            for source in _sources()
                        ]
                    )
                elif len(parts) == 4 and parts[1] == 'live' and parts[3] == 'stream':
                    self._relay(parts[2])
                elif parts == ['api', 'info']:
                    self._json({'version': __version__})
                elif parts == ['api', 'scenarios']:
                    self._json(_scenarios())
                elif parts == ['api', 'teams']:
                    self._json(_teams.teams())
                elif len(parts) == 4 and parts[1] == 'teams' and parts[3] == 'model':
                    self._model(parts[2])
                elif len(parts) == 5 and parts[1] == 'teams' and parts[3] == 'images':
                    self._image(parts[2], parts[4])
                elif len(parts) == 5 and parts[1] == 'teams' and parts[3] == 'sounds':
                    self._sound(parts[2], parts[4])
                elif len(parts) == 4 and parts[1] == 'teams' and parts[3] == 'export':
                    self._export(parts[2])
                elif parts == ['api', 'models']:
                    self._json(_teams.shipped_models())
                elif parts == ['api', 'requests']:
                    self._json(requests.entries())
                elif parts == ['api', 'game']:
                    self._json(games.current())
                elif len(parts) == 4 and parts[1] == 'game' and parts[3] == 'stream':
                    port = games.port(parts[2])
                    if port is None:
                        self._error(404, 'no such game')
                    else:
                        self._pipe(port, 'game')
                else:
                    self._error(404, 'no such resource')
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_POST(self) -> None:
            self._handle(self._post)

        def _post(self) -> None:
            parts, query = self._route()
            try:
                if parts == ['api', 'teams']:
                    entry = _teams.add(
                        query.get('name', ''),
                        query.get('filename', 'main.py'),
                        self._body(),
                        query.get('format', '11v11'),
                        _control(query),
                    )
                    self._json(entry, 201)
                elif parts == ['api', 'requests']:
                    document = self._document()
                    entry = requests.add(
                        str(document.get('left', _teams.BUILTIN)),
                        str(document.get('right', _teams.BUILTIN)),
                        str(document.get('scenario', '11_vs_11_kaggle')),
                        _seed(document.get('seed')),
                        document.get('ability', [None, None]),
                    )
                    self._json(entry, 201)
                elif parts == ['api', 'game']:
                    document = self._document()
                    entry = games.start(
                        str(document.get('scenario', '11_vs_11_kaggle')),
                        document.get('humans', [True, False]),
                        document.get('ability', [None, None]),
                        _seed(document.get('seed')),
                        document.get('record', True),
                    )
                    self._json(entry, 201)
                elif len(parts) == 4 and parts[1] == 'game' and parts[3] == 'input':
                    self._json(games.input(parts[2], self._document()))
                elif len(parts) == 4 and parts[1] == 'game' and parts[3] == 'pause':
                    paused = self._document().get('paused')
                    if not isinstance(paused, bool):
                        raise ArgumentError(f'expected paused as true or false, found {paused!r}')
                    self._json(games.send(parts[2], {'paused': paused}))
                else:
                    self._error(404, 'no such resource')
            except GameBusyError as error:
                self._error(409, str(error))
            except NoGameError as error:
                self._error(404, str(error))
            except (ArgumentError, ValueError) as error:
                self._error(400, str(error))

        def do_PUT(self) -> None:
            self._handle(self._put)

        def _put(self) -> None:
            parts, query = self._route()
            try:
                if len(parts) == 4 and parts[1] == 'teams' and parts[3] == 'model':
                    joints = json.loads(query.get('joints') or '{}')
                    self._json(_teams.set_model(parts[2], self._body(), joints))
                elif len(parts) == 5 and parts[1] == 'teams' and parts[3] == 'images':
                    self._json(_teams.set_image(parts[2], parts[4], self._body()))
                elif len(parts) == 5 and parts[1] == 'teams' and parts[3] == 'sounds':
                    self._json(_teams.set_sound(parts[2], parts[4], self._body()))
                else:
                    self._error(404, 'no such resource')
            except (ArgumentError, ValueError) as error:
                self._error(400, str(error))

        def do_PATCH(self) -> None:
            self._handle(self._patch)

        def _patch(self) -> None:
            parts, _ = self._route()
            try:
                if len(parts) == 3 and parts[1] == 'matches':
                    document = self._document()
                    self._json(
                        _library.update(parts[2], document.get('title'), document.get('folder'))
                    )
                elif len(parts) == 3 and parts[1] == 'teams':
                    document = self._document()
                    self._json(
                        _teams.update(
                            parts[2],
                            document.get('name'),
                            document.get('style'),
                            document.get('format'),
                            document.get('control'),
                        )
                    )
                else:
                    self._error(404, 'no such resource')
            except (ArgumentError, ValueError) as error:
                self._error(400, str(error))

        def do_DELETE(self) -> None:
            self._handle(self._delete)

        def _delete(self) -> None:
            parts, _ = self._route()
            if len(parts) == 3 and parts[1] == 'game':
                try:
                    self._json(games.quit(parts[2]))
                except NoGameError:
                    self._error(404, 'no such game')
                return
            if len(parts) == 3 and parts[1] == 'requests':
                removed = requests.cancel(parts[2])
            elif len(parts) == 4 and parts[1] == 'teams' and parts[3] == 'model':
                try:
                    removed = _teams.remove_model(parts[2])
                except ArgumentError:
                    removed = False
            elif len(parts) == 5 and parts[1] == 'teams' and parts[3] == 'images':
                removed = _teams.remove_image(parts[2], parts[4])
            elif len(parts) == 5 and parts[1] == 'teams' and parts[3] == 'sounds':
                removed = _teams.remove_sound(parts[2], parts[4])
            elif len(parts) == 3 and parts[1] == 'matches':
                removed = _library.remove(parts[2])
            elif len(parts) == 3 and parts[1] == 'teams':
                removed = _teams.remove(parts[2])
            else:
                self._error(404, 'no such resource')
                return
            if removed:
                self.send_response(204)
                self.end_headers()
            else:
                self._error(404, 'no such item')

        def _static(self, parts: list[str]) -> None:
            relative = Path(*parts) if parts else Path('index.html')
            path = (_WEB / relative).resolve()
            if _WEB.resolve() not in path.parents or not path.is_file():
                self._error(404, 'no such file')
                return
            data = path.read_bytes()
            kind = _TYPES.get(path.suffix) or mimetypes.guess_type(path.name)[0]
            self.send_response(200)
            self.send_header('Content-Type', kind or 'application/octet-stream')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _model(self, identifier: str) -> None:
            path = _teams.model_path(identifier)
            if path is None:
                self._error(404, 'no such model')
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'model/gltf-binary')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def _export(self, identifier: str) -> None:
            try:
                filename, data = _teams.export(identifier)
            except ArgumentError as error:
                self._error(404, str(error))
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/zip')
            # A header carries Latin-1 alone, so the name travels encoded with an ASCII fallback.
            fallback = filename.encode('ascii', 'replace').decode().replace('?', '-')
            quoted = urllib.parse.quote(filename)
            self.send_header(
                'Content-Disposition',
                f'attachment; filename="{fallback}"; filename*=UTF-8\'\'{quoted}',
            )
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _image(self, identifier: str, slot: str) -> None:
            path = _teams.image_path(identifier, slot)
            if path is None:
                self._error(404, 'no such image')
                return
            data = path.read_bytes()
            kind = {'.png': 'image/png', '.jpg': 'image/jpeg', '.webp': 'image/webp'}
            self.send_response(200)
            self.send_header('Content-Type', kind.get(path.suffix, 'application/json'))
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def _sound(self, identifier: str, slot: str) -> None:
            path = _teams.sound_path(identifier, slot)
            if path is None:
                self._error(404, 'no such sound')
                return
            data = path.read_bytes()
            kind = {'.ogg': 'audio/ogg', '.mp3': 'audio/mpeg', '.wav': 'audio/wav'}
            self.send_response(200)
            self.send_header('Content-Type', kind[path.suffix])
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def _recording(self, identifier: str) -> None:
            path = _library.path_of(identifier)
            if path is None:
                self._error(404, 'no such match')
                return
            size = path.stat().st_size
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(size))
            self.end_headers()
            with path.open('rb') as stream:
                while chunk := stream.read(1 << 20):
                    self.wfile.write(chunk)

        def _relay(self, identifier: str) -> None:
            source = next((entry for entry in _sources() if entry['id'] == identifier), None)
            if source is None:
                self._error(404, 'no such live source')
                return
            self._pipe(int(source['port']), 'live source')

        def _pipe(self, port: int, kind: str) -> None:
            """Relays the stream of a live source or a game, which listens on `port`."""
            connection = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
            try:
                connection.connect()
                # A batch pauses between steps for as long as training needs, so once connected
                # the relay waits on the stream without a timeout.
                connection.sock.settimeout(None)
                connection.request('GET', '/stream')
                response = connection.getresponse()
            except OSError as error:
                connection.close()
                self._error(502, f'the {kind} did not answer, {error}')
                return
            if response.status != 200:
                connection.close()
                self._error(502, f'the {kind} answered {response.status}')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                while chunk := response.read1(1 << 16):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except OSError:
                pass
            finally:
                connection.close()

    return Handler


def serve(port: int = 8780, open_browser: bool = False) -> None:
    """Serves the renderer's page and API on the loopback interface until interrupted.

    The API deletes files and plays submitted code, so it listens on the loopback interface alone,
    and another machine reaches it through a tunnel, such as the port forwarding of SSH.

    Args:
        port: The port, and 0 picks a free one.
        open_browser: Whether to open the page in the default browser.
    """
    server = http.server.ThreadingHTTPServer(('127.0.0.1', port), make_handler(_Requests()))
    server.daemon_threads = True
    # The library receives each shipped sample match once, and keeps a deleted one deleted.
    try:
        _library.add_samples(again=False)
    except OSError as error:
        print(f'cuju render could not add the sample matches: {error}', flush=True)
    address = f'http://127.0.0.1:{server.server_address[1]}/'
    print(f'cuju render serves {address} from {_home.home()}', flush=True)
    if open_browser:
        import webbrowser

        threading.Timer(0.5, webbrowser.open, (address,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

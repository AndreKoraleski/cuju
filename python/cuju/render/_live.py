# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""A live source, which streams one environment of a running batch to the renderer's page.

The source listens on a port of the loopback interface and announces itself with a small file in
the renderer's folder, which the server lists. It reads a state only while a viewer watches, in
passages of five seconds of play paced to real time, so an unwatched batch pays one attribute check
per step and a watched one reads about ten states per second of wall time. Reading a state leaves
every output of the batch as it is, and a failure of the source turns it off, never the step.

The source keeps the last ten seconds of the steps it read, so a viewer who joins late starts with
that history. `CUJU_RENDER_BACKLOG` makes it read every step even unwatched, at the cost of one
state read per step, so that the first viewer finds a history as well.
"""

from __future__ import annotations

import atexit
import collections
import contextlib
import http.server
import json
import logging
import os
import queue
import struct
import threading
import time
import urllib.parse
import weakref
from typing import Any

from . import _home, _recording

# The variable that turns live sources on for every batch, with the environment to show.
LIVE_VARIABLE = 'CUJU_RENDER'
# Frames a viewer may hold before the source skips steps, five seconds of play.
_BUFFER = 50
# The variable that makes a source read every step, watched or not, for a history of its seconds.
BACKLOG_VARIABLE = 'CUJU_RENDER_BACKLOG'
# The history that a late viewer receives, ten seconds of steps.
_HISTORY = 100
# A watched source reads passages of this many consecutive steps, five seconds of play, and starts
# the next one when the wall clock has caught up, so a fast batch reads ten states a second.
_PASSAGE = 50
_STEPS_PER_SECOND = 10
_LOOPBACK = ('127.0.0.1', 'localhost', '::1')
_LOGGER = logging.getLogger('cuju')


class _Viewer:
    """One connected viewer, with its queue of frames."""

    def __init__(self) -> None:
        self.frames: queue.Queue[bytes | None] = queue.Queue(_BUFFER)
        self.header: bytes | None = None


class LiveSource:
    """Streams one environment of a batch to the renderer while a viewer watches.

    Attributes:
        identifier: The name of the source in the renderer's list.
        environment: The index of the environment that the source shows.
        title: The title that the viewer shows.
        port: The port of the loopback interface that the source listens on.
    """

    def __init__(
        self,
        batch: Any,
        environment: int = 0,
        title: str | None = None,
        teams: list[dict[str, Any]] | None = None,
        scenario: str | None = None,
        backlog: bool = False,
        announce: bool = True,
    ) -> None:
        """Starts a source for `environment` of `batch` and announces it to the renderer.

        Args:
            batch: The batch, which calls `after_step` after each of its steps.
            environment: The environment to show.
            title: The title, by default the scenario, the environment and the process.
            teams: The names and colors of the left and right team, as for a recording.
            scenario: The name of the scenario, for the title and the list of sources, by default
                the batch's.
            backlog: Whether to read every step, watched or not, so that any viewer starts with
                ten seconds of history, at the cost of one state read per step.
            announce: Whether the source joins the renderer's list of live sources. A game that a
                person plays streams through the server alone, so it stays off the list.
        """
        self._batch = weakref.ref(batch)
        self.environment = int(environment)
        players = int(batch.left_players[self.environment] + batch.right_players[self.environment])
        self._reader = _recording.StateReader(players)
        self._scenario = scenario or scenario_name(batch)
        self.title = title or (
            f'{self._scenario}, environment {self.environment}, process {os.getpid()}'
        )
        self._teams = teams
        # The scenario's timing comes with the first state read, since a batch may announce its
        # source before its first reset.
        self._timing: dict[str, Any] | None = None
        self._backlog = backlog
        self._history: collections.deque[bytes] = collections.deque(maxlen=_HISTORY)
        self._lineup: list[dict[str, Any]] | None = None
        self._viewers: list[_Viewer] = []
        self._lock = threading.Lock()
        self._passage = 0
        self._passage_start = 0.0
        self.identifier = _home.new_identifier('live')
        self._server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), self._handler())
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._closed = False
        self._thread.start()
        self._announcement = _home.folder('live') / f'{self.identifier}.json'
        if announce:
            self._write_announcement()
        atexit.register(self.close)

    def _write_announcement(self) -> None:
        """Writes the small file through which the server lists the source."""
        self._announcement.write_text(
            json.dumps(
                {
                    'id': self.identifier,
                    'title': self.title,
                    'scenario': self._scenario,
                    'environment': self.environment,
                    'port': self.port,
                    'pid': os.getpid(),
                }
            )
        )

    def _handler(self) -> type[http.server.BaseHTTPRequestHandler]:
        """Returns the request handler class, bound to this source."""
        source = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *arguments: Any) -> None:
                pass

            def do_GET(self) -> None:
                # The same guard as the server's, against pages of other sites that resolve here.
                host = urllib.parse.urlsplit('//' + (self.headers.get('Host') or '')).hostname
                if host not in _LOOPBACK:
                    self.send_error(403)
                    return
                if self.path != '/stream':
                    self.send_error(404)
                    return
                viewer = _Viewer()
                # The history and the registration share the lock, so no step is sent twice.
                with source._lock:
                    history = list(source._history)
                    if source._lineup is not None:
                        viewer.header = source._header()
                    if source._closed:
                        return
                    source._viewers.append(viewer)
                try:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/octet-stream')
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    if viewer.header is not None:
                        self.wfile.write(viewer.header + b''.join(history))
                        self.wfile.flush()
                    while True:
                        frame = viewer.frames.get()
                        if frame is None:
                            break
                        self.wfile.write(frame)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                finally:
                    with source._lock:
                        if viewer in source._viewers:
                            source._viewers.remove(viewer)

        return Handler

    def _header(self) -> bytes:
        """Returns the header of a stream, once a state has given the lineup."""
        return _recording.header(
            self.title, self._scenario, self._lineup, self._teams, timing=self._timing
        )

    def _paced(self) -> bool:
        """Returns whether this step belongs to a passage that a viewer receives."""
        now = time.monotonic()
        if self._passage >= _PASSAGE:
            if now < self._passage_start + _PASSAGE / _STEPS_PER_SECOND:
                return False
            self._passage = 0
        if self._passage == 0:
            self._passage_start = now
        self._passage += 1
        return True

    def after_step(self, done: Any) -> None:
        """Captures the environment's frame for every viewer with room in its queue.

        Args:
            done: The done flags of the step.
        """
        with self._lock:
            viewers = [viewer for viewer in self._viewers if not viewer.frames.full()]
        if viewers and not self._paced():
            viewers = []
        if not viewers and not self._backlog:
            return
        batch = self._batch()
        if batch is None:
            return
        state = batch._state(self.environment)
        frame = self._reader.frame(state, _end_flags(batch, done, self.environment))
        with self._lock:
            if self._lineup is None:
                self._lineup = self._reader.lineup(state)
            if self._timing is None:
                self._timing = _recording.scenario_timing(state)
        self._deliver(frame, viewers)

    def push(self, frame: bytes, lineup: list[dict[str, Any]]) -> None:
        """Sends a frame that the caller read to every viewer at once, without pacing.

        A game that a person plays reads each frame for its recording and streams the same bytes,
        so the source reads no state of its own and the page receives every step as it happens.

        Args:
            frame: The frame, as `StateReader.frame` returns it.
            lineup: The header entry of each player, as `StateReader.lineup` returns it.
        """
        with self._lock:
            if self._lineup is None:
                self._lineup = lineup
            viewers = list(self._viewers)
        self._deliver(frame, viewers)

    def _deliver(self, frame: bytes, viewers: list[_Viewer]) -> None:
        """Adds a frame to the history and queues it for `viewers`, a header first for new ones."""
        with self._lock:
            # The history holds consecutive steps only, so a gap starts it again.
            if self._history:
                (last,) = struct.unpack_from('<i', self._history[-1])
                (step,) = struct.unpack_from('<i', frame)
                if step != last + 1:
                    self._history.clear()
            if self._closed:
                return
            self._history.append(frame)
            for viewer in viewers:
                if viewer.header is None:
                    viewer.header = self._header()
                    payload = viewer.header + frame
                else:
                    payload = frame
                with contextlib.suppress(queue.Full):
                    viewer.frames.put_nowait(payload)

    def close(self) -> None:
        """Ends every stream, stops listening and withdraws the announcement."""
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        batch = self._batch()
        if batch is not None and getattr(batch, '_live', None) is self:
            batch._live = None
        with self._lock:
            viewers, self._viewers = self._viewers, []
        for viewer in viewers:
            while True:
                try:
                    viewer.frames.get_nowait()
                except queue.Empty:
                    break
            viewer.frames.put_nowait(None)
        self._server.shutdown()
        self._server.server_close()
        with contextlib.suppress(OSError):
            self._announcement.unlink()


def _end_flags(batch: Any, done: Any, environment: int) -> int:
    """Returns the flags of a frame for the done flags of its step.

    With automatic resets the step that ends an episode already shows the next kick-off, so its
    frame carries 2, an episode ended just before it, and otherwise the frame that ends one carries
    1.
    """
    if done is None or not bool(done[environment]):
        return 0
    return 2 if batch.auto_reset else 1


def disable(batch: Any, error: BaseException) -> None:
    """Turns a batch's live source off after a failure, warning once, so its steps go on."""
    _LOGGER.warning('cuju render: the live source stopped, %s: %s', type(error).__name__, error)
    live, batch._live = getattr(batch, '_live', None), None
    if live is not None:
        with contextlib.suppress(Exception):
            live.close()


def from_environment(batch: Any, assets: list[Any]) -> LiveSource | None:
    """Returns a source for `batch` when `CUJU_RENDER` asks for one, else `None`.

    `CUJU_RENDER` holds `on` for environment 0, or the index of the environment to show, and an
    empty value or `off` leaves the batch without a source. A non-empty `CUJU_RENDER_BACKLOG` other
    than `0` or `off` makes the source read every step, so that any viewer starts with a history.

    Args:
        batch: The batch.
        assets: The bundles or scenarios the batch was created with, which name the scenario.
    """
    value = os.environ.get(LIVE_VARIABLE, '').strip().lower()
    if value in ('', 'off', 'false', 'no'):
        return None
    environment = int(value) if value.isdigit() else 0
    if not 0 <= environment < batch.environment_count:
        environment = 0
    backlog = os.environ.get(BACKLOG_VARIABLE, '').strip().lower() not in ('', '0', 'off', 'no')
    return LiveSource(batch, environment, scenario=scenario_name(batch), backlog=backlog)


def scenario_name(batch: Any) -> str:
    """Returns the scenarios a batch was created with, as their names without paths or suffixes."""
    names = []
    for asset in getattr(batch, '_requested', None) or []:
        if isinstance(asset, (str, os.PathLike)):
            name = os.path.basename(str(asset))
            for suffix in ('.cjb', '.py'):
                name = name.removesuffix(suffix)
            if name not in names:
                names.append(name)
    return ', '.join(names) or 'match'

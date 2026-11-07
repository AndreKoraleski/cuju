# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Watches cuju's matches in a browser, saved or live, in 3D with sound.

The renderer reads GRF states and never writes one, so a match plays the same bits watched or not.
`Recorder` saves the matches of a batch, `watch` streams one environment of a running batch,
`add_team` registers a Kaggle submission folder, `play` records a match between two teams of the
registry, and `serve` runs the local page that plays them all. The guide is in the repository's
`docs/rendering.md`.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from ..errors import ArgumentError
from . import _library, _live, _recording
from ._home import home
from ._live import LiveSource
from ._server import serve
from ._teams import add_folder as add_team

__all__ = ['LiveSource', 'Recorder', 'add_team', 'home', 'play', 'serve', 'watch']


class Recorder:
    """Records the matches of one environment of a batch into the renderer's library.

    Call `capture` after `reset` and after each `step`. A recording ends when its episode ends, and
    `save` writes it, so a batch with automatic resets yields one recording per episode.

    Attributes:
        environment: The index of the environment.
        saved: The paths of the recordings written so far.
    """

    def __init__(
        self,
        batch: Any,
        environment: int = 0,
        title: str | None = None,
        teams: list[dict[str, Any]] | None = None,
        scenario: str | None = None,
    ) -> None:
        """Prepares a recorder of `environment` of `batch`.

        Args:
            batch: The batch.
            environment: The environment to record.
            title: The title of each recording, by default the scenario and the environment.
            teams: The name, `color` and `secondary` color of the left and right team, each an
                RGB list of three integers from 0 to 255, each optional.
            scenario: The name of the scenario that the header records, by default the batch's.
        """
        if not hasattr(batch, '_state'):
            raise ArgumentError(
                'expected a BatchEnvironment, found a batch whose states the recorder cannot read; '
                'record one of the batches of a MultiDeviceBatch instead'
            )
        if not 0 <= environment < batch.environment_count:
            raise ArgumentError(
                f'expected an environment from 0 to {batch.environment_count - 1}, '
                f'found {environment}'
            )
        self._batch = batch
        self.environment = environment
        self._title = title
        self._teams = teams
        self._scenario = scenario or _live.scenario_name(batch)
        players = int(batch.left_players[environment] + batch.right_players[environment])
        self._reader = _recording.StateReader(players)
        self._lineup: list[dict[str, Any]] | None = None
        self._frames: list[bytes] = []
        self._timing = None
        self.saved: list[Path] = []

    def capture(self, done: Any = None) -> Path | None:
        """Adds the environment's current step, and saves the recording when its episode ended.

        With automatic resets, the step that ends an episode already shows the next one's start,
        so the recording ends at the step before, marked as the end, and the new start opens the
        next recording.

        Args:
            done: The done flags that `step` returned, or `None` after `reset`.

        Returns:
            The path of the recording saved by this call, or `None`.
        """
        ended = done is not None and bool(done[self.environment])
        saved = None
        if ended and self._batch.auto_reset:
            if self._frames:
                last = bytearray(self._frames[-1])
                (flags,) = struct.unpack_from('<i', last, 28)
                struct.pack_into('<i', last, 28, flags | 1)
                # GRF: the scoring signal is 1 when the left team scores and -1 for the right.
                signal = self._batch._score_reward
                scored = int(signal[self.environment]) if signal is not None else 0
                if scored:
                    offset = 8 if scored > 0 else 12
                    (goals,) = struct.unpack_from('<i', last, offset)
                    struct.pack_into('<i', last, offset, goals + 1)
                    struct.pack_into('<i', last, 28, (flags | 1) & ~(31 << 8))
                self._frames[-1] = bytes(last)
            saved = self.save()
        state = self._batch._state(self.environment)
        self._timing = _recording.scenario_timing(state)
        if self._lineup is None:
            self._lineup = self._reader.lineup(state)
        closing = ended and not self._batch.auto_reset
        self._frames.append(self._reader.frame(state, 1 if closing else 0))
        if closing:
            saved = self.save()
        return saved

    def save(self) -> Path | None:
        """Writes the frames captured so far as one recording and starts a new one.

        Returns:
            The path of the recording, or `None` when nothing was captured.
        """
        if not self._frames or self._lineup is None:
            return None
        title = self._title or f'{self._scenario}, environment {self.environment}'
        head = _recording.header(
            title, self._scenario, self._lineup, self._teams, len(self._frames), self._timing
        )
        path = _library.save(head, self._frames)
        self._frames, self._lineup = [], None
        self.saved.append(path)
        return path


def play(
    left: str,
    right: str,
    scenario: str = '11_vs_11_kaggle',
    seed: int | None = None,
    title: str | None = None,
    device: str = 'cpu',
    ability: tuple[float | None, float | None] = (None, None),
) -> Path:
    """Plays a match between two teams of the registry, saves it and returns the recording.

    Args:
        left: The team of the left side, a registry identifier or `builtin`.
        right: The team of the right side.
        scenario: A shipped scenario, without agent counts, which the teams decide.
        seed: The seed of the match, or `None` for one drawn from the clock.
        title: The title of the recording.
        device: `cpu` or `cuda`.
        ability: The ability of the built-in AI on the left and on the right, each from 0 to 1, or
            `None` for the scenario's, GRF's team difficulty for every player the built-in AI
            controls.

    Raises:
        ArgumentError: If a team is unknown, plays another format than the scenario's, or an
            ability lies outside 0 to 1.
    """
    from ._play import play as run

    return run(left, right, scenario, seed, title, device, ability)


def watch(
    batch: Any,
    environment: int = 0,
    title: str | None = None,
    teams: list[dict[str, Any]] | None = None,
    backlog: bool = False,
) -> LiveSource:
    """Streams one environment of a running batch to the renderer's page.

    The batch publishes after each step while a viewer watches, as `CUJU_RENDER` makes every batch
    do. Close the source, or the batch, to withdraw it.

    Args:
        batch: The batch.
        environment: The environment to show.
        title: The title that the page lists.
        teams: The names and colors of the left and right team, as for `Recorder`.
        backlog: Whether to read every step, watched or not, so that any viewer starts with ten
            seconds of history, at the cost of one state read per step.

    Returns:
        The source, which the batch holds as well.
    """
    if not isinstance(getattr(batch, '_live', False), (LiveSource, type(None))):
        raise ArgumentError(
            'expected a BatchEnvironment, found a batch whose steps show no live source; watch '
            'one of the batches of a MultiDeviceBatch instead'
        )
    if batch._live is not None:
        batch._live.close()
    batch._live = LiveSource(batch, environment, title, teams, backlog=backlog)
    return batch._live


# The modules keep `cuju.render` as the module of the public names.
for _name in (Recorder, watch, home, play, serve, LiveSource, add_team):
    _name.__module__ = __name__
del _name

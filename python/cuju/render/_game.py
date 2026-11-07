# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Plays a game in which a person controls one player of a side, against GRF's built-in AI.

The server runs this module in a process of its own for the page's Play a game. The process steps
one environment at real time, ten steps a second, and streams every frame to the page through a
live source that stays off the list of sources. Commands come from the server as one JSON line at
a time on standard input, the input of each person's side, pause and quit, and reports go back as
one JSON line at a time on a channel of their own, as for a requested match.

Each person's side is a scenario's agent, as `{scenario}_l1_r0` names it, so the person controls
one player and the engine hands that control to the designated player as play moves, the player
nearest the ball among them, as GRF does for its own human players. The keys and buttons that
the page reads become GRF's 19 default actions through GRF's own controller logic, so a person's
actions are actions like any agent's and the match stays the match the engine would play for them.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO

import torch

from .. import BatchEnvironment, bundled_scenarios
from ..errors import ArgumentError
from . import _library, _live, _recording, _teams

# The names of the inputs that the page sends, four directions on the screen and the buttons.
DIRECTIONS = ('left', 'right', 'up', 'down')
BUTTONS = ('long_pass', 'high_pass', 'short_pass', 'shot', 'sliding', 'sprint', 'dribble')
INPUTS = DIRECTIONS + BUTTONS
# GRF: the indices of the default action set.
_IDLE, _LEFT, _TOP_LEFT, _TOP, _TOP_RIGHT, _RIGHT, _BOTTOM_RIGHT, _BOTTOM, _BOTTOM_LEFT = range(9)
_PRESS = {
    'long_pass': 9,
    'high_pass': 10,
    'short_pass': 11,
    'shot': 12,
    'sprint': 13,
    'sliding': 16,
    'dribble': 17,
}
_RELEASE_DIRECTION = 14
# GRF: releasing sprint and dribble are actions of the default set. The engine releases the other
# buttons itself after every step in play, so their releases change nothing and take the place of
# GRF's release actions as idle.
_RELEASE = {
    'long_pass': _IDLE,
    'high_pass': _IDLE,
    'short_pass': _IDLE,
    'shot': _IDLE,
    'sliding': _IDLE,
    'sprint': 15,
    'dribble': 18,
}
# GRF: `get_env_action` checks the buttons in this order, which decides the one action of a step.
_ORDER = ('long_pass', 'high_pass', 'short_pass', 'shot', 'sliding', 'sprint', 'dribble')
STEPS_PER_SECOND = 10
# Without any message from the server for this long the game pauses, and after this long it ends.
PAUSE_AFTER_SECONDS = 8.0
END_AFTER_SECONDS = 120.0
_SCORE = struct.Struct('<2i')

# The channel of the reports, standard output until `main` moves everything else away.
_reports: TextIO = sys.stdout


def _report(**entries: Any) -> None:
    """Writes one line of progress for the server."""
    _reports.write(json.dumps(entries) + '\n')
    _reports.flush()


class Controller:
    """GRF's controller of a person's player, `controller_base.Controller`, on the 19 actions.

    Each step yields one action. A change of direction comes first, the new direction or a release
    of the direction, and otherwise the first button in GRF's order whose state changed, its press
    or its release. Sprint and dribble stay pressed in the engine until released, while the passes,
    the shot and the slide act on the step they are pressed.
    """

    def __init__(self) -> None:
        self._direction = _IDLE
        self._buttons: dict[str, bool] = {}

    def act(self, held: frozenset[str], pressed: frozenset[str]) -> int:
        """Returns the action of one step.

        Args:
            held: The inputs held at the step, directions and buttons, in the side's own frame.
            pressed: The buttons pressed since the last step, even if already let go, as GRF's
                queue of key presses holds them.
        """
        left, right = 'left' in held, 'right' in held
        top, bottom = 'up' in held, 'down' in held
        direction = _IDLE
        for action, on in (
            (_TOP_LEFT, top and left),
            (_TOP_RIGHT, top and right),
            (_BOTTOM_LEFT, bottom and left),
            (_BOTTOM_RIGHT, bottom and right),
            (_RIGHT, right),
            (_LEFT, left),
            (_TOP, top),
            (_BOTTOM, bottom),
        ):
            if on:
                direction = action
                break
        if direction != self._direction:
            self._direction = direction
            return _RELEASE_DIRECTION if direction == _IDLE else direction
        active = held | pressed
        for name in _ORDER:
            state = name in active
            if self._buttons.get(name, False) != state:
                self._buttons[name] = state
                return _PRESS[name] if state else _RELEASE[name]
        return _IDLE


def own_frame(held: frozenset[str], side: int) -> frozenset[str]:
    """Returns the inputs of a side in its own frame from the directions on the screen.

    The page's broadcast camera looks across the pitch from the main stand, so right on the screen
    points along the pitch's x axis and up points away from the camera. GRF's right team plays in a
    frame turned half a circle, so its directions swap left with right and up with down.
    """
    if side == 0:
        return held
    swap = {'left': 'right', 'right': 'left', 'up': 'down', 'down': 'up'}
    return frozenset(swap.get(name, name) for name in held)


class Game:
    """One environment in which people control one player of a side each, stepped on request.

    Attributes:
        humans: Whether a person plays the left side and the right side.
        actions: The actions of every step so far, one list per step, the left agent first.
        frames: The frames of every step so far, the reset first.
        lineup: The header entry of each player.
        teams: The left and right team, as a recording's header holds them.
    """

    def __init__(
        self,
        scenario: str,
        humans: tuple[bool, bool],
        ability: tuple[float | None, float | None] = (None, None),
        seed: int | None = None,
        device: str = 'cpu',
    ) -> None:
        """Creates and resets the environment.

        Args:
            scenario: A shipped scenario, without agent counts.
            humans: Whether a person plays the left side and the right side, at least one.
            ability: The ability of the built-in AI on the left and on the right, each from 0 to
                1, or `None` for the scenario's.
            seed: The seed of the game, or `None` for one drawn from the clock.
            device: `cpu` or `cuda`.

        Raises:
            ArgumentError: If the scenario is not shipped, no side has a person or an ability lies
                outside 0 to 1.
        """
        ability = _teams.check_ability(ability)
        if scenario not in [*bundled_scenarios(), *_teams.renderer_scenarios()]:
            raise ArgumentError(f'expected a shipped scenario, found {scenario!r}')
        if not any(humans):
            raise ArgumentError('expected a person on at least one side')
        self.humans = (bool(humans[0]), bool(humans[1]))
        self.scenario = scenario
        self.seed = int(time.time()) % 2**31 if seed is None else int(seed)
        counts = [1 if human else 0 for human in self.humans]
        self.batch = BatchEnvironment(
            (
                str(_teams.scenario_path(scenario))
                if scenario in _teams.renderer_scenarios()
                else f'{scenario}_l{counts[0]}_r{counts[1]}'
            ),
            agent_counts=[tuple(counts)],
            environment_count=1,
            representation='raw',
            seed=self.seed,
            auto_reset=False,
            device=device,
            difficulty=None if ability == (None, None) else [ability],
        )
        self.sides = [side for side in (0, 1) if self.humans[side]]
        self.controllers = {side: Controller() for side in self.sides}
        self.reader = _recording.StateReader(
            int(self.batch.left_players[0] + self.batch.right_players[0])
        )
        self.batch.reset()
        state = self.batch.get_state(0)
        self.lineup = self.reader.lineup(state)
        self.frames = [self.reader.frame(state)]
        self.actions: list[list[int]] = []
        self.done = False
        played = [round(float(value), 4) for value in self.batch.difficulty[0].tolist()]
        two = all(self.humans)
        self.teams = []
        for side, team in enumerate(_teams.match_teams(_teams.BUILTIN, _teams.BUILTIN)):
            name = (f'Player {side + 1}' if two else 'You') if self.humans[side] else team['name']
            self.teams.append(
                {**team, 'name': name, 'ability': played[side], 'person': self.humans[side]}
            )

    @property
    def title(self) -> str:
        """Returns the title of the game, its teams by name."""
        return f'{self.teams[0]["name"]} against {self.teams[1]["name"]}'

    def step(self, inputs: dict[int, tuple[frozenset[str], frozenset[str]]]) -> bytes:
        """Plays one step with the held and pressed inputs of each person's side, on the screen.

        Returns:
            The frame of the step.
        """
        row = []
        for side in self.sides:
            held, pressed = inputs.get(side, (frozenset(), frozenset()))
            row.append(self.controllers[side].act(own_frame(held, side), pressed))
        self.actions.append(row)
        _, _, finished, _ = self.batch.step(torch.tensor([row], dtype=torch.int32))
        self.done = bool(finished[0])
        frame = self.reader.frame(self.batch.get_state(0), 1 if self.done else 0)
        self.frames.append(frame)
        return frame

    def save(self) -> Path:
        """Saves the game so far as a recording in the library and returns its path."""
        head = _recording.header(
            self.title,
            self.scenario,
            self.lineup,
            self.teams,
            len(self.frames),
            _recording.scenario_timing(self.batch.get_state(0)),
        )
        return _library.save(head, self.frames)

    def close(self) -> None:
        """Frees the environment."""
        self.batch.close()


class _Commands:
    """Reads the server's commands from standard input on a thread of its own."""

    def __init__(self, stream: TextIO) -> None:
        self.queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        threading.Thread(target=self._read, args=(stream,), daemon=True).start()

    def _read(self, stream: TextIO) -> None:
        for line in stream:
            try:
                command = json.loads(line)
            except ValueError:
                continue
            if isinstance(command, dict):
                self.queue.put(command)
        # The server closed the channel, so the game ends.
        self.queue.put(None)


def run(game: Game, commands: _Commands, source: _live.LiveSource, pace: float) -> str:
    """Steps the game at `pace` steps a second while it plays, and returns why it ended.

    The game waits for its first resume, so the page sees the kick-off. Each step takes the inputs
    that arrived last, together with every button pressed since the step before.
    """
    held: dict[int, frozenset[str]] = {side: frozenset() for side in game.sides}
    pressed: dict[int, set[str]] = {side: set() for side in game.sides}
    paused, reason = True, 'ready'
    contact = time.monotonic()
    due = contact
    score = None
    _report(state='ready', step=0, score=[0, 0])
    while not game.done:
        timeout = max(0.0, due - time.monotonic()) if not paused else 0.25
        try:
            command = commands.queue.get(timeout=timeout)
        except queue.Empty:
            command = {}
        now = time.monotonic()
        if command is None or command.get('quit'):
            return 'quit'
        if command:
            contact = now
            # A pause for silence lifts by itself once the page is heard again, and a pause the
            # person chose waits for the person.
            if paused and reason == 'input' and 'paused' not in command:
                paused, reason = False, 'page'
                due = now
                _report(state='playing', reason='input', step=len(game.actions))
        side = command.get('side')
        if isinstance(side, int) and side in held:
            held[side] = frozenset(command.get('held', ())) & frozenset(INPUTS)
            pressed[side].update(frozenset(command.get('pressed', ())) & frozenset(BUTTONS))
        if 'paused' in command and bool(command['paused']) != paused:
            paused, reason = bool(command['paused']), 'page'
            due = now
            _report(state='paused' if paused else 'playing', reason=reason, step=len(game.actions))
        if now - contact > END_AFTER_SECONDS:
            return 'timeout'
        if not paused and now - contact > PAUSE_AFTER_SECONDS:
            paused, reason = True, 'input'
            _report(state='paused', reason=reason, step=len(game.actions))
        if paused or time.monotonic() < due:
            continue
        inputs = {side: (held[side], frozenset(pressed[side])) for side in game.sides}
        for side in game.sides:
            pressed[side].clear()
        frame = game.step(inputs)
        source.push(frame, game.lineup)
        # A late step starts the clock again, so a slow moment never brings a burst of steps.
        due = max(due + 1 / pace, time.monotonic() - 2 / pace)
        now_score = list(_SCORE.unpack_from(frame, 8))
        if now_score != score or len(game.actions) % 10 == 0:
            score = now_score
            _report(state='playing', step=len(game.actions), score=score)
    # A full match ends at full time, and any other scenario at the end of its episode.
    timing = _recording.scenario_timing(game.batch.get_state(0))
    return 'full time' if timing['full_match'] else 'episode'


def main(arguments: list[str] | None = None) -> None:
    """Plays the game that the command line names, as the server asks."""
    global _reports
    parser = argparse.ArgumentParser(prog='python -m cuju.render._game')
    parser.add_argument('--scenario', default='11_vs_11_kaggle')
    parser.add_argument('--left', choices=('person', 'builtin'), default='person')
    parser.add_argument('--right', choices=('person', 'builtin'), default='builtin')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--left-ability', type=float, default=None)
    parser.add_argument('--right-ability', type=float, default=None)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--pace', type=float, default=STEPS_PER_SECOND)
    options = parser.parse_args(arguments)
    # The reports keep a copy of standard output, and everything else written there goes to
    # standard error, so nothing printed by a library can pass for a report.
    sys.stdout.flush()
    _reports = os.fdopen(os.dup(1), 'w', encoding='utf-8')
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    commands = _Commands(sys.stdin)
    try:
        game = Game(
            options.scenario,
            (options.left == 'person', options.right == 'person'),
            (options.left_ability, options.right_ability),
            options.seed,
        )
    except Exception as error:
        _report(error=f'{type(error).__name__}: {error}')
        sys.exit(1)
    source = _live.LiveSource(game.batch, 0, game.title, game.teams, game.scenario, announce=False)
    try:
        source.push(game.frames[0], game.lineup)
        _report(port=source.port, seed=game.seed, teams=[team['name'] for team in game.teams])
        ending = run(game, commands, source, max(0.1, options.pace))
        saved = None
        if options.record and len(game.frames) > 1:
            saved = game.save().stem
        _report(state='ended', ending=ending, step=len(game.actions), match=saved)
        # The page reads the last frames before the stream closes.
        time.sleep(0.5)
    except Exception as error:
        _report(error=f'{type(error).__name__}: {error}')
        sys.exit(1)
    finally:
        source.close()
        game.close()


if __name__ == '__main__':
    main()

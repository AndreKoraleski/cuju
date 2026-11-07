# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Plays one match between two teams of the registry and saves it.

The server runs this module in a process of its own for each requested match, so a submission that
fails ends only that process, and the server ends one that runs past its time. Progress goes to the
server as one JSON line at a time on a channel of its own, while whatever a submission prints goes
to standard error, so no submission can forge a report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, TextIO

import torch

from .. import BatchEnvironment, bundled_scenarios
from ..errors import ArgumentError
from ..kaggle import KaggleAgent, _accepts_configuration, _action, _actions
from . import _library, _recording, _teams

# The channel of the reports, standard output until `main` moves the submissions' prints away.
_reports: TextIO = sys.stdout
# GRF: a full match of the shipped scenarios lasts 3000 steps, so it takes 3001 frames.
_FRAMES = 3001
# The v2 action that hands a player to the builtin AI for one step.
_BUILTIN_AI = 19


def _report(**entries: Any) -> None:
    """Writes one line of progress for the server."""
    _reports.write(json.dumps(entries) + '\n')
    _reports.flush()


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
            `None` for the scenario's. It is GRF's team difficulty, which steers every player that
            the built-in AI controls, the whole team of the built-in AI and the teammates of a
            submission's player alike.

    Returns:
        The path of the recording in the renderer's folder.

    Raises:
        ArgumentError: If a team is unknown, plays another format than the scenario's, or an
            ability lies outside 0 to 1.
    """
    ability = _teams.check_ability(ability)
    if scenario not in [*bundled_scenarios(), *_teams.renderer_scenarios()]:
        raise ArgumentError(
            f'expected a scenario from the renderer scenario list, found {scenario!r}'
        )
    sides = [_teams.entry(left), _teams.entry(right)]
    _teams.check_scenario(sides, scenario)
    agents = [_teams.agent_count(team, scenario, side) for side, team in enumerate(sides)]
    controls = [_teams.check_control(team.get('control'), team['format']) for team in sides]
    extended = any(
        team['kind'] != 'builtin' and (c['assignment'] == 'fixed' or c['action_set'] == 'v2')
        for team, c in zip(sides, controls, strict=True)
    )
    seed = int(time.time()) % 2**31 if seed is None else int(seed)
    batch = BatchEnvironment(
        (
            str(_teams.scenario_path(scenario))
            if scenario in _teams.renderer_scenarios()
            else f'{scenario}_l{agents[0]}_r{agents[1]}'
        ),
        agent_counts=[tuple(agents)],
        environment_count=1,
        representation='raw',
        seed=seed,
        auto_reset=False,
        device=device,
        difficulty=None if ability == (None, None) else [ability],
        # GRF uses builtin AI for each player outside a fixed selection.
        action_set='v2' if extended else 'default',
    )
    try:
        reader = _recording.StateReader(int(batch.left_players[0] + batch.right_players[0]))
        batch.reset()
        players = []
        for side, team in enumerate(sides):
            if team['kind'] == 'builtin':
                continue
            first = 0 if side == 0 else agents[0]
            columns = list(range(first, first + agents[side]))
            control = controls[side]
            selected = None
            if control['assignment'] == 'fixed':
                selected = _teams.selected_players(control, _teams.scenario_players(scenario, side))
                columns = columns[: len(selected)]
            function = _teams.load_agent(team['id'])
            if extended and control['action_set'] == 'default':
                function = _default_actions(function)

            if len(columns) == 1:
                players.append((side, selected, KaggleAgent(function, batch, columns[0]), {}))
            else:
                players.append(
                    (side, selected, KaggleAgent(function, batch, first, agents=columns), {})
                )
        state = batch.get_state(0)
        lineup = reader.lineup(state)
        timing = _recording.scenario_timing(state)
        frames = [reader.frame(state)]
        done = False
        while not done:
            actions = torch.zeros((1, agents[0] + agents[1]), dtype=torch.int32)
            raw = {name: tensor.cpu() for name, tensor in batch.raw().items()}
            for side, selected, player, held in players:
                first = 0 if side == 0 else agents[0]
                columns = list(range(first, first + agents[side]))
                missing: list[int] = []
                if selected is not None:
                    actions[:, columns] = _BUILTIN_AI
                    found = _player_columns(raw, columns, selected, held)
                    if found is None:
                        # GRF is handing the players to the agents again, so the built-in AI
                        # plays this one step and the submission sees its players once more on
                        # the next.
                        continue
                    columns, missing = found
                    player.agents = columns
                    player.agent = columns[0]
                actions[:, columns] = player.act().reshape(1, len(columns))
                # A selected player off the pitch, after a red card for one, is left to the
                # built-in AI, and the submission keeps playing the others.
                actions[:, missing] = _BUILTIN_AI
            _, _, finished, _ = batch.step(actions)
            done = bool(finished[0])
            frames.append(reader.frame(batch.get_state(0), 1 if done else 0))
            if len(frames) % 50 == 0:
                _report(progress=min(len(frames) / _FRAMES, 0.99))
        difficulty = batch.difficulty[0].tolist()
    finally:
        batch.close()
    names = [team['name'] for team in sides]
    title = title or f'{names[0]} against {names[1]}'
    # The ability that played, the scenario's where none was given, goes with each team.
    played = [round(float(value), 4) for value in difficulty]
    teams = [
        {**team, 'ability': played[side]}
        for side, team in enumerate(_teams.match_teams(left, right))
    ]
    head = _recording.header(title, scenario, lineup, teams, len(frames), timing)
    return _library.save(head, frames)


def _player_columns(
    raw: dict[str, torch.Tensor], columns: list[int], selected: list[int], held: dict[int, int]
) -> tuple[list[int], list[int]] | None:
    """Finds the agents holding the selected players in submission order, and those left over.

    GRF clears the selection of a side at a set piece, and for that step an agent reports no player,
    a step that returns `None`. A selected player that no agent reports for another reason, such as
    a red card, keeps the agent it held last, or else an agent that holds no selected player, and
    that agent is the second list. `held` keeps the agent each selected player held last.
    """
    active = [int(raw['active'][0, column]) for column in columns]
    if -1 in active:
        return None
    by_player = {}
    for column, index in zip(columns, active, strict=True):
        if index in selected and index not in by_player:
            by_player[index] = column
    held.update(by_player)
    free = [column for column in columns if column not in by_player.values()]
    missing = []
    for index in selected:
        if index in by_player:
            continue
        column = held.get(index)
        if column not in free:
            column = free[0]
        free.remove(column)
        by_player[index] = column
        missing.append(column)
    return [by_player[index] for index in selected], missing


def _default_actions(function: Any) -> Any:
    """Checks the submission set before translating negative indices to the batch set."""
    two = _accepts_configuration(function)

    def run(obs: Any, configuration: Any) -> list[int]:
        answer = function(obs, configuration) if two else function(obs)
        count = len(obs['players_raw'])
        values = [_action(answer, 19)] if count == 1 else _actions(answer, 19, count)
        return [value % 19 for value in values]

    return run


def main(arguments: list[str] | None = None) -> None:
    """Plays the match that the command line names and reports the recording."""
    global _reports
    parser = argparse.ArgumentParser(prog='python -m cuju.render._play')
    parser.add_argument('--left', required=True)
    parser.add_argument('--right', required=True)
    parser.add_argument('--scenario', default='11_vs_11_kaggle')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--left-ability', type=float, default=None)
    parser.add_argument('--right-ability', type=float, default=None)
    options = parser.parse_args(arguments)
    # The reports keep a copy of standard output, and everything else written there, by Python or
    # by native code, goes to standard error.
    sys.stdout.flush()
    _reports = os.fdopen(os.dup(1), 'w', encoding='utf-8')
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    try:
        path = play(
            options.left,
            options.right,
            options.scenario,
            options.seed,
            None,
            options.device,
            (options.left_ability, options.right_ability),
        )
    except Exception as error:
        _report(error=f'{type(error).__name__}: {error}')
        sys.exit(1)
    _report(progress=1.0, match=path.stem)


if __name__ == '__main__':
    main()

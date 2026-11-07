# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""The command line of the renderer, `cuju render`, also `python -m cuju.render`.

`cuju render` serves the page, `cuju render team add FOLDER` registers a Kaggle submission folder,
`cuju render play` records a match between two registered teams, and `cuju render sample` adds the
shipped sample matches to the library again.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ..errors import Error
from . import _home


def main(arguments: list[str] | None = None) -> int:
    """Runs the command that the arguments name and returns its exit status."""
    parser = argparse.ArgumentParser(
        prog='cuju render', description="Watch cuju's matches in a browser."
    )
    parser.add_argument(
        '--home', help=f'the renderer folder, by default {_home.home()} or ${_home.HOME_VARIABLE}'
    )
    commands = parser.add_subparsers(dest='command')
    server = commands.add_parser('serve', help='serve the page, the default command')
    server.add_argument('--port', type=int, default=8780, help='the port, 8780 by default')
    server.add_argument('--open', action='store_true', help='open the page in the browser')
    team = commands.add_parser('team', help='add, list, export or remove teams of the registry')
    team.add_argument('action', choices=('add', 'list', 'export', 'remove'))
    team.add_argument(
        'target', nargs='?', help='a submission folder or archive to add, or a team identifier'
    )
    team.add_argument('--name', help='the name of an added team, by default the folder name')
    team.add_argument(
        '--format',
        choices=('5v5', '11v11'),
        default='11v11',
        help='the size of an added team, 11v11 by default, unless its folder holds an export',
    )
    team.add_argument('--out', help='the zip file of an export, by default the team name')
    match = commands.add_parser('play', help='record a match between two registered teams')
    match.add_argument('--left', default='builtin', help='a team identifier, or builtin')
    match.add_argument('--right', default='builtin', help='a team identifier, or builtin')
    match.add_argument(
        '--scenario',
        default='11_vs_11_kaggle',
        help='a shipped scenario, 11_vs_11_kaggle by default',
    )
    match.add_argument('--seed', type=int, default=None, help='the seed, drawn when absent')
    match.add_argument('--device', default='cpu', help='cpu or cuda, cpu by default')
    for side in ('left', 'right'):
        match.add_argument(
            f'--{side}-ability',
            type=float,
            default=None,
            help=f"the {side} built-in AI's ability from 0 to 1, the scenario's when absent",
        )
    sample = commands.add_parser(
        'sample', help='add the sample matches to the library again, after a deletion'
    )
    sample.add_argument(
        'name', nargs='?', choices=('11v11', '5v5'), help='one sample, by default both'
    )
    options = parser.parse_args(arguments)
    if options.home:
        os.environ[_home.HOME_VARIABLE] = str(Path(options.home).expanduser().resolve())
    try:
        return _run(parser, options)
    except (Error, OSError) as error:
        print(f'cuju render: {error}', file=sys.stderr)
        return 1


def _run(parser: argparse.ArgumentParser, options: argparse.Namespace) -> int:
    """Runs one command of the parsed command line."""
    if options.command == 'team':
        from . import _teams

        if options.action in ('export', 'remove') and not options.target:
            parser.error(f'team {options.action} needs the identifier of a team')
        if options.action == 'add' and not options.target:
            parser.error('team add needs a submission folder or archive')
        if options.action == 'add':
            entry = _teams.add_folder(options.target, options.name, options.format)
            print(f'added {entry["id"]} {entry["name"]}')
        elif options.action == 'export':
            filename, data = _teams.export(options.target)
            path = options.out or filename
            with open(path, 'wb') as stream:
                stream.write(data)
            print(f'exported {path}')
        elif options.action == 'remove':
            if not _teams.remove(options.target):
                parser.error(f'no team {options.target!r} to remove')
        else:
            for entry in _teams.teams():
                print(f'{entry["id"]}  {entry["format"]}  {entry["name"]}')
        return 0
    if options.command == 'sample':
        from . import _library

        names = None if options.name is None else [f'sample-{options.name}']
        added = _library.add_samples(names)
        for path in added:
            print(f'added {path.stem}')
        if not added:
            print('the library already holds the samples')
        return 0
    if options.command == 'play':
        from ._play import play

        path = play(
            options.left,
            options.right,
            options.scenario,
            options.seed,
            None,
            options.device,
            (options.left_ability, options.right_ability),
        )
        print(f'saved {path}')
        return 0
    from ._server import serve

    port = getattr(options, 'port', 8780)
    if not 0 <= port <= 65535:
        parser.error('expected a port from 0 to 65535')
    serve(port, getattr(options, 'open', False))
    return 0


if __name__ == '__main__':
    sys.exit(main())

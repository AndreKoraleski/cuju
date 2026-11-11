# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""The command `cuju`, also `python -m cuju`, whose subcommands run cuju's tools.

`cuju render` runs the renderer, `cuju check` the shipped self-check, and `cuju replay` and
`cuju dump-to-txt` GRF's scripts for dumps, as their modules do with `python -m`. `cuju help` lists
the commands, and `cuju help NAME` or `cuju NAME --help` describes one. A new command is one entry
of `COMMANDS`, a module whose `main(arguments)` parses the rest of the command line and returns an
exit status or `None`.
"""

from __future__ import annotations

import importlib
import sys
from typing import NamedTuple

from . import _renderer
from ._version import __version__


class Command(NamedTuple):
    """One subcommand of `cuju`.

    Attributes:
        module: The module whose `main(arguments)` runs the command, relative to `cuju`.
        summary: The line that `cuju help` shows.
    """

    module: str
    summary: str


COMMANDS = {
    'render': Command(
        '.render.__main__', 'watch matches in a browser, saved or live, and keep a team registry'
    ),
    'check': Command('.check', 'check that this machine plays the matches of GRF, bit for bit'),
    'replay': Command(
        '.gfootball.replay', "replay a dump of GRF's and check that it plays the same"
    ),
    'dump-to-txt': Command('.gfootball.dump_to_txt', "write a dump of GRF's as readable text"),
}


def _usage() -> str:
    """Returns the help text that lists the commands."""
    width = max(len(name) for name in COMMANDS)
    lines = [
        f'cuju {__version__}, batched Google Research Football, bit for bit',
        '',
        'usage: cuju <command> [arguments]',
        '',
        'commands:',
    ]
    lines += [f'  {name:<{width}}  {command.summary}' for name, command in COMMANDS.items()]
    lines += [f'  {"help":<{width}}  describe the commands, or one with cuju help <command>']
    lines += ['', "Run 'cuju <command> --help' for the arguments of a command."]
    return '\n'.join(lines)


def _run(name: str, arguments: list[str]) -> int:
    """Runs the command `name` with its arguments and returns its exit status."""
    module = importlib.import_module(COMMANDS[name].module, __package__)
    status = module.main(arguments)
    return 0 if status is None else int(status)


def main(arguments: list[str] | None = None) -> int:
    """Runs the command that the arguments name and returns its exit status."""
    arguments = sys.argv[1:] if arguments is None else list(arguments)
    if not arguments or arguments[0] in ('-h', '--help'):
        print(_usage())
        return 0
    if arguments[0] in ('-V', '--version'):
        print(f'cuju {__version__}')
        return 0
    name, rest = arguments[0], arguments[1:]
    if name == 'help':
        if not rest:
            print(_usage())
            return 0
        if rest[0] in COMMANDS:
            return _guarded(rest[0], ['--help'])
        if rest[0] == 'help':
            print(_usage())
            return 0
        name = rest[0]
    if name not in COMMANDS:
        print(f'cuju: no command {name!r}\n\n{_usage()}', file=sys.stderr)
        return 2
    return _guarded(name, rest)


def _guarded(name: str, arguments: list[str]) -> int:
    """Runs a command as `_run` does, and turns a missing renderer and argparse's exit into
    an exit status."""
    try:
        return _run(name, arguments)
    except ModuleNotFoundError as error:
        # The renderer is optional, and a command of a missing one names the package to install.
        if not _renderer.is_missing(error):
            raise
        print(f'cuju {name}: {_renderer.MESSAGE}', file=sys.stderr)
        return 1
    except SystemExit as exit:
        # argparse leaves with `SystemExit`, whose code is the status.
        return exit.code if isinstance(exit.code, int) else 0 if exit.code is None else 1


if __name__ == '__main__':
    sys.exit(main())

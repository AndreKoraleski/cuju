# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""GRF's `gfootball.dump_to_txt` script, which writes a dump as text, with GRF's flags.

It runs as GRF's script does, with `cuju.gfootball` for `gfootball`.

    python -m cuju.gfootball.dump_to_txt --trace_file=/tmp/input.dump --output=/tmp/output.txt
"""

from __future__ import annotations

import argparse

from .env import script_helpers

__all__ = ['main']


def _flag(text: str) -> bool:
    """Returns the value of a boolean flag written as absl reads it."""
    lowered = text.lower()
    if lowered in ('true', 't', '1'):
        return True
    if lowered in ('false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError(f'expected true or false, found {text!r}')


def main(argv: list[str] | None = None) -> None:
    """Runs the script on `argv`, or on the command line without it."""
    parser = argparse.ArgumentParser(description='Converts a dump file to a readable text file.')
    parser.add_argument('--trace_file', required=True, help='Trace file to convert')
    parser.add_argument('--output', required=True, help='Output txt file')
    parser.add_argument(
        '--include_debug',
        nargs='?',
        const=True,
        default=True,
        type=_flag,
        help='Include debug information for each step',
    )
    parser.add_argument('--noinclude_debug', dest='include_debug', action='store_false')
    arguments = parser.parse_args(argv)
    script_helpers.ScriptHelpers().dump_to_txt(
        arguments.trace_file, arguments.output, arguments.include_debug
    )


if __name__ == '__main__':
    main()

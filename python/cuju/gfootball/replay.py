# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""GRF's `gfootball.replay` without rendering, which plays a dump again and checks it.

    python -m cuju.gfootball.replay --trace_file=/tmp/dumps/episode_done_20261026-120000000000.dump

It replays cuju's and GRF's dumps alike and compares every step with the dump's bit for bit, apart
from the wall-clock time. It prints the first difference and exits 1 if there is one, and
`check(trace_file)` does the same from Python.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from .. import _replays, dumps

__all__ = ['check', 'main']


def check(trace_file: str, device: Any = 'cpu', threads: int = 1) -> _replays.ReplayResult:
    """Plays a dump again and compares every step with the dump's.

    Args:
        trace_file: A dump that starts at the start of its episode.
        device: The device that plays the replay. Every device plays the same match.
        threads: The CPU threads of the replay.

    Returns:
        `(steps, ended, difference)`, the steps found equal, whether the replay's episode ended
        with the dump's last step, and the first difference as a text, or `None` if the replay
        equals the dump.

    Raises:
        ValueError: If the dump holds no configuration, does not start at the start of its
            episode, or holds an action outside its action set.
    """
    return _replays.replay(trace_file, device=device, threads=threads)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='python -m cuju.gfootball.replay', description='Replays a dump and checks it.'
    )
    parser.add_argument('--trace_file', required=True, help='Trace file to replay')
    parser.add_argument(
        '--fps',
        type=int,
        default=None,
        help="Frames per second of GRF's replay, where only 10 plays the match of a dump of 10 "
        'ticks a step. The default replays any dump at its own ticks.',
    )
    parser.add_argument('--device', default='cpu', help='cpu or cuda, which play the same match')
    arguments = parser.parse_args(argv)
    if arguments.fps is not None:
        steps = dumps.read_dump(arguments.trace_file)
        ticks = steps[0]['debug']['config']['physics_steps_per_frame'] if steps else None
        # GRF: other rates add idle steps, or play steps of 100 / fps ticks.
        if arguments.fps != ticks or 100 // arguments.fps != ticks:
            print(
                f"at {arguments.fps} frames per second GRF's replay plays another match than the "
                f'dump of {ticks} ticks a step',
                file=sys.stderr,
            )
            return 2
    result = check(arguments.trace_file, device=arguments.device)
    if result.difference is not None:
        print(f'The replay differs from the dump at {result.difference}')
        return 1
    ending = 'and ends the episode there' if result.ended else 'before the end of the episode'
    print(f"The replay equals the dump's {result.steps} steps, {ending}.")
    return 0


if __name__ == '__main__':
    sys.exit(main())

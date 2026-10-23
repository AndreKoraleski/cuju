# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""GRF's `gfootball.env.script_helpers`, with the `load_dump` and `dump_to_txt` of `ScriptHelpers`.

`load_dump` reads a dump without running its code and gives GRF's numpy arrays and `CoreAction`s,
so `dump_to_txt` writes GRF's text. `replay(dump, render=False)` plays a dump again and checks
every step bit for bit. `dump_to_video` renders, so cuju lacks it, and GRF's runs on cuju's dumps.
"""

from __future__ import annotations

from typing import Any

import torch

from ... import _replays, dumps

__all__ = ['ScriptHelpers']


def _grf_value(value: Any) -> Any:
    """Returns a value of `read_dump` with numpy arrays for tensors, as GRF's pickle holds it."""
    if isinstance(value, torch.Tensor):
        return value.numpy()
    if isinstance(value, dict):
        return {key: _grf_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_grf_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_grf_value(item) for item in value)
    return value


class ScriptHelpers:
    """The methods of GRF's `ScriptHelpers` that need no rendering."""

    def load_dump(self, dump_file: str) -> list[dict[str, Any]]:
        """Returns the steps of a dump as GRF's method, with numpy arrays and `CoreAction`s."""
        steps = [_grf_value(step) for step in dumps.read_dump(dump_file)]
        for step in steps:
            debug = step.get('debug')
            # GRF: a dump holds actions only in the debug information of each step.
            if isinstance(debug, dict) and 'action' in debug:
                debug['action'] = [dumps._CORE_ACTIONS[name] for name in debug['action']]
        return steps

    def dump_to_txt(self, dump_file: str, output: str, include_debug: bool) -> None:
        """Writes the steps of a dump to `output` as text, as GRF's method."""
        dump = self.load_dump(dump_file)
        if not include_debug:
            for step in dump:
                if 'debug' in step:
                    del step['debug']
        with open(output, 'w') as handle:
            handle.write(str(dump))

    def replay(
        self,
        dump: str,
        fps: int = 10,
        config_update: dict[str, Any] | None = None,
        directory: str | None = None,
        render: bool = True,
    ) -> None:
        """Plays a dump again and checks it, as GRF's replay without rendering.

        The replay plays the dump's actions in a new environment of the dump's configuration, as
        GRF's replay players do, and compares each step with the dump's, every observation, action
        and reward bit for bit, apart from the wall-clock time.

        Args:
            dump: A dump that starts at the start of its episode.
            fps: GRF's frame rate of the replay. Only 10 plays the match of a dump of 10 ticks a
                step. GRF's other rates add idle steps or change the ticks, another match.
            config_update: Values that replace those of the dump's configuration.
            directory: Where GRF's replay writes its own dumps and video. cuju writes neither, so
                it is ignored.
            render: Must be `False`, since cuju renders nothing.

        Raises:
            ValueError: If `render` is set, `fps` plays another match than the dump's, the dump
                cannot be replayed, or a step of the replay differs from the dump's, naming the
                first such step and value.
        """
        if render:
            raise ValueError('cuju renders nothing, so it replays with render=False only')
        steps = dumps.read_dump(dump)
        if not steps or 'config' not in steps[0].get('debug', {}):
            raise ValueError(f'expected a dump whose first step holds the configuration in {dump}')
        ticks = steps[0]['debug']['config']['physics_steps_per_frame']
        # GRF: `__modify_trace` adds idle steps unless the rate equals the dump's, and the replay
        # plays steps of 100 / fps ticks.
        if fps != ticks or 100 // fps != ticks:
            raise ValueError(
                f"at {fps} frames per second GRF's replay plays another match than the dump of "
                f'{ticks} ticks a step, and only 10 replays a dump of 10 ticks a step'
            )
        result = _replays.replay(dump, overrides=config_update)
        if result.difference is not None:
            raise ValueError(f'the replay differs from the dump at {result.difference}')

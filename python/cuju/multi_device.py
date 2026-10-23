# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""One batch of environments over several devices, each device stepping a contiguous range.

The guide is in the repository's `docs/backends.md`, under Several GPUs.
"""

from __future__ import annotations

from typing import Any

import torch

from . import BatchEnvironment
from .errors import (
    ActionRangeError,
    ArgumentError,
    ChoiceError,
    ClosedError,
    ConfigurationError,
    EnvironmentIndexError,
)

__all__ = ['MultiDeviceBatch']


def _device(device):
    """Returns a name, an index or a `torch.device` as a `torch.device` with an index."""
    device = torch.device('cuda', device) if isinstance(device, int) else torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    if device.type not in ('cuda', 'cpu'):
        raise ChoiceError(f'expected a CUDA device or the CPU, found {device}')
    return device


def _per_environment(values, start, end, count, error=ConfigurationError):
    """Returns the values of environments `start` to `end - 1`, or the one value for every one."""
    array = torch.as_tensor(values)
    if array.dim() == 0:
        return array.item()
    if array.shape[0] != count:
        raise error(
            f'expected one value, or one for each of the {count} environments, found the shape '
            f'{tuple(array.shape)}'
        )
    return array[start:end]


class MultiDeviceBatch:
    """A batch whose environments run on several devices, each a `BatchEnvironment` of its own.

    The part on device k plays environments `start` to `end - 1` of the seed rule, through
    `first_index`, so environment `i` plays as environment `i` of one `BatchEnvironment` of
    `environment_count` environments and the same options. A step starts on every device before it
    waits for any, so the GPUs step at once, and a CPU part steps in its own thread.

    The outputs of `reset` and `step` are tensors on `output_device`, which the next call
    overwrites, as a `BatchEnvironment`'s are. Gathering them copies each part's outputs, and
    `parts` gives each part's own outputs without the copy.

    Attributes:
        parts: The `BatchEnvironment` of each device, in the order of `devices`.
        ranges: The `(start, end)` environments of each part.
        devices: The device of each part.
        output_device: The device of the outputs.
        environment_count, agents, players, action_count, squeeze, stacked, auto_reset: As in
            `BatchEnvironment`.
        left_agents, right_agents, left_players, right_players, agent_mask, difficulty: As in
            `BatchEnvironment`, over every environment.
    """

    def __init__(
        self,
        assets: str | list[str],
        devices: list[Any],
        environment_count: int = 1024,
        split: list[int] | None = None,
        output_device: Any = None,
        assignment: Any = None,
        game_engine_random_seed: Any = None,
        **options: Any,
    ) -> None:
        """Creates a part of the batch on each device.

        Args:
            assets: As for `BatchEnvironment`.
            devices: The devices, each `cpu`, `cuda`, `cuda:N`, an index N or a `torch.device`,
                and a device may repeat.
            environment_count: The number of environments over every device.
            split: The number of environments of each device. `None` splits them evenly, the first
                devices taking one more where they do not divide.
            output_device: Where the outputs of `reset` and `step` live. `None` takes the first
                device.
            assignment, game_engine_random_seed: As for `BatchEnvironment`, over every
                environment, and each part takes its range.
            **options: The other options of `BatchEnvironment`, the same for every part, except
                `device` and `first_index`, which the parts take from `devices` and `split`.

        Raises:
            ConfigurationError: If `devices` is empty, `split` does not add up to
                `environment_count` or holds a part of no environment, `options` names `device` or
                `first_index`, or `assignment` or `game_engine_random_seed` holds neither one value
                nor one per environment. It is a `ValueError`.
            ChoiceError: If a device is neither a CUDA device nor the CPU. It is a `ValueError`.
        """
        self.devices = [_device(device) for device in devices]
        if not self.devices:
            raise ConfigurationError('expected at least one device')
        for name in ('device', 'first_index'):
            if name in options:
                raise ConfigurationError(
                    f'expected no {name}, which MultiDeviceBatch sets for each part'
                )
        count = len(self.devices)
        if split is None:
            split = [
                environment_count // count + (part < environment_count % count)
                for part in range(count)
            ]
        if len(split) != count or sum(split) != environment_count or min(split) < 1:
            raise ConfigurationError(
                f'expected a split of the {environment_count} environments into {count} parts of '
                f'at least one, found {split}'
            )
        self.ranges = []
        start = 0
        for size in split:
            self.ranges.append((start, start + size))
            start += size
        self.output_device = self.devices[0] if output_device is None else _device(output_device)
        self.environment_count = environment_count
        self.parts = []
        try:
            for device, (start, end) in zip(self.devices, self.ranges, strict=True):
                part_options = dict(options)
                if assignment is not None:
                    part_options['assignment'] = _per_environment(
                        assignment, start, end, environment_count
                    )
                if game_engine_random_seed is not None:
                    part_options['game_engine_random_seed'] = _per_environment(
                        game_engine_random_seed, start, end, environment_count
                    )
                # Environment 0 of a part keeps the seed rule's index of its range.
                if start:
                    part_options['first_index'] = start
                if device.type == 'cuda':
                    with torch.cuda.device(device):
                        part = BatchEnvironment(
                            assets, environment_count=end - start, device='cuda', **part_options
                        )
                else:
                    part = BatchEnvironment(
                        assets, environment_count=end - start, device='cpu', **part_options
                    )
                self.parts.append(part)
        except BaseException:
            self.close()
            raise
        first = self.parts[0]
        for name in ('agents', 'players', 'action_count', 'squeeze', 'stacked', 'auto_reset'):
            setattr(self, name, getattr(first, name))
        for name in (
            'left_agents',
            'right_agents',
            'left_players',
            'right_players',
            'agent_mask',
            'difficulty',
        ):
            setattr(self, name, torch.cat([getattr(part, name) for part in self.parts]))
        self._gathered = {}
        # The event after the last gather from the page-locked outputs of a CPU part, which a copy
        # to the GPU reads only when its stream gets there, and the event itself, made once.
        self._copied = self._copy_event = None

    def close(self) -> None:
        """Closes every part. A second call does nothing.

        Every later call of another method raises `ClosedError`, a `RuntimeError`, as on a closed
        `BatchEnvironment`.
        """
        for part in getattr(self, 'parts', []):
            part.close()
        self.parts = []
        self._copied = None

    def _open_parts(self):
        """Returns the parts, or raises `ClosedError` once the batch is closed."""
        if not getattr(self, 'parts', None):
            raise ClosedError('the batch is closed')
        return self.parts

    def _before_write(self, part):
        """Waits, before a call rewrites the outputs of `part`, for the copies that read them.

        A step, a reset or a state of a CPU part with page-locked outputs rewrites them, which a
        copy of the last gather that the GPU has not yet run would then read. Other parts need no
        wait, since a GPU part's next call runs after the copies on the same stream, and the copy
        of pageable memory reads it before it returns.
        """
        if self._copied is not None and part.device == 'cpu' and part._native.pinned:
            self._copied.synchronize()
            self._copied = None

    def __reduce__(self):
        raise TypeError(
            'a batch cannot be pickled; create it inside the worker process that uses it'
        )

    def _gather(self, key, outputs):
        """Gathers the parts' outputs on `output_device` into tensors that `key` reuses."""
        if isinstance(outputs[0], dict):
            return {
                name: self._gather((key, name), [output[name] for output in outputs])
                for name in outputs[0]
            }
        gathered = self._gathered.get(key)
        if gathered is None:
            shape = (self.environment_count, *outputs[0].shape[1:])
            gathered = torch.empty(shape, dtype=outputs[0].dtype, device=self.output_device)
            self._gathered[key] = gathered
        # A copy into the host's pageable memory must wait, else it may land after the return.
        non_blocking = self.output_device.type == 'cuda'
        pinned = False
        for part, (start, end), output in zip(self.parts, self.ranges, outputs, strict=True):
            gathered[start:end].copy_(output, non_blocking=non_blocking)
            pinned |= part.device == 'cpu' and bool(part._native.pinned)
        if non_blocking and pinned:
            if self._copy_event is None:
                self._copy_event = torch.cuda.Event()
            self._copy_event.record(torch.cuda.current_stream(self.output_device))
            self._copied = self._copy_event
        return gathered

    def _split(self, values):
        """Returns `values` per part, a slice of one value per environment or the one value."""
        if values is None:
            return [None] * len(self.parts)
        return [
            _per_environment(values, start, end, self.environment_count, ArgumentError)
            for start, end in self.ranges
        ]

    def _step_outputs(self, results):
        observation, reward, done, information = zip(*results, strict=True)
        return (
            self._gather('observation', observation),
            self._gather('reward', reward),
            self._gather('done', done),
            {
                name: self._gather(name, [part[name] for part in information])
                for name in information[0]
            },
        )

    def reset(self, seeds: Any = None, reverse: Any = None) -> Any:
        """Starts the next episode of every environment, as `BatchEnvironment.reset` does.

        Args:
            seeds, reverse: As for `BatchEnvironment.reset`, over every environment.

        Returns:
            The observation of every environment on `output_device`.

        Raises:
            ArgumentError: If `seeds` or `reverse` holds neither one value nor one per environment,
                or as `BatchEnvironment.reset` raises it. It is a `ValueError`.
            ClosedError: If the batch is closed. It is a `RuntimeError`.
        """
        parts = self._open_parts()
        observations = []
        for part, part_seeds, part_reverse in zip(
            parts, self._split(seeds), self._split(reverse), strict=True
        ):
            self._before_write(part)
            observations.append(part.reset(seeds=part_seeds, reverse=part_reverse))
        return self._gather('observation', observations)

    def reset_done(self) -> None:
        """Starts the next episode of every environment whose episode ended.

        Raises:
            ClosedError: If the batch is closed. It is a `RuntimeError`.
        """
        for part in self._open_parts():
            self._before_write(part)
            part.reset_done()

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        """Runs one step of every environment, as `BatchEnvironment.step` does.

        Args:
            actions: `[environment_count, agents]` action indices, or `[environment_count]` for
                one agent, on any device. Each part takes its rows on its own device.

        Returns:
            The observation, the reward, the done flags and the information of every environment,
            on `output_device`.

        Raises:
            ActionRangeError: If a part on the CPU meets an action outside the action set, before
                any part steps. It is an `IndexError`.
            ClosedError: If the batch is closed. It is a `RuntimeError`.
        """
        self._open_parts()
        actions = torch.as_tensor(actions)
        # A CUDA copy to the host could return before it lands, so only CUDA to CUDA skips the wait.
        rows = [
            actions[start:end].to(device, non_blocking=device.type == 'cuda' and actions.is_cuda)
            for device, (start, end) in zip(self.devices, self.ranges, strict=True)
        ]
        # GRF: an action outside `[-action_count, action_count)` raises. A part checks its host
        # actions too, but only once the parts before it stepped, so every part's are checked here
        # first.
        count = self.action_count
        for part, part_rows in zip(self.parts, rows, strict=True):
            # The part itself raises for a wrong shape or type.
            if (
                part.device == 'cpu'
                and not part_rows.is_floating_point()
                and part_rows.numel() == part.agent_mask.numel()
            ):
                values = part_rows.reshape(part.agent_mask.shape)[part.agent_mask]
                if bool(((values < -count) | (values >= count)).any()):
                    raise ActionRangeError(
                        f'expected action indices from {-count} to {count - 1}, found '
                        f'{values[(values < -count) | (values >= count)][0].item()}'
                    )
        # The CPU parts step in threads of their own, so every device steps at once.
        started = []
        try:
            for part, part_rows in zip(self.parts, rows, strict=True):
                self._before_write(part)
                part.step_async(part_rows)
                started.append(part)
        except BaseException:
            for part in started:
                part.step_wait()
            raise
        return self._step_outputs([part.step_wait() for part in self.parts])

    def observation(self) -> Any:
        """Returns the observation of every environment of the last `reset` or `step`.

        Raises:
            ClosedError: If the batch is closed. It is a `RuntimeError`.
        """
        return self._gather('observation', [part.observation() for part in self._open_parts()])

    def _part(self, index):
        """Returns the part of environment `index` and the index there."""
        for part, (start, end) in zip(self._open_parts(), self.ranges, strict=True):
            if start <= index < end:
                return part, index - start
        raise EnvironmentIndexError(
            f'expected an environment from 0 to {self.environment_count - 1}, found {index}'
        )

    def get_state(self, index: int) -> bytes:
        """Returns the state of environment `index`, as `BatchEnvironment.get_state` does.

        Raises:
            EnvironmentIndexError: If `index` lies outside the batch. It is an `IndexError` and a
                `RuntimeError`.
            ClosedError: If the batch is closed. It is a `RuntimeError`.
        """
        part, local = self._part(index)
        return part.get_state(local)

    def set_state(self, index: int, state: bytes) -> None:
        """Sets environment `index` to a state, as `BatchEnvironment.set_state` does.

        Raises:
            EnvironmentIndexError: If `index` lies outside the batch. It is an `IndexError` and a
                `RuntimeError`.
            ClosedError: If the batch is closed. It is a `RuntimeError`.
        """
        part, local = self._part(index)
        self._before_write(part)
        part.set_state(local, state)

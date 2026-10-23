# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Digests of a batch's outputs, and logs of a run's actions that replay it bit for bit.

`Digest` hashes outputs as the package's self-check does, and `digest` plays the self-check's
run of a scenario and returns its SHA-256, the same on every backend and machine. An `ActionLog`
creates a batch, takes its calls in place of the batch's and logs them with their actions, and
`replay` plays a saved log again, on either backend, and checks every digest.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import json
import zipfile
from typing import Any

import torch

from . import BatchEnvironment

__all__ = ['ActionLog', 'Digest', 'digest', 'replay']

# The version of the log's format.
_FORMAT = 1
# The options that change no output, which a log leaves to the machine that replays it.
_MACHINE_OPTIONS = (
    'device',
    'library',
    'threads',
    'pin_memory',
    'allocator',
    'pin_threads',
    'numa',
    'debug',
    'check_actions',
)
# The machine options that a replay takes, the device being an argument of its own.
_REPLAY_OPTIONS = _MACHINE_OPTIONS[1:]


class Digest:
    """The SHA-256 of a sequence of outputs.

    Tensors hash as the bytes of their elements in order, on any device. A dictionary hashes its
    values in the order of its sorted keys, and bytes hash as they are.
    """

    def __init__(self) -> None:
        self._hash = hashlib.sha256()

    def update(self, value: Any) -> None:
        """Adds a tensor, a dictionary or sequence of them, or bytes.

        Raises:
            TypeError: If `value` is none of these.
        """
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to('cpu').contiguous()
            self._hash.update(
                ctypes.string_at(tensor.data_ptr(), tensor.numel() * tensor.element_size())
            )
        elif isinstance(value, dict):
            for key in sorted(value):
                self.update(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.update(item)
        elif isinstance(value, (bytes, bytearray)):
            self._hash.update(value)
        else:
            raise TypeError(f'expected tensors, dictionaries, sequences or bytes, found {value!r}')

    def copy(self) -> Digest:
        """Returns a digest of the same outputs, which goes on apart."""
        other = Digest()
        other._hash = self._hash.copy()
        return other

    def hexdigest(self) -> str:
        """Returns the SHA-256 so far in hexadecimal."""
        return self._hash.hexdigest()


def _states(batch, value):
    """Adds the state of every environment of a batch without a stack to the digest `value`."""
    if not batch.stacked:
        for index in range(batch.environment_count):
            value.update(batch.get_state(index))
    return value


def _digest_actions(step: int, environment_count: int, agents: int, action_count: int) -> Any:
    """Returns the actions of a step of `digest`, `[environment_count, agents]` `int32`."""
    values = [
        [(step * 7 + environment * 3 + agent * 5) % action_count for agent in range(agents)]
        for environment in range(environment_count)
    ]
    return torch.tensor(values, dtype=torch.int32)


def digest(
    assets: Any,
    device: str = 'cpu',
    steps: int = 100,
    seed: int = 7,
    environment_count: int = 4,
    **options: Any,
) -> str:
    """Plays a batch with fixed actions and returns the SHA-256 of its outputs.

    Agent `a` of environment `e` takes action `(7 * step + 3 * e + 5 * a) % action_count` at each
    step, the same on every machine. The digest covers the observation of the reset, the
    observation, reward and done flags of every step, then the state of every environment unless
    the batch stacks its observations. The package's self-check compares such digests, 100 steps
    of 4 environments with seed 7, with those in its `digests.json`.

    Args:
        assets: The assets of the batch, as `BatchEnvironment` takes them.
        device: `cpu` or `cuda`, which give the same digest.
        steps: The steps after the reset.
        seed: The seed of the batch.
        environment_count: The number of environments.
        **options: Other keyword arguments of `BatchEnvironment`.

    Returns:
        The SHA-256 in hexadecimal.
    """
    batch = BatchEnvironment(
        assets, environment_count=environment_count, device=device, seed=seed, **options
    )
    try:
        value = Digest()
        value.update(batch.reset())
        for step in range(steps):
            actions = _digest_actions(step, environment_count, batch.agents, batch.action_count)
            observation, reward, done, _ = batch.step(actions.to(batch._torch_device))
            value.update(observation)
            value.update(reward)
            value.update(done)
        return _states(batch, value).hexdigest()
    finally:
        batch.close()


def _plain(value):
    """Returns an option as a value of JSON, with tensors and tuples as lists."""
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _host_values(values):
    """Returns per-environment arguments of `reset` as values of JSON, or `None`."""
    return None if values is None else _plain(torch.as_tensor(values))


class ActionLog:
    """A batch whose calls, actions and outputs are logged for `replay`.

    Call the log's `reset`, `step`, `reset_done` and `set_state` in place of the batch's, and they
    return what the batch's return. `save` writes the options of the batch, every call with its
    arguments, and the digest of the outputs after each call. The digest after the last call adds
    the state of every environment, as `digest` does.

    Attributes:
        batch: The `BatchEnvironment` that the log created.
        calls: The calls so far, each a dictionary with its name and arguments.

    Examples:
        >>> import torch
        >>> import cuju
        >>> log = cuju.ActionLog('5_vs_5_l4_r0', environment_count=8, device='cpu')
        >>> _ = log.reset()
        >>> for _ in range(100):
        ...     _ = log.step(torch.randint(0, 19, (8, 4), dtype=torch.int32))
        >>> log.save('run.zip')
        >>> cuju.replay('run.zip') == log.hexdigest()
        True
    """

    def __init__(self, assets: Any, **options: Any) -> None:
        """Creates the batch.

        Args:
            assets: The assets of the batch, as `BatchEnvironment` takes them.
            **options: The keyword arguments of `BatchEnvironment`. `device`, `library`,
                `threads`, `pin_memory`, `allocator`, `pin_threads`, `numa`, `debug` and
                `check_actions` change no output, so the log leaves them out and a replay chooses
                its own.

        Raises:
            TypeError: If an option is no value of JSON, a tensor, or a sequence of them.
        """
        self._assets = _plain(assets)
        self._options = {
            key: _plain(value) for key, value in options.items() if key not in _MACHINE_OPTIONS
        }
        json.dumps([self._assets, self._options])
        self.batch = BatchEnvironment(assets, **options)
        self.calls: list[dict[str, Any]] = []
        self._actions: list[bytes] = []
        self._states: list[bytes] = []
        self._digest = Digest()
        self._digests: list[bytes] = []

    def _logged(self, call, *outputs):
        """Logs a call with the digest of its outputs."""
        for output in outputs:
            self._digest.update(output)
        self.calls.append(call)
        self._digests.append(bytes.fromhex(self._digest.hexdigest()))

    def reset(self, seeds: Any = None, reverse: Any = None) -> Any:
        """Calls the batch's `reset` and logs it."""
        observation = self.batch.reset(seeds=seeds, reverse=reverse)
        call = {'call': 'reset', 'seeds': _host_values(seeds), 'reverse': _host_values(reverse)}
        self._logged(call, observation)
        return observation

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        """Calls the batch's `step` and logs it with its actions."""
        result = self.batch.step(actions)
        host = torch.as_tensor(actions).to('cpu', torch.int32).contiguous()
        self._actions.append(ctypes.string_at(host.data_ptr(), host.numel() * host.element_size()))
        self._logged({'call': 'step', 'shape': list(host.shape)}, *result[:3])
        return result

    def reset_done(self) -> None:
        """Calls the batch's `reset_done` and logs it."""
        self.batch.reset_done()
        self._logged({'call': 'reset_done'}, self.batch.observation())

    def set_state(self, index: int, state: bytes) -> None:
        """Calls the batch's `set_state` and logs it with the state."""
        self.batch.set_state(index, state)
        self._states.append(bytes(state))
        self._logged({'call': 'set_state', 'index': int(index)}, self.batch.observation())

    def hexdigest(self) -> str:
        """Returns the SHA-256 of every output so far and of the state of every environment."""
        return _states(self.batch, self._digest.copy()).hexdigest()

    def save(self, path: str) -> None:
        """Writes the log to a ZIP file at `path`, which `replay` reads."""
        try:
            version = importlib.metadata.version('cuju')
        except importlib.metadata.PackageNotFoundError:
            version = None
        header = {
            'format': _FORMAT,
            'cuju': version,
            'assets': self._assets,
            'options': self._options,
            'device': self.batch.device,
            'calls': self.calls,
            'digest': self.hexdigest(),
        }
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('log.json', json.dumps(header, indent=1))
            archive.writestr('actions.bin', b''.join(self._actions))
            archive.writestr('digests.bin', b''.join(self._digests))
            for number, state in enumerate(self._states):
                archive.writestr(f'states/{number}.bin', state)

    def close(self) -> None:
        """Closes the batch."""
        self.batch.close()

    def __enter__(self) -> ActionLog:
        return self

    def __exit__(self, *exception: Any) -> None:
        self.close()


def replay(path: str, device: str | None = None, **options: Any) -> str:
    """Plays a saved `ActionLog` again and checks the digest after every call.

    Args:
        path: The file that `ActionLog.save` wrote.
        device: `cpu` or `cuda`, or `None` for the device of the log, which give the same outputs.
        **options: `library`, `threads`, `pin_memory`, `allocator`, `pin_threads`, `numa`,
            `debug` or `check_actions` for the batch, which change no output. A log that an
            earlier version wrote with any of them replays without them unless given here.

    Returns:
        The digest of the replay, equal to the log's.

    Raises:
        ValueError: If the file holds no log of this format, an option changes outputs, or the
            outputs of a call differ from the log's, naming the first such call.
    """
    unknown = sorted(set(options) - set(_REPLAY_OPTIONS))
    if unknown:
        raise ValueError(
            f'expected only {", ".join(_REPLAY_OPTIONS)}, which change no output, found {unknown}'
        )
    with zipfile.ZipFile(path) as archive:
        header = json.loads(archive.read('log.json'))
        if header.get('format') != _FORMAT:
            raise ValueError(
                f'expected an action log of format {_FORMAT}, found {header.get("format")}'
            )
        actions = archive.read('actions.bin')
        digests = archive.read('digests.bin')
        states = [
            archive.read(f'states/{number}.bin')
            for number in range(sum(call['call'] == 'set_state' for call in header['calls']))
        ]
    # An earlier version logged the machine options, which the replaying machine chooses instead.
    batch_options = {
        key: value for key, value in header['options'].items() if key not in _MACHINE_OPTIONS
    }
    for key in ('channel_dimensions',):
        if key in batch_options:
            batch_options[key] = tuple(batch_options[key])
    for key in ('difficulty', 'agent_counts'):
        if batch_options.get(key) is not None:
            batch_options[key] = [tuple(pair) for pair in batch_options[key]]
    batch = BatchEnvironment(
        header['assets'], device=device or header['device'], **batch_options, **options
    )
    try:
        value = Digest()
        offset = 0
        state = 0
        for number, call in enumerate(header['calls']):
            kind = call['call']
            if kind == 'reset':
                value.update(batch.reset(seeds=call['seeds'], reverse=call['reverse']))
            elif kind == 'step':
                count = 1
                for size in call['shape']:
                    count *= size
                step_actions = torch.frombuffer(
                    bytearray(actions[offset : offset + 4 * count]), dtype=torch.int32
                ).reshape(call['shape'])
                offset += 4 * count
                observation, reward, done, _ = batch.step(step_actions.to(batch._torch_device))
                value.update((observation, reward, done))
            elif kind == 'reset_done':
                batch.reset_done()
                value.update(batch.observation())
            elif kind == 'set_state':
                batch.set_state(call['index'], states[state])
                state += 1
                value.update(batch.observation())
            else:
                raise ValueError(f'expected the calls of an action log, found {kind!r}')
            expected = digests[32 * number : 32 * number + 32].hex()
            if value.hexdigest() != expected:
                raise ValueError(
                    f'the outputs of call {number} ({kind}) differ from the log: the digest is '
                    f'{value.hexdigest()}, the log holds {expected}'
                )
        result = _states(batch, value).hexdigest()
    finally:
        batch.close()
    if result != header['digest']:
        raise ValueError(
            f'the states after the last call differ from the log: the digest is {result}, the log '
            f'holds {header["digest"]}'
        )
    return result

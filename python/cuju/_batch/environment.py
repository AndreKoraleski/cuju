# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""`BatchEnvironment`, the batch of environments on the GPU or the CPU, and `memory_estimate`."""

from __future__ import annotations

import contextlib
import ctypes
import importlib
import importlib.util
import io
import operator
import os
import pickle
import struct
import sys
import tempfile
import warnings
from typing import Any

import torch

from ..errors import (
    ActionRangeError,
    ArgumentError,
    ChoiceError,
    ClosedError,
    ConfigurationError,
    DeviceError,
    EnvironmentIndexError,
    LibraryError,
    StateError,
    _from_library,
)
from . import frame_stack as _frame_stack
from . import log as _log
from .assets import _DATA_DIRECTORY, _resolve_asset
from .library import (
    _CONFIGURATION_VERSION,
    _automatic,
    _backend,
    _Configuration,
    _load_library,
    _placement,
    library_path,
)
from .memory import (
    _LOCAL_MEMORY_HELD,
    _free_device_memory,
    _local_memory_bytes,
    _MemoryEstimate,
    _warn_memory,
)
from .state import _pickle_episode, _StateUnpickler

_REPRESENTATIONS = {'simple115': 1, 'simple115v2': 2, 'extracted': 4, 'raw': 8}
# By the type code of `cuju_environment_raw_field`.
_RAW_TYPES = (torch.float32, torch.int32, torch.bool, torch.uint8)
# `MAXIMUM_AGENTS` of the library.
_MAXIMUM_AGENTS = 22
_TYPE_STRINGS = {torch.float32: '<f4', torch.int32: '<i4', torch.bool: '|b1', torch.uint8: '|u1'}
_ACTION_SETS = {'default': 0, 'v2': 1, 'full': 2}
_ACTION_COUNTS = {'default': 19, 'v2': 20, 'full': 33}
# Whether this PyTorch has `torch.library.custom_op` (2.4 and later), which `torch.compile` traces
# a step through.
_CUSTOM_OPERATORS = hasattr(torch.library, 'custom_op')


class _Native:
    """The library's handle of a batch and the files it built, which the views of its buffers keep.

    A tensor that views a buffer holds this object through its storage, where Python's cycle
    collector cannot see it, so this object never refers to the batch. The batch goes when the
    caller drops it, and the memory when the last view of it goes.
    """

    def __init__(self, library, device):
        self.library, self.device = library, device
        self.environment = None
        self.scenario_directory = None
        # The allocator whose functions the library calls, kept until the batch is freed.
        self.allocator = None
        # The page-locked output buffers of a CPU batch, released before the batch.
        self.pinned = None

    def free(self):
        """Frees the batch and its buffers and removes the bundles it built, at most once."""
        if self.pinned:
            from . import pinned as _pinned

            _pinned.unregister(self.pinned)
            self.pinned = None
        if self.environment:
            context = (
                torch.cuda.device(self.device)
                if self.device.type == 'cuda'
                else contextlib.nullcontext()
            )
            with context:
                self.library.cuju_environment_destroy(self.environment)
            self.environment = None
            # The library released its memory through the allocator inside the call.
            self.allocator = None
        if self.scenario_directory is not None:
            self.scenario_directory.cleanup()
            self.scenario_directory = None

    def __del__(self, _finalizing=sys.is_finalizing):
        # At interpreter exit the modules that `free` uses may already be torn down, and the
        # process frees the batch with everything else. The default argument outlives them.
        if _finalizing():
            return
        self.free()


class _CudaArray:
    """A device buffer that `torch.as_tensor` reads through `__cuda_array_interface__`."""

    def __init__(self, pointer, shape, dtype, owner):
        # The tensor holds this object, and this object the owner of the memory it views.
        self.owner = owner
        self.__cuda_array_interface__ = {
            'data': (pointer, False),
            'shape': tuple(shape),
            'typestr': _TYPE_STRINGS[dtype],
            'version': 2,
            'strides': None,
        }


def _view(pointer, shape, dtype, device, owner):
    """Returns a tensor that views the batch memory at `pointer` on `device` without a copy.

    The tensor keeps `owner`, the `_Native` of the batch, alive, so the memory outlives the batch
    while a tensor views it. The owner must not refer to the batch, or the two form a cycle that
    nothing collects.
    """
    shape = tuple(shape)
    if not all(shape):
        return torch.zeros(shape, dtype=dtype, device=device)
    if device.type == 'cuda':
        return torch.as_tensor(_CudaArray(pointer, shape, dtype, owner), device=device)
    elements = 1
    for size in shape:
        elements *= size
    memory = (ctypes.c_char * (elements * dtype.itemsize)).from_address(pointer)
    memory.owner = owner
    return torch.frombuffer(memory, dtype=dtype, count=elements).view(shape)


def _spaces():
    """Returns the spaces of gymnasium, or of gym without it, and their `batch_space`."""
    try:
        import gymnasium as gym
    except ImportError:
        try:
            import gym
        except ImportError:
            raise ModuleNotFoundError(
                'expected gymnasium or gym for the spaces of a batch'
            ) from None
    return gym.spaces, importlib.import_module(gym.__name__ + '.vector.utils').batch_space


def _host(values, dtype):
    """Returns `values` as a contiguous CPU tensor of `dtype`, which the library reads."""
    return torch.as_tensor(values, dtype=dtype).to('cpu').contiguous()


def _per_environment(values, dtype, name, count, error=ConfigurationError):
    """Returns `values` as one host value per environment; a single value serves every one."""
    array = _host(values, dtype)
    if array.dim() == 0:
        array = array.repeat(count)
    if tuple(array.shape) != (count,):
        raise error(
            f'expected {name} with one value, or one for each of the {count} environments, '
            f'found the shape {tuple(array.shape)}'
        )
    return array.contiguous()


def _integers(values, dtype, name, count, error):
    """Returns integer `values` as one host value per environment, and refuses floating ones."""
    array = torch.as_tensor(values)
    # GRF: the seed of a scenario is a C++ `unsigned int`, which refuses a float, 5.0 included.
    if array.is_floating_point() or array.is_complex():
        raise TypeError(f'expected integer {name}, found {array.dtype}')
    return _per_environment(array, dtype, name, count, error)


def _is_pair(value):
    """Returns whether `value` is a `(left, right)` pair."""
    return isinstance(value, (tuple, list)) and len(value) == 2


def _choice(kind, value, choices, hints=None):
    """Returns `choices[value]`, or raises `ChoiceError` naming the valid values and any hint."""
    if value in choices:
        return choices[value]
    hint = (hints or {}).get(value, '')
    raise ChoiceError(
        f'expected the {kind} {", ".join(map(repr, choices))}, found {value!r}'
        + (f'; {hint}' if hint else '')
    )


class BatchEnvironment:
    """A batch of football environments on the GPU or the CPU.

    Every tensor that a method returns views a buffer of the batch on its device, without a copy,
    and the next `reset` or `step` overwrites it, so a caller clones what it keeps. The tensor
    attributes live on the CPU.

    Attributes:
        device: `cuda` or `cpu`, the backend that runs the batch.
        torch_device: The `torch.device` that holds the buffers of the batch, such as `cuda:0`, and
            on which every call runs.
        threads: The CPU threads that run the environments, the caller included, or 0 on the GPU.
        environment_count: The number of environments.
        left_agents: The agents of the left team of each environment, `[environment_count]` `int32`.
        right_agents: The agents of the right team of each environment, `[environment_count]`
            `int32`.
        agents: The size of the agent axis, the largest number of agents of any bundle.
        agent_mask: `[environment_count, agents]`, true for each agent that exists.
        left_players: The players of the left team of each environment, `[environment_count]`
            `int32`, as GRF's `len(observation['left_team'])` for a left agent.
        right_players: The players of the right team of each environment, `[environment_count]`
            `int32`. A right agent sees its own team as the left team, so its counts swap.
        players: The size of each team array of `raw`, the largest team of any bundle; the array
            holds the players of its team first and zeros after them.
        single_observation_space, single_action_space, observation_space, action_space: The spaces
            of one environment and of the batch, as gymnasium describes them.
        squeeze, stacked, channel_dimensions, auto_reset, check_actions, debug: The options of
            the batch.
        action_count: The number of actions: 19 for `default`, 20 for `v2` and 33 for `full`.
        difficulty: `[environment_count, 2]`, the difficulty of the left and of the right team of
            each environment. It is the one that `difficulty` sets, else the scenario's, and for a
            scenario whose difficulty follows the parity of the episode, that of its first episode.

    Examples:
        >>> import torch
        >>> import cuju
        >>> batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=1024, device='cpu')
        >>> observation = batch.reset()
        >>> tuple(observation.shape)
        (1024, 4, 115)
        >>> actions = torch.zeros((1024, 4), dtype=torch.int32)
        >>> observation, reward, done, information = batch.step(actions)
        >>> tuple(reward.shape), tuple(done.shape)
        ((1024, 4), (1024,))
    """

    def __init__(
        self,
        assets: str | list[str],
        animations: str | None = None,
        environment_count: int = 1024,
        representation: str = 'simple115v2',
        rewards: str = 'scoring',
        action_set: str = 'default',
        seed: int = 0,
        numpy_legacy: bool = True,
        auto_reset: bool = True,
        library: str | None = None,
        assignment: Any = None,
        with_raw: bool = False,
        difficulty: list[tuple[float | None, float | None]] | None = None,
        device: str | torch.device = 'cuda',
        threads: int = 0,
        physics_steps_per_frame: int = 10,
        channel_dimensions: tuple[int, int] = (96, 72),
        stacked: bool = False,
        game_engine_random_seed: Any = None,
        squeeze: bool = False,
        agent_counts: list[tuple[int | None, int | None]] | None = None,
        episode_number: int = 0,
        first_index: int = 0,
        allocator: str = 'cuda',
        pin_threads: bool = False,
        numa: bool = False,
        pin_memory: bool = False,
        check_actions: bool = False,
        debug: bool = False,
    ) -> None:
        """Creates the batch and allocates its buffers.

        Args:
            assets: A bundle or a list of bundles, each a bundle path, a shipped name
                `<scenario>_l<left>_r<right>` with any agent counts, a bare shipped scenario such as
                `5_vs_5`, or a GRF scenario file; the last two take one left agent, as in GRF.
            animations: The path of `animations.bin`. `None` takes the shipped tables.
            environment_count: The number of environments.
            representation: `simple115`, `simple115v2`, `extracted` (the SMM) or `raw` (a dictionary
                of named tensors).
            rewards: `scoring`, or `scoring,checkpoints` to add the checkpoint reward.
            action_set: `default`, `v2` or `full`.
            seed: The seed of the batch, from 0 to 2**64 - 1. Environment `index` plays as GRF's
                `create_environment` after `random.seed(seed * 2**32 + first_index + index)`.
            numpy_legacy: Whether the checkpoint reward adds in the precision of numpy before 2, as
                GRF does with such a numpy. `False` follows numpy 2.
            auto_reset: Whether a step resets every environment whose episode ended. `False` leaves
                each such environment at its end until `reset_done` or `reset`.
            library: The path of the library of `device`. `None` takes `library_path(device)`.
            assignment: The index into `assets` of the bundle of each environment,
                `[environment_count]`. `None` gives every environment the first bundle.
            with_raw: Whether the batch also writes the `raw` representation, which `raw` and
                `final_raw` return.
            difficulty: One `(left, right)` pair of team difficulties per bundle, each from 0 to 1
                or `None` for the scenario's; every reset writes them. `None` keeps the scenarios'
                own.
            device: `cuda` for the current CUDA device through `libcuju_cuda.so`, `cuda:N` or a
                `torch.device` for GPU N, or `cpu` for the CPU through `libcuju_cpu.so`. Both give
                the same outputs bit for bit, and the device decides only where the batch runs.
                `auto` takes PyTorch's current CUDA device where PyTorch sees a GPU and the CUDA
                library is installed, else `cpu`, and `batch.torch_device` says which. Only an
                explicit `auto` picks, never the default.
            threads: The CPU threads, the caller included, or 0 for one per logical processor that
                the process may run on. The GPU ignores it.
            physics_steps_per_frame: The ticks of 10 milliseconds in a step, as GRF's option.
            channel_dimensions: The width and the height of the SMM in cells, as GRF's option.
            stacked: Whether the observation stacks the last 4 along its last axis, the oldest
                first, as GRF's `stacked`; a reset fills the stack. `raw` has no stack.
            game_engine_random_seed: The seed of every episode, as GRF's option: an integer for
                every environment, or one per environment with -1 to draw. `None` draws every seed.
            squeeze: Whether the observation (except `raw`) and the reward drop the agent axis and
                `step` takes `[environment_count]` actions, as GRF for one agent. Needs one agent
                each.
            agent_counts: One `(left, right)` pair of agent counts per bundle, each `None` for what
                the name or the bundle gives. `None` keeps the agents of every bundle.
            episode_number: GRF's `episode_number`, the episode before the first; its parity picks
                the even or the odd form of a scenario.
            first_index: The index of environment 0 in the seed rule, so the batch plays the
                environments `first_index` to `first_index + environment_count - 1` of `seed`, as a
                larger batch plays them. Every index stays below 2**32.
            allocator: Where the GPU buffers come from, `cuda` from `cudaMalloc`, or `torch` from
                PyTorch's caching allocator, so that `torch.cuda.memory_allocated`, its statistics
                and its limits count them. The outputs are the same bit for bit. The local memory
                that the driver reserves for the kernels stays outside either. The CPU ignores it.
            pin_threads: Whether each CPU thread that the batch starts stays on a processor of its
                own, taken in order from those that `os.sched_setaffinity` allows. The calling
                thread stays where it is. The GPU ignores it.
            numa: Whether the CPU threads form a group per NUMA node, on that node's processors,
                each stepping a fixed range of environments whose memory its threads touch first.
                With one node it changes nothing. The GPU ignores it.
            pin_memory: Whether the output buffers of a CPU batch are page-locked, so that
                `tensor.to('cuda', non_blocking=True)` copies them without waiting. Needs CUDA. The
                GPU ignores it.
            check_actions: Whether a GPU batch also checks actions in GPU memory against
                `[-action_count, action_count)`, as it checks actions in host memory, so that `step`
                raises `ActionRangeError` before any environment steps. The check waits for the
                device at every step. Without it, such an action steps as the idle action. The CPU
                checks every action anyway.
            debug: Whether every `step`, `step_async`, `reset_done` and masked `reset` of a GPU
                batch waits for the device, so that a failure on the device raises at the call that
                caused it. It checks the actions as `check_actions` does, and costs the waits. A
                CUDA graph capture never waits. The CPU, which waits anyway, ignores it.

        Raises:
            TypeError: If `game_engine_random_seed` holds floating-point values, which GRF refuses.
            ChoiceError: If `representation`, `action_set` or `device` names no known value. It is a
                `ValueError` and a `KeyError`.
            ConfigurationError: If `difficulty` or `agent_counts` lacks a pair per bundle,
                `squeeze` meets an environment without exactly one agent, `stacked` meets `raw`,
                `rewards` lacks `scoring`, `seed` lies outside `[0, 2**64)` or `first_index`
                outside its range. It is a `ValueError`, and its subclass
                `scenarios.ScenarioError` reports a refused scenario file.
            MissingFileError: If an asset names no bundle or the library is missing. It is a
                `FileNotFoundError`.
            DeviceError: If `device` names a CUDA GPU and PyTorch sees none or fewer, or the CPU
                lacks AVX2, BMI2 or FMA. It is a `RuntimeError`.
            LibraryError: If the library fails to create the batch, for example on an `assignment`
                outside `assets`, a team with fewer players than agents, a scenario with
                `control_all_players` and agents that switch players, which GRF asserts against, or
                if the library runs another backend. It is a `RuntimeError`.
        """
        device = _automatic(device)
        backend = _backend(device)
        representation_flag = _choice(
            'representation',
            representation,
            _REPRESENTATIONS,
            {
                'pixels': 'cuju renders nothing',
                'pixels_gray': 'cuju renders nothing',
                'extracted_stacked': "pass representation='extracted' with stacked=True",
                'simple115_stacked': "pass representation='simple115' with stacked=True",
            },
        )
        _choice('action set', action_set, _ACTION_SETS)
        # GRF: an assertion requires `scoring` in the rewards.
        if 'scoring' not in rewards.split(','):
            raise ConfigurationError(
                f'expected the rewards scoring or scoring,checkpoints, found {rewards!r}'
            )
        # GRF: the other parts are ignored, so a typo silently drops a reward.
        unknown = [part for part in rewards.split(',') if part not in ('scoring', 'checkpoints')]
        if unknown:
            warnings.warn(
                f'the rewards {rewards!r} hold {unknown}, which GRF ignores, as cuju does; '
                "write them without spaces, as 'scoring,checkpoints'",
                stacklevel=2,
            )
        # The library holds the seed in 64 bits, where a larger or negative one would wrap.
        if not 0 <= seed < 2**64:
            raise ConfigurationError(f'expected a seed from 0 to 2**64 - 1, found {seed}')
        if environment_count < 1:
            raise ConfigurationError(
                f'expected at least one environment, found {environment_count}'
            )
        # GRF: the index and the seed share one key, so an index of 2**32 would be another seed's.
        if not 0 <= first_index <= 2**32 - environment_count:
            raise ConfigurationError(
                f'expected a first index from 0 to 2**32 - environment_count, found {first_index}'
            )
        if threads < 0:
            raise ConfigurationError(
                f'expected 0 threads for one per processor, or more, found {threads}'
            )
        if allocator not in ('cuda', 'torch'):
            raise ChoiceError(f'expected the allocator cuda or torch, found {allocator!r}')
        if pin_memory and backend == 'cpu' and not torch.cuda.is_available():
            raise DeviceError(
                'expected a CUDA GPU for pin_memory=True, found none that PyTorch can use'
            )
        self.device = backend
        # A GPU batch stays on its CUDA device, and every call runs there.
        self._torch_device = _placement(device)
        logged = _log.start(backend, self._torch_device)
        self._library = _load_library(library, backend)
        # What the views of the buffers keep alive, without the batch.
        self._native = _Native(self._library, self._torch_device)
        requested = [assets] if isinstance(assets, str) or callable(assets) else list(assets)
        resolved = [_resolve_asset(asset) for asset in requested]
        paths = [path for path, _ in resolved]
        if animations is None:
            animations = os.path.join(_DATA_DIRECTORY, 'animations.bin')
        # A scenario file becomes a bundle without agents, which the batch reads at its creation,
        # built with the batch's animation tables.
        for index, path in enumerate(paths):
            if path.endswith('.py'):
                from .. import scenarios

                # With `CUJU_BUNDLE_CACHE` set, a file builds once and later batches load it.
                cached = scenarios._batch_bundle(path, backend, library, animations)
                if cached is not None:
                    paths[index] = cached
                    continue
                if self._native.scenario_directory is None:
                    self._native.scenario_directory = tempfile.TemporaryDirectory(prefix='cuju-')
                name = os.path.splitext(os.path.basename(path))[0]
                paths[index] = scenarios.build_bundle(
                    path,
                    os.path.join(self._native.scenario_directory.name, f'{index}_{name}_l0_r0.cjb'),
                    device=backend,
                    library=library,
                    animations=animations,
                )
        if agent_counts is not None and (
            len(agent_counts) != len(paths) or not all(map(_is_pair, agent_counts))
        ):
            raise ConfigurationError(
                f'expected a (left, right) pair of agent counts for each of the {len(paths)} '
                f'bundles, such as [(4, 0)] for one, found {agent_counts!r}'
            )
        counts = [
            [
                named if given is None else given
                for named, given in zip(named_pair, given_pair, strict=True)
            ]
            for (_, named_pair), given_pair in zip(
                resolved, agent_counts or [(None, None)] * len(paths), strict=True
            )
        ]
        counts_array = (ctypes.c_int * (2 * len(paths)))(
            *(count for pair in counts for count in pair)
        )
        if difficulty is not None:
            if len(difficulty) != len(paths) or not all(map(_is_pair, difficulty)):
                raise ConfigurationError(
                    f'expected a (left, right) pair of difficulties for each of the {len(paths)} '
                    f'bundles, such as [(None, 0.5)] for one, found {difficulty!r}'
                )
            if any(
                value is not None and not 0 <= value <= 1 for pair in difficulty for value in pair
            ):
                raise ConfigurationError(
                    f"expected difficulties from 0 to 1, or None for the scenario's, found "
                    f'{difficulty!r}'
                )
        if assignment is None and len(paths) > 1:
            warnings.warn(
                f'the batch has {len(paths)} bundles and no assignment, so every environment plays '
                'the first; pass assignment, the bundle index of each environment',
                stacklevel=2,
            )
        if with_raw:
            representation_flag |= _REPRESENTATIONS['raw']
        if stacked and representation == 'raw':
            raise ConfigurationError(
                'expected simple115, simple115v2 or extracted for a stack, found raw'
            )
        path_array = (ctypes.c_char_p * len(paths))(*(path.encode() for path in paths))
        assignment_array = (
            None
            if assignment is None
            else _per_environment(assignment, torch.int32, 'assignment', environment_count)
        )
        fixed_seeds = None
        if game_engine_random_seed is not None:
            fixed_seeds = _integers(
                game_engine_random_seed,
                torch.int64,
                'game_engine_random_seed',
                environment_count,
                ConfigurationError,
            )
        width, height = channel_dimensions
        configuration = _Configuration(
            version=_CONFIGURATION_VERSION,
            assets=path_array,
            asset_count=len(paths),
            assignment=None if assignment_array is None else assignment_array.data_ptr(),
            animations=animations.encode(),
            environment_count=environment_count,
            seed=seed,
            action_set=_ACTION_SETS[action_set],
            observation=representation_flag,
            checkpoints=int('checkpoints' in rewards.split(',')),
            numpy_legacy=int(numpy_legacy),
            auto_reset=int(auto_reset),
            threads=threads,
            ticks_per_step=physics_steps_per_frame,
            smm_width=width,
            smm_height=height,
            stack=4 if stacked else 1,
            fixed_seeds=None if fixed_seeds is None else fixed_seeds.data_ptr(),
            agent_counts=ctypes.addressof(counts_array),
            episode_number=episode_number,
            first_index=first_index,
            pin_threads=int(pin_threads),
            numa=int(numa),
            page_aligned=int(pin_memory and backend == 'cpu'),
        )
        estimate_only = getattr(self, '_estimate_only', False)
        # The functions must outlive the batch, which calls them until it is freed.
        if allocator == 'torch' and backend == 'cuda' and not estimate_only:
            from .allocator import TorchAllocator

            self._native.allocator = TorchAllocator(self._torch_device)
            configuration.allocate, configuration.release = self._native.allocator.pointers()
        # The query loads the kernels, so the free memory that follows leaves their code out.
        local_memory = _local_memory_bytes(self._library, self._torch_device)
        free_memory = None
        if backend == 'cuda' and not estimate_only:
            free_memory = _free_device_memory(self._torch_device, allocator)
        with self._on_device():
            if estimate_only:
                estimate = ctypes.c_ulonglong()
                status = self._library.cuju_memory_estimate(
                    ctypes.byref(configuration), ctypes.byref(estimate)
                )
                environment = None if status != 0 else 1
            else:
                environment = self._library.cuju_environment_create(ctypes.byref(configuration))
        if not environment:
            message = self._library.cuju_last_error().decode()
            # The library names the bundle it read; name what the caller passed instead.
            for path, asset in zip(paths, requested, strict=True):
                message = message.replace(path, repr(asset))
            if self._native.allocator is not None and self._native.allocator.errors:
                raise self._native.allocator.errors[0]
            raise _from_library(message)
        if estimate_only:
            bundles = (
                torch.zeros(environment_count, dtype=torch.int64)
                if assignment_array is None
                else assignment_array.long()
            )
            # The difficulty set below takes `[environment_count, 2]` floats of the library.
            sets_difficulty = any(
                value is not None and bool((bundles == bundle).any())
                for bundle, pair in enumerate(difficulty or [])
                for value in pair
            )
            self._estimated_bytes = _MemoryEstimate(
                estimate.value + 8 * environment_count * sets_difficulty, local_memory
            )
            return
        self._native.environment = environment
        # The creation launched the kernels, and the driver holds their local memory from then on.
        if backend == 'cuda':
            _LOCAL_MEMORY_HELD.add(self._torch_device.index)
        if free_memory is not None:
            _warn_memory(
                self.memory_bytes + local_memory, free_memory, self._torch_device, local_memory
            )
        self.threads = self._library.cuju_environment_threads(environment)
        self.environment_count = environment_count
        self.left_agents = torch.zeros(environment_count, dtype=torch.int32)
        self.right_agents = torch.zeros(environment_count, dtype=torch.int32)
        self.agents = self._library.cuju_environment_agents(
            environment, self.left_agents.data_ptr(), self.right_agents.data_ptr()
        )
        self.left_players = torch.zeros(environment_count, dtype=torch.int32)
        self.right_players = torch.zeros(environment_count, dtype=torch.int32)
        self.players = self._library.cuju_environment_players(
            environment, self.left_players.data_ptr(), self.right_players.data_ptr()
        )
        slots = torch.arange(self.agents)
        self.agent_mask = slots < (self.left_agents + self.right_agents)[:, None]
        self.action_count = _ACTION_COUNTS[action_set]
        if squeeze and not bool(((self.left_agents + self.right_agents) == 1).all()):
            self.close()
            raise ConfigurationError(
                'expected one agent in every environment to squeeze the agent axis'
            )
        self.squeeze = squeeze
        self.stacked = stacked
        self.channel_dimensions = (width, height)
        bundles = (
            torch.zeros(environment_count, dtype=torch.int64)
            if assignment_array is None
            else assignment_array.long()
        )
        chosen = torch.full((environment_count, 2), float('nan'))
        for bundle, pair in enumerate(difficulty or []):
            for side, value in enumerate(pair):
                if value is not None:
                    chosen[bundles == bundle, side] = value
        if not torch.isnan(chosen).all():
            # The library takes a negative difficulty for that of the scenario.
            values = torch.nan_to_num(chosen, nan=-1.0).contiguous()
            with self._on_device():
                self._check(
                    self._library.cuju_environment_set_difficulty(self._handle(), values.data_ptr())
                )
        self.difficulty = torch.empty((environment_count, 2), dtype=torch.float32)
        self._library.cuju_environment_difficulty(self._handle(), self.difficulty.data_ptr())
        self.auto_reset = auto_reset
        self.check_actions, self.debug = check_actions, debug
        self._checks_device_actions = backend == 'cuda' and (check_actions or debug)
        self._agent_mask_on_device = None
        self._checkpoints = 'checkpoints' in rewards.split(',')
        self._representation = representation
        self._raw_fields = (
            self._describe_raw() if representation_flag & _REPRESENTATIONS['raw'] else None
        )
        stack = 4 if stacked else 1
        if representation == 'raw':
            self._observation_name, self._observation_shape, self._observation_type = (
                'raw',
                None,
                None,
            )
        elif representation == 'extracted':
            self._observation_name, self._observation_shape, self._observation_type = (
                'smm',
                (environment_count, self.agents, height, width, 4 * stack),
                torch.uint8,
            )
        else:
            self._observation_name, self._observation_shape, self._observation_type = (
                'simple115',
                (environment_count, self.agents, 115 * stack),
                torch.float32,
            )
        # The tensors share the buffers, which never move, so they are made once.
        self._cached_outputs = None
        self._score_reward = None
        _log.created(self, requested, library or library_path(backend), logged)
        # The key of the batch for the operators of `torch.compile`, and the outputs they mutate.
        self._operator_key = None
        self._operator_outputs = None
        self._operator_static = False
        self._step_in_flight = False
        if pin_memory and backend == 'cpu':
            from . import pinned as _pinned

            try:
                self._native.pinned = _pinned.register(self._output_ranges())
            except BaseException:
                self.close()
                raise

    def _output_ranges(self):
        """Returns the address and the size in bytes of every output buffer of the batch."""
        count, agents = self.environment_count, self.agents
        sizes = {'reward': count * agents * 4, 'done': count, 'score_reward': count * 4}
        if self._observation_shape is not None:
            elements = 1
            for size in self._observation_shape:
                elements *= size
            observation = elements * self._observation_type.itemsize
            sizes[self._observation_name] = sizes['final_' + self._observation_name] = observation
        if self._raw_fields is not None:
            raw = 0
            for _, dtype, shape, offset in self._raw_fields:
                elements = 1
                for size in shape:
                    elements *= size
                raw = max(raw, offset + elements * dtype.itemsize)
            sizes['raw'] = sizes['final_raw'] = raw
        return [
            (self._library.cuju_environment_buffer(self._handle(), name.encode()), size)
            for name, size in sizes.items()
        ]

    def close(self) -> None:
        """Frees the batch and its buffers at once. A second call does nothing.

        The tensors that the batch returned view those buffers, so none may be used after it, and
        every later call on the batch raises `ClosedError`, a `RuntimeError`. A batch dropped
        without `close` frees them once no tensor that it returned remains.
        """
        native = getattr(self, '_native', None)
        if native is not None:
            native.free()

    def __enter__(self) -> BatchEnvironment:
        self._handle()
        return self

    def __exit__(self, *exception: object) -> None:
        # `with` frees the batch at the end of its block, an exception included.
        self.close()

    def __reduce__(self):
        # A batch owns device memory and threads, which no pickle can carry to another process.
        raise TypeError(
            'a batch cannot be pickled; create it inside the worker process that uses it'
        )

    @property
    def memory_bytes(self) -> int:
        """The bytes of memory that the batch has requested so far, its tables included.

        It counts device memory on the GPU and host memory on the CPU. The allocators round each
        request up. The first batch of the process on a GPU also makes the driver reserve the local
        memory of the kernels, which `memory_estimate` gives as `local_memory_bytes` and this count
        leaves out. A few later calls add a small buffer each, on the GPU the first
        `reset` with seeds, with a reverse flag or with a host mask, and on the CPU the first
        `step_async`.
        """
        return int(self._library.cuju_environment_memory_bytes(self._handle()))

    def _handle(self):
        """Returns the library's handle of the batch, or raises `ClosedError` once it is closed."""
        environment = getattr(getattr(self, '_native', None), 'environment', None)
        if not environment:
            raise ClosedError('the batch is closed')
        return environment

    def _check(self, status):
        """Raises the error of the library if `status` reports a failure."""
        if status != 0:
            raise _from_library(self._library.cuju_last_error().decode())

    def _describe_raw(self):
        """Returns the name, the type, the shape and the offset of each raw array."""
        fields = []
        element_type, offset = ctypes.c_int(), ctypes.c_ulonglong()
        dimensions = (ctypes.c_int * 2)()
        for field in range(64):
            name = self._library.cuju_environment_raw_field(
                self._handle(),
                field,
                ctypes.byref(element_type),
                dimensions,
                ctypes.byref(offset),
            )
            if name is None:
                break
            shape = (
                self.environment_count,
                self.agents,
                *(dimension for dimension in dimensions if dimension),
            )
            fields.append((name.decode(), _RAW_TYPES[element_type.value], shape, offset.value))
        return fields

    def _raw(self, name):
        """Returns the raw arrays of a buffer as tensors that share it."""
        pointer = self._library.cuju_environment_buffer(self._handle(), name.encode())
        return {
            field: _view(pointer + offset, shape, dtype, self._torch_device, self._native)
            for field, dtype, shape, offset in self._raw_fields
        }

    def _buffer(self, name, shape, dtype):
        """Returns a named output buffer as a tensor that shares it."""
        pointer = self._library.cuju_environment_buffer(self._handle(), name.encode())
        return _view(pointer, shape, dtype, self._torch_device, self._native)

    def _outputs(self):
        """Returns the observation, the reward, the done flags and the final observation."""
        # The cached tensors view the buffers, which a closed batch has freed.
        self._handle()
        if self._cached_outputs is None:
            reward = self._buffer('reward', (self.environment_count, self.agents), torch.float32)
            done = self._buffer('done', (self.environment_count,), torch.uint8)
            if self._representation == 'raw':
                observation, final = self._raw('raw'), self._raw('final_raw')
            else:
                observation = self._buffer(
                    self._observation_name, self._observation_shape, self._observation_type
                )
                final = self._buffer(
                    'final_' + self._observation_name,
                    self._observation_shape,
                    self._observation_type,
                )
            if self.squeeze:
                # GRF: the raw observation of one agent keeps its agent axis.
                if self._representation != 'raw':
                    observation, final = observation.squeeze(1), final.squeeze(1)
                reward = reward.squeeze(1)
            self._cached_outputs = observation, reward, done, final
            self._prepare_operators()
        # The buffers never move, so the compiler's CUDA graphs may write them in place. The
        # compiler takes seconds to load, so a batch marks them only once something else loaded it.
        if not self._operator_static and 'torch._dynamo' in sys.modules:
            for tensor in self._operator_outputs:
                torch._dynamo.mark_static_address(tensor)
            self._operator_static = True
        return self._cached_outputs

    @property
    def torch_device(self) -> torch.device:
        """The `torch.device` of the buffers of the batch, such as `cuda:0` or `cpu`."""
        return self._torch_device

    def _prepare_operators(self):
        """Registers the batch with the operators of `torch.compile` and lists what they mutate."""
        from . import operators as _operators

        if self._score_reward is None:
            self._score_reward = self._buffer(
                'score_reward', (self.environment_count,), torch.int32
            )
        self._operator_outputs = [
            tensor
            for output in (*self._cached_outputs, self._score_reward)
            for tensor in (output.values() if isinstance(output, dict) else (output,))
        ]
        self._operator_key = _operators.register(self)

    def _operator_step(self, actions, outputs):
        """Runs `cuju::step`, then fills each of `outputs` that copies a tensor of the batch."""
        if self._checks_device_actions:
            self._check_device_actions(actions)
        with self._on_device():
            status = self._library.cuju_environment_step(
                self._handle(), actions.data_ptr(), 1, self._stream(), 0
            )
        if status == -2:
            raise ActionRangeError(self._library.cuju_last_error().decode())
        self._check(status)
        self._wait_in_debug()
        self._copy_operator_outputs(outputs)

    def _operator_reset_done(self, outputs):
        """Runs `cuju::reset_done`, then fills `outputs` as `_operator_step` does."""
        with self._on_device():
            self._check(self._library.cuju_environment_reset_done(self._handle(), self._stream()))
        self._wait_in_debug()
        self._copy_operator_outputs(outputs)

    def _copy_operator_outputs(self, outputs):
        """Copies each output into the tensor of `outputs` in its place, unless it is the same."""
        for given, own in zip(outputs, self._operator_outputs, strict=True):
            if given.data_ptr() != own.data_ptr():
                given.copy_(own)

    def _compiled_step(self, actions):
        """`step` under `torch.compile`, through the operator `cuju::step`."""
        shape = (self.environment_count, self.agents)
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        if actions.is_floating_point() or actions.is_complex():
            raise TypeError(f'expected integer action indices, found {actions.dtype}')
        if tuple(actions.shape) != shape and not (
            self.agents == 1 and tuple(actions.shape) == (self.environment_count,)
        ):
            raise ArgumentError(
                f'expected actions of the shape {shape}, [environment_count, agents], found '
                f'{tuple(actions.shape)}'
            )
        actions = actions.to(self._torch_device, torch.int32).reshape(shape).contiguous()
        # Before the first reset this is a graph break, and the step then raises as `step` does.
        if self._operator_outputs is None:
            self._outputs()
        torch.ops.cuju.step(self._operator_key, actions, self._operator_outputs)
        observation, reward, done, final = self._cached_outputs
        return (
            observation,
            reward,
            done,
            {'final_observation': final, 'score_reward': self._score_reward},
        )

    def _compiled_reset_done(self):
        """`reset_done` under `torch.compile`, through the operator `cuju::reset_done`."""
        if self._operator_outputs is None:
            self._outputs()
        torch.ops.cuju.reset_done(self._operator_key, self._operator_outputs)

    def _stream(self):
        """Returns PyTorch's current CUDA stream on the batch's device, or `None` on the CPU."""
        if self.device == 'cpu':
            return None
        return torch.cuda.current_stream(self._torch_device).cuda_stream

    def _on_device(self):
        """Returns a context that runs the calls of the library on the batch's CUDA device."""
        if self.device == 'cpu':
            return contextlib.nullcontext()
        return torch.cuda.device(self._torch_device)

    def _per_environment(self, values, dtype, name):
        """Returns `values` as one host value per environment of the batch."""
        return _per_environment(values, dtype, name, self.environment_count, ArgumentError)

    def _staged(self, array):
        """Returns a host input of a GPU batch that the caller may reuse once the call returns.

        The library copies host inputs to the device with `cudaMemcpyAsync`, which reads pinned
        memory later, when the stream reaches the copy, and stages pageable memory before it
        returns. So a pinned input becomes a pageable clone, and any other stays as it is.
        """
        if array is not None and self.device == 'cuda' and array.is_pinned():
            return array.clone()
        return array

    def reset(self, seeds: Any = None, reverse: Any = None, mask: Any = None) -> Any:
        """Starts the next episode of every environment, or of those that `mask` selects.

        Args:
            seeds: One seed per environment for this episode, as `game_engine_random_seed` in the
                configuration of GRF, or one seed for every environment. A seed of -1 draws that
                environment's seed exactly as `None` does, the draws of its first reset included.
                `None` draws every seed.
            reverse: One `reverse_team_processing` flag per environment, or one for every
                environment. `None` follows the seeds, as GRF does. Only the first reset of an
                environment reads it.
            mask: `[environment_count]` booleans. Each environment where it is true starts its next
                episode as GRF's `reset()` does, with its seed draw, and the others keep their
                episode, outputs and done flags. A mask on the batch's device takes no copy, and
                the call then waits for nothing, so a CUDA graph can capture it once every
                environment has had its first reset. `None` resets every environment and waits.
                Host arguments, pinned or not, may be reused as soon as the call returns.

        Returns:
            The observation of every agent: `[environment_count, agents, 115]` for `simple115` and
            `simple115v2`, `[environment_count, agents, height, width, 4]` for `extracted`, or a
            dictionary of `[environment_count, agents, ...]` tensors for `raw`. A stack multiplies
            the last axis by 4, and `squeeze` drops the agent axis except for `raw`.

        Raises:
            TypeError: If `seeds`, `reverse` or `mask` holds floating-point values, which GRF
                refuses for a seed, 5.0 included.
            ArgumentError: If `seeds`, `reverse` or `mask` holds neither one value nor one per
                environment, a seed lies outside `[0, 4294967295]` and is not -1, or a reverse flag
                is neither 0 nor 1. It is a `ValueError`.
            LibraryError: If the library fails to reset the batch. It is a `RuntimeError`.
            UndefinedPathError: If an environment reaches a path that GRF leaves undefined. It is a
                `RuntimeError`.
            DeviceError: If the device fails in a `debug` batch. It is a `RuntimeError`.
        """
        seed_array = None
        if seeds is not None:
            seed_array = _integers(
                seeds, torch.int64, 'seeds', self.environment_count, ArgumentError
            )
            # GRF: the scenario holds its seed as a 32-bit `unsigned int`.
            if bool(((seed_array < -1) | (seed_array > 4294967295)).any()):
                raise ArgumentError(
                    f'expected seeds from 0 to 4294967295, or -1 to draw, found '
                    f'{seed_array.tolist()}'
                )
        reverse_array = None
        if reverse is not None:
            reverse_array = _integers(
                reverse, torch.int64, 'reverse', self.environment_count, ArgumentError
            )
            if bool(((reverse_array != 0) & (reverse_array != 1)).any()):
                raise ArgumentError(
                    f'expected reverse flags of 0 or 1, found {reverse_array.tolist()}'
                )
            reverse_array = reverse_array.to(torch.int32)
        if mask is not None:
            return self._reset_masked(mask, seed_array, reverse_array)
        if seed_array is not None and bool((seed_array == -1).any()):
            # Only the masked entry takes seeds that draw.
            return self._reset_masked(None, seed_array, reverse_array, synchronize=True)
        if seed_array is not None:
            seed_array = seed_array.to(torch.int32)
        with self._on_device():
            self._check(
                self._library.cuju_environment_reset(
                    self._handle(),
                    None if seed_array is None else seed_array.data_ptr(),
                    None if reverse_array is None else reverse_array.data_ptr(),
                    self._stream(),
                )
            )
        return self._outputs()[0]

    def _reset_masked(self, mask, seed_array, reverse_array, synchronize=False):
        """Resets the environments of `mask`, or all for `None`, through the masked entry."""
        mask_array, on_device = None, 0
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask)
            if mask.is_floating_point() or mask.is_complex():
                raise TypeError(f'expected a boolean or integer mask, found {mask.dtype}')
            if mask.device == self._torch_device and mask.dim() == 1:
                if tuple(mask.shape) != (self.environment_count,):
                    raise ArgumentError(
                        f'expected a mask of the shape ({self.environment_count},), found '
                        f'{tuple(mask.shape)}'
                    )
                mask_array, on_device = mask.to(torch.uint8).contiguous(), 1
            else:
                mask_array = self._staged(self._per_environment(mask, torch.uint8, 'mask'))
                on_device = int(self.device == 'cpu')
        seed_array, reverse_array = self._staged(seed_array), self._staged(reverse_array)
        with self._on_device():
            self._check(
                self._library.cuju_environment_reset_masked(
                    self._handle(),
                    None if mask_array is None else mask_array.data_ptr(),
                    on_device,
                    None if seed_array is None else seed_array.data_ptr(),
                    None if reverse_array is None else reverse_array.data_ptr(),
                    self._stream(),
                    int(synchronize),
                )
            )
        self._wait_in_debug()
        return self._outputs()[0]

    def reset_done(self) -> None:
        """Starts the next episode of every environment whose episode ended.

        A batch with `auto_reset=False` leaves such environments at their end. The reset reads the
        done flags in the batch's memory, so on the GPU it does not wait for the device; it writes
        the new observations and clears the done flags of the environments it resets.

        Raises:
            LibraryError: If the library fails to run the reset. It is a `RuntimeError`.
            UndefinedPathError: If an environment reaches a path that GRF leaves undefined. It is a
                `RuntimeError`.
            DeviceError: If the device fails in a `debug` batch. It is a `RuntimeError`.
        """
        # Under `torch.compile` the call goes through the operator `cuju::reset_done`.
        if _CUSTOM_OPERATORS and torch.compiler.is_compiling():
            return self._compiled_reset_done()
        with self._on_device():
            self._check(self._library.cuju_environment_reset_done(self._handle(), self._stream()))
        self._wait_in_debug()

    def observation(self) -> Any:
        """Returns the observation of every agent that the last `reset` or `step` returned."""
        return self._outputs()[0]

    @property
    def single_observation_space(self) -> Any:
        """The observation space of one environment, as GRF's with the same options, or `None` for
        `raw`, which has none in GRF. The spaces need gymnasium or gym."""
        if self._representation == 'raw':
            return None
        spaces, _ = _spaces()
        import numpy as np

        shape = self._observation_shape[2 if self.squeeze else 1 :]
        if self._representation == 'extracted':
            return spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        return spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

    @property
    def single_action_space(self) -> Any:
        """The action space of one environment, `Discrete` with `squeeze` and `MultiDiscrete` over
        the agents without, as in GRF for one agent and for several."""
        spaces, _ = _spaces()
        if self.squeeze:
            return spaces.Discrete(self.action_count)
        return spaces.MultiDiscrete([self.action_count] * self.agents)

    @property
    def observation_space(self) -> Any:
        """The observation space of the batch, `single_observation_space` for each environment, or
        `None` for `raw`."""
        single = self.single_observation_space
        return None if single is None else _spaces()[1](single, self.environment_count)

    @property
    def action_space(self) -> Any:
        """The action space of the batch, `single_action_space` for each environment."""
        return _spaces()[1](self.single_action_space, self.environment_count)

    def _raw_of(self, name):
        """Returns the raw tensors of a buffer, as `raw` returns them."""
        if self._raw_fields is None:
            raise ConfigurationError('expected a batch with the raw representation or `with_raw`')
        return self._raw(name)

    def raw(self) -> dict[str, torch.Tensor]:
        """Returns the `raw` representation of the last observation of every agent.

        The tensors view buffers of the batch, which the next call to `reset` or `step` overwrites.

        Raises:
            ConfigurationError: If the batch writes no raw representation. It is a `ValueError`.
        """
        return self._raw_of('raw')

    def final_raw(self) -> dict[str, torch.Tensor]:
        """Returns the `raw` representation of the step that ended each episode, as `raw` does.

        Raises:
            ConfigurationError: If the batch writes no raw representation. It is a `ValueError`.
        """
        return self._raw_of('final_raw')

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        """Runs one step of every environment.

        On the GPU the step runs on the current stream of PyTorch and does not wait for the device.
        Under `torch.compile` it traces as the operator `cuju::step`, so a compiled function that
        steps the batch needs no graph break, and `mode='reduce-overhead'` may capture it.

        Args:
            actions: `[environment_count, agents]` indices into the action set. As in GRF, an index
                from `-action_count` to `-1` counts from the end. A tensor on the batch's device
                takes no copy. Actions in host memory, pinned or not, may be reused as soon as
                the call returns, since a GPU batch copies pinned ones first.

        Returns:
            The observation, the reward `[environment_count, agents]`, the done flags
            `[environment_count]` and the information, which maps `final_observation` to the
            observation of the step that ended each episode and `score_reward` to GRF's
            `info['score_reward']`, the left team's scoring reward, `[environment_count]` `int32`.

        Raises:
            TypeError: If the actions are not integers.
            ArgumentError: If the actions are not `[environment_count, agents]`, or
                `[environment_count]` for one agent. It is a `ValueError`.
            ActionRangeError: If an action in host memory, or with `check_actions` or `debug` in
                GPU memory, lies outside `[-action_count, action_count)`, and then no environment
                steps. It is an `IndexError`.
            DeviceError: If the device fails in a `debug` batch. It is a `RuntimeError`.
            UndefinedPathError: If an environment reaches a path that GRF leaves undefined. It is a
                `RuntimeError`. On the GPU the device traps, which ends the CUDA context of the
                process, so the next CUDA call reports it and the process must restart.
            LibraryError: If the library fails to run the step. It is a `RuntimeError`.
        """
        # Under `torch.compile` the step goes through the operator `cuju::step`, without a break.
        if _CUSTOM_OPERATORS and torch.compiler.is_compiling():
            return self._compiled_step(actions)
        shape = (self.environment_count, self.agents)
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        # GRF: an action must be an integer.
        if actions.is_floating_point() or actions.is_complex():
            raise TypeError(f'expected integer action indices, found {actions.dtype}')
        # With one agent, `[environment_count]` actions serve as well, as `squeeze` returns them.
        if tuple(actions.shape) != shape and not (
            self.agents == 1 and tuple(actions.shape) == (self.environment_count,)
        ):
            raise ArgumentError(
                f'expected actions of the shape {shape}, [environment_count, agents], found '
                f'{tuple(actions.shape)}'
            )
        if actions.device == self._torch_device:
            contiguous_actions = actions.to(torch.int32).reshape(shape).contiguous()
            on_device = 1
            if self._checks_device_actions:
                self._check_device_actions(contiguous_actions)
        else:
            contiguous_actions = self._staged(_host(actions, torch.int32).reshape(shape))
            on_device = int(self.device == 'cpu')
        # GRF: an action indexes a Python list, so one outside `[-action_count, action_count)`
        # raises. The library checks host actions only, since a check on the GPU would wait for it.
        with self._on_device():
            status = self._library.cuju_environment_step(
                self._handle(), contiguous_actions.data_ptr(), on_device, self._stream(), 0
            )
        if status == -2:
            raise ActionRangeError(self._library.cuju_last_error().decode())
        self._check(status)
        self._wait_in_debug()
        observation, reward, done, final = self._outputs()
        if self._score_reward is None:
            self._score_reward = self._buffer(
                'score_reward', (self.environment_count,), torch.int32
            )
        return (
            observation,
            reward,
            done,
            {'final_observation': final, 'score_reward': self._score_reward},
        )

    def _check_device_actions(self, actions):
        """Raises `ActionRangeError` for an action in GPU memory outside the set, before a step."""
        with self._on_device():
            # The check reads its result on the host, which a CUDA graph capture cannot wait for.
            if torch.cuda.is_current_stream_capturing():
                return
            # As the library's check of host actions, the agent slots past an environment's agents
            # go unchecked.
            if self._agent_mask_on_device is None:
                self._agent_mask_on_device = self.agent_mask.to(self._torch_device)
            count = self.action_count
            outside = ((actions < -count) | (actions >= count)) & self._agent_mask_on_device
            if not bool(outside.any()):
                return
            environment, agent = (int(index) for index in outside.nonzero()[0])
            raise ActionRangeError(
                f'expected action indices from {-count} to {count - 1}, found '
                f'{int(actions[environment, agent])} on the device for agent {agent} of '
                f'environment {environment}'
            )

    def _wait_in_debug(self):
        """Waits for the device in a `debug` GPU batch, so that a failure raises at its call."""
        if not self.debug or self.device == 'cpu':
            return
        with self._on_device():
            if torch.cuda.is_current_stream_capturing():
                return
            self._check(self._library.cuju_environment_synchronize(self._handle(), self._stream()))

    def _environment_index(self, index):
        """Returns `index` as an integer, or raises `EnvironmentIndexError` outside the batch."""
        index = operator.index(index)
        if not 0 <= index < self.environment_count:
            raise EnvironmentIndexError(
                f'expected an environment from 0 to {self.environment_count - 1}, found {index}'
            )
        return index

    def _check_stack(self, stacked):
        """Raises `StateError` unless `stacked` matches the batch, as GRF pickles its stack."""
        if self.stacked and not stacked:
            raise StateError(
                'expected stacked=True for the state of a batch that stacks its observations, '
                "whose state holds GRF's FrameStack"
            )
        if stacked and not self.stacked:
            raise StateError('expected a batch with stacked=True for a stacked state')

    def _frame_shape(self):
        """Returns the shape of one agent's frame of the stack and its type."""
        shape = self._observation_shape[2:]
        dtype = 'uint8' if self._observation_type == torch.uint8 else 'float32'
        return (*shape[:-1], shape[-1] // _frame_stack.FRAMES), dtype

    def step_async(self, actions: Any) -> None:
        """Starts a step of every environment and returns at once. `step_wait` ends the step.

        On the CPU the batch steps in a thread of its own, with a copy of the actions, while the
        caller goes on, for example to run a policy on the GPU. On the GPU the step runs on the
        current stream, as `step` does. Until `step_wait` returns, the outputs are not ready, and
        any other call on the batch waits for the step first.

        Args:
            actions: As for `step`, and the caller may reuse them at once.

        Raises:
            TypeError, ArgumentError, ActionRangeError: As `step` raises them, before the step
                starts.
            RuntimeError: If a step is already in flight, or the library fails to start the step.
        """
        if self._step_in_flight:
            raise RuntimeError('a step is in flight; call step_wait before the next step_async')
        shape = (self.environment_count, self.agents)
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        if actions.is_floating_point() or actions.is_complex():
            raise TypeError(f'expected integer action indices, found {actions.dtype}')
        if tuple(actions.shape) != shape and not (
            self.agents == 1 and tuple(actions.shape) == (self.environment_count,)
        ):
            raise ArgumentError(
                f'expected actions of the shape {shape}, [environment_count, agents], found '
                f'{tuple(actions.shape)}'
            )
        if actions.device == self._torch_device:
            contiguous_actions = actions.to(torch.int32).reshape(shape).contiguous()
            on_device = 1
            if self._checks_device_actions:
                self._check_device_actions(contiguous_actions)
        else:
            contiguous_actions = self._staged(_host(actions, torch.int32).reshape(shape))
            on_device = int(self.device == 'cpu')
        with self._on_device():
            status = self._library.cuju_environment_step_async(
                self._handle(), contiguous_actions.data_ptr(), on_device, self._stream()
            )
        if status == -2:
            raise ActionRangeError(self._library.cuju_last_error().decode())
        self._check(status)
        self._wait_in_debug()
        self._step_in_flight = True

    def step_wait(self) -> tuple[Any, Any, Any, dict[str, Any]]:
        """Waits for the step that `step_async` started and returns what `step` returns.

        Raises:
            RuntimeError: If no step is in flight, or the step failed as `step` would.
        """
        if not self._step_in_flight:
            raise RuntimeError('no step is in flight; call step_async first')
        self._step_in_flight = False
        self._check(self._library.cuju_environment_step_wait(self._handle()))
        observation, reward, done, final = self._outputs()
        return (
            observation,
            reward,
            done,
            {'final_observation': final, 'score_reward': self._score_reward},
        )

    def _wait_for_device(self):
        """Waits for every stream of the device, since state copies bypass PyTorch's streams."""
        if self.device == 'cuda':
            torch.cuda.synchronize(self._torch_device)

    def get_state(self, index: int, stacked: bool = False) -> bytes:
        """Returns the state of one environment in the byte format of GRF.

        The bytes hold the length of a pickle as `int32`, the pickle of the episode logic and the
        serialized match.

        Args:
            index: The index of the environment.
            stacked: Whether the pickle also holds GRF's `FrameStack`, as the state of GRF with
                `stacked=True` does. It is a `deque` of the last four observations as numpy arrays,
                each as GRF's wrappers return it, such as `[agents, 115]` or `[115]` for one agent.
                It needs numpy, and a batch with `stacked=True`.

        Returns:
            The bytes, which `set_state` here or in GRF reads.

        Raises:
            EnvironmentIndexError: If the index is out of range. It is an `IndexError` and a
                `RuntimeError`.
            StateError: If `stacked` differs from the batch's `stacked`. It is a `ValueError` and a
                `RuntimeError`.
        """
        index = self._environment_index(index)
        self._check_stack(stacked)
        episode = (ctypes.c_int * (3 + _MAXIMUM_AGENTS))()
        size = ctypes.c_ulonglong()
        self._wait_for_device()
        with self._on_device():
            pointer = self._library.cuju_environment_get_state(
                self._handle(), index, ctypes.byref(size), episode
            )
        if not pointer:
            raise _from_library(self._library.cuju_last_error().decode())
        match = ctypes.string_at(pointer, size.value)
        collected = list(episode[3 : 3 + self.agents]) if self._checkpoints else None
        frames = None
        if stacked:
            agents = int(self.left_agents[index] + self.right_agents[index])
            stack = self._buffer(
                self._observation_name, self._observation_shape, self._observation_type
            )[index, :agents].cpu()
            frames = _frame_stack.pickled_frames(_frame_stack.frames_of(stack, agents))
        data = _pickle_episode(list(episode[:3]), collected, frames)
        return struct.pack('<i', len(data)) + data + match

    def set_state(self, index: int, state: bytes, stacked: bool = False) -> None:
        """Sets one environment to a state in the byte format of GRF.

        The observation and the sticky actions follow from the state. The reward is 0 and the done
        flag is clear until the next step.

        Args:
            index: The index of the environment.
            state: The bytes that `get_state` of this package or of GRF returned.
            stacked: Whether the state holds GRF's `FrameStack`, as `get_state(stacked=True)` here
                or a GRF with `stacked=True` writes it, and its four frames become the observation
                stack. It needs numpy, and a batch with `stacked=True`.

        Raises:
            EnvironmentIndexError: If the index is out of range. It is an `IndexError` and a
                `RuntimeError`.
            StateError: If the bytes hold no pickle of the episode logic or do not load, the
                state belongs to another scenario or to other difficulties, `stacked` differs from
                the batch's, or a stacked state holds no four frames of the batch's observation.
                It is a `ValueError` and a `RuntimeError`.
        """
        index = self._environment_index(index)
        self._check_stack(stacked)
        if len(state) < 4 or not 0 < struct.unpack_from('<i', state)[0] <= len(state) - 4:
            raise StateError(
                f'expected the bytes of a state: a pickle length, the pickle and the match, '
                f'found {len(state)} bytes that hold none'
            )
        (length,) = struct.unpack_from('<i', state)
        try:
            unpickler = _StateUnpickler(io.BytesIO(state[4 : 4 + length]))
            unpickler.stacked = stacked
            unpickled = unpickler.load()
            logic = unpickled['FootballEnvCore'].__dict__
        except (pickle.UnpicklingError, KeyError, AttributeError, EOFError) as error:
            raise StateError(
                f'expected the pickle of the episode logic in the state, found {error}'
            ) from error
        episode = (ctypes.c_int * (3 + _MAXIMUM_AGENTS))(
            logic['previous_score_diff'], logic['previous_game_mode'], logic['prev_ball_owned_team']
        )
        # GRF: the checkpoint reward reads its counts from the state.
        if self._checkpoints and 'CheckpointRewardWrapper' not in unpickled:
            raise StateError('expected the checkpoints of the checkpoint reward in the state')
        for agent, count in unpickled.get('CheckpointRewardWrapper', {}).items():
            episode[3 + agent] = count
        stack = None
        if stacked and 0 <= index < self.environment_count:
            # GRF: `FrameStack.set_state` takes the deque of the state as its observations.
            if 'FrameStack' not in unpickled:
                raise StateError("expected GRF's FrameStack in the stacked state")
            shape, dtype = self._frame_shape()
            agents = int(self.left_agents[index] + self.right_agents[index])
            try:
                stack = _frame_stack.stack_bytes(unpickled['FrameStack'], agents, shape, dtype)
            except ValueError as error:
                raise StateError(str(error)) from error
        match = state[4 + length :]
        self._wait_for_device()
        with self._on_device():
            status = self._library.cuju_environment_set_state(
                self._handle(), index, match, len(match), episode
            )
        if status != 0:
            message = self._library.cuju_last_error().decode()
            if 'found another' in message:
                # GRF: a state loads only where the constant part of the scenario is the same.
                message += (
                    '; a state loads only into an environment of the same scenario, agent counts, '
                    'episode end flags, duration, ball position, formation and difficulties'
                )
            error = _from_library(message)
            raise StateError(message) if type(error) is LibraryError else error
        if stack is not None:
            with self._on_device():
                self._check(
                    self._library.cuju_environment_set_stack(
                        self._handle(), index, self._observation_name.encode(), stack, len(stack)
                    )
                )


def memory_estimate(assets: str | list[str], **options: Any) -> int:
    """Returns the bytes that `BatchEnvironment(assets, **options)` requests, without creating it.

    The count is what `memory_bytes` of that batch reports right after its creation, its buffers,
    worlds and tables, in device memory on the GPU and in host memory on the CPU. The allocators
    round each request up. It reads the bundles and the animation tables, as the creation does, so
    it takes a fraction of a second, and it needs the library of the device, and on the GPU a CUDA
    device, as the creation does.

    The first batch of the process on a GPU also makes the driver reserve the local memory of the
    kernels for every thread that the GPU runs at once, and holds it until the process ends. The
    result is an `int` whose attribute `local_memory_bytes` gives that reservation, from the local
    memory of the kernels and the size of the device. It is 0 once a batch was created on the
    device, and always 0 on the CPU, so that a batch created next takes the result plus
    `local_memory_bytes`.

    Args:
        assets: As `BatchEnvironment` takes them.
        **options: Any other argument of `BatchEnvironment`, such as `environment_count`,
            `representation` or `device`.

    Returns:
        The bytes of memory that the batch would request, with the attribute
        `local_memory_bytes`.

    Raises:
        LibraryError: If the library refuses the configuration. The other errors are those that
            `BatchEnvironment` raises for the same arguments, except the check of `squeeze`
            against the agents, which follows the creation.

    Examples:
        >>> import cuju
        >>> per_environment = cuju.memory_estimate(
        ...     '5_vs_5_l4_r0', environment_count=2, device='cpu'
        ... ) - cuju.memory_estimate('5_vs_5_l4_r0', environment_count=1, device='cpu')
    """
    batch = BatchEnvironment.__new__(BatchEnvironment)
    batch._estimate_only = True
    try:
        batch.__init__(assets, **options)
        return batch._estimated_bytes
    finally:
        batch.close()

# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks stacked states, GRF's `FrameStack` pickled beside the state and loaded back."""

import collections
import pickle
import struct
import sys
import types

import pytest

import cuju
from conftest import actions_at, tensor_bytes

np = pytest.importorskip('numpy')


@pytest.fixture
def grf_classes(monkeypatch):
    """Registers a stand-in for GRF's `EnvState`, so that `pickle` reads and writes states."""
    names = ('gfootball', 'gfootball.env', 'gfootball.env.football_env_core')
    modules = {name: types.ModuleType(name) for name in names}

    class EnvState:
        pass

    # `pickle` finds a class by its module and qualified name.
    EnvState.__module__, EnvState.__qualname__ = names[-1], 'EnvState'
    modules[names[-1]].EnvState = EnvState
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return EnvState


def split(state):
    """Returns the pickled dictionary of a state and its match bytes."""
    (length,) = struct.unpack_from('<i', state)
    return pickle.loads(state[4 : 4 + length]), state[4 + length :]


def join(unpickled, match):
    data = pickle.dumps(unpickled, protocol=2)
    return struct.pack('<i', len(data)) + data + match


def steps(batch, start, count):
    """Steps every environment of `batch` with the same actions and returns the observations."""
    seen = []
    for step in range(start, start + count):
        actions = actions_at(step, 1, batch.agents, batch.action_count).repeat(
            batch.environment_count, 1
        )
        observation, _, _, _ = batch.step(actions.to(batch.device))
        seen.append(observation.clone())
    return seen


@pytest.mark.parametrize(
    'assets, representation',
    [
        ('5_vs_5_l1_r0', 'simple115v2'),
        ('5_vs_5_l2_r1', 'simple115v2'),
        ('5_vs_5_l2_r0', 'extracted'),
    ],
)
def test_a_stacked_state_continues_the_stack(device, assets, representation):
    batch = cuju.BatchEnvironment(
        assets, environment_count=2, device=device, representation=representation, stacked=True
    )
    batch.reset()
    steps(batch, 0, 25)
    batch.set_state(1, batch.get_state(0, stacked=True), stacked=True)
    assert tensor_bytes(batch.observation()[0]) == tensor_bytes(batch.observation()[1])
    for observation in steps(batch, 25, 20):
        assert tensor_bytes(observation[0]) == tensor_bytes(observation[1])


def test_the_pickle_holds_grf_frame_stack(grf_classes):
    batch = cuju.BatchEnvironment('5_vs_5_l2_r1', environment_count=1, device='cpu', stacked=True)
    batch.reset()
    steps(batch, 0, 6)
    unpickled, _ = split(batch.get_state(0, stacked=True))
    frames = unpickled['FrameStack']
    assert isinstance(frames, collections.deque) and frames.maxlen == 4 and len(frames) == 4
    # GRF: `FrameStack` concatenates its frames along the last axis, the oldest first.
    stack = np.concatenate(list(frames), axis=-1)
    assert all(frame.dtype == np.float32 and frame.shape == (3, 115) for frame in frames)
    assert stack.tobytes() == tensor_bytes(batch.observation()[0])
    assert isinstance(unpickled['FootballEnvCore'], grf_classes)


def test_one_agent_frames_drop_the_agent_axis(grf_classes):
    batch = cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=1, device='cpu', stacked=True)
    batch.reset()
    unpickled, _ = split(batch.get_state(0, stacked=True))
    assert [frame.shape for frame in unpickled['FrameStack']] == [(115,)] * 4


def test_a_stack_from_elsewhere_sets_the_observation(grf_classes):
    batch = cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=1, device='cpu', stacked=True)
    batch.reset()
    steps(batch, 0, 3)
    unpickled, match = split(batch.get_state(0, stacked=True))
    # A deque as GRF's reset leaves it, the same observation four times.
    frame = np.arange(115, dtype=np.float32)
    unpickled['FrameStack'] = collections.deque([frame] * 4, maxlen=4)
    batch.set_state(0, join(unpickled, match), stacked=True)
    assert tensor_bytes(batch.observation()[0]) == np.tile(frame, 4).tobytes()


def test_malformed_stacks(grf_classes):
    batch = cuju.BatchEnvironment(
        '5_vs_5_l1_r0', environment_count=1, device='cpu', stacked=True, representation='extracted'
    )
    batch.reset()
    unpickled, match = split(batch.get_state(0, stacked=True))
    frames = list(unpickled['FrameStack'])
    unpickled['FrameStack'] = collections.deque(frames[:3], maxlen=4)
    with pytest.raises(ValueError, match='frames'):
        batch.set_state(0, join(unpickled, match), stacked=True)
    unpickled['FrameStack'] = collections.deque([frame[:, :10] for frame in frames], maxlen=4)
    with pytest.raises(ValueError, match='shape'):
        batch.set_state(0, join(unpickled, match), stacked=True)
    # GRF's minimap marks a cell with 255 and leaves the others 0.
    marked = [frame.copy() for frame in frames]
    marked[0][0, 0, 0] = 7
    unpickled['FrameStack'] = collections.deque(marked, maxlen=4)
    with pytest.raises(RuntimeError, match='minimap'):
        batch.set_state(0, join(unpickled, match), stacked=True)
    del unpickled['FrameStack']
    with pytest.raises(ValueError, match='FrameStack'):
        batch.set_state(0, join(unpickled, match), stacked=True)


def test_the_stack_option_must_match():
    stacked = cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=1, device='cpu', stacked=True)
    stacked.reset()
    with pytest.raises(ValueError, match='stacked=True'):
        stacked.get_state(0)
    plain = cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=1, device='cpu')
    plain.reset()
    with pytest.raises(ValueError, match='stacked=True'):
        plain.get_state(0, stacked=True)
    with pytest.raises(ValueError, match='stacked=True'):
        stacked.set_state(0, plain.get_state(0))


def test_an_unstacked_state_is_unchanged():
    # The pickle without a stack keeps its bytes, as the digests read them.
    batch = cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=1, device='cpu')
    batch.reset()
    state = batch.get_state(0)
    (length,) = struct.unpack_from('<i', state)
    assert b'FrameStack' not in state[4 : 4 + length]

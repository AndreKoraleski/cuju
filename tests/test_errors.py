# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that each failure raises its cuju class, which is also the built-in type it replaced."""

import pytest
import torch

import cuju
from cuju import errors, scenarios


@pytest.fixture(scope='module')
def batch():
    batch = cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=2, device='cpu')
    batch.reset()
    yield batch
    batch.close()


def test_every_class_is_an_error():
    for name in errors.__all__:
        assert issubclass(getattr(cuju, name), cuju.Error)
    assert issubclass(scenarios.ScenarioError, cuju.ConfigurationError)


def test_an_environment_out_of_range_is_an_index_error(batch):
    for call in (lambda: batch.get_state(99), lambda: batch.set_state(-1, b'')):
        with pytest.raises(cuju.EnvironmentIndexError) as raised:
            call()
        assert isinstance(raised.value, IndexError) and isinstance(raised.value, RuntimeError)
    with pytest.raises(IndexError, match='from 0 to 1, found 99'):
        batch.get_state(99)


def test_an_unknown_device_is_a_value_error():
    for options in ({'device': 'tpu'}, {'device': 'cpu', 'representation': 'pixels'}):
        with pytest.raises(cuju.ChoiceError) as raised:
            cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=1, **options)
        assert isinstance(raised.value, ValueError) and isinstance(raised.value, KeyError)
        # Unlike a `KeyError`, the message prints without quotes.
        assert str(raised.value).startswith('expected the ')
    with pytest.raises(ValueError):
        cuju.library_path('tpu')


def test_arguments_of_a_call(batch):
    with pytest.raises(cuju.ArgumentError) as raised:
        batch.step(torch.zeros((2, 2), dtype=torch.int32))
    assert isinstance(raised.value, ValueError)
    with pytest.raises(cuju.ActionRangeError) as raised:
        batch.step(torch.full((2, 1), 19, dtype=torch.int32))
    assert isinstance(raised.value, IndexError)
    with pytest.raises(cuju.ArgumentError):
        batch.reset(seeds=[1, 2, 3])


def test_states(batch):
    with pytest.raises(cuju.StateError) as raised:
        batch.set_state(0, b'\x00' * 8)
    assert isinstance(raised.value, ValueError) and isinstance(raised.value, RuntimeError)
    other = cuju.BatchEnvironment('academy_empty_goal_close', environment_count=1, device='cpu')
    other.reset()
    # The library refuses a state of another scenario.
    with pytest.raises(cuju.StateError, match='found another'):
        batch.set_state(0, other.get_state(0))
    other.close()


def test_options_files_and_closed_batches():
    with pytest.raises(cuju.ConfigurationError) as raised:
        cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=0, device='cpu')
    assert isinstance(raised.value, ValueError)
    with pytest.raises(cuju.MissingFileError) as raised:
        cuju.BatchEnvironment('no_such_scenario.py', device='cpu')
    assert isinstance(raised.value, FileNotFoundError)
    closed = cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=1, device='cpu')
    closed.close()
    with pytest.raises(cuju.ClosedError) as raised:
        closed.reset()
    assert isinstance(raised.value, RuntimeError)
    with cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=1, device='cpu') as scoped:
        scoped.reset()
    with pytest.raises(cuju.ClosedError):
        scoped.reset()


def test_library_messages_map_onto_classes():
    undefined = errors._from_library('environment 3 reached a path that GRF leaves undefined: x')
    assert type(undefined) is cuju.UndefinedPathError
    assert type(errors._from_library('cudaMalloc(&pointer): out of memory')) is cuju.DeviceError
    assert type(errors._from_library('expected a bundle from 0 to 1')) is cuju.LibraryError

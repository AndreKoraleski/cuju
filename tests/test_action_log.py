# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the digests of outputs, and that a saved action log plays its run again bit for bit."""

import json
import os
import zipfile

import pytest
import torch

import cuju
from conftest import CUDA, actions_at, play_digest, tensor_bytes

DIGESTS = os.path.join(os.path.dirname(os.path.abspath(cuju.__file__)), 'data', 'digests.json')
with open(DIGESTS, encoding='utf-8') as handle:
    EXPECTED = json.load(handle)


def test_digest_is_the_digest_of_the_self_check():
    entry = EXPECTED['cases']['academy_empty_goal_close/simple115v2']
    value = cuju.digest(
        entry['assets'],
        'cpu',
        EXPECTED['steps'],
        seed=EXPECTED['seed'],
        environment_count=EXPECTED['environments'],
        **entry['options'],
    )
    assert value == entry['digest']
    assert cuju.digest('5_vs_5_l1_r1', steps=30, with_raw=True) == play_digest(
        '5_vs_5_l1_r1', 'cpu', 30, with_raw=True
    )


def test_digest_hashes_dictionaries_by_sorted_keys():
    first, second = cuju.Digest(), cuju.Digest()
    first.update({'b': torch.ones(2), 'a': torch.zeros(3, dtype=torch.int32)})
    second.update(torch.zeros(3, dtype=torch.int32))
    second.update(torch.ones(2))
    assert first.hexdigest() == second.hexdigest()
    with pytest.raises(TypeError):
        first.update(1.5)


def run(target, batch, steps=50):
    """Makes every call that a log logs on `target`, a log or a batch, and returns the outputs."""
    outputs = [tensor_bytes(target.reset(seeds=[11, 12, 13]))]
    for step in range(steps):
        observation, reward, done, _ = target.step(actions_at(step, 3, batch.agents, 19))
        outputs += [tensor_bytes(observation), tensor_bytes(reward), tensor_bytes(done)]
        if step == 20:
            target.reset_done()
        if step == 30:
            target.set_state(2, batch.get_state(0))
    return outputs


def test_an_action_log_replays_its_run(tmp_path):
    log = cuju.ActionLog('5_vs_5_l2_r1', environment_count=3, device='cpu', auto_reset=False)
    logged = run(log, log.batch)
    log.save(tmp_path / 'run.zip')
    assert cuju.replay(tmp_path / 'run.zip', threads=2) == log.hexdigest()
    # The calls of a log return what the batch's return.
    batch = cuju.BatchEnvironment(
        '5_vs_5_l2_r1', environment_count=3, device='cpu', auto_reset=False
    )
    assert run(batch, batch) == logged


def test_machine_options_stay_out_of_a_log(tmp_path):
    machine = {'pin_threads': True, 'numa': True, 'debug': True, 'check_actions': True}
    log = cuju.ActionLog(
        '5_vs_5_l2_r1', environment_count=3, device='cpu', allocator='torch', **machine
    )
    run(log, log.batch, steps=10)
    log.save(tmp_path / 'run.zip')
    with zipfile.ZipFile(tmp_path / 'run.zip') as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    header = json.loads(files['log.json'])
    # None of them changes an output, so the replaying machine chooses its own.
    assert not set(header['options']) & {*machine, 'allocator', 'pin_memory'}
    assert cuju.replay(tmp_path / 'run.zip', threads=2, **machine) == log.hexdigest()
    with pytest.raises(ValueError, match='change no output'):
        cuju.replay(tmp_path / 'run.zip', seed=3)
    # A log of an earlier version, which holds them among its options, replays without them.
    header['options'].update({**machine, 'pin_memory': True, 'allocator': 'torch'})
    files['log.json'] = json.dumps(header).encode()
    with zipfile.ZipFile(tmp_path / 'old.zip', 'w') as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    assert cuju.replay(tmp_path / 'old.zip') == log.hexdigest()
    assert cuju.replay(tmp_path / 'old.zip', pin_threads=False) == log.hexdigest()
    log.close()


def test_a_changed_action_fails_its_replay(tmp_path):
    log = cuju.ActionLog('academy_empty_goal_close_l1_r0', environment_count=3, device='cpu')
    run(log, log.batch, steps=10)
    log.save(tmp_path / 'run.zip')
    with zipfile.ZipFile(tmp_path / 'run.zip') as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    # The first environment's agent runs right from step 2 on.
    actions = bytearray(files['actions.bin'])
    for step in range(2, 10):
        actions[4 * 3 * step] = 5
    files['actions.bin'] = bytes(actions)
    with zipfile.ZipFile(tmp_path / 'changed.zip', 'w') as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    # Call 0 is the reset, so step 2 is call 3.
    with pytest.raises(ValueError, match=r'the outputs of call \d+ \(step\) differ') as raised:
        cuju.replay(tmp_path / 'changed.zip')
    assert int(str(raised.value).split()[4]) >= 3


@pytest.mark.skipif(not CUDA, reason='no CUDA device or library')
def test_a_gpu_log_replays_on_the_cpu(tmp_path):
    log = cuju.ActionLog('5_vs_5_l2_r1', environment_count=3, device='cuda', auto_reset=False)
    run(log, log.batch)
    log.save(tmp_path / 'run.zip')
    assert cuju.replay(tmp_path / 'run.zip', device='cpu') == log.hexdigest()

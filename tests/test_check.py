# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks `python -m cuju.check`, its diagnostics, and the log of `CUJU_LOG=1`."""

import logging
import os
import subprocess
import sys

import pytest
import torch

import cuju
from conftest import CUDA, play_digest
from cuju import check

CASE = '5_vs_5/simple115v2'


def test_its_digest_is_that_of_the_self_check():
    entry = check.load_cases()[CASE]
    assert check.run_case(entry, 'cpu', None) == entry['digest']
    assert play_digest(entry['assets'], 'cpu', entry['steps']) == entry['digest']


def test_the_long_cases_are_extra():
    short, every = check.load_cases(), check.load_cases(long=True)
    assert set(short) < set(every)
    assert all(every[name]['steps'] > short[CASE]['steps'] for name in set(every) - set(short))


def test_a_case_that_matches_passes(capsys):
    assert check.main(['--device', 'cpu', '--case', CASE]) == 0
    out = capsys.readouterr().out
    assert f'{CASE}: ok' in out and 'PASSED' in out
    assert 'files match data/SHA256SUMS' in out


def test_a_different_digest_fails(capsys, monkeypatch):
    cases = check.load_cases()
    cases[CASE]['digest'] = '0' * 64
    monkeypatch.setattr(check, 'load_cases', lambda long=False: cases)
    assert check.main(['--device', 'cpu', '--case', CASE]) == 1
    out = capsys.readouterr().out
    assert f'{CASE}: DIFFERENT' in out and 'FAILED' in out


def test_a_damaged_data_file_fails(monkeypatch):
    monkeypatch.setattr(check._checksums, 'file_sha256', lambda path: '0' * 64)
    assert check.check_data() and all('differs' in problem for problem in check.check_data())


def test_an_absent_backend_fails_only_when_named(capsys, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(check, 'load_cases', lambda long=False: {})
    assert check.main([]) == 0
    assert check.main(['--device', 'cuda']) == 1
    assert 'the GPU cases did not run' in capsys.readouterr().out


def test_the_diagnostics_name_the_setup():
    text = '\n'.join(check.diagnostics())
    for part in ('cuju', 'torch', 'CPU:', 'flags:', 'cpu library:', 'cuda library:'):
        assert part in text
    if torch.cuda.is_available():
        assert 'compute capability' in text


@pytest.mark.skipif(not CUDA, reason='no CUDA device or library')
def test_the_cuda_library_holds_code_for_this_gpu():
    architectures = check.cuda_architectures(cuju.library_path('cuda'))
    assert architectures
    capability = torch.cuda.get_device_capability()
    assert 'build cuju' not in check.code_for(capability, architectures)


def test_the_code_a_gpu_runs():
    wheel = ['sm_75', 'sm_80', 'sm_86', 'sm_89']
    assert check.code_for((8, 6), wheel) == 'runs the machine code of sm_86'
    assert check.code_for((8, 7), wheel) == 'runs the machine code of sm_86'
    assert 'build cuju' in check.code_for((9, 0), wheel)
    assert 'PTX of compute_86' in check.code_for((9, 0), ['sm_86', 'compute_86'])
    assert check.cuda_architectures(cuju.library_path('cpu')) in ([], None)


def test_the_module_runs():
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run(
        [sys.executable, '-m', 'cuju.check', '--list'],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert CASE in result.stdout


def test_cuju_log_logs_each_creation(caplog, monkeypatch):
    monkeypatch.setenv('CUJU_LOG', '1')
    with caplog.at_level(logging.INFO, logger='cuju'):
        cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=2, device='cpu', threads=1).close()
    (record,) = [record for record in caplog.records if record.name == 'cuju']
    assert "created 2 environments of ['5_vs_5_l4_r0']" in record.getMessage()
    assert 'the CPU with 1 threads' in record.getMessage()


def test_cuju_log_measures_the_memory_of_the_batch_gpu(monkeypatch):
    from cuju._batch import log

    monkeypatch.setenv('CUJU_LOG', '1')
    asked = []

    def free_memory(device=None):
        asked.append(device)
        return 1024, 2048

    # A batch on GPU 3, which is not the current device, reads the free memory of GPU 3.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', free_memory)
    _, free = log.start('cuda', torch.device('cuda', 3))
    assert asked == [3] and free == 1024


def test_no_log_by_default(caplog, monkeypatch):
    monkeypatch.delenv('CUJU_LOG', raising=False)
    with caplog.at_level(logging.INFO, logger='cuju'):
        cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=2, device='cpu').close()
    assert not [record for record in caplog.records if record.name == 'cuju']

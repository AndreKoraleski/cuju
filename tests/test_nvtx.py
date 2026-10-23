# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that a build with NVTX ranges changes no output, with or without a profiler."""

import os
import subprocess
import sys

from conftest import play_digest

# A child process plays a batch as `play_digest` does, with NVTX not yet loaded.
CHILD = """
import sys
sys.path.insert(0, sys.argv[1])
from conftest import play_digest
print(play_digest('5_vs_5_l4_r0', sys.argv[2], 10))
"""


def play_in_child(device, environment):
    child = subprocess.run(
        [sys.executable, '-c', CHILD, os.path.dirname(__file__), device],
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert child.returncode == 0, child.stderr[-2000:]
    return child.stdout.split()[-1]


def test_a_fresh_process_plays_the_same_outputs(device):
    assert play_in_child(device, dict(os.environ)) == play_digest('5_vs_5_l4_r0', device, 10)


def test_a_missing_profiler_changes_no_output():
    # NVTX looks for a tool at this path, finds none and records nothing. The CPU library alone,
    # since on the GPU the outcome also depends on how CUDA and PyTorch treat the variable.
    environment = dict(os.environ, NVTX_INJECTION64_PATH='/nonexistent/libcuju-no-tool.so')
    assert play_in_child('cpu', environment) == play_digest('5_vs_5_l4_r0', 'cpu', 10)

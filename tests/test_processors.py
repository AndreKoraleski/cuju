# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that `threads=0` takes no more threads than the CPU quota of the cgroup allows."""

import ctypes
import os

import pytest

import cuju

V2_MOUNT = (
    '30 23 0:26 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime shared:4 - cgroup2 cgroup2 rw'
)
V1_MOUNT = (
    '40 30 0:35 {root} /sys/fs/cgroup/cpu,cpuacct ro,nosuid,nodev,noexec,relatime master:18 - '
    'cgroup cgroup rw,cpu,cpuacct'
)


@pytest.fixture(scope='module')
def quota():
    """Returns the library's reader of the quota, which reads a copy of `/proc` and `/sys`."""
    library = ctypes.CDLL(cuju.library_path('cpu'))
    library.cuju_cpu_quota.argtypes = [ctypes.c_char_p]
    return lambda root=None: library.cuju_cpu_quota(None if root is None else str(root).encode())


def write(root, path, text):
    path = root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_version_2_takes_the_smallest_level_rounded_up(tmp_path, quota):
    write(tmp_path, 'proc/self/cgroup', '0::/kubepods/pod1/container\n')
    write(tmp_path, 'proc/self/mountinfo', V2_MOUNT + '\n')
    write(tmp_path, 'sys/fs/cgroup/kubepods/cpu.max', '800000 100000\n')
    write(tmp_path, 'sys/fs/cgroup/kubepods/pod1/cpu.max', '250000 100000\n')
    write(tmp_path, 'sys/fs/cgroup/kubepods/pod1/container/cpu.max', 'max 100000\n')
    assert quota(tmp_path) == 3


def test_version_2_inside_a_namespace(tmp_path, quota):
    write(tmp_path, 'proc/self/cgroup', '0::/\n')
    write(tmp_path, 'proc/self/mountinfo', V2_MOUNT + '\n')
    write(tmp_path, 'sys/fs/cgroup/cpu.max', '50000 100000\n')
    assert quota(tmp_path) == 1


def test_version_1_mounted_at_the_container_cgroup(tmp_path, quota):
    write(tmp_path, 'proc/self/cgroup', '5:cpu,cpuacct:/docker/abc\n4:memory:/docker/abc\n')
    write(tmp_path, 'proc/self/mountinfo', V1_MOUNT.format(root='/docker/abc') + '\n')
    write(tmp_path, 'sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us', '150000\n')
    write(tmp_path, 'sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us', '100000\n')
    assert quota(tmp_path) == 2


def test_no_quota(tmp_path, quota):
    write(tmp_path, 'proc/self/cgroup', '0::/user.slice\n4:cpu,cpuacct:/user.slice\n')
    write(tmp_path, 'proc/self/mountinfo', V2_MOUNT + '\n' + V1_MOUNT.format(root='/') + '\n')
    write(tmp_path, 'sys/fs/cgroup/user.slice/cpu.max', 'max 100000\n')
    write(tmp_path, 'sys/fs/cgroup/cpu,cpuacct/user.slice/cpu.cfs_quota_us', '-1\n')
    write(tmp_path, 'sys/fs/cgroup/cpu,cpuacct/user.slice/cpu.cfs_period_us', '100000\n')
    assert quota(tmp_path) == 0
    assert quota(tmp_path / 'missing') == 0


@pytest.mark.skipif(not hasattr(os, 'sched_getaffinity'), reason='no affinity mask')
def test_threads_zero_honours_the_quota(quota):
    processors = len(os.sched_getaffinity(0))
    limit = quota()
    batch = cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=1, device='cpu', threads=0)
    assert batch.threads == (min(processors, limit) if limit else processors)
    batch.close()

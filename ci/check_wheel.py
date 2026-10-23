# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks wheels of cuju: the license files, and the libraries each one holds.

Usage: `python ci/check_wheel.py [--cpu-only] <wheel>...`. A wheel of `cuju` holds the CPU and the
CUDA library; with `--cpu-only`, one of `cuju-cpu` holds the CPU library and no CUDA library.
"""

import sys
import zipfile

LICENSES = [
    'LICENSE',
    'NOTICE',
    'LICENSES/LGPL-2.1-or-later.txt',
    'LICENSES/MIT.txt',
    'LICENSES/SunPro.txt',
    'LICENSES/BSL-1.0.txt',
    'LICENSES/BSD-3-Clause.txt',
]
CPU_LIBRARIES = ('libcuju_cpu.so', 'cuju_cpu.dll', 'libcuju_cpu.dylib')
CUDA_LIBRARIES = ('libcuju_cuda.so', 'cuju_cuda.dll')


def problems(path, cpu_only):
    """Returns what is wrong with the wheel at `path`, or an empty list."""
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
    found = [
        f'missing {name}'
        for name in LICENSES
        if not any(n.endswith(f'licenses/{name}') for n in names)
    ]
    libraries = {name.rpartition('/')[2] for name in names if name.startswith('cuju/')}
    if not libraries & set(CPU_LIBRARIES):
        found.append('missing the CPU library')
    cuda = sorted(libraries & set(CUDA_LIBRARIES))
    if cpu_only and cuda:
        found.append(f'expected no CUDA library, found {", ".join(cuda)}')
    if not cpu_only and not cuda:
        found.append('missing the CUDA library')
    return found


def main():
    arguments = sys.argv[1:]
    cpu_only = '--cpu-only' in arguments
    wheels = [argument for argument in arguments if argument != '--cpu-only']
    if not wheels:
        sys.exit('usage: python ci/check_wheel.py [--cpu-only] <wheel>...')
    failed = False
    for path in wheels:
        for problem in problems(path, cpu_only):
            print(f'{path}: {problem}')
            failed = True
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()

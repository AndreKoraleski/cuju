#!/bin/bash
# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
# Installs the CUDA compiler and Clang into the manylinux_2_28 image (AlmaLinux 8) that builds
# the wheel. Clang builds the CPU library, about 10% faster than GCC and as exact.
set -euo pipefail
# NVIDIA names its aarch64 server repository `sbsa`.
case $(uname -m) in
    x86_64) repository=x86_64 ;;
    aarch64) repository=sbsa ;;
    *) echo "no CUDA repository for $(uname -m)"; exit 1 ;;
esac
# A mirror slower than 500 kB/s gives way to the next one rather than holding the build for an hour.
cat >> /etc/dnf/dnf.conf << 'EOF'
fastestmirror=True
minrate=500k
timeout=30
max_parallel_downloads=10
EOF
dnf install -y dnf-plugins-core
dnf config-manager --add-repo \
    "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/$repository/cuda-rhel8.repo"
# NVIDIA's mirrors can serve a repository index for a moment before the files it names, so a failed
# download tries again a minute later.
for attempt in 1 2 3 4 5; do
    if dnf install -y cuda-nvcc-12-9 cuda-cudart-devel-12-9 clang; then
        exit 0
    fi
    echo "attempt $attempt of 5 failed"
    dnf clean metadata
    sleep 60
done
exit 1

# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

# The image of cuju and its Python package. Both backends are included.
# The build has no GPU, so `CUDA_ARCHITECTURES` names the targets. `-real` builds machine code
# without the PTX that a driver would compile for other GPUs. The runtime stage needs no
# CUDA toolkit, since the libraries link the CUDA runtime statically, and keeps the sdist
# in /usr/share/cuju so that the image carries the LGPL source. The self-check runs offline, as
#     docker run --rm --gpus all cuju python -m pytest -p no:cacheprovider /usr/share/cuju/tests
ARG CUDA_VERSION=12.9.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build
ARG CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real"
# The time of the commit, which dates every file of the wheel so that a rebuild gives its bytes.
ARG SOURCE_DATE_EPOCH
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install --no-install-recommends -y \
    cmake g++ python3 python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /cuju
COPY . .
RUN python3 -m venv /venv && /venv/bin/pip install --no-cache-dir build \
    && CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" /venv/bin/python -m build --outdir /dist .

FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu24.04
# PyTorch comes from this index; a site whose driver is older than PyTorch's default CUDA build
# names another, such as https://download.pytorch.org/whl/cu126.
ARG TORCH_INDEX_URL=https://pypi.org/simple
LABEL org.opencontainers.image.title="cuju" \
    org.opencontainers.image.description="Batched Google Research Football, bit for bit" \
    org.opencontainers.image.source="https://github.com/AndreKoraleski/cuju" \
    org.opencontainers.image.licenses="Apache-2.0 AND LGPL-2.1-or-later AND MIT AND SunPro AND BSL-1.0 AND BSD-3-Clause"
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install --no-install-recommends -y python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /dist /tmp/dist
COPY --from=build /cuju/tests /usr/share/cuju/tests
RUN python3 -m venv /venv \
    && /venv/bin/pip install --no-cache-dir --index-url "$TORCH_INDEX_URL" torch \
    && /venv/bin/pip install --no-cache-dir "$(ls /tmp/dist/*.whl)[test]" \
    && mv /tmp/dist/*.tar.gz /usr/share/cuju/ \
    && rm -rf /tmp/dist
ENV PATH=/venv/bin:$PATH

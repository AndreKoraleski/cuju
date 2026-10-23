// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// A read-only view of a contiguous array, through which the simulation reads the static tables.
#pragma once

#include <cstddef>
#include <cstdint>

#include "host_device.h"

namespace cuju {

template <class T>
struct Span {
    const T *pointer = nullptr;
    int32_t count = 0;
    CUJU_HOST_DEVICE size_t size() const { return (size_t)count; }
    CUJU_HOST_DEVICE bool empty() const { return count == 0; }
    CUJU_HOST_DEVICE const T &operator[](size_t index) const { return pointer[index]; }
    CUJU_HOST_DEVICE const T &at(size_t index) const { return pointer[index]; }
    CUJU_HOST_DEVICE const T *begin() const { return pointer; }
    CUJU_HOST_DEVICE const T *end() const { return pointer + count; }
    CUJU_HOST_DEVICE const T &back() const { return pointer[count - 1]; }
};

}  // namespace cuju

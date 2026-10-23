// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Float to integer conversions as GRF's x86-64 build makes them.
// Exact: NaN and out-of-range inputs give the integer indefinite value, where the device saturates.
#pragma once

#include "host_device.h"

namespace cuju {

// `cvttss2si` with a 32-bit destination.
CUJU_HOST_DEVICE int32_t FloatToInt32(float value) {
    if (!(value >= -2147483648.0f && value < 2147483648.0f)) return INT32_MIN;
    return (int32_t)value;
}

// A float to `unsigned` cast on x86-64, as `cvttss2si` with a 64-bit destination, low 32 bits kept,
// so NaN and values outside `int64_t` give 0.
CUJU_HOST_DEVICE uint32_t FloatToUnsigned32(float value) {
    if (!(value >= -9223372036854775808.0f && value < 9223372036854775808.0f)) return 0u;
    return (uint32_t)(uint64_t)(int64_t)value;
}

}  // namespace cuju

namespace cuju {

// `cvttsd2si` with a 32-bit destination.
CUJU_HOST_DEVICE int32_t DoubleToInt32(double value) {
    if (!(value > -2147483649.0 && value < 2147483648.0)) return INT32_MIN;
    return (int32_t)value;
}

}  // namespace cuju

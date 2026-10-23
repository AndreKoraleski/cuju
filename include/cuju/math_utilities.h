// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// GRF's scalar float helpers, with the comparisons and the operation order the bits depend on.
#pragma once

#include "host_device.h"

namespace cuju {

constexpr float PI = 3.1415926535897932384626433832795028841972f;

CUJU_HOST_DEVICE float clamp(float value, float min, float max) {
    if (min > value) return min;
    if (max < value) return max;
    return value;
}

CUJU_HOST_DEVICE float NormalizedClamp(float value, float min, float max) {
    float clamped = clamp(value, min, max);
    clamped = (clamped - min) / (max - min);
    return clamped;
}

CUJU_HOST_DEVICE int SignSide(float value) { return value >= 0 ? 1 : -1; }

}  // namespace cuju

namespace cuju {

// Exact: one span per step, so the result carries the rounding of each step.
CUJU_HOST_DEVICE float ModulateIntoRange(float min, float max, float value) {
    float step = max - min;
    float newValue = value;
    while (newValue < min) newValue += step;
    while (newValue > max) newValue -= step;
    return newValue;
}

}  // namespace cuju

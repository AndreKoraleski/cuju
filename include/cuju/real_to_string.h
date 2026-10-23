// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The round trip of a float through six-decimal text, as GRF stores some settings.
#pragma once

#include <cstdint>
#include <cstring>

#include "host_device.h"

namespace cuju {

// GRF: `snprintf("%f")`, then `atof` and a conversion to float. The C library prints the exact
// value rounded to six decimals, ties to even, and `strtod` returns the double nearest to it. For
// a magnitude below 2^33 that decimal times 10^6 is an integer N below 2^53, so the double is the
// IEEE division `(double)N / 1e6`. Callers pass 0 to 1, and infinity and NaN return unchanged.
CUJU_HOST_DEVICE_NO_INLINE float RoundTripSixDecimals(float value) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    const bool negative = bits >> 31;
    const int biasedExponent = (bits >> 23) & 0xff;
    uint64_t mantissa = bits & 0x7fffff;
    if (biasedExponent == 0xff) return value;
    int exponent;
    if (biasedExponent == 0) {
        exponent = -149;
    } else {
        mantissa |= 0x800000;
        exponent = biasedExponent - 150;
    }
    const uint64_t scaledMantissa = mantissa * 1000000ull;  // < 2^44
    uint64_t rounded;
    if (exponent >= 0) {
        rounded = scaledMantissa << exponent;  // A magnitude below 2^33 keeps this below 2^53.
    } else {
        const int shift = -exponent;
        if (shift >= 64) {
            rounded = 0;
        } else {
            // To the nearest integer, ties to even.
            rounded = scaledMantissa >> shift;
            const uint64_t droppedBits = scaledMantissa & ((1ull << shift) - 1),
                           half = 1ull << (shift - 1);
            if (droppedBits > half || (droppedBits == half && (rounded & 1))) rounded++;
        }
    }
    const float result = (float)((double)rounded / 1e6);
    return negative ? -result : result;
}

}  // namespace cuju

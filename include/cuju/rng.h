// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// Copyright Jens Maurer 2000-2001, Steven Watanabe 2010-2011 (Boost.Random)
// SPDX-License-Identifier: Apache-2.0 AND BSL-1.0

// GRF's RNG, Boost.Random's 32-bit Mersenne Twister and uniform float, in place in the world.
#pragma once

#include <cstdint>
#include <cstring>

#include "host_device.h"

namespace cuju {

constexpr int RNG_BYTES = 2512;

struct RNGView {
    uint8_t *bytes;

    CUJU_HOST_DEVICE uint32_t Word(int index) const {
        uint32_t value;
        memcpy(&value, bytes + 4 * index, 4);
        return value;
    }
    CUJU_HOST_DEVICE void SetWord(int index, uint32_t value) {
        memcpy(bytes + 4 * index, &value, 4);
    }
    CUJU_HOST_DEVICE uint64_t Index() const {
        uint64_t value;
        memcpy(&value, bytes + 2496, 8);
        return value;
    }
    CUJU_HOST_DEVICE void SetIndex(uint64_t value) { memcpy(bytes + 2496, &value, 8); }
    CUJU_HOST_DEVICE float Minimum() const {
        float value;
        memcpy(&value, bytes + 2504, 4);
        return value;
    }
    CUJU_HOST_DEVICE float Maximum() const {
        float value;
        memcpy(&value, bytes + 2508, 4);
        return value;
    }

    // GRF: the seed normalizes the state as Boost.Random does. The first word keeps only its top
    // bit from the recurrence, and a state of zeros gets the first word `1 << 31`.
    CUJU_HOST_DEVICE_NO_INLINE void Seed(uint32_t value) {
        const int stateSize = 624, shiftSize = 397;
        SetWord(0, value);
        for (int index = 1; index < stateSize; index++)
            SetWord(index,
                    1812433253u * (Word(index - 1) ^ (Word(index - 1) >> 30)) + (uint32_t)index);
        SetIndex(stateSize);
        const uint32_t upperMask = ~0u << 31, lowerMask = ~upperMask, twistMask = 0x9908b0dfu;
        uint32_t mixed = Word(shiftSize - 1) ^ Word(stateSize - 1);
        if (mixed & (1u << 31))
            mixed = ((mixed ^ twistMask) << 1) | 1;
        else
            mixed = mixed << 1;
        SetWord(0, (Word(0) & upperMask) | (mixed & lowerMask));
        for (int index = 0; index < stateSize; index++)
            if (Word(index) != 0) return;
        SetWord(0, 1u << 31);
    }

    CUJU_HOST_DEVICE_NO_INLINE void Twist() {
        const int stateSize = 624, shiftSize = 397;
        const uint32_t upperMask = ~0u << 31, lowerMask = ~upperMask, twistMask = 0x9908b0dfu;
        for (int index = 0; index < stateSize - shiftSize; index++) {
            uint32_t mixed = (Word(index) & upperMask) | (Word(index + 1) & lowerMask);
            SetWord(index,
                    Word(index + shiftSize) ^ (mixed >> 1) ^ ((Word(index + 1) & 1) * twistMask));
        }
        for (int index = stateSize - shiftSize; index < stateSize - 1; index++) {
            uint32_t mixed = (Word(index) & upperMask) | (Word(index + 1) & lowerMask);
            SetWord(index, Word(index - (stateSize - shiftSize)) ^ (mixed >> 1) ^
                               ((Word(index + 1) & 1) * twistMask));
        }
        uint32_t mixed = (Word(stateSize - 1) & upperMask) | (Word(0) & lowerMask);
        SetWord(stateSize - 1, Word(shiftSize - 1) ^ (mixed >> 1) ^ ((Word(0) & 1) * twistMask));
        SetIndex(0);
    }

    CUJU_HOST_DEVICE uint32_t Next() {
        if (Index() == 624) Twist();
        uint64_t index = Index();
        uint32_t tempered = Word((int)index);
        SetIndex(index + 1);
        tempered ^= (tempered >> 11) & 0xffffffffu;
        tempered ^= (tempered << 7) & 0x9d2c5680u;
        tempered ^= (tempered << 15) & 0xefc60000u;
        tempered ^= (tempered >> 18);
        return tempered;
    }

    // Exact: the division by 2^32 is in float. A result that rounds up to `Maximum` draws again.
    CUJU_HOST_DEVICE float Uniform() {
        const float minimum = Minimum(), maximum = Maximum();
        for (;;) {
            float numerator = (float)(Next() - 0u);
            float divisor = (float)(0xffffffffu - 0u) + 1;
            float result = numerator / divisor * (maximum - minimum) + minimum;
            if (result < maximum) return result;
        }
    }
};

CUJU_HOST_DEVICE float RandomInRange(uint8_t *stateBytes, float min, float max) {
    RNGView generator{stateBytes};
    float stretch = max - min;
    return min + (generator.Uniform() * stretch);
}

}  // namespace cuju

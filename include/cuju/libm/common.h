// Copyright (c) 2017-2020, Arm Limited.
//
// Ported to C++ for host and device from Arm Optimized Routines v21.02
// math/math_config.h and math_errf.c by the cuju Authors, 2026. glibc 2.39 carries the
// same code in sysdeps/ieee754/flt-32/math_config.h.
// SPDX-License-Identifier: MIT

// Bit casts, fused steps, tables and exception helpers of the ports of glibc 2.39.
// Exact: a NaN keeps its class but not its payload, which the device makes canonical; no `errno`.
#pragma once

#include <cmath>
#include <cstring>

#include "../host_device.h"

// The ports follow glibc's build for FMA and AVX2, so each fused step of that build is one `fma`.
#define CUJU_FUSED_MULTIPLY_ADD(left, right, addend) fma((left), (right), (addend))
#define CUJU_FUSED_MULTIPLY_SUBTRACT(left, right, subtrahend) fma((left), (right), -(subtrahend))
#define CUJU_FUSED_NEGATED_MULTIPLY_ADD(left, right, addend) fma(-(left), (right), (addend))

// Each table has a host copy and a device copy, since a `__host__ __device__` function cannot read
// `__constant__` memory on the host; `CUJU_TABLE` picks the copy at compile time.
#if defined(__CUDACC__)
#define CUJU_TABLE_DEFINITION(Type, name, ...)        \
    static constexpr Type name##Host[] = __VA_ARGS__; \
    static __constant__ const Type name##Device[] = __VA_ARGS__;
#else
#define CUJU_TABLE_DEFINITION(Type, name, ...) static constexpr Type name##Host[] = __VA_ARGS__;
#endif

// The device copy sits in global memory, for a table that does not fit beside the others in the
// 64 KB constant bank.
#if defined(__CUDACC__)
#define CUJU_TABLE_DEFINITION_GLOBAL(Type, name, ...) \
    static constexpr Type name##Host[] = __VA_ARGS__; \
    static __device__ const Type name##Device[] = __VA_ARGS__;
#else
#define CUJU_TABLE_DEFINITION_GLOBAL(Type, name, ...) \
    static constexpr Type name##Host[] = __VA_ARGS__;
#endif

#if defined(__CUDA_ARCH__)
#define CUJU_TABLE(name) (name##Device)
#else
#define CUJU_TABLE(name) (name##Host)
#endif

namespace cuju {
namespace libm {

CUJU_HOST_DEVICE uint32_t FloatBits(float value) {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(value);
#else
    uint32_t bits;
    memcpy(&bits, &value, 4);
    return bits;
#endif
}

CUJU_HOST_DEVICE float FloatFromBits(uint32_t bits) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(bits);
#else
    float value;
    memcpy(&value, &bits, 4);
    return value;
#endif
}

// The signed words of Sun's `GET_FLOAT_WORD` and `SET_FLOAT_WORD`.
CUJU_HOST_DEVICE int32_t FloatWord(float value) { return (int32_t)FloatBits(value); }
CUJU_HOST_DEVICE float FloatFromWord(int32_t word) { return FloatFromBits((uint32_t)word); }

CUJU_HOST_DEVICE uint64_t DoubleBits(double value) {
#if defined(__CUDA_ARCH__)
    return (uint64_t)__double_as_longlong(value);
#else
    uint64_t bits;
    memcpy(&bits, &value, 8);
    return bits;
#endif
}

CUJU_HOST_DEVICE double DoubleFromBits(uint64_t bits) {
#if defined(__CUDA_ARCH__)
    return __longlong_as_double((long long)bits);
#else
    double value;
    memcpy(&value, &bits, 8);
    return value;
#endif
}

CUJU_HOST_DEVICE int IsSignalingFloat(float value) {
    // A clear top bit of the significand marks a signaling NaN on x86.
    uint32_t bits = FloatBits(value);
    return 2 * (bits ^ 0x00400000) > 2u * 0x7fc00000;
}

// glibc's `xflowf`: the product overflows or underflows on purpose, to raise that exception.
CUJU_HOST_DEVICE float FloatRangeError(uint32_t sign, float magnitude) {
    return (sign ? -magnitude : magnitude) * magnitude;
}
CUJU_HOST_DEVICE float FloatUnderflow(uint32_t sign) { return FloatRangeError(sign, 0x1p-95f); }
// Rounds to the smallest subnormal, not to zero.
CUJU_HOST_DEVICE float FloatPossibleUnderflow(uint32_t sign) {
    return FloatRangeError(sign, 0x1.4p-75f);
}
CUJU_HOST_DEVICE float FloatOverflow(uint32_t sign) { return FloatRangeError(sign, 0x1p97f); }
// The zero is a variable, which keeps nvcc from warning about the division.
CUJU_HOST_DEVICE float FloatDivisionByZero(uint32_t sign) {
    float zero = 0.0f;
    return (sign ? -1.0f : 1.0f) / zero;
}
CUJU_HOST_DEVICE float FloatInvalid(float value) { return (value - value) / (value - value); }

}  // namespace libm
}  // namespace cuju

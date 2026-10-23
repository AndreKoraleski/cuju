// Copyright (c) 2017-2019, Arm Limited.
//
// Ported to C++ for host and device from Arm Optimized Routines v21.02 math/powf.c,
// powf_log2_data.c and exp2f_data.c by the cuju Authors, 2026. glibc 2.39 carries the
// same code in sysdeps/ieee754/flt-32/e_powf.c.
// SPDX-License-Identifier: MIT

// Float powf of glibc 2.39's build for FMA, bit for bit on host and device.
// Exact: each `a * b + c` whose product has no other use is fused, as that build's compiler does.
#pragma once

#include "../host_device.h"
#include "common.h"

namespace cuju {
namespace libm {

struct PowfBinaryLogarithmEntry {
    double inverseCenter, logarithmOfCenter;
};

// glibc's `__powf_log2_data.tab`: 1 / c and log2(c) for the centers c of 16 subintervals around 1.
CUJU_TABLE_DEFINITION(PowfBinaryLogarithmEntry, POWF_BINARY_LOGARITHM_TABLE,
                      {
                          {0x1.661ec79f8f3bep+0, -0x1.efec65b963019p-2},
                          {0x1.571ed4aaf883dp+0, -0x1.b0b6832d4fca4p-2},
                          {0x1.49539f0f010bp+0, -0x1.7418b0a1fb77bp-2},
                          {0x1.3c995b0b80385p+0, -0x1.39de91a6dcf7bp-2},
                          {0x1.30d190c8864a5p+0, -0x1.01d9bf3f2b631p-2},
                          {0x1.25e227b0b8eap+0, -0x1.97c1d1b3b7afp-3},
                          {0x1.1bb4a4a1a343fp+0, -0x1.2f9e393af3c9fp-3},
                          {0x1.12358f08ae5bap+0, -0x1.960cbbf788d5cp-4},
                          {0x1.0953f419900a7p+0, -0x1.a6f9db6475fcep-5},
                          {0x1p+0, 0x0p+0},
                          {0x1.e608cfd9a47acp-1, 0x1.338ca9f24f53dp-4},
                          {0x1.ca4b31f026aap-1, 0x1.476a9543891bap-3},
                          {0x1.b2036576afce6p-1, 0x1.e840b4ac4e4d2p-3},
                          {0x1.9c2d163a1aa2dp-1, 0x1.40645f0c6651cp-2},
                          {0x1.886e6037841edp-1, 0x1.88e9c2c1b9ff8p-2},
                          {0x1.767dcf5534862p-1, 0x1.ce0a44eb17bccp-2},
                      });

// The polynomial of the binary logarithm near 1, highest degree first.
CUJU_TABLE_DEFINITION(double, POWF_BINARY_LOGARITHM_POLYNOMIAL,
                      {
                          0x1.27616c9496e0bp-2,
                          -0x1.71969a075c67ap-2,
                          0x1.ec70a6ca7baddp-2,
                          -0x1.7154748bef6c8p-1,
                          0x1.71547652ab82bp0,
                      });

// The bits of 2^(i / 32) minus `i << 47`, for i from 0 to 31; adding the rounded argument shifted
// by 47 bits restores the exponent.
CUJU_TABLE_DEFINITION(
    uint64_t, POWF_BINARY_EXPONENTIAL_TABLE,
    {
        0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51,
        0x3fef72b83c7d517b, 0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1,
        0x3fef06fe0a31b715, 0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d,
        0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429, 0x3feea47eb03a5585,
        0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
        0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
        0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069,
        0x3fef5818dcfba487, 0x3fef7c97337b9b5f, 0x3fefa4afa2a490da, 0x3fefd0765b6e4540,
    });

// The polynomial of 2^r for |r| <= 1 / 64, highest degree first.
CUJU_TABLE_DEFINITION(double, POWF_BINARY_EXPONENTIAL_POLYNOMIAL,
                      {0x1.c6af84b912394p-5, 0x1.ebfce50fac4f3p-3, 0x1.62e42ff0c52d6p-1});
// 1.5 * 2^52 / 32: the sum with a double rounds it to a multiple of 1 / 32.
constexpr double POWF_BINARY_EXPONENTIAL_SHIFT = 0x1.8p+52 / 32;

// e_powf.c's `log2_inline`, in double. `bits` holds a positive normal float, or a subnormal float
// times 2^23 with 23 taken from its exponent field.
CUJU_HOST_DEVICE double PowfBinaryLogarithm(uint32_t bits) {
    // The offset centers the subintervals on 1, so that `bits - offset` gives the exponent and the
    // subinterval of the argument.
    constexpr uint32_t offset = 0x3f330000;
    const double *coefficients = CUJU_TABLE(POWF_BINARY_LOGARITHM_POLYNOMIAL);
    uint32_t temporary = bits - offset;
    int index = (temporary >> (23 - 4)) % 16;
    uint32_t top = temporary & 0xff800000;
    uint32_t normalizedBits = bits - top;
    // The shift is arithmetic so that a negative exponent keeps its sign.
    int exponent = (int32_t)top >> 23;
    double inverseCenter = CUJU_TABLE(POWF_BINARY_LOGARITHM_TABLE)[index].inverseCenter;
    double logarithmOfCenter = CUJU_TABLE(POWF_BINARY_LOGARITHM_TABLE)[index].logarithmOfCenter;
    double normalized = (double)FloatFromBits(normalizedBits);

    double reduced = CUJU_FUSED_MULTIPLY_SUBTRACT(normalized, inverseCenter, 1.0);
    double logarithmHead = logarithmOfCenter + (double)exponent;
    double reducedSquare = reduced * reduced;
    double result = CUJU_FUSED_MULTIPLY_ADD(coefficients[0], reduced, coefficients[1]);
    double middle = CUJU_FUSED_MULTIPLY_ADD(coefficients[2], reduced, coefficients[3]);
    double reducedFourth = reducedSquare * reducedSquare;
    double lower = CUJU_FUSED_MULTIPLY_ADD(coefficients[4], reduced, logarithmHead);
    lower = CUJU_FUSED_MULTIPLY_ADD(middle, reducedSquare, lower);
    result = CUJU_FUSED_MULTIPLY_ADD(result, reducedFourth, lower);
    return result;
}

// Arm's `exp2_inline`: 2^argument rounded to float, with the sign bit that `signBias` sets when it
// is not 0.
CUJU_HOST_DEVICE float PowfBinaryExponential(double argument, uint32_t signBias) {
    const double *coefficients = CUJU_TABLE(POWF_BINARY_EXPONENTIAL_POLYNOMIAL);
    double rounded = argument + POWF_BINARY_EXPONENTIAL_SHIFT;
    uint64_t roundedBits = DoubleBits(rounded);
    rounded -= POWF_BINARY_EXPONENTIAL_SHIFT;
    double reduced = argument - rounded;
    uint64_t scaleBits = CUJU_TABLE(POWF_BINARY_EXPONENTIAL_TABLE)[roundedBits % 32];
    uint64_t biasedBits = roundedBits + signBias;
    scaleBits += biasedBits << (52 - 5);
    double scale = DoubleFromBits(scaleBits);
    double upper = CUJU_FUSED_MULTIPLY_ADD(coefficients[0], reduced, coefficients[1]);
    double reducedSquare = reduced * reduced;
    double result = CUJU_FUSED_MULTIPLY_ADD(coefficients[2], reduced, 1.0);
    result = CUJU_FUSED_MULTIPLY_ADD(upper, reducedSquare, result);
    result = result * scale;
    return result;
}

// e_powf.c's `checkint`: 0 for a float that is not an integer, 1 for an odd integer and 2 for an
// even one.
CUJU_HOST_DEVICE int PowfIntegerKind(uint32_t bits) {
    int exponent = bits >> 23 & 0xff;
    if (exponent < 0x7f) return 0;
    if (exponent > 0x7f + 23) return 2;
    if (bits & ((1 << (0x7f + 23 - exponent)) - 1)) return 0;
    if (bits & (1 << (0x7f + 23 - exponent))) return 1;
    return 2;
}

CUJU_HOST_DEVICE int PowfIsZeroInfinityOrNaN(uint32_t bits) {
    return 2 * bits - 1 >= 2u * 0x7f800000 - 1;
}

CUJU_HOST_DEVICE_NO_INLINE float powf(float base, float exponent) {
    // The bias sets the sign bit of the result once `PowfBinaryExponential` shifts it into place.
    constexpr uint32_t SIGN_BIAS = 1 << (5 + 11);
    uint32_t signBias = 0;
    uint32_t baseBits = FloatBits(base);
    uint32_t exponentBits = FloatBits(exponent);
    if (baseBits - 0x00800000 >= 0x7f800000 - 0x00800000 || PowfIsZeroInfinityOrNaN(exponentBits)) {
        if (PowfIsZeroInfinityOrNaN(exponentBits)) {
            if (2 * exponentBits == 0) return IsSignalingFloat(base) ? base + exponent : 1.0f;
            if (baseBits == 0x3f800000) return IsSignalingFloat(exponent) ? base + exponent : 1.0f;
            if (2 * baseBits > 2u * 0x7f800000 || 2 * exponentBits > 2u * 0x7f800000)
                return base + exponent;
            if (2 * baseBits == 2 * 0x3f800000) return 1.0f;
            if ((2 * baseBits < 2 * 0x3f800000) == !(exponentBits & 0x80000000)) return 0.0f;
            return exponent * exponent;
        }
        if (PowfIsZeroInfinityOrNaN(baseBits)) {
            float baseSquare = base * base;
            if (baseBits & 0x80000000 && PowfIntegerKind(exponentBits) == 1) {
                baseSquare = -baseSquare;
                signBias = 1;
            }
            if (2 * baseBits == 0 && exponentBits & 0x80000000)
                return FloatDivisionByZero(signBias);
            return exponentBits & 0x80000000 ? 1 / baseSquare : baseSquare;
        }
        if (baseBits & 0x80000000) {
            int integerKind = PowfIntegerKind(exponentBits);
            if (integerKind == 0) return FloatInvalid(base);
            if (integerKind == 1) signBias = SIGN_BIAS;
            baseBits &= 0x7fffffff;
        }
        // The product with 2^23 makes a subnormal base normal, and the subtraction takes the 23
        // back from its exponent field.
        if (baseBits < 0x00800000) {
            baseBits = FloatBits(base * 0x1p23f);
            baseBits &= 0x7fffffff;
            baseBits -= 23 << 23;
        }
    }
    double logarithm = PowfBinaryLogarithm(baseBits);
    double product = (double)exponent * logarithm;
    if ((DoubleBits(product) >> 47 & 0xffff) >= DoubleBits(126.0) >> 47) {
        if (product > 0x1.fffffffd1d571p+6) return FloatOverflow(signBias);
        // The bounds assume rounding to nearest, where `1 + 2^-25 == 1`.
        if (product <= -150.0) return FloatUnderflow(signBias);
        if (product < -149.0) return FloatPossibleUnderflow(signBias);
    }
    return PowfBinaryExponential(product, signBias);
}

// `powf` for a positive `exponent` that is not an odd integer. It skips the double work for a base
// of +0 or -0, which gives +0, and of 1, which gives 1; clamped arguments often are one of them.
CUJU_HOST_DEVICE float PowfPositiveExponent(float base, float exponent) {
    if (base == 0.0f) return 0.0f;
    if (base == 1.0f) return 1.0f;
    return powf(base, exponent);
}

}  // namespace libm
}  // namespace cuju

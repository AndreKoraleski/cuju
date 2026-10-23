// Copyright (c) 2018-2019, Arm Limited.
//
// Ported to C++ for host and device from Arm Optimized Routines v21.02 math/sinf.c,
// cosf.c, sincosf.c, sincosf.h and sincosf_data.c by the cuju Authors, 2026. glibc 2.39
// carries the same code in sysdeps/ieee754/flt-32/s_sinf.c and s_cosf.c.
// SPDX-License-Identifier: MIT

// Float sinf and cosf of glibc 2.39's build for FMA, bit for bit on host and device.
// Exact: a product with one use that feeds an addition or a subtraction is fused, as in that build.
#pragma once

#include "../host_device.h"
#include "common.h"

namespace cuju {
namespace libm {

// glibc's `sincos_t` with the coefficients in another order, which the initializers follow.
struct SinCosFloatConstants {
    double sign[4];
    // `inverseHalfPiScaled` is `2^24 * 2 / PI`.
    double inverseHalfPiScaled, halfPi;
    double cosine0, cosine1;
    double sine1, cosine2;
    double sine2, cosine3;
    double sine3, cosine4;
};

// The second set negates the cosine coefficients, for the quadrants whose result changes sign.
CUJU_TABLE_DEFINITION(SinCosFloatConstants, SIN_COS_FLOAT_CONSTANTS,
                      {
                          {{1.0, -1.0, -1.0, 1.0},
                           0x1.45F306DC9C883p+23,
                           0x1.921FB54442D18p0,
                           0x1p0,
                           -0x1.ffffffd0c621cp-2,
                           -0x1.555545995a603p-3,
                           0x1.55553e1068f19p-5,
                           0x1.1107605230bc4p-7,
                           -0x1.6c087e89a359dp-10,
                           -0x1.994eb3774cf24p-13,
                           0x1.99343027bf8c3p-16},
                          {{1.0, -1.0, -1.0, 1.0},
                           0x1.45F306DC9C883p+23,
                           0x1.921FB54442D18p0,
                           -0x1p0,
                           0x1.ffffffd0c621cp-2,
                           -0x1.555545995a603p-3,
                           -0x1.55553e1068f19p-5,
                           0x1.1107605230bc4p-7,
                           0x1.6c087e89a359dp-10,
                           -0x1.994eb3774cf24p-13,
                           -0x1.99343027bf8c3p-16},
                      });

// The bits of 4 / pi, in overlapping 32-bit words that start 8 bits apart.
CUJU_TABLE_DEFINITION(uint32_t, FOUR_OVER_PI_BITS,
                      {
                          0xa2,       0xa2f9,     0xa2f983,   0xa2f9836e, 0xf9836e4e, 0x836e4e44,
                          0x6e4e4415, 0x4e441529, 0x441529fc, 0x1529fc27, 0x29fc2757, 0xfc2757d1,
                          0x2757d1f5, 0x57d1f534, 0xd1f534dd, 0xf534ddc0, 0x34ddc0db, 0xddc0db62,
                          0xc0db6295, 0xdb629599, 0x6295993c, 0x95993c43, 0x993c4390, 0x3c439041,
                      });

// Turns the fixed-point remainder of `ReduceLarge` into radians.
constexpr double PI_TIMES_TWO_TO_MINUS63 = 0x1.921FB54442D18p-62;
constexpr double QUARTER_PI = 0x1.921FB54442D18p-1;

// glibc's `abstop12`: the exponent and the 3 leading significand bits, without the sign.
CUJU_HOST_DEVICE uint32_t MagnitudeTopBits(float value) { return (FloatBits(value) >> 20) & 0x7ff; }

// glibc's `reduce_fast` for |argument| < 120: returns the remainder and leaves the rounded multiple
// of pi / 2 in `*quadrant`.
CUJU_HOST_DEVICE double ReduceSmall(double argument, const SinCosFloatConstants *constants,
                                    int *quadrant) {
    double scaled = argument * constants->inverseHalfPiScaled;
    int rounded = ((int32_t)scaled + 0x800000) >> 24;
    *quadrant = rounded;
    // The fused step computes `argument - rounded * halfPi`.
    return CUJU_FUSED_NEGATED_MULTIPLY_ADD((double)rounded, constants->halfPi, argument);
}

// glibc's `reduce_large` for finite arguments from 120 up in magnitude: returns the remainder and
// leaves the multiple of pi / 2 modulo 4, before the sign of the argument, in `*quadrant`.
CUJU_HOST_DEVICE_NO_INLINE double ReduceLarge(uint32_t bits, int *quadrant) {
    const uint32_t *digits = &CUJU_TABLE(FOUR_OVER_PI_BITS)[(bits >> 26) & 15];
    int shift = (bits >> 23) & 7;
    uint64_t rounded, result0, result1, result2;
    bits = (bits & 0xffffff) | 0x800000;
    bits <<= shift;
    // The product wraps to 32 bits, since only its low word matters.
    result0 = (uint32_t)(bits * digits[0]);
    result1 = (uint64_t)bits * digits[4];
    result2 = (uint64_t)bits * digits[8];
    result0 = (result2 >> 32) | (result0 << 32);
    result0 += result1;
    rounded = (result0 + (1ULL << 61)) >> 62;
    result0 -= rounded << 62;
    double remainder = (double)(int64_t)result0;
    *quadrant = (int)rounded;
    return remainder * PI_TIMES_TWO_TO_MINUS63;
}

// glibc's `sinf_poly`: the sine of `reduced` for an even `quadrant` and its cosine otherwise, with
// the constants whose signs fit the quadrant.
CUJU_HOST_DEVICE float SinCosPolynomial(double reduced, double square,
                                        const SinCosFloatConstants *constants, int quadrant) {
    if ((quadrant & 1) == 0) {
        double cube = reduced * square;
        double upper = CUJU_FUSED_MULTIPLY_ADD(square, constants->sine3, constants->sine2);
        double seventhPower = cube * square;
        double sine = CUJU_FUSED_MULTIPLY_ADD(cube, constants->sine1, reduced);
        return (float)CUJU_FUSED_MULTIPLY_ADD(seventhPower, upper, sine);
    } else {
        double fourthPower = square * square;
        double upper = CUJU_FUSED_MULTIPLY_ADD(square, constants->cosine4, constants->cosine3);
        double lower = CUJU_FUSED_MULTIPLY_ADD(square, constants->cosine1, constants->cosine0);
        double sixthPower = fourthPower * square;
        double cosine = CUJU_FUSED_MULTIPLY_ADD(fourthPower, constants->cosine2, lower);
        return (float)CUJU_FUSED_MULTIPLY_ADD(sixthPower, upper, cosine);
    }
}

CUJU_HOST_DEVICE_NO_INLINE float sinf(float argument) {
    double reduced = argument;
    // `factor` holds the square of a small argument or the sign of the quadrant.
    double factor;
    int quadrant;
    const SinCosFloatConstants *constants = &CUJU_TABLE(SIN_COS_FLOAT_CONSTANTS)[0];
    if (MagnitudeTopBits(argument) < MagnitudeTopBits(QUARTER_PI)) {
        factor = reduced * reduced;
        if (MagnitudeTopBits(argument) < MagnitudeTopBits(0x1p-12f)) return argument;
        return SinCosPolynomial(reduced, factor, constants, 0);
    } else if (MagnitudeTopBits(argument) < MagnitudeTopBits(120.0f)) {
        reduced = ReduceSmall(reduced, constants, &quadrant);
        factor = constants->sign[quadrant & 3];
        if (quadrant & 2) constants = &CUJU_TABLE(SIN_COS_FLOAT_CONSTANTS)[1];
        return SinCosPolynomial(reduced * factor, reduced * reduced, constants, quadrant);
    } else if (MagnitudeTopBits(argument) < MagnitudeTopBits(INFINITY)) {
        uint32_t bits = FloatBits(argument);
        int sign = bits >> 31;
        reduced = ReduceLarge(bits, &quadrant);
        factor = constants->sign[(quadrant + sign) & 3];
        if ((quadrant + sign) & 2) constants = &CUJU_TABLE(SIN_COS_FLOAT_CONSTANTS)[1];
        return SinCosPolynomial(reduced * factor, reduced * reduced, constants, quadrant);
    }
    return FloatInvalid(argument);
}

CUJU_HOST_DEVICE_NO_INLINE float cosf(float argument) {
    double reduced = argument;
    double factor;
    int quadrant;
    const SinCosFloatConstants *constants = &CUJU_TABLE(SIN_COS_FLOAT_CONSTANTS)[0];
    if (MagnitudeTopBits(argument) < MagnitudeTopBits(QUARTER_PI)) {
        double square = reduced * reduced;
        if (MagnitudeTopBits(argument) < MagnitudeTopBits(0x1p-12f)) return 1.0f;
        return SinCosPolynomial(reduced, square, constants, 1);
    } else if (MagnitudeTopBits(argument) < MagnitudeTopBits(120.0f)) {
        reduced = ReduceSmall(reduced, constants, &quadrant);
        factor = constants->sign[quadrant & 3];
        if (quadrant & 2) constants = &CUJU_TABLE(SIN_COS_FLOAT_CONSTANTS)[1];
        return SinCosPolynomial(reduced * factor, reduced * reduced, constants, quadrant ^ 1);
    } else if (MagnitudeTopBits(argument) < MagnitudeTopBits(INFINITY)) {
        uint32_t bits = FloatBits(argument);
        int sign = bits >> 31;
        reduced = ReduceLarge(bits, &quadrant);
        factor = constants->sign[(quadrant + sign) & 3];
        if ((quadrant + sign) & 2) constants = &CUJU_TABLE(SIN_COS_FLOAT_CONSTANTS)[1];
        return SinCosPolynomial(reduced * factor, reduced * reduced, constants, quadrant ^ 1);
    }
    return FloatInvalid(argument);
}

struct SineCosine {
    float sine, cosine;
};

// Not glibc's `sincosf`: the `sinf` and `cosf` above from one reduction. They pick the same
// reduction and constants and differ only in the polynomial, so each result is theirs bit for bit.
CUJU_HOST_DEVICE_NO_INLINE SineCosine sincosf(float argument) {
    double reduced = argument;
    double factor;
    int quadrant;
    const SinCosFloatConstants *constants = &CUJU_TABLE(SIN_COS_FLOAT_CONSTANTS)[0];
    if (MagnitudeTopBits(argument) < MagnitudeTopBits(QUARTER_PI)) {
        if (MagnitudeTopBits(argument) < MagnitudeTopBits(0x1p-12f))
            return SineCosine{argument, 1.0f};
        double square = reduced * reduced;
        return SineCosine{SinCosPolynomial(reduced, square, constants, 0),
                          SinCosPolynomial(reduced, square, constants, 1)};
    } else if (MagnitudeTopBits(argument) < MagnitudeTopBits(120.0f)) {
        reduced = ReduceSmall(reduced, constants, &quadrant);
        factor = constants->sign[quadrant & 3];
        if (quadrant & 2) constants = &CUJU_TABLE(SIN_COS_FLOAT_CONSTANTS)[1];
    } else if (MagnitudeTopBits(argument) < MagnitudeTopBits(INFINITY)) {
        uint32_t bits = FloatBits(argument);
        int sign = bits >> 31;
        reduced = ReduceLarge(bits, &quadrant);
        factor = constants->sign[(quadrant + sign) & 3];
        if ((quadrant + sign) & 2) constants = &CUJU_TABLE(SIN_COS_FLOAT_CONSTANTS)[1];
    } else {
        float invalid = FloatInvalid(argument);
        return SineCosine{invalid, invalid};
    }
    double signedReduced = reduced * factor, square = reduced * reduced;
    return SineCosine{SinCosPolynomial(signedReduced, square, constants, quadrant),
                      SinCosPolynomial(signedReduced, square, constants, quadrant ^ 1)};
}

}  // namespace libm
}  // namespace cuju

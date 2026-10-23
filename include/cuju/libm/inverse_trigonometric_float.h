// Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
//
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
//
// Ported to C++ for host and device from FreeBSD lib/msun/src/e_acosf.c, s_atanf.c and e_atan2f.c
// at commit 5aa554c by the cuju Authors, 2026, with the |x| thresholds of glibc 2.39 in acosf
// (0x32800000) and atanf (0x4c000000).
// SPDX-License-Identifier: SunPro

// Float acosf, atanf and atan2f with the results of glibc 2.39, which runs the same code.
// Exact: float arithmetic, no contraction; `(float)` marks each double literal the source converts.
#pragma once

#include "../host_device.h"
#include "common.h"

namespace cuju {
namespace libm {

CUJU_HOST_DEVICE_NO_INLINE float acosf(float argument) {
    const float one = (float)1.0000000000e+00, PI = (float)3.1415925026e+00,
                halfPiHigh = (float)1.5707962513e+00, halfPiLow = (float)7.5497894159e-08,
                numerator0 = (float)1.6666667163e-01, numerator1 = (float)-3.2556581497e-01,
                numerator2 = (float)2.0121252537e-01, numerator3 = (float)-4.0055535734e-02,
                numerator4 = (float)7.9153501429e-04, numerator5 = (float)3.4793309169e-05,
                denominator1 = (float)-2.4033949375e+00, denominator2 = (float)2.0209457874e+00,
                denominator3 = (float)-6.8828397989e-01, denominator4 = (float)7.7038154006e-02;
    // `square` is the source's `z`: `x * x` for |x| < 0.5 and `(1 - |x|) / 2` above.
    float square, numerator, denominator, ratio, correction, root, rootLow, rootHigh;
    int32_t word = FloatWord(argument), magnitudeWord = word & 0x7fffffff;
    if (magnitudeWord == 0x3f800000) {
        if (word > 0) return 0.0;
        return PI + (float)2.0 * halfPiLow;
    } else if (magnitudeWord > 0x3f800000) {
        return (argument - argument) / (argument - argument);
    }
    if (magnitudeWord < 0x3f000000) {
        if (magnitudeWord <= 0x32800000) return halfPiHigh + halfPiLow;
        square = argument * argument;
        numerator =
            square *
            (numerator0 +
             square *
                 (numerator1 +
                  square * (numerator2 +
                            square * (numerator3 + square * (numerator4 + square * numerator5)))));
        denominator =
            one +
            square * (denominator1 +
                      square * (denominator2 + square * (denominator3 + square * denominator4)));
        ratio = numerator / denominator;
        return halfPiHigh - (argument - (halfPiLow - argument * ratio));
    } else if (word < 0) {
        square = (one + argument) * (float)0.5;
        numerator =
            square *
            (numerator0 +
             square *
                 (numerator1 +
                  square * (numerator2 +
                            square * (numerator3 + square * (numerator4 + square * numerator5)))));
        denominator =
            one +
            square * (denominator1 +
                      square * (denominator2 + square * (denominator3 + square * denominator4)));
        root = sqrtf(square);
        ratio = numerator / denominator;
        correction = ratio * root - halfPiLow;
        return PI - (float)2.0 * (root + correction);
    } else {
        square = (one - argument) * (float)0.5;
        root = sqrtf(square);
        // `rootHigh` is `root` cut to its high 12 significand bits, and `rootHigh + rootLow` is the
        // square root to more precision.
        rootHigh = FloatFromWord(FloatWord(root) & 0xfffff000);
        rootLow = (square - rootHigh * rootHigh) / (root + rootHigh);
        numerator =
            square *
            (numerator0 +
             square *
                 (numerator1 +
                  square * (numerator2 +
                            square * (numerator3 + square * (numerator4 + square * numerator5)))));
        denominator =
            one +
            square * (denominator1 +
                      square * (denominator2 + square * (denominator3 + square * denominator4)));
        ratio = numerator / denominator;
        correction = ratio * root + rootLow;
        return (float)2.0 * (rootHigh + correction);
    }
}

}  // namespace libm
}  // namespace cuju

// Included here to keep the definition order acosf, asinf, atanf, which the emitted code follows.
#include "asinf.h"

namespace cuju {
namespace libm {

// atan(0.5), atan(1), atan(1.5) and atan(inf), rounded to float.
CUJU_TABLE_DEFINITION(float, ATAN_HIGH,
                      {
                          (float)4.6364760399e-01,
                          (float)7.8539812565e-01,
                          (float)9.8279368877e-01,
                          (float)1.5707962513e+00,
                      });
// What the floats of `ATAN_HIGH` leave out of those arc tangents.
CUJU_TABLE_DEFINITION(float, ATAN_LOW,
                      {
                          (float)5.0121582440e-09,
                          (float)3.7748947079e-08,
                          (float)3.4473217170e-08,
                          (float)7.5497894159e-08,
                      });
// The series of the arc tangent on [-7 / 16, 7 / 16].
CUJU_TABLE_DEFINITION(float, ATAN_COEFFICIENTS,
                      {
                          (float)3.3333334327e-01,
                          (float)-2.0000000298e-01,
                          (float)1.4285714924e-01,
                          (float)-1.1111110449e-01,
                          (float)9.0908870101e-02,
                          (float)-7.6918758452e-02,
                          (float)6.6610731184e-02,
                          (float)-5.8335702866e-02,
                          (float)4.9768779427e-02,
                          (float)-3.6531571299e-02,
                          (float)1.6285819933e-02,
                      });

CUJU_HOST_DEVICE_NO_INLINE float atanf(float argument) {
    const float *atanHigh = CUJU_TABLE(ATAN_HIGH), *atanLow = CUJU_TABLE(ATAN_LOW),
                *coefficients = CUJU_TABLE(ATAN_COEFFICIENTS);
    const float one = (float)1.0, huge = (float)1.0e30;
    // `square` is the source's `z`, which takes the result in the last step.
    float fourthPower, oddSum, evenSum, square;
    // `interval` picks the reduction point 0.5, 1, 1.5 or infinity, and -1 means no reduction.
    int32_t word = FloatWord(argument), magnitudeWord = word & 0x7fffffff, interval;
    if (magnitudeWord >= 0x4c000000) {
        if (magnitudeWord > 0x7f800000) return argument + argument;
        if (word > 0) return atanHigh[3] + atanLow[3];
        return -atanHigh[3] - atanLow[3];
    }
    if (magnitudeWord < 0x3ee00000) {
        if (magnitudeWord < 0x31000000) {
            // The comparison raises the inexact exception for a nonzero argument.
            if (huge + argument > one) return argument;
        }
        interval = -1;
    } else {
        argument = fabsf(argument);
        if (magnitudeWord < 0x3f980000) {
            if (magnitudeWord < 0x3f300000) {
                interval = 0;
                argument = ((float)2.0 * argument - one) / ((float)2.0 + argument);
            } else {
                interval = 1;
                argument = (argument - one) / (argument + one);
            }
        } else {
            if (magnitudeWord < 0x401c0000) {
                interval = 2;
                argument = (argument - (float)1.5) / (one + (float)1.5 * argument);
            } else {
                interval = 3;
                argument = -(float)1.0 / argument;
            }
        }
    }
    square = argument * argument;
    fourthPower = square * square;
    oddSum = square *
             (coefficients[0] +
              fourthPower *
                  (coefficients[2] +
                   fourthPower * (coefficients[4] +
                                  fourthPower * (coefficients[6] +
                                                 fourthPower * (coefficients[8] +
                                                                fourthPower * coefficients[10])))));
    evenSum =
        fourthPower *
        (coefficients[1] +
         fourthPower *
             (coefficients[3] +
              fourthPower * (coefficients[5] +
                             fourthPower * (coefficients[7] + fourthPower * coefficients[9]))));
    if (interval < 0) return argument - argument * (oddSum + evenSum);
    square = atanHigh[interval] - ((argument * (oddSum + evenSum) - atanLow[interval]) - argument);
    return (word < 0) ? -square : square;
}

CUJU_HOST_DEVICE_NO_INLINE float atan2f(float y, float x) {
    const float tiny = (float)1.0e-30, zero = (float)0.0, quarterPi = (float)7.8539818525e-01,
                halfPi = (float)1.5707963705e+00, PI = (float)3.1415927410e+00,
                piLow = (float)-8.7422776573e-08;
    float angle;
    // `signs` holds the sign of `y` in bit 0 and the sign of `x` in bit 1.
    int32_t exponentDifference, signs, xWord = FloatWord(x), yWord = FloatWord(y);
    int32_t xMagnitude = xWord & 0x7fffffff, yMagnitude = yWord & 0x7fffffff;
    if ((xMagnitude > 0x7f800000) || (yMagnitude > 0x7f800000)) return x + y;
    if (xWord == 0x3f800000) return atanf(y);
    signs = ((yWord >> 31) & 1) | ((xWord >> 30) & 2);
    if (yMagnitude == 0) {
        switch (signs) {
            case 0:
            case 1:
                return y;
            case 2:
                return PI + tiny;
            case 3:
                return -PI - tiny;
        }
    }
    if (xMagnitude == 0) return (yWord < 0) ? -halfPi - tiny : halfPi + tiny;
    if (xMagnitude == 0x7f800000) {
        if (yMagnitude == 0x7f800000) {
            switch (signs) {
                case 0:
                    return quarterPi + tiny;
                case 1:
                    return -quarterPi - tiny;
                case 2:
                    return (float)3.0 * quarterPi + tiny;
                case 3:
                    return (float)-3.0 * quarterPi - tiny;
            }
        } else {
            switch (signs) {
                case 0:
                    return zero;
                case 1:
                    return -zero;
                case 2:
                    return PI + tiny;
                case 3:
                    return -PI - tiny;
            }
        }
    }
    if (yMagnitude == 0x7f800000) return (yWord < 0) ? -halfPi - tiny : halfPi + tiny;
    exponentDifference = (yMagnitude - xMagnitude) >> 23;
    if (exponentDifference > 60)
        angle = halfPi + (float)0.5 * piLow;
    else if (xWord < 0 && exponentDifference < -60)
        angle = 0.0;
    else
        angle = atanf(fabsf(y / x));
    switch (signs) {
        case 0:
            return angle;
        case 1:
            return FloatFromWord(FloatWord(angle) ^ (int32_t)0x80000000);
        case 2:
            return PI - (angle - piLow);
        default:
            return (angle - piLow) - PI;
    }
}

}  // namespace libm
}  // namespace cuju

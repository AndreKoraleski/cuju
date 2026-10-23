// IBM Accurate Mathematical Library
// written by International Business Machines Corp.
// Copyright (C) 2001-2024 Free Software Foundation, Inc.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation; either version 2.1 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU  Lesser General Public License
// along with this program; if not, see <https://www.gnu.org/licenses/>.
//
// Ported to C++ for host and device from glibc 2.39 sysdeps/ieee754/dbl-64/s_sin.c, usncs.h,
// branred.c and branred.h by the cuju Authors, 2026.
// SPDX-License-Identifier: LGPL-2.1-or-later

// Double sin and cos of glibc 2.39's build for FMA and AVX2, bit for bit on host and device.
// Exact: each `fma` is a contraction of that build and nothing else fuses; `__branred` has none.
#pragma once

#include "../host_device.h"
#include "common.h"
#include "trigonometric_tables_double.h"

namespace cuju {
namespace libm {

constexpr double SIN_SERIES1 = -0x1.5555555555555p-3;
constexpr double SIN_SERIES2 = 0x1.1111111110ECEp-7;
constexpr double SIN_SERIES3 = -0x1.A01A019DB08B8p-13;
constexpr double SIN_SERIES4 = 0x1.71DE27B9A7ED9p-19;
constexpr double SIN_SERIES5 = -0x1.ADDFFC2FCDF59p-26;
// 1.5 * 2^45: the sum with |x| rounds |x| to a multiple of 1 / 128 and leaves `round(|x| * 128)`
// in its low word.
constexpr double SIN_COS_TABLE_SHIFT = 0x1.8000000000000p45;
constexpr double SIN_COS_HALF_PI_HIGH = 0x1.921FB54442D18p0;
constexpr double SIN_COS_HALF_PI_LOW = 0x1.1A62633145C07p-54;
constexpr double SIN_COS_HALF_PI_PART1 = 0x1.921FB58000000p0;
constexpr double SIN_COS_HALF_PI_PART2 = -0x1.DDE973C000000p-27;
constexpr double SIN_COS_HALF_PI_PART3 = -0x1.CB3B398000000p-55;
constexpr double SIN_COS_HALF_PI_PART4 = -0x1.d747f23e32ed7p-83;
constexpr double SIN_COS_INVERSE_HALF_PI = 0x1.45F306DC9C883p-1;
// 1.5 * 2^52: the sum with a double rounds it to an integer.
constexpr double SIN_COS_ROUND_SHIFT = 0x1.8000000000000p52;
constexpr double SIN_COS_SINE3 = -0x1.5555555555515p-3;   // -1.66666666666664880952546298448555E-01
constexpr double SIN_COS_SINE5 = 0x1.11110e829872fp-7;    //  8.33333214285722277379541354343671E-03
constexpr double SIN_COS_COSINE2 = 0x1.0000000000000p-1;  //  4.99999999999999999999950396842453E-01
constexpr double SIN_COS_COSINE4 =
    -0x1.5555555555535p-5;  // -4.16666666666664434524222570944589E-02
constexpr double SIN_COS_COSINE6 =
    0x1.6c16bedd9e239p-10;  //  1.38888874007937613028114285595617E-03

CUJU_HOST_DEVICE int SinCosTableIndex(double shifted) {
    return (int)(uint32_t)DoubleBits(shifted) << 2;
}

// s_sin.c's `do_cos`: the cosine of `head + tail` for |head| <= 0.855469.
CUJU_HOST_DEVICE double CosOfSum(double head, double tail) {
    const double *table = CUJU_TABLE(SIN_COS_TABLE);
    if (head < 0) tail = -tail;
    double shifted = SIN_COS_TABLE_SHIFT + fabs(head);
    head = fabs(head) - (shifted - SIN_COS_TABLE_SHIFT) + tail;
    double square = head * head;
    // The steps are `sine = head + head * square * (s3 + square * s5)` and
    // `versine = square * (c2 + square * (c4 + square * c6))`.
    double sine = fma(head * square, fma(square, SIN_COS_SINE5, SIN_COS_SINE3), head);
    double versine =
        square * fma(square, fma(square, SIN_COS_COSINE6, SIN_COS_COSINE4), SIN_COS_COSINE2);
    int index = SinCosTableIndex(shifted);
    double tableSine = table[index], tableSineTail = table[index + 1],
           tableCosine = table[index + 2], tableCosineTail = table[index + 3];
    // The correction is
    // `(tableCosineTail - sine * tableSineTail - tableCosine * versine) - tableSine * sine`.
    double correction = fma(-sine, tableSine,
                            fma(-versine, tableCosine, fma(-sine, tableSineTail, tableCosineTail)));
    return tableCosine + correction;
}

// s_sin.c's `do_sin`: the sine of `head + tail` for |head| <= 0.855469.
CUJU_HOST_DEVICE double SinOfSum(double head, double tail) {
    double original = head;
    if (fabs(head) < 0.126) {
        // The Taylor series gives `head + ((series * head - 0.5 * tail) * square + tail)`.
        double square = head * head;
        double series =
            fma(square,
                fma(square, fma(square, fma(square, SIN_SERIES5, SIN_SERIES4), SIN_SERIES3),
                    SIN_SERIES2),
                SIN_SERIES1);
        double correction = fma(square, fma(series, head, -(0.5 * tail)), tail);
        return head + correction;
    }
    const double *table = CUJU_TABLE(SIN_COS_TABLE);
    if (head <= 0) tail = -tail;
    double shifted = SIN_COS_TABLE_SHIFT + fabs(head);
    head = fabs(head) - (shifted - SIN_COS_TABLE_SHIFT);
    double square = head * head;
    // The sine is `head + (tail + head * square * (s3 + square * s5))`.
    double sine = head + fma(head * square, fma(square, SIN_COS_SINE5, SIN_COS_SINE3), tail);
    // The versine is `head * tail + square * (c2 + square * (c4 + square * c6))`.
    double versine =
        fma(head, tail,
            square * fma(square, fma(square, SIN_COS_COSINE6, SIN_COS_COSINE4), SIN_COS_COSINE2));
    int index = SinCosTableIndex(shifted);
    double tableSine = table[index], tableSineTail = table[index + 1],
           tableCosine = table[index + 2], tableCosineTail = table[index + 3];
    // The correction is
    // `(tableSineTail + sine * tableCosineTail - tableSine * versine) + tableCosine * sine`.
    double correction =
        fma(sine, tableCosine, fma(-versine, tableSine, fma(sine, tableCosineTail, tableSineTail)));
    return copysign(tableSine + correction, original);
}

// s_sin.c's `reduce_sincos` for |argument| < 105414350: returns the quadrant modulo 4 and leaves
// the remainder in `head + tail`.
CUJU_HOST_DEVICE int ReduceHalfPi(double argument, double *head, double *tail) {
    double shifted = fma(argument, SIN_COS_INVERSE_HALF_PI, SIN_COS_ROUND_SHIFT);
    double multiple = shifted - SIN_COS_ROUND_SHIFT;
    // The remainder is `(argument - multiple * part1) - multiple * part2`.
    double remainder =
        fma(-multiple, SIN_COS_HALF_PI_PART2, fma(-multiple, SIN_COS_HALF_PI_PART1, argument));
    int quadrant = (int)(uint32_t)DoubleBits(shifted) & 3;
    // The product `multiple * part3` feeds two subtractions, and the FMA build fuses it into both.
    double partial = fma(-multiple, SIN_COS_HALF_PI_PART3, remainder);
    double reducedTail = fma(-multiple, SIN_COS_HALF_PI_PART3, remainder - partial);
    double reduced = fma(-multiple, SIN_COS_HALF_PI_PART4, partial);
    reducedTail = reducedTail + fma(-multiple, SIN_COS_HALF_PI_PART4, partial - reduced);
    *head = reduced;
    *tail = reducedTail;
    return quadrant;
}

// s_sin.c's `do_sincos`: the sine of `quadrant * PI / 2 + (head + tail)`.
CUJU_HOST_DEVICE double SinOfQuadrant(double head, double tail, int quadrant) {
    double result = (quadrant & 1) ? CosOfSum(head, tail) : SinOfSum(head, tail);
    return (quadrant & 2) ? -result : result;
}

// One half of the split argument of `__branred` times 2 / pi; `__branred` repeats these steps for
// the other half.
CUJU_HOST_DEVICE void ReduceHalfPiPart(double part, double *fractionOut, double *fractionTailOut,
                                       double *sumOut) {
    const double *twoOverPi = CUJU_TABLE(TWO_OVER_PI);
    constexpr double twoTo576 = 0x1p576, twoToMinus24 = 0x1p-24;
    constexpr double roundShift = 0x1.8p52, fourRoundShift = 0x1.8p54;
    double digits[6], rounded, total, sum = 0, fraction, fractionTail;
    int digitIndex = (int)((DoubleBits(part) >> 52) & 2047);
    digitIndex = (digitIndex - 450) / 24;
    if (digitIndex < 0) digitIndex = 0;
    double scale = DoubleFromBits(DoubleBits(twoTo576) - ((uint64_t)(digitIndex * 24) << 52));
    for (int digit = 0; digit < 6; digit++) {
        digits[digit] = part * twoOverPi[digitIndex + digit] * scale;
        scale *= twoToMinus24;
    }
    for (int digit = 0; digit < 3; digit++) {
        rounded = (digits[digit] + roundShift) - roundShift;
        sum += rounded;
        digits[digit] -= rounded;
    }
    total = 0;
    for (int digit = 0; digit < 6; digit++) total += digits[5 - digit];
    fractionTail =
        (((((digits[0] - total) + digits[1]) + digits[2]) + digits[3]) + digits[4]) + digits[5];
    rounded = (total + roundShift) - roundShift;
    sum += rounded;
    total -= rounded;
    fraction = total + fractionTail;
    fractionTail = (total - fraction) + fractionTail;
    rounded = (sum + fourRoundShift) - fourRoundShift;
    sum -= rounded;
    *fractionOut = fraction;
    *fractionTailOut = fractionTail;
    *sumOut = sum;
}

// branred.c's `__branred`: reduces a huge `argument` to `quadrant * PI / 2 + (head + tail)` with
// 2 / pi in base 2^24 digits. glibc builds it without FMA, so no operation here is fused.
CUJU_HOST_DEVICE_NO_INLINE int ReduceHalfPiLarge(double argument, double *head, double *tail) {
    constexpr double twoToMinus600 = 0x1p-600, splitter = 134217729.0;
    constexpr double halfPiHigh = 0x1.921FB54442D18p0, halfPiLow = 0x1.1A62633145C07p-54;
    constexpr double halfPiPart1 = 0x1.921FB58000000p0, halfPiPart2 = -0x1.DDE9740000000p-27;
    double result, resultTail, sum, fraction, fractionTail, sum1, sum2, fraction1, fractionTail1,
        fraction2, fractionTail2, argumentHigh, argumentLow, resultHigh, resultLow;
    argument *= twoToMinus600;
    // `resultTail` holds the product that splits the argument until the result needs it.
    resultTail = argument * splitter;
    argumentHigh = resultTail - (resultTail - argument);
    argumentLow = argument - argumentHigh;
    ReduceHalfPiPart(argumentHigh, &fraction1, &fractionTail1, &sum1);
    ReduceHalfPiPart(argumentLow, &fraction2, &fractionTail2, &sum2);
    sum = sum1 + sum2;
    fraction = fraction1 + fraction2;
    fractionTail = (fabs(fraction1) > fabs(fraction2)) ? (fraction1 - fraction) + fraction2
                                                       : (fraction2 - fraction) + fraction1;
    if (fraction > 0.5) {
        fraction -= 1.0;
        sum += 1.0;
    } else if (fraction < -0.5) {
        fraction += 1.0;
        sum -= 1.0;
    }
    result = fraction + ((fractionTail + fractionTail1) + fractionTail2);
    resultTail = ((fraction - result) + fractionTail) + (fractionTail1 + fractionTail2);
    fraction = result * splitter;
    resultHigh = fraction - (fraction - result);
    resultLow = result - resultHigh;
    fraction = result * halfPiHigh;
    fractionTail = (((resultHigh * halfPiPart1 - fraction) + resultHigh * halfPiPart2) +
                    resultLow * halfPiPart1) +
                   ((resultLow * halfPiPart2 + result * halfPiLow) + resultTail * halfPiHigh);
    result = fraction + fractionTail;
    resultTail = (fraction - result) + fractionTail;
    *head = result;
    *tail = resultTail;
    return ((int)sum) & 3;
}

CUJU_HOST_DEVICE_NO_INLINE double sin(double argument) {
    int32_t magnitudeWord = (int32_t)(DoubleBits(argument) >> 32) & 0x7fffffff;
    if (magnitudeWord < 0x3e500000) return argument;  // |argument| < 2^-26
    if (magnitudeWord < 0x3feb6000) return SinOfSum(argument, 0);
    if (magnitudeWord < 0x400368fd) {
        double complement = SIN_COS_HALF_PI_HIGH - fabs(argument);
        return copysign(CosOfSum(complement, SIN_COS_HALF_PI_LOW), argument);
    }
    double head, tail;
    if (magnitudeWord < 0x419921FB) {
        int quadrant = ReduceHalfPi(argument, &head, &tail);
        return SinOfQuadrant(head, tail, quadrant);
    }
    if (magnitudeWord < 0x7ff00000) {
        int quadrant = ReduceHalfPiLarge(argument, &head, &tail);
        return SinOfQuadrant(head, tail, quadrant);
    }
    return argument / argument;
}

CUJU_HOST_DEVICE_NO_INLINE double cos(double argument) {
    int32_t magnitudeWord = (int32_t)(DoubleBits(argument) >> 32) & 0x7fffffff;
    if (magnitudeWord < 0x3e400000) return 1.0;  // |argument| < 2^-27
    if (magnitudeWord < 0x3feb6000) return CosOfSum(argument, 0);
    if (magnitudeWord < 0x400368fd) {
        double complement = SIN_COS_HALF_PI_HIGH - fabs(argument);
        double head = complement + SIN_COS_HALF_PI_LOW;
        double tail = (complement - head) + SIN_COS_HALF_PI_LOW;
        return SinOfSum(head, tail);
    }
    double head, tail;
    if (magnitudeWord < 0x419921FB) {
        int quadrant = ReduceHalfPi(argument, &head, &tail);
        return SinOfQuadrant(head, tail, quadrant + 1);
    }
    if (magnitudeWord < 0x7ff00000) {
        int quadrant = ReduceHalfPiLarge(argument, &head, &tail);
        return SinOfQuadrant(head, tail, quadrant + 1);
    }
    return argument / argument;
}

}  // namespace libm
}  // namespace cuju

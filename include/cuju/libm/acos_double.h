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
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, see <https://www.gnu.org/licenses/>.
//
// Ported to C++ for host and device from glibc 2.39 sysdeps/ieee754/dbl-64/e_asin.c and uasncs.h
// by the cuju Authors, 2026.
// SPDX-License-Identifier: LGPL-2.1-or-later

// Double acos of glibc 2.39's build for FMA and AVX2, bit for bit on host and device.
// Exact: each `fma` is a contraction of that build and nothing else fuses; constants are exact hex.
#pragma once

#include "../host_device.h"
#include "common.h"
#include "trigonometric_tables_double.h"

namespace cuju {
namespace libm {

constexpr double ACOS_HALF_PI_HIGH = 0x1.921FB54442D18p0;
constexpr double ACOS_HALF_PI_LOW = 0x1.1A62633145C07p-54;
// asin(x) = x + x^3 * series(x^2).
constexpr double ACOS_SERIES1 = 0x1.55555555554f9p-3;  // 1.66666666666664110590506577996662E-01
constexpr double ACOS_SERIES2 = 0x1.333333336127dp-4;  // 7.50000000026122686814431784722623E-02
constexpr double ACOS_SERIES3 = 0x1.6db6dae42c0e4p-5;  // 4.46428561421059750978517350006940E-02
constexpr double ACOS_SERIES4 = 0x1.f1c7e04f4ad99p-6;  // 3.03821268582119319911193410625235E-02
constexpr double ACOS_SERIES5 = 0x1.6e442c822d419p-6;  // 2.23551211026525610742786300334557E-02
constexpr double ACOS_SERIES6 = 0x1.292d80f453c72p-6;  // 1.81382903404565056280372531963613E-02
// These refine a seed of the inverse square root.
constexpr double ACOS_ROOT0 = 0x1.fffffffecc1ddp-1;  // 9.99999999859990725855365213134618E-01
constexpr double ACOS_ROOT1 = 0x1.fffffff757304p-2;  // 4.99999999495955425917856814202739E-01
constexpr double ACOS_ROOT2 = 0x1.800496769c91ap-2;  // 3.75017500867345182581453026130850E-01
constexpr double ACOS_ROOT3 = 0x1.4006318d1dab9p-2;  // 3.12523626554518656309172508769531E-01
constexpr double ACOS_SPLITTER = 134217728.0;

CUJU_HOST_DEVICE double AcosSeries(double value) {
    return fma(fma(fma(fma(fma(value, ACOS_SERIES6, ACOS_SERIES5), value, ACOS_SERIES4), value,
                       ACOS_SERIES3),
                   value, ACOS_SERIES2),
               value, ACOS_SERIES1);
}

// e_asin.c's table branches for 0.125 <= |x| < 0.96875. The expansion at `index` holds its point,
// the linear term, the terms up to `index + last`, a constant and the arc sine of the point.
CUJU_HOST_DEVICE double AcosFromTable(const double *table, int index, int last, double argument,
                                      int32_t highWord) {
    double distance = (highWord > 0) ? argument - table[index] : -argument - table[index];
    double series = table[index + last];
    for (int term = last - 1; term >= 2; term--)
        series = fma(series, distance, table[index + term]);
    double sum = fma(distance * distance, series, table[index + last + 1]);
    double low = fma(table[index + 1], distance, sum);
    double high = (highWord > 0) ? (ACOS_HALF_PI_HIGH - table[index + last + 2])
                                 : (ACOS_HALF_PI_HIGH + table[index + last + 2]);
    low = (highWord > 0) ? (ACOS_HALF_PI_LOW - low) : (ACOS_HALF_PI_LOW + low);
    return high + low;
}

CUJU_HOST_DEVICE_NO_INLINE double acos(double argument) {
    const double *table = CUJU_TABLE(ASIN_COS_TABLE);
    uint64_t bits = DoubleBits(argument);
    int32_t highWord = (int32_t)(bits >> 32);
    int32_t magnitudeWord = 0x7fffffff & highWord;
    if (magnitudeWord < 0x3c880000) return ACOS_HALF_PI_HIGH;  // |argument| < 2.77556e-17
    if (magnitudeWord < 0x3fc00000) {                          // |argument| < 0.125
        double square = argument * argument;
        // The fused step computes
        // `(((halfPiHigh - difference) - argument) + halfPiLow) - series * (argument * square)`.
        double difference = ACOS_HALF_PI_HIGH - argument;
        double correction = fma(-AcosSeries(square), argument * square,
                                ((ACOS_HALF_PI_HIGH - difference) - argument) + ACOS_HALF_PI_LOW);
        return difference + correction;
    }
    if (magnitudeWord < 0x3fe00000) {  // 0.125 <= |argument| < 0.5
        int index = (magnitudeWord < 0x3fd00000) ? 11 * ((magnitudeWord & 0x000fffff) >> 15)
                                                 : 11 * ((magnitudeWord & 0x000fffff) >> 14) + 352;
        return AcosFromTable(table, index, 6, argument, highWord);
    }
    if (magnitudeWord < 0x3fe80000)  // 0.5 <= |argument| < 0.75
        return AcosFromTable(table, 1056 + ((magnitudeWord & 0x000fe000) >> 11) * 3, 7, argument,
                             highWord);
    if (magnitudeWord < 0x3fed8000)  // 0.75 <= |argument| < 0.921875
        return AcosFromTable(table, 992 + ((magnitudeWord & 0x000fe000) >> 13) * 13, 8, argument,
                             highWord);
    if (magnitudeWord < 0x3fee8000)  // 0.921875 <= |argument| < 0.953125
        return AcosFromTable(table, 884 + ((magnitudeWord & 0x000fe000) >> 13) * 14, 9, argument,
                             highWord);
    if (magnitudeWord < 0x3fef0000)  // 0.953125 <= |argument| < 0.96875
        return AcosFromTable(table, 768 + ((magnitudeWord & 0x000fe000) >> 13) * 15, 10, argument,
                             highWord);
    if (magnitudeWord < 0x3ff00000) {  // 0.96875 <= |argument| < 1
        const double *inverseRootSeeds = CUJU_TABLE(INVERSE_ROOT_SEEDS);
        const double *powersOfTwo = CUJU_TABLE(POWERS_OF_TWO);
        double halfComplement = 0.5 * ((highWord > 0) ? (1.0 - argument) : (1.0 + argument));
        int32_t halfComplementWord = (int32_t)(DoubleBits(halfComplement) >> 32);
        double inverseRoot = inverseRootSeeds[(halfComplementWord & 0x001fffff) >> 14] *
                             powersOfTwo[511 - (halfComplementWord >> 21)];
        // The residual is `1 - inverseRoot * inverseRoot * halfComplement`.
        double residual = fma(-(inverseRoot * inverseRoot), halfComplement, 1.0);
        inverseRoot =
            inverseRoot * fma(fma(fma(residual, ACOS_ROOT3, ACOS_ROOT2), residual, ACOS_ROOT1),
                              residual, ACOS_ROOT0);
        double root = inverseRoot * halfComplement;
        // The refined root `root * refinement` has one use, the sum with `rootHigh`, so the FMA
        // build fuses the product into that sum.
        double refinement = fma(-(inverseRoot * 0.5), root, 1.5);
        // The two fused steps compute `(splitter * root + root) - splitter * root`.
        double rootHigh = fma(-root, ACOS_SPLITTER, fma(root, ACOS_SPLITTER, root));
        // The low part is `(halfComplement - rootHigh * rootHigh) / (refined root + rootHigh)`.
        double rootLow = fma(-rootHigh, rootHigh, halfComplement) / fma(refinement, root, rootHigh);
        double series = AcosSeries(halfComplement) * halfComplement;
        // Both branches read this product, so it stays unfused.
        double product = series * (rootHigh + rootLow);
        if (highWord < 0) {
            double correction = (ACOS_HALF_PI_LOW - rootLow) - product;
            double difference = ACOS_HALF_PI_HIGH - rootHigh;
            double result = difference + correction;
            return result + result;
        }
        double correction = rootLow + product;
        double result = rootHigh + correction;
        return result + result;
    }
    if (magnitudeWord == 0x3ff00000 && (uint32_t)bits == 0)
        return (highWord > 0) ? 0 : 2.0 * ACOS_HALF_PI_HIGH;
    return (argument - argument) / (argument - argument);
}

}  // namespace libm
}  // namespace cuju

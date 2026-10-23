// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
//
// Modifications for single precision expansion are
// Copyright (C) 2001 Stephen L. Moshier <moshier@na-net.ornl.gov>
// and are incorporated herein by permission of the author.  The author
// reserves the right to distribute this material elsewhere under different
// copying permissions.  These modifications are distributed here under
// the following terms:
//
//   This library is free software; you can redistribute it and/or
//   modify it under the terms of the GNU Lesser General Public
//   License as published by the Free Software Foundation; either
//   version 2.1 of the License, or (at your option) any later version.
//
//   This library is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//   Lesser General Public License for more details.
//
//   You should have received a copy of the GNU Lesser General Public
//   License along with this library; if not, see
//   <https://www.gnu.org/licenses/>.
//
// Ported to C++ for host and device from glibc 2.39 sysdeps/ieee754/flt-32/e_asinf.c by the cuju
// Authors, 2026.
// SPDX-License-Identifier: LGPL-2.1-or-later AND SunPro

// Float asinf of glibc 2.39, bit for bit on host and device.
// Exact: float arithmetic, no contraction; `(float)` marks each double literal that glibc converts.
#pragma once

#include "../host_device.h"
#include "common.h"

namespace cuju {
namespace libm {

CUJU_HOST_DEVICE_NO_INLINE float asinf(float argument) {
    const float one = (float)1.0000000000e+00, huge = (float)1.000e+30,
                halfPiHigh = 1.57079637050628662109375f, halfPiLow = -4.37113900018624283e-8f,
                quarterPiHigh = 0.785398185253143310546875f, coefficient0 = 1.666675248e-1f,
                coefficient1 = 7.495297643e-2f, coefficient2 = 4.547037598e-2f,
                coefficient3 = 2.417951451e-2f, coefficient4 = 4.216630880e-2f;
    // `square` is glibc's `t`, which takes the magnitude of the result in the last steps.
    float square, term, series, difference, rootLow, ratio, root;
    int32_t word = FloatWord(argument), magnitudeWord = word & 0x7fffffff;
    if (magnitudeWord == 0x3f800000) {
        return argument * halfPiHigh + argument * halfPiLow;
    } else if (magnitudeWord > 0x3f800000) {
        return (argument - argument) / (argument - argument);
    } else if (magnitudeWord < 0x3f000000) {
        if (magnitudeWord < 0x32000000) {
            // The comparison raises the inexact exception for a nonzero argument.
            if (huge + argument > one) return argument;
        } else {
            square = argument * argument;
            term = square * (coefficient0 +
                             square * (coefficient1 +
                                       square * (coefficient2 +
                                                 square * (coefficient3 + square * coefficient4))));
            return argument + argument * term;
        }
    }
    term = one - fabsf(argument);
    square = term * 0.5f;
    series = square *
             (coefficient0 +
              square * (coefficient1 +
                        square * (coefficient2 + square * (coefficient3 + square * coefficient4))));
    root = sqrtf(square);
    if (magnitudeWord >= 0x3F79999A) {
        square = halfPiHigh - (2.0f * (root + root * series) - halfPiLow);
    } else {
        // `term` is `root` cut to its high 12 significand bits, and `term + rootLow` is the square
        // root to more precision.
        term = FloatFromWord(FloatWord(root) & 0xfffff000);
        rootLow = (square - term * term) / (root + term);
        ratio = series;
        series = 2.0f * root * ratio - (halfPiLow - 2.0f * rootLow);
        difference = quarterPiHigh - 2.0f * term;
        square = quarterPiHigh - (series - difference);
    }
    if (word > 0) return square;
    return -square;
}

}  // namespace libm
}  // namespace cuju

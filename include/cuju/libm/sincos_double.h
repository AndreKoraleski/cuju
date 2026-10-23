// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Double sincos of glibc 2.39, whose results are those of its `sin` and `cos`.
#pragma once

#include "../host_device.h"
#include "sin_double.h"

namespace cuju {
namespace libm {

CUJU_HOST_DEVICE void sincos(double argument, double *sine, double *cosine) {
    *sine = sin(argument);
    *cosine = cos(argument);
}

}  // namespace libm
}  // namespace cuju

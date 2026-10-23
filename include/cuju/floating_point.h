// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The floating point environment that every host result needs.
#pragma once

#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#elif !defined(__aarch64__)
#error "cuju sets the floating point environment of x86-64 and aarch64 hosts only"
#endif

namespace cuju {

#if defined(__aarch64__)

// Exact: sets the C library's default FPCR, with round to nearest, no flush of subnormals and no
// default NaN, for its lifetime, as on x86-64 below. `torch.set_flush_denormal` sets FPCR.FZ there.
class DefaultFloatingPoint {
public:
    static constexpr unsigned long long DEFAULT = 0;

    DefaultFloatingPoint() : saved_(Read()) { Write(DEFAULT); }
    ~DefaultFloatingPoint() { Write(saved_); }
    DefaultFloatingPoint(const DefaultFloatingPoint &) = delete;
    DefaultFloatingPoint &operator=(const DefaultFloatingPoint &) = delete;

private:
    static unsigned long long Read() {
        unsigned long long value;
        __asm__ __volatile__("mrs %0, fpcr" : "=r"(value) : : "memory");
        return value;
    }
    static void Write(unsigned long long value) {
        __asm__ __volatile__("msr fpcr, %0" : : "r"(value) : "memory");
    }

    unsigned long long saved_;
};

#else

// Exact: sets the C library's default MXCSR, round to nearest and no flush of subnormals, for its
// lifetime. GRF runs there, and a caller may have changed it, as `torch.set_flush_denormal` does.
class DefaultFloatingPoint {
public:
    static constexpr unsigned DEFAULT = 0x1F80;

    DefaultFloatingPoint() : saved_(_mm_getcsr()) { _mm_setcsr(DEFAULT); }
    ~DefaultFloatingPoint() { _mm_setcsr(saved_); }
    DefaultFloatingPoint(const DefaultFloatingPoint &) = delete;
    DefaultFloatingPoint &operator=(const DefaultFloatingPoint &) = delete;

private:
    unsigned saved_;
};

#endif

}  // namespace cuju

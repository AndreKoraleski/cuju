// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Macros that compile each function once for the host and the device.
// Exact: both agree only under the flags of `CMakeLists.txt`, which mean no contraction, IEEE
// division and square root, and subnormals kept.
#pragma once

// nvcc defines no macro for its flags, so `CMakeLists.txt` guards the device ones.
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "cuju needs the exact floating-point semantics of IEEE 754, without fast math"
#endif
#if defined(__FLT_EVAL_METHOD__) && __FLT_EVAL_METHOD__ != 0
#error "cuju needs float and double operations evaluated in their own types"
#endif

#if defined(__CUDACC__) && !defined(__CUDA_ARCH__) && defined(__GNUC__)
// nvcc's host pass makes host and device declarations inline, so GCC warns about a `noinline` that
// still holds.
#pragma GCC diagnostic ignored "-Wattributes"
#endif

#if defined(__CUDACC__)
#define CUJU_HOST_DEVICE __host__ __device__ __forceinline__
#define CUJU_HOST_DEVICE_NO_INLINE __host__ __device__ __noinline__ inline
#define CUJU_FUNCTION __host__ __device__
// Speed: a step inlined whole overwhelms the register allocator and the instruction cache.
#define CUJU_FUNCTION_NO_INLINE __host__ __device__ __noinline__
#else
#define CUJU_HOST_DEVICE inline
#define CUJU_HOST_DEVICE_NO_INLINE inline
#define CUJU_FUNCTION
#define CUJU_FUNCTION_NO_INLINE
#endif

#include <cstdint>
#if !defined(__CUDA_ARCH__)
#include <stdexcept>
#endif

// A state no match reaches. The host throws `std::runtime_error`, the device traps and drops it.
#if defined(__CUDA_ARCH__)
#define CUJU_FATAL(message) __trap()
#else
#define CUJU_FATAL(message) throw std::runtime_error(message)
#endif

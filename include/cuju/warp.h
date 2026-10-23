// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Loops that split the work of one environment over the 32 lanes of its warp.
#pragma once

#include "host_device.h"

#include <cstring>
#include <type_traits>

namespace cuju {

// A device build with `CUJU_WARP_ENVIRONMENT` runs each environment on one warp. Every function
// of the simulation runs on every lane of the warp, converged and with the same locals, unless its
// comment says otherwise. Each helper here hands every lane the results of the sequential loop in
// its order, and is that loop in every other build. Inside a region the lanes diverge, so no
// shuffle, vote or barrier runs there, and a helper called there loops on the calling lane alone.
// Speed: each helper instantiates its lambdas once for both loops, since code size bounds a step.
#if defined(__CUDA_ARCH__) && defined(CUJU_WARP_ENVIRONMENT)
#define CUJU_WARP_LANES 1
CUJU_HOST_DEVICE int Lane() { return (int)(threadIdx.x & 31); }
// One byte of shared memory per warp, for blocks of up to 1024 threads.
__device__ __forceinline__ unsigned char &WarpNestFlag() {
    __shared__ unsigned char nestedFlags[32];
    return nestedFlags[threadIdx.x >> 5];
}
// A kernel calls it before any other helper of this file.
CUJU_HOST_DEVICE void WarpInitialize() {
    if (Lane() == 0) WarpNestFlag() = 0;
    __syncwarp();
}
CUJU_HOST_DEVICE void WarpRegionBegin() {
    __syncwarp();
    if (Lane() == 0) WarpNestFlag() = 1;
    __syncwarp();
}
// Every write made inside the region is visible to every lane afterwards.
CUJU_HOST_DEVICE void WarpRegionEnd() {
    __syncwarp();
    if (Lane() == 0) WarpNestFlag() = 0;
    __syncwarp();
}
CUJU_HOST_DEVICE bool WarpNested() { return WarpNestFlag() != 0; }
#else
#define CUJU_WARP_LANES 0
CUJU_HOST_DEVICE int Lane() { return 0; }
CUJU_HOST_DEVICE void WarpInitialize() {}
#endif

// A loop split by hand runs on 32 lanes outside a region of a warp build. Inside a region and in
// every other build it runs on one lane, and `WarpSync` does nothing.
#if CUJU_WARP_LANES
#define WARP_LANES (WarpNested() ? 1 : 32)
CUJU_HOST_DEVICE int SplitLane() { return WarpNested() ? 0 : Lane(); }
CUJU_HOST_DEVICE void WarpSync() {
    if (!WarpNested()) __syncwarp();
}
#else
#define WARP_LANES 1
CUJU_HOST_DEVICE int SplitLane() { return 0; }
CUJU_HOST_DEVICE void WarpSync() {}
#endif

// Speed: blocks of 128 threads, at least 6 per multiprocessor, cap a thread at 80 registers, 6.5%
// faster than 128 registers on the RTX 4090 with 16384 environments, while 72 and 64 are slower.
#ifndef CUJU_BLOCK_THREADS
#define CUJU_BLOCK_THREADS 128
#endif
#ifndef CUJU_MINIMUM_BLOCKS
#define CUJU_MINIMUM_BLOCKS 6
#endif
#if defined(__CUDACC__)
#define CUJU_ENVIRONMENT_KERNEL_BOUNDS __launch_bounds__(CUJU_BLOCK_THREADS, CUJU_MINIMUM_BLOCKS)
#else
#define CUJU_ENVIRONMENT_KERNEL_BOUNDS
#endif

#if CUJU_WARP_LANES
CUJU_HOST_DEVICE void WarpSplit(bool &nested, int &lane, int &step) {
    nested = WarpNested();
    lane = nested ? 0 : Lane();
    step = nested ? 1 : 32;
}
#endif

// `emit` runs in index order for each index that `predicate` selects. The predicates of a chunk
// of 32 run in a region before its emits, so `emit(index)` may overwrite data up to `index` and a
// list compacts in place. A predicate writes nothing that another lane reads.
template <class Predicate, class Emit>
CUJU_HOST_DEVICE void WarpFilterEmit(int first, int end, Predicate predicate, Emit emit) {
#if CUJU_WARP_LANES
    bool nested;
    int lane, step;
    WarpSplit(nested, lane, step);
    for (int base = first; base < end; base += step) {
        const int index = base + lane;
        if (!nested) WarpRegionBegin();
        const bool selected = index < end && predicate(index);
        if (!nested) WarpRegionEnd();
        unsigned selectedMask = nested ? (unsigned)selected : __ballot_sync(0xffffffffu, selected);
        while (selectedMask) {
            emit(base + __ffs(selectedMask) - 1);
            selectedMask &= selectedMask - 1;
        }
    }
#else
    for (int index = first; index < end; index++)
        if (predicate(index)) emit(index);
#endif
}

// The calls run in a region in no fixed order, so no iteration reads what another writes, and a
// result goes to memory that every lane addresses, since a caller's local changes only in one lane.
template <class Body>
CUJU_HOST_DEVICE void WarpFor(int count, Body body) {
#if CUJU_WARP_LANES
    bool nested;
    int lane, step;
    WarpSplit(nested, lane, step);
    if (!nested) WarpRegionBegin();
    for (int index = lane; index < count; index += step) body(index);
    if (!nested) WarpRegionEnd();
#else
    for (int index = 0; index < count; index++) body(index);
#endif
}

// `body` runs in index order on every lane. `prefetch` runs up to 32 iterations ahead in a
// region, so it reads nothing that `body` writes. `Prefetched` is trivially copyable.
template <class Prefetched, class Prefetch, class Body>
CUJU_HOST_DEVICE void WarpPipeline(int count, Prefetch prefetch, Body body) {
#if CUJU_WARP_LANES
    static_assert(sizeof(Prefetched) % 4 == 0, "WarpPipeline: 4-byte words");
    bool nested;
    int lane, step;
    WarpSplit(nested, lane, step);
    for (int base = 0; base < count; base += step) {
        Prefetched ownValue;
        if (!nested) WarpRegionBegin();
        if (base + lane < count) ownValue = prefetch(base + lane);
        if (!nested) WarpRegionEnd();
        // Speed: words through `memcpy` stay in registers. The value's own address would put it in
        // local memory.
        constexpr int WORD_COUNT = (int)(sizeof(Prefetched) / 4);
        uint32_t ownWords[WORD_COUNT];
        memcpy(ownWords, &ownValue, sizeof(Prefetched));
        const int chunkCount = count - base < step ? count - base : step;
        for (int sourceLane = 0; sourceLane < chunkCount; sourceLane++) {
            Prefetched value;
            if (nested) {
                value = ownValue;
            } else {
                uint32_t words[WORD_COUNT];
#pragma unroll
                for (int word = 0; word < WORD_COUNT; word++)
                    words[word] = __shfl_sync(0xffffffffu, ownWords[word], sourceLane);
                memcpy(&value, words, sizeof(Prefetched));
            }
            body(base + sourceLane, value);
        }
    }
#else
    for (int index = 0; index < count; index++) body(index, prefetch(index));
#endif
}

// Every lane writes the whole `output`, a caller's local or memory every lane addresses.
// `compute` runs in a region and writes nothing that another lane reads.
template <class Value, class Compute>
CUJU_HOST_DEVICE void WarpMap(int count, Compute compute, Value *output) {
#if CUJU_WARP_LANES
    static_assert(sizeof(Value) == 4 || sizeof(Value) == 8, "WarpMap: 4- or 8-byte values");
    bool nested;
    int lane, step;
    WarpSplit(nested, lane, step);
    for (int base = 0; base < count; base += step) {
        const int index = base + lane;
        if (!nested) WarpRegionBegin();
        Value value = index < count ? compute(index) : Value();
        if (!nested) WarpRegionEnd();
        if (nested) {
            output[index] = value;
        } else {
            const int chunkCount = count - base < 32 ? count - base : 32;
            for (int sourceLane = 0; sourceLane < chunkCount; sourceLane++)
                output[base + sourceLane] = __shfl_sync(0xffffffffu, value, sourceLane);
        }
    }
#else
    for (int index = 0; index < count; index++) output[index] = compute(index);
#endif
}

// Zeros when `source` is null. Both arrays lie in memory every lane addresses, never in a lane's
// local. Speed: one loop, not an unrolled struct copy, to save code size.
CUJU_HOST_DEVICE_NO_INLINE void WarpCopyWords(uint32_t *destination, const uint32_t *source,
                                              int wordCount) {
#if CUJU_WARP_LANES
    WarpSync();
    const int lane = SplitLane(), step = WARP_LANES;
    for (int word = lane; word < wordCount; word += step)
        destination[word] = source ? source[word] : 0u;
    WarpSync();
#else
    if (source)
        memcpy(destination, source, sizeof(uint32_t) * wordCount);
    else
        memset(destination, 0, sizeof(uint32_t) * wordCount);
#endif
}
template <class T>
CUJU_HOST_DEVICE void WarpCopy(T &destination, const T &source) {
    static_assert(sizeof(T) % 4 == 0 && alignof(T) >= 4, "WarpCopy: 4-byte words");
    WarpCopyWords(reinterpret_cast<uint32_t *>(&destination),
                  reinterpret_cast<const uint32_t *>(&source), (int)(sizeof(T) / 4));
}
template <class T>
CUJU_HOST_DEVICE void WarpZero(T &destination) {
    static_assert(sizeof(T) % 4 == 0 && alignof(T) >= 4, "WarpZero: 4-byte words");
    static_assert(std::is_trivially_default_constructible_v<T>,
                  "WarpZero: a type with default member values takes WarpReset");
    WarpCopyWords(reinterpret_cast<uint32_t *>(&destination), nullptr, (int)(sizeof(T) / 4));
}

#if defined(__CUDACC__)
// A constant initializer, so the value exists before any kernel runs. nvcc refuses a const device
// variable template on Windows, and nothing writes this one there either.
#if defined(_WIN32)
template <class T>
__device__ T devicePrototype{};
#else
template <class T>
__device__ const T devicePrototype{};
#endif
#endif
template <class T>
inline const T hostPrototype{};

// Copies a prototype of `T{}`, so member defaults survive, at one load per word more than
// `WarpZero`. `T{}` is a constant expression.
template <class T>
CUJU_HOST_DEVICE void WarpReset(T &destination) {
#if defined(__CUDA_ARCH__)
    const T &prototype = devicePrototype<T>;
#else
    const T &prototype = hostPrototype<T>;
#endif
    WarpCopy(destination, prototype);
}

}  // namespace cuju

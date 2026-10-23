// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The device copies of the read-only tables, which `batch::UploadTables` fills.
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../batch/tables.h"

// Throws `std::runtime_error` with the CUDA message, which the C API reports as a failed call.
#define CUJU_CUDA_CHECK(expression)                                       \
    do {                                                                  \
        cudaError_t cujuCudaStatus = (expression);                        \
        if (cujuCudaStatus != cudaSuccess)                                \
            throw std::runtime_error(std::string(#expression) + ": " +    \
                                     cudaGetErrorString(cujuCudaStatus)); \
    } while (0)

namespace cuju {
namespace gpu {

using batch::Rebase;
using batch::UploadTables;

// `bytes` as `12.3 MB (12345678 bytes)`.
inline std::string ByteCount(size_t bytes) {
    const char *const units[] = {"bytes", "KB", "MB", "GB", "TB"};
    double value = (double)bytes;
    int unit = 0;
    while (value >= 1000 && unit < 4) {
        value /= 1000;
        unit++;
    }
    char text[64];
    if (unit == 0)
        snprintf(text, sizeof(text), "%zu bytes", bytes);
    else
        snprintf(text, sizeof(text), "%.1f %s (%zu bytes)", value, units[unit], bytes);
    return text;
}

// The error of a GPU without the memory that `what` needs, with the bytes asked for, those of the
// batch so far and those free, and what can make room.
inline std::runtime_error OutOfMemory(const std::string &what, size_t bytes, size_t held) {
    cudaGetLastError();
    size_t available = 0, total = 0;
    const bool known = cudaMemGetInfo(&available, &total) == cudaSuccess;
    cudaGetLastError();
    return std::runtime_error(
        "the GPU is out of memory: " + what + " needs " + ByteCount(bytes) +
        (held ? " more, after the " + ByteCount(held) + " that the batch holds," : "") +
        (known ? " and the GPU has " + ByteCount(available) + " free of " + ByteCount(total) : "") +
        ". Create the batch before other large allocations, release PyTorch's cached memory "
        "with torch.cuda.empty_cache(), see what is free with torch.cuda.mem_get_info(), or "
        "make fewer environments");
}

// `cudaMalloc`, whose failure for lack of memory says what it needed. `held` counts the bytes the
// caller holds already.
// Throws `std::runtime_error` with `what` and the CUDA message unless `status` is a success.
inline void CheckCuda(cudaError_t status, const std::string &what) {
    if (status != cudaSuccess) throw std::runtime_error(what + ": " + cudaGetErrorString(status));
}

inline void *DeviceAllocate(size_t bytes, const std::string &what, size_t held = 0) {
    void *pointer = nullptr;
    const cudaError_t status = cudaMalloc(&pointer, bytes);
    if (status == cudaErrorMemoryAllocation) throw OutOfMemory(what, bytes, held);
    if (status != cudaSuccess)
        throw std::runtime_error("cannot allocate " + ByteCount(bytes) + " on the GPU for " + what +
                                 ": " + cudaGetErrorString(status));
    return pointer;
}

// Device memory from `cudaMalloc`, or from the caller's `allocate` and `release`, such as those of
// PyTorch's caching allocator.
struct DeviceAllocator {
    void *(*allocate)(size_t bytes, void *context) = nullptr;
    void (*release)(void *pointer, void *context) = nullptr;
    void *context = nullptr;

    // `what` names the memory and `held` counts the bytes the caller holds already, for the error
    // of a GPU out of memory.
    void *Allocate(size_t bytes, const std::string &what, size_t held = 0) const {
        if (!allocate) return DeviceAllocate(bytes, what, held);
        void *pointer = allocate(bytes, context);
        if (!pointer) throw OutOfMemory(what + " through the allocator of the batch", bytes, held);
        return pointer;
    }
    // `cudaFree` waits for the device, but memory that `release` takes back may serve another
    // allocation at once, so the device must finish with it first.
    void Free(const std::vector<void *> &pointers) const {
        if (release && !pointers.empty()) cudaDeviceSynchronize();
        for (void *pointer : pointers)
            release ? release(pointer, context) : (void)cudaFree(pointer);
    }
};

// Owns the device copies and frees them. `Upload` copies synchronously.
struct DeviceTables {
    // The host-only tables stay behind.
    static constexpr bool HOST = false;
    std::vector<void *> allocations;
    simulation::SimulationContext context{};  // Device pointers only.
    DeviceAllocator allocator;
    size_t bytes = 0;  // Requested by `Upload`.

    DeviceTables() = default;
    DeviceTables(const DeviceTables &) = delete;
    DeviceTables &operator=(const DeviceTables &) = delete;
    ~DeviceTables() { allocator.Free(allocations); }

    template <class T>
    T *Upload(const T *data, size_t count) {
        static_assert(std::is_trivially_copyable<T>::value,
                      "device tables must be trivially copyable");
        if (count == 0) return nullptr;
        T *device = static_cast<T *>(allocator.Allocate(count * sizeof(T), "a table of the batch"));
        allocations.push_back(device);
        bytes += count * sizeof(T);
        CUJU_CUDA_CHECK(cudaMemcpy(device, data, count * sizeof(T), cudaMemcpyHostToDevice));
        return device;
    }
    template <class T>
    T *Upload(const std::vector<T> &values) {
        return Upload(values.data(), values.size());
    }
};

}  // namespace gpu
}  // namespace cuju

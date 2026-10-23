// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// GPU backend, with one environment per warp under `CUJU_WARP_ENVIRONMENT`, else one per thread.
#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "../batch/batch_environment.h"
#include "device_data.h"

namespace cuju {
namespace gpu {

using namespace batch;

#if defined(CUJU_WARP_ENVIRONMENT)
constexpr int ENVIRONMENT_THREADS = 32;
#else
constexpr int ENVIRONMENT_THREADS = 1;
#endif

__device__ inline int EnvironmentIndex() {
    return (int)((blockIdx.x * blockDim.x + threadIdx.x) / ENVIRONMENT_THREADS);
}

// A null `mask` means every environment. `seeds` and `reverse` are as in `ResetEnvironment`.
__global__ void CUJU_ENVIRONMENT_KERNEL_BOUNDS ResetKernel(EnvironmentBuffers buffers,
                                                           Variants variants, const uint8_t *mask,
                                                           const int64_t *seeds,
                                                           const int32_t *reverse) {
    const int environment = EnvironmentIndex();
    if (environment >= buffers.environmentCount || (mask && !mask[environment])) return;
    WarpInitialize();
    ResetAndWriteEnvironment(buffers, variants, environment, seeds, reverse);
}

// With `autoReset`, an ended episode resets inside this kernel while the other warps keep
// stepping, so a reset adds no wait to the batch. `actions` as in `StepEnvironment`.
__global__ void CUJU_ENVIRONMENT_KERNEL_BOUNDS StepKernel(EnvironmentBuffers buffers,
                                                          Variants variants, const int32_t *actions,
                                                          bool autoReset) {
    const int environment = EnvironmentIndex();
    if (environment >= buffers.environmentCount) return;
    WarpInitialize();
    StepEnvironment(buffers, variants, environment, actions, autoReset);
}

// One thread per agent, not per environment. `mask` is as in `ResetKernel`.
__global__ void OutputKernel(EnvironmentBuffers buffers, const uint8_t *mask, bool afterStep) {
    const int output = blockIdx.x * blockDim.x + threadIdx.x;
    if (output >= buffers.environmentCount * buffers.agents) return;
    const int environment = output / buffers.agents, agent = output % buffers.agents;
    if (mask && !mask[environment]) return;
    const EnvironmentSlot &slot = buffers.slots[environment];
    if (agent >= slot.leftAgents + slot.rightAgents) return;
    WriteAgentOutputs(buffers, environment, agent, afterStep);
}

// Device-memory buffers. `Reset` and `Step` launch on the caller's stream without waiting.
class CudaBackend {
public:
    using Stream = cudaStream_t;
    static constexpr bool HOST_MEMORY = false;

    CudaBackend() = default;
    CudaBackend(const CudaBackend &) = delete;
    CudaBackend &operator=(const CudaBackend &) = delete;
    ~CudaBackend() { tables_.allocator.Free(allocations_); }

    void Configure(const BatchEnvironmentConfiguration &configuration) {
        RequireKernels();
        threadsPerBlock_ = configuration.threadsPerBlock;
        if (!configuration.allocate != !configuration.release)
            throw std::invalid_argument("expected both an allocate and a release function");
        tables_.allocator = DeviceAllocator{configuration.allocate, configuration.release,
                                            configuration.allocatorContext};
        if (configuration.stackBytes) {
            const cudaError_t status =
                cudaDeviceSetLimit(cudaLimitStackSize, configuration.stackBytes);
            if (status == cudaErrorMemoryAllocation)
                throw OutOfMemory("a CUDA stack of " + ByteCount(configuration.stackBytes) +
                                      " for each thread the GPU runs at once",
                                  configuration.stackBytes * ResidentThreads(), 0);
            CheckCuda(status, "cannot set the CUDA stack to " +
                                  ByteCount(configuration.stackBytes) + " per thread");
        }
    }
    int Threads() const { return 0; }
    template <class T>
    T *Allocate(size_t count) {
        const size_t bytes = sizeof(T) * (count ? count : 1);
        T *pointer =
            static_cast<T *>(tables_.allocator.Allocate(bytes, "a buffer of the batch", held_));
        allocations_.push_back(pointer);
        held_ += bytes;
        CUJU_CUDA_CHECK(cudaMemset(pointer, 0, bytes));
        return pointer;
    }
    // The bytes that `Allocate` and the tables requested. The driver rounds each up.
    size_t AllocatedBytes() const { return held_ + tables_.bytes; }
    DeviceTables &Tables() { return tables_; }
    void ToDevice(void *destination, const void *source, size_t bytes) const {
        CUJU_CUDA_CHECK(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice));
    }
    void ToDeviceAsync(void *destination, const void *source, size_t bytes, Stream stream) const {
        CUJU_CUDA_CHECK(
            cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream));
    }
    void ToHost(void *destination, const void *source, size_t bytes) const {
        CUJU_CUDA_CHECK(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost));
    }
    void Copy(void *destination, const void *source, size_t bytes) const {
        CUJU_CUDA_CHECK(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToDevice));
    }
    void Fill(void *destination, int value, size_t bytes) const {
        CUJU_CUDA_CHECK(cudaMemset(destination, value, bytes));
    }
    // Launches both kernels on no environment so the driver reserves their local memory now. A
    // batch that does not fit fails at its creation rather than at its first reset.
    void Reserve(const EnvironmentBuffers &buffers, const Variants &variants) {
        EnvironmentBuffers none = buffers;
        none.environmentCount = 0;
        ResetKernel<<<1, threadsPerBlock_>>>(none, variants, nullptr, nullptr, nullptr);
        StepKernel<<<1, threadsPerBlock_>>>(none, variants, nullptr, false);
        for (const cudaError_t status : {cudaGetLastError(), cudaDeviceSynchronize()}) {
            // The driver reserves the local memory of every thread that the GPU runs at once.
            if (status == cudaErrorMemoryAllocation) {
                size_t stack = 0;
                cudaDeviceGetLimit(&stack, cudaLimitStackSize);
                throw OutOfMemory("the local memory of the kernels, about " + ByteCount(stack) +
                                      " for each of the " + std::to_string(ResidentThreads()) +
                                      " threads that the GPU runs at once,",
                                  stack * ResidentThreads(), held_);
            }
            CheckCuda(status, "cannot launch the kernels of cuju");
        }
    }
    // The local memory that the driver adds for the kernels on the current device at their first
    // launch, which `Reserve` makes. The driver holds the stack limit for each thread that the GPU
    // runs at once, from the creation of the context, and a launch raises the limit to the local
    // memory of the kernel and keeps it, so this is 0 once a batch has launched on the device.
    static size_t LocalMemoryBytes() {
        RequireKernels();
        size_t perThread = 0, limit = 0;
        for (const void *kernel :
             {(const void *)ResetKernel, (const void *)StepKernel, (const void *)OutputKernel}) {
            cudaFuncAttributes attributes;
            CheckCuda(cudaFuncGetAttributes(&attributes, kernel), "cannot read the CUDA kernels");
            perThread = std::max(perThread, attributes.localSizeBytes);
        }
        CheckCuda(cudaDeviceGetLimit(&limit, cudaLimitStackSize), "cannot read the CUDA stack");
        return perThread > limit ? (perThread - limit) * ResidentThreads() : 0;
    }
    void Synchronize(Stream stream) const { CUJU_CUDA_CHECK(cudaStreamSynchronize(stream)); }
    void SynchronizeAll() const { CUJU_CUDA_CHECK(cudaDeviceSynchronize()); }

    void Reset(const EnvironmentBuffers &buffers, const Variants &variants, const uint8_t *mask,
               const int64_t *seeds, const int32_t *reverse, Stream stream) {
        ResetKernel<<<EnvironmentGrid(buffers.environmentCount), threadsPerBlock_, 0, stream>>>(
            buffers, variants, mask, seeds, reverse);
        CUJU_CUDA_CHECK(cudaGetLastError());
    }
    void Step(const EnvironmentBuffers &buffers, const Variants &variants, const int32_t *actions,
              bool autoReset, Stream stream) {
        StepKernel<<<EnvironmentGrid(buffers.environmentCount), threadsPerBlock_, 0, stream>>>(
            buffers, variants, actions, autoReset);
        CUJU_CUDA_CHECK(cudaGetLastError());
    }
    void Outputs(const EnvironmentBuffers &buffers, const uint8_t *mask, bool afterStep) {
        const int grid = std::max(1, (buffers.environmentCount * buffers.agents + 127) / 128);
        OutputKernel<<<grid, 128>>>(buffers, mask, afterStep);
        CUJU_CUDA_CHECK(cudaGetLastError());
    }

private:
    // Without this, a library built for other GPUs fails only at its first launch, naming no cause.
    static void RequireKernels() {
        cudaFuncAttributes attributes;
        const cudaError_t found = cudaFuncGetAttributes(&attributes, StepKernel);
        if (found == cudaErrorInvalidDeviceFunction || found == cudaErrorNoKernelImageForDevice) {
            cudaGetLastError();
            int device = 0, major = 0, minor = 0;
            cudaGetDevice(&device);
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
            throw std::runtime_error(
                "this build of cuju holds no CUDA kernel for the GPU of compute "
                "capability " +
                std::to_string(major) + "." + std::to_string(minor) +
                "; build cuju from its repository for this GPU");
        }
        CheckCuda(found, "cannot read the CUDA kernels of cuju");
    }
    static size_t ResidentThreads() {
        int device = 0, processors = 0, threads = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&processors, cudaDevAttrMultiProcessorCount, device);
        cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerMultiProcessor, device);
        return (size_t)processors * threads;
    }
    int EnvironmentGrid(int count) const {
        const long threads = (long)count * ENVIRONMENT_THREADS;
        return (int)((threads + threadsPerBlock_ - 1) / threadsPerBlock_);
    }

    DeviceTables tables_;
    std::vector<void *> allocations_;
    int threadsPerBlock_ = 64;
    size_t held_ = 0;  // The bytes of `allocations_`.
};

// A step or a reset launches one kernel. The outputs stay in device buffers until the next call.
using BatchEnvironment = batch::BackendBatchEnvironment<CudaBackend>;

}  // namespace gpu
}  // namespace cuju

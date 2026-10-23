// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The memory a batch requests at its creation, counted without allocating it.
#pragma once

#include <type_traits>
#include <utility>
#include <vector>

#include "batch_environment.h"

namespace cuju {
namespace batch {

// Counts what a backend's tables upload, as `HostTables` and `DeviceTables` do, and keeps none.
template <bool HOST_TABLES>
struct MeasuringTables {
    static constexpr bool HOST = HOST_TABLES;
    simulation::SimulationContext context{};
    size_t bytes = 0;

    // The host values stand in for the copies, which the creation never reads.
    template <class T>
    T *Upload(const T *data, size_t count) {
        if (count == 0) return nullptr;
        bytes += count * sizeof(T);
        return const_cast<T *>(data);
    }
    template <class T>
    T *Upload(const std::vector<T> &values) {
        return Upload(values.data(), values.size());
    }
};

// Stands in for `Real`, `CpuBackend` or `CudaBackend`, through a batch's creation. It counts every
// `Allocate` as `Real` does and allocates nothing.
template <class Real>
class MeasuringBackend {
public:
    using Stream = typename Real::Stream;
    static constexpr bool HOST_MEMORY = Real::HOST_MEMORY;
    using Counter = MeasuringTables<
        std::remove_reference<decltype(std::declval<Real &>().Tables())>::type::HOST>;

    void Configure(const BatchEnvironmentConfiguration &) {}
    void Reserve(const EnvironmentBuffers &, const Variants &) {}
    int Threads() const { return 0; }
    template <class T>
    T *Allocate(size_t count) {
        allocatedBytes_ += sizeof(T) * (count ? count : 1);
        return nullptr;
    }
    size_t AllocatedBytes() const { return allocatedBytes_ + tables_.bytes; }
    Counter &Tables() { return tables_; }
    void ToDevice(void *, const void *, size_t) const {}

private:
    Counter tables_;
    size_t allocatedBytes_ = 0;
};

// The bytes that a batch of `configuration` on `Real` requests at its creation, as
// `BackendBatchEnvironment::MemoryBytes` reports right after it. It reads the bundles and the
// animation tables, as the creation does.
template <class Real>
size_t EstimateMemory(const BatchEnvironmentConfiguration &configuration) {
    return BackendBatchEnvironment<MeasuringBackend<Real>>(configuration).MemoryBytes();
}

}  // namespace batch
}  // namespace cuju

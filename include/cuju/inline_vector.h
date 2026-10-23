// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// A list of fixed capacity stored inline. Outgrowing the capacity stops as `CUJU_FATAL` does.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "host_device.h"

namespace cuju {

#if defined(__CUDACC__)
CUJU_HOST_DEVICE void InlineVectorOverflow() {
#elif defined(_MSC_VER)
// Speed: the throw stays out of line. MSVC has no `cold`.
__declspec(noinline) inline void InlineVectorOverflow() {
#else
// Speed: the throw stays cold and out of line, away from the hot code of every list.
__attribute__((cold, noinline)) inline void InlineVectorOverflow() {
#endif
    CUJU_FATAL("a fixed-capacity list outgrows its capacity");
}

// Speed: the elements stay unconstructed, as `T` is trivially copyable. Zero-filling the capacity
// of every list, as a default `Vector3` does, was a large share of the device's local memory
// traffic.
template <class T, int Capacity>
struct InlineVector {
    int32_t count = 0;
    union {
        T elements[Capacity];
    };
    CUJU_HOST_DEVICE InlineVector() {}

    CUJU_HOST_DEVICE size_t size() const { return (size_t)count; }
    CUJU_HOST_DEVICE bool empty() const { return count == 0; }
    CUJU_HOST_DEVICE static constexpr int capacity() { return Capacity; }
    CUJU_HOST_DEVICE void clear() { count = 0; }
    CUJU_HOST_DEVICE void push_back(const T &value) {
        if (count >= Capacity) InlineVectorOverflow();
        elements[count++] = value;
    }
    CUJU_HOST_DEVICE void pop_back() { count--; }
    // Added elements stay unwritten.
    CUJU_HOST_DEVICE void resize(size_t newCount) {
        if ((int)newCount > Capacity) InlineVectorOverflow();
        count = (int32_t)newCount;
    }
    CUJU_HOST_DEVICE T &operator[](size_t index) { return elements[index]; }
    CUJU_HOST_DEVICE const T &operator[](size_t index) const { return elements[index]; }
    CUJU_HOST_DEVICE T &at(size_t index) { return elements[index]; }
    CUJU_HOST_DEVICE const T &at(size_t index) const { return elements[index]; }
    CUJU_HOST_DEVICE T &front() { return elements[0]; }
    CUJU_HOST_DEVICE const T &front() const { return elements[0]; }
    CUJU_HOST_DEVICE T &back() { return elements[count - 1]; }
    CUJU_HOST_DEVICE const T &back() const { return elements[count - 1]; }
    CUJU_HOST_DEVICE T *begin() { return elements; }
    CUJU_HOST_DEVICE T *end() { return elements + count; }
    CUJU_HOST_DEVICE const T *begin() const { return elements; }
    CUJU_HOST_DEVICE const T *end() const { return elements + count; }
    CUJU_HOST_DEVICE T *data() { return elements; }
    CUJU_HOST_DEVICE const T *data() const { return elements; }
    CUJU_HOST_DEVICE T *erase(T *position) {
        for (T *element = position; element + 1 < elements + count; element++)
            *element = *(element + 1);
        count--;
        return position;
    }
    CUJU_HOST_DEVICE T *insert(T *position, const T &value) {
        if (count >= Capacity) InlineVectorOverflow();
        for (T *element = elements + count; element > position; element--)
            *element = *(element - 1);
        *position = value;
        count++;
        return position;
    }
    // Speed: only the live elements move. A whole copy would move the full capacity.
    CUJU_HOST_DEVICE void swap(InlineVector &other) {
        const int liveCount = count > other.count ? count : other.count;
        for (int index = 0; index < liveCount; index++) {
            T element = elements[index];
            elements[index] = other.elements[index];
            other.elements[index] = element;
        }
        const int32_t ownCount = count;
        count = other.count;
        other.count = ownCount;
    }
};

}  // namespace cuju

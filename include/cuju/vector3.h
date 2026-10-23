// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// GRF's float vector of three components.
// Exact: only `GetLength` uses double. `Normalize` multiplies by `1.0f / sqrtf(dot)`.
#pragma once

#include <cmath>

#include "host_device.h"

namespace cuju {

// The float after the double 1e-6, so `value < ABOVE_ONE_MILLIONTH` exactly when
// `(double)value < 0.000001`.
constexpr float ABOVE_ONE_MILLIONTH = 0x1.0c6f7cp-20f;
static_assert((double)0x1.0c6f7ap-20f < 0.000001 && 0.000001 < (double)ABOVE_ONE_MILLIONTH,
              "the double 1e-6 lies between two neighboring floats");

struct Vector3 {
    float coordinates[3];

    CUJU_HOST_DEVICE constexpr Vector3() : coordinates{0, 0, 0} {}
    CUJU_HOST_DEVICE constexpr Vector3(float value) : coordinates{value, value, value} {}
    CUJU_HOST_DEVICE constexpr Vector3(float x, float y, float z) : coordinates{x, y, z} {}

    CUJU_HOST_DEVICE void Mirror() {
        coordinates[0] = -coordinates[0];
        coordinates[1] = -coordinates[1];
    }
    CUJU_HOST_DEVICE void Set(float value) {
        coordinates[0] = coordinates[1] = coordinates[2] = value;
    }
    CUJU_HOST_DEVICE void Set(float x, float y, float z) {
        coordinates[0] = x;
        coordinates[1] = y;
        coordinates[2] = z;
    }

    CUJU_HOST_DEVICE bool operator==(const Vector3 &other) const {
        return coordinates[0] == other.coordinates[0] && coordinates[1] == other.coordinates[1] &&
               coordinates[2] == other.coordinates[2];
    }
    CUJU_HOST_DEVICE bool operator!=(const Vector3 &other) const { return !(*this == other); }

    CUJU_HOST_DEVICE Vector3 operator*(float factor) const {
        return Vector3(coordinates[0] * factor, coordinates[1] * factor, coordinates[2] * factor);
    }
    CUJU_HOST_DEVICE Vector3 operator*(const Vector3 &factors) const {
        return Vector3(coordinates[0] * factors.coordinates[0],
                       coordinates[1] * factors.coordinates[1],
                       coordinates[2] * factors.coordinates[2]);
    }
    CUJU_HOST_DEVICE Vector3 &operator*=(float factor) {
        coordinates[0] *= factor;
        coordinates[1] *= factor;
        coordinates[2] *= factor;
        return *this;
    }
    CUJU_HOST_DEVICE Vector3 &operator*=(const Vector3 &factors) {
        coordinates[0] *= factors.coordinates[0];
        coordinates[1] *= factors.coordinates[1];
        coordinates[2] *= factors.coordinates[2];
        return *this;
    }
    CUJU_HOST_DEVICE Vector3 operator/(float divisor) const {
        return Vector3(coordinates[0] / divisor, coordinates[1] / divisor,
                       coordinates[2] / divisor);
    }
    CUJU_HOST_DEVICE Vector3 &operator/=(const Vector3 &divisors) {
        coordinates[0] /= divisors.coordinates[0];
        coordinates[1] /= divisors.coordinates[1];
        coordinates[2] /= divisors.coordinates[2];
        return *this;
    }
    CUJU_HOST_DEVICE Vector3 &operator+=(float addend) {
        coordinates[0] += addend;
        coordinates[1] += addend;
        coordinates[2] += addend;
        return *this;
    }
    CUJU_HOST_DEVICE Vector3 &operator+=(const Vector3 &other) {
        coordinates[0] += other.coordinates[0];
        coordinates[1] += other.coordinates[1];
        coordinates[2] += other.coordinates[2];
        return *this;
    }
    CUJU_HOST_DEVICE Vector3 &operator-=(const Vector3 &other) {
        coordinates[0] -= other.coordinates[0];
        coordinates[1] -= other.coordinates[1];
        coordinates[2] -= other.coordinates[2];
        return *this;
    }
    CUJU_HOST_DEVICE Vector3 operator+(const Vector3 &other) const {
        return Vector3(coordinates[0] + other.coordinates[0], coordinates[1] + other.coordinates[1],
                       coordinates[2] + other.coordinates[2]);
    }
    CUJU_HOST_DEVICE Vector3 operator+(float addend) const {
        return Vector3(coordinates[0] + addend, coordinates[1] + addend, coordinates[2] + addend);
    }
    CUJU_HOST_DEVICE Vector3 operator-(const Vector3 &other) const {
        return Vector3(coordinates[0] - other.coordinates[0], coordinates[1] - other.coordinates[1],
                       coordinates[2] - other.coordinates[2]);
    }
    CUJU_HOST_DEVICE Vector3 operator-() const {
        return Vector3(-coordinates[0], -coordinates[1], -coordinates[2]);
    }
    CUJU_HOST_DEVICE Vector3 operator-(float subtrahend) const {
        return Vector3(coordinates[0] - subtrahend, coordinates[1] - subtrahend,
                       coordinates[2] - subtrahend);
    }

    CUJU_HOST_DEVICE Vector3 GetCrossProduct(const Vector3 &other) const {
        return Vector3(
            coordinates[1] * other.coordinates[2] - coordinates[2] * other.coordinates[1],
            coordinates[2] * other.coordinates[0] - coordinates[0] * other.coordinates[2],
            coordinates[0] * other.coordinates[1] - coordinates[1] * other.coordinates[0]);
    }
    CUJU_HOST_DEVICE float GetDotProduct(const Vector3 &other) const {
        return coordinates[0] * other.coordinates[0] + coordinates[1] * other.coordinates[1] +
               coordinates[2] * other.coordinates[2];
    }

    CUJU_HOST_DEVICE void Normalize(const Vector3 &fallback) {
        if (fabsf(coordinates[0]) < 0.000001f && fabsf(coordinates[1]) < 0.000001f &&
            fabsf(coordinates[2]) < 0.000001f) {
            coordinates[0] = fallback.coordinates[0];
            coordinates[1] = fallback.coordinates[1];
            coordinates[2] = fallback.coordinates[2];
        } else {
            float inverseLength = 1.0f / sqrtf(GetDotProduct(*this));
            coordinates[0] *= inverseLength;
            coordinates[1] *= inverseLength;
            coordinates[2] *= inverseLength;
        }
    }
    // A zero vector becomes NaN.
    CUJU_HOST_DEVICE void Normalize() {
        float inverseLength = 1.0f / sqrtf(GetDotProduct(*this));
        coordinates[0] *= inverseLength;
        coordinates[1] *= inverseLength;
        coordinates[2] *= inverseLength;
    }
    CUJU_HOST_DEVICE Vector3 GetNormalized(const Vector3 &fallback) const {
        Vector3 temporary(*this);
        temporary.Normalize(fallback);
        return temporary;
    }
    CUJU_HOST_DEVICE Vector3 GetNormalized() const {
        Vector3 temporary(*this);
        temporary.Normalize();
        return temporary;
    }

    CUJU_HOST_DEVICE float GetLength() const {
        double x = coordinates[0], y = coordinates[1], z = coordinates[2];
        float length = (float)sqrt(x * x + y * y + z * z);
        if (length < ABOVE_ONE_MILLIONTH) length = 0;
        return length;
    }
    CUJU_HOST_DEVICE float GetSquaredLength() const {
        return coordinates[0] * coordinates[0] + coordinates[1] * coordinates[1] +
               coordinates[2] * coordinates[2];
    }
    CUJU_HOST_DEVICE Vector3 Get2D() const { return Vector3(coordinates[0], coordinates[1], 0); }

    CUJU_HOST_DEVICE Vector3 GetRotated2D(float angle) const;
    CUJU_HOST_DEVICE void Rotate2D(float angle);
    CUJU_HOST_DEVICE float GetAngle2D() const;
    CUJU_HOST_DEVICE float GetAngle2D(const Vector3 &test) const;
    CUJU_HOST_DEVICE Vector3 EnforceMaximumDeviation(const Vector3 &deviant,
                                                     float maximumDeviation) const {
        Vector3 result = *this;
        Vector3 difference = deviant - *this;
        float differenceDistance = difference.GetLength();
        if (differenceDistance > maximumDeviation) {
            result += difference.GetNormalized() * (differenceDistance - maximumDeviation);
        }
        return result;
    }
    CUJU_HOST_DEVICE Vector3 GetAbsolute() const {
        return Vector3(fabsf(coordinates[0]), fabsf(coordinates[1]), fabsf(coordinates[2]));
    }
};

}  // namespace cuju

#include "libm/inverse_trigonometric_float.h"
#include "libm/sincosf.h"
#include "math_utilities.h"

namespace cuju {

// `GetRotated2D(angle)` for `libm::cosf(angle)` and `libm::sinf(angle)`, which `libm::sincosf`
// gives too, so several vectors can share one angle's sine and cosine.
CUJU_HOST_DEVICE Vector3 RotatedByCosineSine(const Vector3 &vector, float cosine, float sine) {
    Vector3 result;
    result.coordinates[0] = (vector.coordinates[0] * cosine) - (vector.coordinates[1] * sine);
    result.coordinates[1] = (vector.coordinates[1] * cosine) + (vector.coordinates[0] * sine);
    result.coordinates[2] = vector.coordinates[2];
    return result;
}

CUJU_HOST_DEVICE Vector3 RotatedBy(const Vector3 &vector, const libm::SineCosine &turn) {
    return RotatedByCosineSine(vector, turn.cosine, turn.sine);
}

CUJU_HOST_DEVICE Vector3 Vector3::GetRotated2D(float angle) const {
    return RotatedBy(*this, libm::sincosf(angle));
}

CUJU_HOST_DEVICE void Vector3::Rotate2D(float angle) {
    const Vector3 rotated = RotatedBy(*this, libm::sincosf(angle));
    coordinates[0] = rotated.coordinates[0];
    coordinates[1] = rotated.coordinates[1];
}

CUJU_HOST_DEVICE float Vector3::GetAngle2D() const {
    float angle = libm::atan2f(coordinates[1], coordinates[0]);
    if (angle < 0) angle += 2 * PI;
    return angle;
}

CUJU_HOST_DEVICE float Vector3::GetAngle2D(const Vector3 &test) const {
    float angle =
        -libm::atan2f(coordinates[0] * test.coordinates[1] - coordinates[1] * test.coordinates[0],
                      coordinates[0] * test.coordinates[0] + coordinates[1] * test.coordinates[1]);
    return ModulateIntoRange(-PI, PI, angle);
}

}  // namespace cuju

// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// GRF's float quaternion, stored x, y, z, then the scalar w.
// Exact: every step keeps GRF's float or double, and its libm.
#pragma once

#include <cmath>

#include "host_device.h"
#include "libm/sincos_double.h"
#include "libm/trigonometric_double.h"
#include "libm/inverse_trigonometric_float.h"
#include "libm/sincosf.h"
#include "math_utilities.h"
#include "vector3.h"
#include "x86.h"

namespace cuju {

struct Quaternion {
    float elements[4];

    CUJU_HOST_DEVICE Quaternion() : elements{0, 0, 0, 1} {}
    CUJU_HOST_DEVICE Quaternion(float x, float y, float z, float w) : elements{x, y, z, w} {}

    CUJU_HOST_DEVICE static Quaternion Identity() { return Quaternion(0, 0, 0, 1); }

    CUJU_HOST_DEVICE bool operator==(const Quaternion &other) const {
        return elements[0] == other.elements[0] && elements[1] == other.elements[1] &&
               elements[2] == other.elements[2] && elements[3] == other.elements[3];
    }
    CUJU_HOST_DEVICE bool operator!=(const Quaternion &other) const { return !(*this == other); }

    CUJU_HOST_DEVICE void Set(float x, float y, float z, float w) {
        elements[0] = x;
        elements[1] = y;
        elements[2] = z;
        elements[3] = w;
    }

    CUJU_HOST_DEVICE Quaternion operator*(float factor) const {
        return Quaternion(elements[0] * factor, elements[1] * factor, elements[2] * factor,
                          elements[3] * factor);
    }
    CUJU_HOST_DEVICE Quaternion operator*(const Quaternion &other) const {
        Quaternion product;
        product.elements[0] = elements[3] * other.elements[0] + elements[0] * other.elements[3] +
                              elements[1] * other.elements[2] - elements[2] * other.elements[1];
        product.elements[1] = elements[3] * other.elements[1] + elements[1] * other.elements[3] +
                              elements[2] * other.elements[0] - elements[0] * other.elements[2];
        product.elements[2] = elements[3] * other.elements[2] + elements[2] * other.elements[3] +
                              elements[0] * other.elements[1] - elements[1] * other.elements[0];
        product.elements[3] = elements[3] * other.elements[3] - elements[0] * other.elements[0] -
                              elements[1] * other.elements[1] - elements[2] * other.elements[2];
        return product;
    }
    // GRF: two cross products with the vector part, not two quaternion products. A rotation only
    // at length 1.
    CUJU_HOST_DEVICE Vector3 operator*(const Vector3 &vector) const {
        Vector3 vectorPart(elements[0], elements[1], elements[2]);
        Vector3 firstCross = vectorPart.GetCrossProduct(vector);
        Vector3 secondCross = vectorPart.GetCrossProduct(firstCross);
        firstCross *= (2.0f * elements[3]);
        secondCross *= 2.0f;
        return vector + firstCross + secondCross;
    }
    CUJU_HOST_DEVICE Quaternion operator+(const Quaternion &other) const {
        return Quaternion(elements[0] + other.elements[0], elements[1] + other.elements[1],
                          elements[2] + other.elements[2], elements[3] + other.elements[3]);
    }
    CUJU_HOST_DEVICE Quaternion operator-() const {
        return Quaternion(-elements[0], -elements[1], -elements[2], -elements[3]);
    }

    CUJU_HOST_DEVICE void Scale(float factor) {
        for (int index = 0; index < 4; index++) elements[index] *= factor;
    }

    // Exact: a float square root, compared with the double 1e-6.
    CUJU_HOST_DEVICE float GetMagnitude() const {
        if (elements[0] == 0.0f && elements[1] == 0.0f && elements[2] == 0.0f &&
            elements[3] == 0.0f)
            return 0.0f;
        float magnitude = sqrtf(elements[0] * elements[0] + elements[1] * elements[1] +
                                elements[2] * elements[2] + elements[3] * elements[3]);
        if (magnitude < ABOVE_ONE_MILLIONTH) return 0.0f;
        return magnitude;
    }

    // GRF takes the reciprocal in double and rounds it to float, which is the float quotient, since
    // a double holds at least twice the digits of a float plus two.
    CUJU_HOST_DEVICE Quaternion GetInverse() const {
        float magnitude = GetMagnitude();
        if (magnitude < 0.000001f) return Identity();
        float inverseMagnitude = 1.0f / magnitude;
        return Quaternion(-elements[0] * inverseMagnitude, -elements[1] * inverseMagnitude,
                          -elements[2] * inverseMagnitude, elements[3] * inverseMagnitude);
    }

    // GRF picks `2 / (1 + s)` over `1 / sqrt(s)` when the magnitude of `(int32_t)(1.0 - s)` is
    // below 2.1e-8, at 0 when s is below 2 or at `INT32_MIN` for NaN or a float above 2^31. The
    // float tests decide the same for every float, with s summed in float, the factors in double.
    CUJU_HOST_DEVICE void Normalize() {
        const float squaredMagnitude = elements[0] * elements[0] + elements[1] * elements[1] +
                                       elements[2] * elements[2] + elements[3] * elements[3];
        if (squaredMagnitude < 0.000001f) {
            Set(0, 0, 0, 1);
        } else if (squaredMagnitude < 2.0f || !(squaredMagnitude <= 0x1p31f)) {
            Scale((float)(2.0 / (1.0 + (double)squaredMagnitude)));
        } else {
            Scale((float)(1.0 / sqrt((double)squaredMagnitude)));
        }
    }
    CUJU_HOST_DEVICE Quaternion GetNormalized() const {
        Quaternion temporary(*this);
        temporary.Normalize();
        return temporary;
    }

    CUJU_HOST_DEVICE float GetSquaredMagnitude() const {
        return elements[0] * elements[0] + elements[1] * elements[1] + elements[2] * elements[2] +
               elements[3] * elements[3];
    }

    CUJU_HOST_DEVICE float GetDotProduct(const Quaternion &other) const {
        return elements[0] * other.elements[0] + elements[1] * other.elements[1] +
               elements[2] * other.elements[2] + elements[3] * other.elements[3];
    }

    // Exact: the dot sums in float in the order w, x, y, z. The rest runs in double and rounds to
    // float at the end.
    CUJU_HOST_DEVICE_NO_INLINE Quaternion GetSphericallyInterpolated(float bias,
                                                                     const Quaternion &to) const {
        Quaternion result;
        Quaternion target = to;
        double cosHalfTheta = elements[3] * target.elements[3] + elements[0] * target.elements[0] +
                              elements[1] * target.elements[1] + elements[2] * target.elements[2];
        if (cosHalfTheta < 0) {
            target = -target;
            cosHalfTheta = -cosHalfTheta;
        }
        if (fabs(cosHalfTheta) >= 1.0) {
            result = *this;
            return result;
        }
        double sinHalfTheta = sqrt(1.0 - cosHalfTheta * cosHalfTheta);
        if (fabs(sinHalfTheta) < 0.000001f) {
            result.elements[3] = (float)(elements[3] * 0.5 + target.elements[3] * 0.5);
            result.elements[0] = (float)(elements[0] * 0.5 + target.elements[0] * 0.5);
            result.elements[1] = (float)(elements[1] * 0.5 + target.elements[1] * 0.5);
            result.elements[2] = (float)(elements[2] * 0.5 + target.elements[2] * 0.5);
            return result;
        }
#if !defined(__CUDA_ARCH__)
        // Speed: a zero bias gives weights within 2^-25 of 1 and a zero signed as `bias`,
        // so each element rounds back to its own, and a zero takes the sign of a sum of two zeros.
        if (bias == 0.0f && sinHalfTheta >= 0.000001f) {
            for (int element = 0; element < 4; element++) {
                const float own = elements[element];
                result.elements[element] =
                    own != 0.0f ? own
                    : (std::signbit(own) &&
                       std::signbit(target.elements[element]) != std::signbit(bias))
                        ? -0.0f
                        : 0.0f;
            }
            return result;
        }
#endif
        double halfTheta = libm::acos(cosHalfTheta);
        // The sine of a signed zero is that zero exactly, so a zero angle skips the call.
        const double angleFrom = (1 - bias) * halfTheta, angleTo = bias * halfTheta;
        double ratioFrom = (angleFrom == 0.0 ? angleFrom : libm::sin(angleFrom)) / sinHalfTheta;
        double ratioTo = (angleTo == 0.0 ? angleTo : libm::sin(angleTo)) / sinHalfTheta;
        result.elements[3] = (float)(elements[3] * ratioFrom + target.elements[3] * ratioTo);
        result.elements[0] = (float)(elements[0] * ratioFrom + target.elements[0] * ratioTo);
        result.elements[1] = (float)(elements[1] * ratioFrom + target.elements[1] * ratioTo);
        result.elements[2] = (float)(elements[2] * ratioFrom + target.elements[2] * ratioTo);
        return result;
    }

    CUJU_HOST_DEVICE Quaternion GetRotationTo(const Quaternion &to) const {
        return to * this->GetInverse();
    }

    CUJU_HOST_DEVICE float GetRotationAngle(const Quaternion &to) const {
        return 2.0f * libm::acosf(clamp(this->GetDotProduct(to), -1.0f, 1.0f));
    }

    CUJU_HOST_DEVICE void SetAngleAxis(float angle, const Vector3 &axis) {
        float halfAngle = 0.5f * angle;
        const libm::SineCosine half = libm::sincosf(halfAngle);
        elements[0] = half.sine * axis.coordinates[0];
        elements[1] = half.sine * axis.coordinates[1];
        elements[2] = half.sine * axis.coordinates[2];
        elements[3] = half.cosine;
    }

    CUJU_HOST_DEVICE_NO_INLINE void GetAngleAxis(float &angle, Vector3 &axis) const {
        angle = 2.0f * libm::acosf(elements[3]);
        float halfAngleSine = sqrtf(1.0f - elements[3] * elements[3]);
        if (halfAngleSine < 0.000001f) {
            axis.coordinates[0] = elements[0];
            axis.coordinates[1] = elements[1];
            axis.coordinates[2] = elements[2];
        } else {
            axis.coordinates[0] = elements[0] / halfAngleSine;
            axis.coordinates[1] = elements[1] / halfAngleSine;
            axis.coordinates[2] = elements[2] / halfAngleSine;
        }
    }

    CUJU_HOST_DEVICE Quaternion GetLinearlyInterpolated(float bias, const Quaternion &to) const {
        return (*this * (1 - bias) + to * bias).GetNormalized();
    }

    CUJU_HOST_DEVICE float MakeSameNeighborhood(const Quaternion &other) {
        float dotProduct = GetDotProduct(other);
        if (dotProduct < 0) {
            dotProduct = -dotProduct;
            *this = -*this;
        }
        return dotProduct;
    }

    CUJU_HOST_DEVICE Quaternion GetRotationMultipliedBy(float factor) const {
        Quaternion result;
        Vector3 axis;
        float angle;
        GetAngleAxis(angle, axis);
        if (angle > PI) angle -= 2.0f * PI;
        angle = fmodf(angle * factor, 2.0f * PI);
        result.SetAngleAxis(angle, axis);
        return result.GetNormalized();
    }

    // GRF: the roles of the y and z components swap, and the singularity test compares in double.
    CUJU_HOST_DEVICE_NO_INLINE void GetAngles(float &angleX, float &angleY, float &angleZ) const {
        const int xIndex = 0, yIndex = 2, zIndex = 1;
        float singularityTest =
            elements[xIndex] * elements[yIndex] + elements[zIndex] * elements[3];
        if ((double)singularityTest > 0.49999 || (double)singularityTest < -0.49999) {
            if (singularityTest > 0) {
                angleZ = 2 * libm::atan2f(elements[xIndex], elements[zIndex]);
                angleY = (float)((double)PI * 0.5);
            }
            if (singularityTest < 0) {
                angleZ = -2 * libm::atan2f(elements[xIndex], elements[zIndex]);
                angleY = (float)((double)-PI * 0.5);
            }
            angleX = 0;
            return;
        }
        float squaredX = elements[xIndex] * elements[xIndex];
        float squaredY = elements[yIndex] * elements[yIndex];
        float squaredZ = elements[zIndex] * elements[zIndex];
        angleZ = libm::atan2f(
            2 * elements[yIndex] * elements[3] - 2 * elements[xIndex] * elements[zIndex],
            1 - 2 * squaredY - 2 * squaredZ);
        angleY = libm::asinf(2 * elements[xIndex] * elements[yIndex] +
                             2 * elements[zIndex] * elements[3]);
        angleX = libm::atan2f(
            2 * elements[xIndex] * elements[3] - 2 * elements[yIndex] * elements[zIndex],
            1 - 2 * squaredX - 2 * squaredZ);
    }

    CUJU_HOST_DEVICE_NO_INLINE void SetAngles(float angleX, float angleY, float angleZ) {
        double sine, cosine;
        libm::sincos((double)angleY / 2.0, &sine, &cosine);
        float cosineY = (float)cosine, sineY = (float)sine;
        libm::sincos((double)angleZ / 2.0, &sine, &cosine);
        float cosineZ = (float)cosine, sineZ = (float)sine;
        libm::sincos((double)angleX / 2.0, &sine, &cosine);
        float cosineX = (float)cosine, sineX = (float)sine;
        float cosineYZ = cosineY * cosineZ;
        float sineYZ = sineY * sineZ;
        elements[3] = cosineYZ * cosineX - sineYZ * sineX;
        elements[0] = cosineYZ * sineX + sineYZ * cosineX;
        elements[1] = sineY * cosineZ * cosineX + cosineY * sineZ * sineX;
        elements[2] = cosineY * sineZ * cosineX - sineY * cosineZ * sineX;
    }
};

}  // namespace cuju

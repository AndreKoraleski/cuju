// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The usual and the optimistic time that a player needs to reach a target position.
// Exact: float to integer conversions go through `FloatToInt32` and `FloatToUnsigned32`.
#pragma once

#include "profiling.h"

#include "game_definitions.h"
#include "math_utilities.h"
#include "vector3.h"
#include "x86.h"
#include "libm/common.h"

// AVX scans the walk eight steps at a time, outside nvcc, whose front end rejects some intrinsics.
#if !defined(__CUDACC__) && defined(__AVX__)
#define CUJU_WALK_AVX 1
#include <immintrin.h>
#endif

namespace cuju {

// Entry `k` is `clamp((float)(10 k) / (float)700, 0, 1)`. The table replaces a division that
// took most of the time of the loop.
CUJU_TABLE_DEFINITION(float, TIME_NEEDED_BIAS,
                      {
                          0x0.0p+0f,
                          0x1.d41d420000000p-7f,
                          0x1.d41d420000000p-6f,
                          0x1.5f15f20000000p-5f,
                          0x1.d41d420000000p-5f,
                          0x1.24924a0000000p-4f,
                          0x1.5f15f20000000p-4f,
                          0x1.99999a0000000p-4f,
                          0x1.d41d420000000p-4f,
                          0x1.0750760000000p-3f,
                          0x1.24924a0000000p-3f,
                          0x1.41d41e0000000p-3f,
                          0x1.5f15f20000000p-3f,
                          0x1.7c57c60000000p-3f,
                          0x1.99999a0000000p-3f,
                          0x1.b6db6e0000000p-3f,
                          0x1.d41d420000000p-3f,
                          0x1.f15f160000000p-3f,
                          0x1.0750760000000p-2f,
                          0x1.15f1600000000p-2f,
                          0x1.24924a0000000p-2f,
                          0x1.3333340000000p-2f,
                          0x1.41d41e0000000p-2f,
                          0x1.5075080000000p-2f,
                          0x1.5f15f20000000p-2f,
                          0x1.6db6dc0000000p-2f,
                          0x1.7c57c60000000p-2f,
                          0x1.8af8b00000000p-2f,
                          0x1.99999a0000000p-2f,
                          0x1.a83a840000000p-2f,
                          0x1.b6db6e0000000p-2f,
                          0x1.c57c580000000p-2f,
                          0x1.d41d420000000p-2f,
                          0x1.e2be2c0000000p-2f,
                          0x1.f15f160000000p-2f,
                          0x1.0000000000000p-1f,
                          0x1.0750760000000p-1f,
                          0x1.0ea0ea0000000p-1f,
                          0x1.15f1600000000p-1f,
                          0x1.1d41d40000000p-1f,
                          0x1.24924a0000000p-1f,
                          0x1.2be2be0000000p-1f,
                          0x1.3333340000000p-1f,
                          0x1.3a83a80000000p-1f,
                          0x1.41d41e0000000p-1f,
                          0x1.4924920000000p-1f,
                          0x1.5075080000000p-1f,
                          0x1.57c57c0000000p-1f,
                          0x1.5f15f20000000p-1f,
                          0x1.6666660000000p-1f,
                          0x1.6db6dc0000000p-1f,
                          0x1.7507500000000p-1f,
                          0x1.7c57c60000000p-1f,
                          0x1.83a83a0000000p-1f,
                          0x1.8af8b00000000p-1f,
                          0x1.9249240000000p-1f,
                          0x1.99999a0000000p-1f,
                          0x1.a0ea0e0000000p-1f,
                          0x1.a83a840000000p-1f,
                          0x1.af8af80000000p-1f,
                          0x1.b6db6e0000000p-1f,
                          0x1.be2be20000000p-1f,
                          0x1.c57c580000000p-1f,
                          0x1.cccccc0000000p-1f,
                          0x1.d41d420000000p-1f,
                          0x1.db6db60000000p-1f,
                          0x1.e2be2c0000000p-1f,
                          0x1.ea0ea00000000p-1f,
                          0x1.f15f160000000p-1f,
                          0x1.f8af8a0000000p-1f,
                          0x1.0000000000000p+0f,
                      });

struct TimeNeeded {
    uint32_t usualMilliseconds = 0;
    uint32_t optimisticMilliseconds = 0;
};

// The smallest doubles whose square roots round to a float above 16 and 48. Both steps are
// monotone, so `(float)sqrt(s) > 16` exactly when `s >= ABOVE_SIXTEEN_SQUARED`.
constexpr double ABOVE_SIXTEEN_SQUARED = 0x1.0000020000012p+8;
constexpr double ABOVE_FORTY_EIGHT_SQUARED = 0x1.2000018000009p+11;

// Exactly `GetLength() > 16`, or 48 when `precise`, without the square root.
CUJU_HOST_DEVICE bool BeyondOptimizeDistance(const Vector3 &playerPosition,
                                             const Vector3 &targetPosition, bool precise) {
    const Vector3 difference = playerPosition - targetPosition;
    // The sum of squares of `GetLength`, in double.
    double x = difference.coordinates[0], y = difference.coordinates[1],
           z = difference.coordinates[2];
    const double squaredLength = x * x + y * y + z * z;
    return squaredLength >= (precise ? ABOVE_FORTY_EIGHT_SQUARED : ABOVE_SIXTEEN_SQUARED);
}

CUJU_HOST_DEVICE uint32_t DirectTimeMilliseconds(const Vector3 &playerPosition,
                                                 const Vector3 &playerMovement,
                                                 const Vector3 &targetPosition,
                                                 float maximumVelocity) {
    return (uint32_t)FloatToInt32(
        roundf((targetPosition - (playerPosition + playerMovement * 0.2f)).GetLength() /
               (maximumVelocity * 0.75f) * 1000));
}

// Velocities are in meters per second. `maximumTimeMilliseconds` is `UINT32_MAX` for no limit.
CUJU_HOST_DEVICE_NO_INLINE TimeNeeded GetTimeNeededForDistanceMilliseconds(
    const Vector3 &playerPosition, const Vector3 &playerMovement, const Vector3 &targetPosition,
    float maximumVelocity, bool precise, uint32_t maximumTimeMilliseconds) {
    CUJU_PROFILE_SCOPE(TIME_NEEDED);
    TimeNeeded result;

#if defined(__CUDA_ARCH__)
    // Speed: on the GPU one direct estimate and the plain length are faster.
    const uint32_t directTimeMilliseconds =
        DirectTimeMilliseconds(playerPosition, playerMovement, targetPosition, maximumVelocity);
    auto directTime = [&] { return directTimeMilliseconds; };
    const bool beyond = (playerPosition - targetPosition).GetLength() > (precise ? 48.0f : 16.0f);
#else
    // Speed: on the host only the two cases that read the direct estimate compute it.
    auto directTime = [&] {
        return DirectTimeMilliseconds(playerPosition, playerMovement, targetPosition,
                                      maximumVelocity);
    };
    const bool beyond = BeyondOptimizeDistance(playerPosition, targetPosition, precise);
#endif
    if (beyond) {
        result.usualMilliseconds = directTime();
        result.optimisticMilliseconds = result.usualMilliseconds - 200;
        return result;
    }

    Vector3 currentPosition = playerPosition;
    Vector3 currentMovement = playerMovement;

    float initialOffset = 0.1f;
    if (currentMovement.GetLength() > IDLE_DRIBBLE_SWITCH) {
        currentPosition += currentMovement.GetNormalized() * initialOffset;
        currentPosition += currentMovement * 0.01f;
    } else {
        currentPosition += (targetPosition - playerPosition).GetNormalized(0) * initialOffset;
    }

    uint32_t currentTimeMilliseconds = 0;
    const uint32_t timeStepMilliseconds = 10;
    float usualRadius = 0.28f;
    float optimisticRadius = 0.9f;
    float resultingUsualRadius = usualRadius;
    bool foundOptimisticTime = false;

    float adaptedMaximumVelocity = maximumVelocity * 0.94f;

    while (true) {
        // `clamp((float)currentTimeMilliseconds / (float)700, 0, 1)`. Entry 70 is 1 and ends the
        // loop.
        float bias = CUJU_TABLE(TIME_NEEDED_BIAS)[currentTimeMilliseconds / timeStepMilliseconds];

        if (bias >= 1.0f) {
            float remainingUsualDistance = clamp(
                (targetPosition - currentPosition).GetLength() - usualRadius, 0.0f, 100000.0f);
            result.usualMilliseconds =
                FloatToUnsigned32((float)currentTimeMilliseconds +
                                  (remainingUsualDistance / adaptedMaximumVelocity) * 1000);
            resultingUsualRadius = usualRadius + remainingUsualDistance;

            if (!foundOptimisticTime) {
                float remainingOptimisticDistance =
                    clamp((targetPosition - currentPosition).GetLength() - optimisticRadius, 0.0f,
                          100000.0f);
                result.optimisticMilliseconds = FloatToUnsigned32(
                    (float)currentTimeMilliseconds +
                    (remainingOptimisticDistance / adaptedMaximumVelocity) * 1000);
            }
            break;
        } else {
            currentMovement.coordinates[0] = playerMovement.coordinates[0] * (1.0f - bias);
            currentMovement.coordinates[1] = playerMovement.coordinates[1] * (1.0f - bias);

            currentPosition.coordinates[0] +=
                currentMovement.coordinates[0] * timeStepMilliseconds * 0.001f;
            currentPosition.coordinates[1] +=
                currentMovement.coordinates[1] * timeStepMilliseconds * 0.001f;

            usualRadius += adaptedMaximumVelocity * bias * timeStepMilliseconds * 0.001f;
            optimisticRadius += adaptedMaximumVelocity * bias * timeStepMilliseconds * 0.001f;

            float targetDistance = (targetPosition - currentPosition).GetSquaredLength();
            if ((targetDistance < optimisticRadius * optimisticRadius ||
                 (maximumTimeMilliseconds != (uint32_t)-1 &&
                  currentTimeMilliseconds > maximumTimeMilliseconds)) &&
                !foundOptimisticTime) {
                result.optimisticMilliseconds = currentTimeMilliseconds;
                foundOptimisticTime = true;
            }
            if (targetDistance < usualRadius * usualRadius ||
                (maximumTimeMilliseconds != (uint32_t)-1 &&
                 currentTimeMilliseconds > maximumTimeMilliseconds)) {
                resultingUsualRadius = usualRadius;
                result.usualMilliseconds = currentTimeMilliseconds;
                break;
            }
        }

        currentTimeMilliseconds += timeStepMilliseconds;
    }

    if (maximumTimeMilliseconds != (uint32_t)-1 &&
        currentTimeMilliseconds > maximumTimeMilliseconds) {
        uint32_t defaultTime = directTime(), extendedTime = (currentTimeMilliseconds + 100) * 2;
        result.usualMilliseconds = defaultTime > extendedTime ? defaultTime : extendedTime;
        if (!foundOptimisticTime) result.optimisticMilliseconds = result.usualMilliseconds;
        return result;
    }

    if (result.usualMilliseconds == 0) {
        result.usualMilliseconds = (uint32_t)FloatToInt32(
            roundf(clamp((targetPosition - playerPosition).GetLength() / resultingUsualRadius, 0.0f,
                         1.0f) *
                   10));
        result.optimisticMilliseconds = result.usualMilliseconds;
    }

    return result;
}

#if !defined(__CUDA_ARCH__)
// Speed: host only. A moving player's walk does not depend on the target, so its steps are computed
// once, with the operations of `GetTimeNeededForDistanceMilliseconds` in order. The optimistic
// radius always exceeds the usual one, so the first step within each decides `Time` bit for bit.
struct TimeNeededWalk {
    static constexpr int STEPS = 70;
    static constexpr int PADDED_STEPS = 72;
    alignas(32) float positionsX[PADDED_STEPS] = {}, positionsY[PADDED_STEPS] = {};
    alignas(32) float squaredUsualRadii[PADDED_STEPS] = {},
                      squaredOptimisticRadii[PADDED_STEPS] = {};
    float usualRadii[STEPS];
    Vector3 lastPosition;
    float lastOptimisticRadius;
    bool computed = false;
    bool shared;
    Vector3 playerMovement, start;
    float adaptedMaximumVelocity;

    TimeNeededWalk(const Vector3 &playerPosition, const Vector3 &movement, float maximumVelocity)
        : shared(movement.GetLength() > IDLE_DRIBBLE_SWITCH),
          playerMovement(movement),
          start(playerPosition),
          adaptedMaximumVelocity(maximumVelocity * 0.94f) {
        if (!shared) return;
        start += movement.GetNormalized() * 0.1f;
        start += movement * 0.01f;
    }

    void Compute() {
        const uint32_t timeStepMilliseconds = 10;
        Vector3 position = start;
        float usualRadius = 0.28f, optimisticRadius = 0.9f;
        for (int step = 0; step < STEPS; step++) {
            const float bias = CUJU_TABLE(TIME_NEEDED_BIAS)[step];
            const float movementX = playerMovement.coordinates[0] * (1.0f - bias);
            const float movementY = playerMovement.coordinates[1] * (1.0f - bias);
            position.coordinates[0] += movementX * timeStepMilliseconds * 0.001f;
            position.coordinates[1] += movementY * timeStepMilliseconds * 0.001f;
            usualRadius += adaptedMaximumVelocity * bias * timeStepMilliseconds * 0.001f;
            optimisticRadius += adaptedMaximumVelocity * bias * timeStepMilliseconds * 0.001f;
            positionsX[step] = position.coordinates[0];
            positionsY[step] = position.coordinates[1];
            usualRadii[step] = usualRadius;
            squaredUsualRadii[step] = usualRadius * usualRadius;
            squaredOptimisticRadii[step] = optimisticRadius * optimisticRadius;
        }
        lastPosition = position;
        lastOptimisticRadius = optimisticRadius;
        computed = true;
    }

    // The first step below `end` within `squaredRadii`, or `end`. The squares sum as
    // `Vector3::GetSquaredLength` sums them.
    int FirstWithin(const float *squaredRadii, int end, float targetX, float targetY,
                    float squaredZ) const {
#if defined(CUJU_WALK_AVX)
        const __m256 x = _mm256_set1_ps(targetX), y = _mm256_set1_ps(targetY),
                     z = _mm256_set1_ps(squaredZ);
        for (int block = 0; block < end; block += 8) {
            const __m256 dx = _mm256_sub_ps(x, _mm256_load_ps(positionsX + block));
            const __m256 dy = _mm256_sub_ps(y, _mm256_load_ps(positionsY + block));
            const __m256 squared =
                _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)), z);
            const int within = _mm256_movemask_ps(
                _mm256_cmp_ps(squared, _mm256_load_ps(squaredRadii + block), _CMP_LT_OQ));
            if (within) {
#if defined(_MSC_VER)
                // `within` is not 0, so `tzcnt` counts what `__builtin_ctz` does.
                const int step = block + (int)_tzcnt_u32((unsigned)within);
#else
                const int step = block + __builtin_ctz((unsigned)within);
#endif
                return step < end ? step : end;
            }
        }
        return end;
#else
        for (int step = 0; step < end; step++) {
            const float dx = targetX - positionsX[step], dy = targetY - positionsY[step];
            if (dx * dx + dy * dy + squaredZ < squaredRadii[step]) return step;
        }
        return end;
#endif
    }

    // Requires `shared`, and the arguments that built the walk.
    TimeNeeded Time(const Vector3 &playerPosition, const Vector3 &targetPosition,
                    float maximumVelocity, bool precise, uint32_t maximumTimeMilliseconds) {
        TimeNeeded result;
        if (BeyondOptimizeDistance(playerPosition, targetPosition, precise)) {
            result.usualMilliseconds = DirectTimeMilliseconds(playerPosition, playerMovement,
                                                              targetPosition, maximumVelocity);
            result.optimisticMilliseconds = result.usualMilliseconds - 200;
            return result;
        }
        if (!computed) Compute();
        // From this step on the time exceeds the limit, which ends the walk as a radius does.
        int limit = STEPS;
        if (maximumTimeMilliseconds != (uint32_t)-1 &&
            maximumTimeMilliseconds / 10 + 1 < (uint32_t)STEPS)
            limit = (int)(maximumTimeMilliseconds / 10 + 1);
        const float differenceZ = targetPosition.coordinates[2] - lastPosition.coordinates[2];
        const float squaredZ = differenceZ * differenceZ;
        const float targetX = targetPosition.coordinates[0],
                    targetY = targetPosition.coordinates[1];
        const int stop = FirstWithin(squaredUsualRadii, limit, targetX, targetY, squaredZ);
        const int optimisticEnd = stop < STEPS ? stop + 1 : STEPS;
        int optimistic =
            FirstWithin(squaredOptimisticRadii, optimisticEnd, targetX, targetY, squaredZ);
        if (optimistic > limit) optimistic = limit;
        const bool foundOptimisticTime = optimistic < optimisticEnd;
        if (foundOptimisticTime) result.optimisticMilliseconds = (uint32_t)optimistic * 10;
        uint32_t currentTimeMilliseconds;
        float resultingUsualRadius;
        if (stop < STEPS) {
            currentTimeMilliseconds = (uint32_t)stop * 10;
            resultingUsualRadius = usualRadii[stop];
            result.usualMilliseconds = currentTimeMilliseconds;
        } else {
            currentTimeMilliseconds = (uint32_t)STEPS * 10;
            const float lastUsualRadius = usualRadii[STEPS - 1];
            float remainingUsualDistance = clamp(
                (targetPosition - lastPosition).GetLength() - lastUsualRadius, 0.0f, 100000.0f);
            result.usualMilliseconds =
                FloatToUnsigned32((float)currentTimeMilliseconds +
                                  (remainingUsualDistance / adaptedMaximumVelocity) * 1000);
            resultingUsualRadius = lastUsualRadius + remainingUsualDistance;
            if (!foundOptimisticTime) {
                float remainingOptimisticDistance =
                    clamp((targetPosition - lastPosition).GetLength() - lastOptimisticRadius, 0.0f,
                          100000.0f);
                result.optimisticMilliseconds = FloatToUnsigned32(
                    (float)currentTimeMilliseconds +
                    (remainingOptimisticDistance / adaptedMaximumVelocity) * 1000);
            }
        }
        if (maximumTimeMilliseconds != (uint32_t)-1 &&
            currentTimeMilliseconds > maximumTimeMilliseconds) {
            uint32_t defaultTime = DirectTimeMilliseconds(playerPosition, playerMovement,
                                                          targetPosition, maximumVelocity),
                     extendedTime = (currentTimeMilliseconds + 100) * 2;
            result.usualMilliseconds = defaultTime > extendedTime ? defaultTime : extendedTime;
            if (!foundOptimisticTime) result.optimisticMilliseconds = result.usualMilliseconds;
            return result;
        }
        if (result.usualMilliseconds == 0) {
            result.usualMilliseconds = (uint32_t)FloatToInt32(
                roundf(clamp((targetPosition - playerPosition).GetLength() / resultingUsualRadius,
                             0.0f, 1.0f) *
                       10));
            result.optimisticMilliseconds = result.usualMilliseconds;
        }
        return result;
    }
};
#endif

}  // namespace cuju

// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The earliest times at which a player can reach the predicted path of the ball.
#pragma once

#include "profiling.h"

#include "ai_time.h"
#include "game_definitions.h"
#include "math_utilities.h"
#include "vector3.h"
#include "x86.h"

namespace cuju {

// GRF: the time compares as unsigned, so a negative time reads the last entry, at 2990 ms, as a
// time past the span does.
CUJU_HOST_DEVICE Vector3 BallPredict(const Vector3 *predictions, int predictTimeMilliseconds) {
    int index = predictTimeMilliseconds;
    if ((uint32_t)index >= BALL_PREDICTION_SPAN_MILLISECONDS)
        index = BALL_PREDICTION_SPAN_MILLISECONDS - 10;
    index = index / 10;
    if (index < 0) index = 0;
    return predictions[index];
}

template <class FlippablePredictions>
CUJU_HOST_DEVICE Vector3 BallPredictFlip(const FlippablePredictions &predictions,
                                         int predictTimeMilliseconds) {
    Vector3 position = BallPredict(predictions.positions, predictTimeMilliseconds);
    if (predictions.flip) position.Mirror();
    return position;
}

struct PossessionScanResult {
    uint32_t timeNeededToGetToBallMilliseconds;
    uint32_t timeNeededToGetToBallOptimisticMilliseconds;
};

CUJU_HOST_DEVICE_NO_INLINE PossessionScanResult PossessionScan(const Vector3 *stored, bool flip,
                                                               const Vector3 &position,
                                                               const Vector3 &movement,
                                                               float maximumVelocity, bool precise,
                                                               uint32_t startTimeMilliseconds) {
    CUJU_PROFILE_SCOPE(POSSESSION_SCAN);
    struct Predictions {
        const Vector3 *positions;
        bool flip;
    } predictions{stored, flip};
    PossessionScanResult scan;
    {
        uint32_t latestTime = BALL_PREDICTION_SPAN_MILLISECONDS;
        uint32_t directTime = FloatToUnsigned32(
            roundf((BallPredictFlip(predictions, BALL_PREDICTION_SPAN_MILLISECONDS - 10).Get2D() -
                    (position + movement * 0.2f))
                       .GetLength() /
                   (maximumVelocity * 0.75f) * 1000));
        scan.timeNeededToGetToBallMilliseconds = latestTime > directTime ? latestTime : directTime;
    }
    scan.timeNeededToGetToBallOptimisticMilliseconds = scan.timeNeededToGetToBallMilliseconds;

    bool refine = false;
    uint32_t timeStepMilliseconds = 10;
    uint32_t previousMilliseconds = 0;
#if !defined(__CUDA_ARCH__)
    // Speed: on the host, the targets of the scan share the walk of a moving player.
    TimeNeededWalk walk(position, movement, maximumVelocity);
#endif
    for (uint32_t milliseconds = startTimeMilliseconds;
         milliseconds < BALL_PREDICTION_SPAN_MILLISECONDS; milliseconds += timeStepMilliseconds) {
        if (BallPredictFlip(predictions, milliseconds).coordinates[2] < 1.5f) {
            const Vector3 target = BallPredictFlip(predictions, milliseconds).Get2D();
#if !defined(__CUDA_ARCH__)
            TimeNeeded result =
                walk.shared
                    ? walk.Time(position, target, maximumVelocity, precise, milliseconds)
                    : GetTimeNeededForDistanceMilliseconds(position, movement, target,
                                                           maximumVelocity, precise, milliseconds);
#else
            TimeNeeded result = GetTimeNeededForDistanceMilliseconds(
                position, movement, target, maximumVelocity, precise, milliseconds);
#endif
            uint32_t timeNeeded = result.usualMilliseconds;
            uint32_t timeNeededOptimistic = result.optimisticMilliseconds;

            if (timeNeededOptimistic <= milliseconds) {
                if (milliseconds < scan.timeNeededToGetToBallOptimisticMilliseconds)
                    scan.timeNeededToGetToBallOptimisticMilliseconds = milliseconds;
            }

            if (timeNeeded <= milliseconds) {
                if (!refine) {
                    milliseconds = previousMilliseconds;
                    timeStepMilliseconds = 10;
                    refine = true;
                } else {
                    scan.timeNeededToGetToBallMilliseconds = milliseconds;
                    break;
                }
            }
        }

        if (!refine) {
            float ballDistance =
                (position - BallPredictFlip(predictions, milliseconds).Get2D()).GetLength() + 0.2f;
            float maximumBallVelocity = 50;
            uint32_t timeToGoMilliseconds =
                (uint32_t)FloatToInt32(roundf((ballDistance / maximumBallVelocity) * 1000.0f));
            // Exact: the clamp runs in float, and the result converts back as x86-64 converts it.
            timeStepMilliseconds = FloatToUnsigned32(clamp((float)timeToGoMilliseconds, 10, 500));
            timeStepMilliseconds = (timeStepMilliseconds / 10) * 10;
        } else {
            timeStepMilliseconds = 10;
        }

        previousMilliseconds = milliseconds;
    }
    return scan;
}

}  // namespace cuju

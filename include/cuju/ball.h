// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The ball's flight in 10 ms prediction steps, then the next tick's momentum and spin.
// Exact: squares are x * x, other powers `libm::powf`. Only `GetLength` uses double.
#pragma once

#include "profiling.h"

#include "game_definitions.h"
#include "host_device.h"
#include "libm/powf.h"
#include "libm/sincosf.h"
#include "math_utilities.h"
#include "quaternion.h"
#include "vector3.h"

namespace cuju {

struct BallSpatialInformation {
    Vector3 momentum;
    Quaternion rotationPerMillisecond;
};

// Entry `k` of `predictions` is the position after `10 k` ms. GRF: when the first step lands
// where the last tick predicted, the predictions shift down one entry instead, at most
// `validPredictions` ticks in a row.
template <class BallLayout>
CUJU_HOST_DEVICE_NO_INLINE BallSpatialInformation BallCalculatePrediction(BallLayout &ball,
                                                                          bool ballIsInGoal) {
    CUJU_PROFILE_SCOPE(BALL_PREDICTION);
    constexpr float bounce = 0.62f;
    constexpr float linearBounce = 0.06f;
    constexpr float drag = 0.015f;
    constexpr float friction = 0.04f;
    constexpr float linearFriction = 1.6f;
    constexpr float gravity = -9.81f;
    constexpr float grassHeight = 0.025f;
    constexpr float timeStep = 0.01f;

    Vector3 newMomentum;
    Quaternion newRotationPerMillisecond;

    Vector3 nextPosition = ball.positionBuffer;
    Quaternion nextOrientation = ball.orientationBuffer;
    Vector3 momentumPredict = ball.momentum;
    Quaternion rotationPredictPerMillisecond = ball.rotationPerMillisecond;

    ball.predictions[0] = nextPosition;

    bool firstTime = true;
    bool useCache = false;
    // `GetAngles` is pure, so its result is cached until the rotation changes.
    Vector3 rotationAngles;
    bool rotationAnglesValid = false;
    auto currentRotationAngles = [&]() {
        if (!rotationAnglesValid) {
            rotationPredictPerMillisecond.GetAngles(rotationAngles.coordinates[0],
                                                    rotationAngles.coordinates[1],
                                                    rotationAngles.coordinates[2]);
            rotationAnglesValid = true;
        }
        return rotationAngles;
    };
    ball.ballTouchesNet = false;

    for (uint32_t predictTimeMilliseconds = int(timeStep * 1000.0f);
         predictTimeMilliseconds < BALL_PREDICTION_SPAN_MILLISECONDS + CACHED_PREDICTIONS * 10;
         predictTimeMilliseconds += int(timeStep * 1000.0f)) {
        if (useCache) {
            // GRF: the last shift reads `predictions[400]`, which no tick writes. The state keeps
            // it.
            ball.predictions[predictTimeMilliseconds / 10] =
                ball.predictions[predictTimeMilliseconds / 10 + 1];
            continue;
        }

        float frictionFactor = 0.0f;

        momentumPredict.coordinates[2] = momentumPredict.coordinates[2] + gravity * timeStep;

        float momentumVelocity = momentumPredict.GetLength();
        float momentumVelocityDragged =
            momentumVelocity - drag * (momentumVelocity * momentumVelocity) * timeStep;
        momentumPredict = momentumPredict.GetNormalized(0) * momentumVelocityDragged;

        float ballBottom = nextPosition.coordinates[2] - 0.11f;
        float grassInfluenceBias = clamp(1.0f - (ballBottom / grassHeight), 0.0f, 1.0f);
        grassInfluenceBias = libm::PowfPositiveExponent(grassInfluenceBias, 0.7f);

        if (nextPosition.coordinates[2] < 0.11f) {
            if (momentumPredict.coordinates[2] < 0.0f) {
                frictionFactor =
                    NormalizedClamp(-momentumPredict.coordinates[2] - 0.5f, 0.0f, 12.0f);
                momentumPredict.coordinates[2] = -momentumPredict.coordinates[2] * bounce;
                float bouncedVertical = momentumPredict.coordinates[2] - linearBounce;
                momentumPredict.coordinates[2] = bouncedVertical > 0.0f ? bouncedVertical : 0.0f;
            }
            nextPosition.coordinates[2] = 0.11f;
        }

        if (nextPosition.coordinates[2] < 0.11f + grassHeight) {
            float adaptedFriction = (friction * grassInfluenceBias);
            Vector3 horizontalMomentum = momentumPredict.Get2D();
            float horizontalVelocity = horizontalMomentum.GetLength();
            float newVelocity = horizontalVelocity - adaptedFriction *
                                                         (horizontalVelocity * horizontalVelocity) *
                                                         timeStep;
            newVelocity = clamp(newVelocity - (linearFriction * grassInfluenceBias * timeStep),
                                0.0f, 100000.0f);
            horizontalMomentum.Normalize(Vector3(0));
            horizontalMomentum *= newVelocity;
            momentumPredict.coordinates[0] = horizontalMomentum.coordinates[0];
            momentumPredict.coordinates[1] = horizontalMomentum.coordinates[1];
        }

        float netAbsorbInverse = 0.95f;
        float netExponent = 2.6f;
        float powerFactor = 1.8f;
        float postAbsorbInverse = 0.8f;
        float ballRadius = 0.11f;
        float postRadius = 0.07f;
        // GRF: `pow(0.95, timeStep * 100)`, exactly 0.95 since the exponent is 1.
        netAbsorbInverse = 0.95f;

        // GRF: the posts and the crossbar act on the first step only.
        if (firstTime) {
            if (nextPosition.coordinates[2] < GOAL_HEIGHT + ballRadius + postRadius &&
                (nextPosition.Get2D().GetAbsolute() -
                 Vector3(PITCH_HALF_LENGTH, GOAL_HALF_WIDTH, 0))
                        .GetLength() < ballRadius + postRadius) {
                Vector3 normal;
                if (nextPosition.coordinates[0] < 0) {
                    if (nextPosition.coordinates[1] < 0) {
                        normal = (nextPosition.Get2D() -
                                  Vector3(-PITCH_HALF_LENGTH, -GOAL_HALF_WIDTH, 0))
                                     .GetNormalized(Vector3(1, 0, 0));
                        float nextPositionZ = nextPosition.coordinates[2];
                        nextPosition = Vector3(-PITCH_HALF_LENGTH, -GOAL_HALF_WIDTH, 0) +
                                       normal * (postRadius + ballRadius);
                        nextPosition.coordinates[2] = nextPositionZ;
                    } else {
                        normal =
                            (nextPosition.Get2D() - Vector3(-PITCH_HALF_LENGTH, GOAL_HALF_WIDTH, 0))
                                .GetNormalized(Vector3(1, 0, 0));
                        float nextPositionZ = nextPosition.coordinates[2];
                        nextPosition = Vector3(-PITCH_HALF_LENGTH, GOAL_HALF_WIDTH, 0) +
                                       normal * (postRadius + ballRadius);
                        nextPosition.coordinates[2] = nextPositionZ;
                    }
                } else {
                    if (nextPosition.coordinates[1] < 0) {
                        normal =
                            (nextPosition.Get2D() - Vector3(PITCH_HALF_LENGTH, -GOAL_HALF_WIDTH, 0))
                                .GetNormalized(Vector3(-1, 0, 0));
                        float nextPositionZ = nextPosition.coordinates[2];
                        nextPosition = Vector3(PITCH_HALF_LENGTH, -GOAL_HALF_WIDTH, 0) +
                                       normal * (postRadius + ballRadius);
                        nextPosition.coordinates[2] = nextPositionZ;
                    } else {
                        normal =
                            (nextPosition.Get2D() - Vector3(PITCH_HALF_LENGTH, GOAL_HALF_WIDTH, 0))
                                .GetNormalized(Vector3(-1, 0, 0));
                        float nextPositionZ = nextPosition.coordinates[2];
                        nextPosition = Vector3(PITCH_HALF_LENGTH, GOAL_HALF_WIDTH, 0) +
                                       normal * (postRadius + ballRadius);
                        nextPosition.coordinates[2] = nextPositionZ;
                    }
                }
                momentumPredict = (momentumPredict.Get2D().GetNormalized(normal) + (normal * 1.1f))
                                          .GetNormalized() *
                                      momentumPredict.Get2D().GetLength() * postAbsorbInverse +
                                  (Vector3(0, 0, 1) * momentumPredict.coordinates[2]);
            }

            Vector3 nextPositionXZ = nextPosition * Vector3(1, 0, 1);
            if ((nextPositionXZ.GetAbsolute() - Vector3(PITCH_HALF_LENGTH, 0, GOAL_HEIGHT))
                        .GetLength() < ballRadius + postRadius &&
                fabsf(nextPosition.coordinates[1]) < GOAL_HALF_WIDTH + ballRadius + postRadius) {
                Vector3 normal;
                if (nextPosition.coordinates[0] < 0) {
                    normal = (nextPositionXZ - Vector3(-PITCH_HALF_LENGTH, 0, GOAL_HEIGHT))
                                 .GetNormalized(Vector3(0, 0, 1));
                    float nextPositionY = nextPosition.coordinates[1];
                    nextPosition = Vector3(-PITCH_HALF_LENGTH, 0, GOAL_HEIGHT) +
                                   normal * (postRadius + ballRadius);
                    nextPosition.coordinates[1] = nextPositionY;
                } else {
                    normal = (nextPositionXZ - Vector3(PITCH_HALF_LENGTH, 0, GOAL_HEIGHT))
                                 .GetNormalized(Vector3(0, 0, -1));
                    float nextPositionY = nextPosition.coordinates[1];
                    nextPosition = Vector3(PITCH_HALF_LENGTH, 0, GOAL_HEIGHT) +
                                   normal * (postRadius + ballRadius);
                    nextPosition.coordinates[1] = nextPositionY;
                }
                Vector3 momentumPredictXZ = momentumPredict * Vector3(1, 0, 1);
                momentumPredict =
                    (momentumPredictXZ.GetNormalized(normal) + (normal * 1.1f)).GetNormalized() *
                        momentumPredictXZ.GetLength() * postAbsorbInverse +
                    (Vector3(0, 1, 0) * momentumPredict.coordinates[1]);
            }
        }

        // GRF: the netting too acts on the first step only.
        if (predictTimeMilliseconds <= 10) {
            int inGoal = ballIsInGoal ? 1 : -1;
            bool behindBackline = fabsf(nextPosition.coordinates[0]) > PITCH_HALF_LENGTH + 0.11f;
            bool beforeGoalBack =
                fabsf(nextPosition.coordinates[0]) < PITCH_HALF_LENGTH + GOAL_DEPTH - 0.11f;
            bool belowGoalHeight = nextPosition.coordinates[2] < GOAL_HEIGHT + 0.11f;
            bool betweenGoalWidth = fabsf(nextPosition.coordinates[1]) < GOAL_HALF_WIDTH - 0.11f;

            if ((ballIsInGoal && !betweenGoalWidth && behindBackline)) {
                float netDistance = fabsf(fabsf(nextPosition.coordinates[1]) - GOAL_HALF_WIDTH);
                netDistance = clamp(netDistance, 0, 1);
                float power = libm::powf(netDistance, netExponent) *
                              -SignSide(nextPosition.coordinates[1]) * inGoal;
                float woodworkTensionBiasInverse = clamp(
                    (fabsf(momentumPredict.coordinates[0]) - PITCH_HALF_LENGTH) * 2.0f, 0.0f, 1.0f);
                float adaptedPowerFactor = powerFactor + (1.0f - woodworkTensionBiasInverse) * 3.0f;
                momentumPredict.coordinates[1] = momentumPredict.coordinates[1] * netAbsorbInverse +
                                                 power * adaptedPowerFactor * (100 * timeStep);
                if (predictTimeMilliseconds == 10) ball.ballTouchesNet = true;
            }

            if ((ballIsInGoal && !beforeGoalBack && behindBackline)) {
                float netDistance =
                    fabsf(fabsf(nextPosition.coordinates[0]) - (PITCH_HALF_LENGTH + GOAL_DEPTH));
                netDistance = clamp(netDistance, 0, 1);
                float power = libm::powf(netDistance, netExponent) *
                              -SignSide(nextPosition.coordinates[0]) * inGoal;
                momentumPredict.coordinates[0] = momentumPredict.coordinates[0] * netAbsorbInverse +
                                                 power * powerFactor * (100 * timeStep);
                if (predictTimeMilliseconds == 10) ball.ballTouchesNet = true;
            }

            if ((ballIsInGoal && !belowGoalHeight && behindBackline)) {
                float netDistance = fabsf(fabsf(nextPosition.coordinates[2]) - GOAL_HEIGHT);
                netDistance = clamp(netDistance, 0, 1);
                float power = libm::powf(netDistance, netExponent) * -inGoal;
                float woodworkTensionBiasInverse = clamp(
                    (fabsf(momentumPredict.coordinates[0]) - PITCH_HALF_LENGTH) * 2.0f, 0.0f, 1.0f);
                float adaptedPowerFactor = powerFactor + (1.0f - woodworkTensionBiasInverse) * 3.0f;
                momentumPredict.coordinates[2] = momentumPredict.coordinates[2] * netAbsorbInverse +
                                                 power * adaptedPowerFactor * (100 * timeStep);
                if (predictTimeMilliseconds == 10) ball.ballTouchesNet = true;
            }
        }

        if (nextPosition.coordinates[2] < 0.11f + grassHeight) {
            constexpr float radius = 0.11f;
            float angularVelocityX = momentumPredict.coordinates[1] / radius;
            float angularVelocityY = momentumPredict.coordinates[0] / radius;

            Quaternion rotationX;
            rotationX.SetAngleAxis(clamp(angularVelocityX * 0.001f, -PI * 0.49f, PI * 0.49f),
                                   Vector3(-1, 0, 0));
            Quaternion rotationY;
            rotationY.SetAngleAxis(clamp(angularVelocityY * 0.001f, -PI * 0.49f, PI * 0.49f),
                                   Vector3(0, 1, 0));
            Quaternion groundRotation = rotationX * rotationY;

            Quaternion oldToNewRotation =
                rotationPredictPerMillisecond.GetRotationTo(groundRotation).GetNormalized();
            float rotationChangePerSecond =
                fabsf(oldToNewRotation.GetRotationAngle(Quaternion::Identity())) * 1000.0f;

            float maximumRotationChangePerSecond = 1.0f * PI * grassInfluenceBias;
            if (frictionFactor > 0.0f) maximumRotationChangePerSecond += 4.0f * PI;
            float factor = 1.0f;
            if (rotationChangePerSecond > maximumRotationChangePerSecond)
                factor = maximumRotationChangePerSecond / rotationChangePerSecond;
            if (factor < 1.0f) oldToNewRotation = oldToNewRotation.GetRotationMultipliedBy(factor);

            Quaternion newRotationPredictPerMillisecond =
                oldToNewRotation * rotationPredictPerMillisecond;

            // The angles of the rotation before this step.
            const Vector3 angles = currentRotationAngles();
            float x = angles.coordinates[0], y = angles.coordinates[1];
            x = -x;

            Vector3 ballRotationMomentum;
            ballRotationMomentum.coordinates[0] = y * radius * 1000.0f;
            ballRotationMomentum.coordinates[1] = x * radius * 1000.0f;

            float rotationBias = 0.01f;
            rotationBias *= grassInfluenceBias;
            if (frictionFactor > 0.0f) rotationBias += 0.5f * frictionFactor;
            rotationBias = clamp(rotationBias, 0.0f, 1.0f);
            momentumPredict.coordinates[0] =
                momentumPredict.coordinates[0] * (1.0f - rotationBias) +
                ballRotationMomentum.coordinates[0] * rotationBias;
            momentumPredict.coordinates[1] =
                momentumPredict.coordinates[1] * (1.0f - rotationBias) +
                ballRotationMomentum.coordinates[1] * rotationBias;

            rotationPredictPerMillisecond = newRotationPredictPerMillisecond;
            rotationAnglesValid = false;
        }

        {
            Vector3 spin = currentRotationAngles();
            spin *= 10.0f;
            float swerveAmount = NormalizedClamp(momentumPredict.GetLength(), 0.0f, 70.0f);
            swerveAmount = libm::PowfPositiveExponent(libm::sinf(swerveAmount * PI * 0.94f), 2.6f);
            Vector3 adaptedMomentumPredict =
                momentumPredict.GetNormalized(0) * swerveAmount * 30.0f;
            Vector3 swerve = adaptedMomentumPredict.GetCrossProduct(-spin) * 1.0f;
            momentumPredict += swerve * timeStep;
        }

        nextPosition += momentumPredict * timeStep;

        // Only the first step's orientation is read, so the later steps skip its update.
        if (predictTimeMilliseconds == 10) {
            Vector3 rotationVector = currentRotationAngles();
            rotationVector *= timeStep / 0.001f;
            Quaternion rotationPredictTimeStepped;
            rotationPredictTimeStepped.SetAngles(rotationVector.coordinates[0],
                                                 rotationVector.coordinates[1],
                                                 rotationVector.coordinates[2]);
            nextOrientation = rotationPredictTimeStepped * nextOrientation;
        }

        if (predictTimeMilliseconds == 10) {
            newMomentum = momentumPredict;
            newRotationPerMillisecond = rotationPredictPerMillisecond;
            ball.orientationPrediction = nextOrientation;
            if (ball.validPredictions > 0 && ball.predictions[2] == nextPosition) {
                ball.validPredictions--;
                useCache = true;
            } else {
                ball.validPredictions = CACHED_PREDICTIONS;
            }
        }
        ball.predictions[predictTimeMilliseconds / 10] = nextPosition;
        firstTime = false;
    }

    BallSpatialInformation information;
    information.momentum = newMomentum;
    information.rotationPerMillisecond = newRotationPerMillisecond;
    return information;
}

}  // namespace cuju

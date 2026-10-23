// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The camera of a match, which never affects play. Its pose is part of the state.
// Exact: while it follows play, its shake draws two numbers a tick from the RNG that play uses.
#pragma once

#include "../profiling.h"

#include "../libm/powf.h"
#include "../libm/sincosf.h"
#include "../rng.h"
#include "ai_functions.h"
#include "humanoid.h"

namespace cuju {
namespace simulation {

constexpr float DEFAULT_CAMERA_ZOOM = 0.5f, DEFAULT_CAMERA_HEIGHT = 0.3f,
                DEFAULT_CAMERA_FIELD_OF_VIEW = 0.4f, DEFAULT_CAMERA_ANGLE_FACTOR = 0.0f;
constexpr unsigned int cameraPositionSize = 150;

// Exact: `SimulationContext::cameraWeights` tabulates this, so its operation order fixes the table.
CUJU_FUNCTION inline float CameraWeight(int index, int positionCount) {
    const float indexSize = positionCount;
    float weight = libm::sinf((index / indexSize - 0.3f) * 1.4f * PI) * 0.5f + 0.5f;
    weight *= libm::powf(1.0f - index / indexSize, 0.3f);
    return weight;
}
// The table holds `CameraWeight(index, count)` at `count * (count - 1) / 2 + index` for every
// `count` from 1 to `cameraPositionSize`.
constexpr int CAMERA_WEIGHTS = cameraPositionSize * (cameraPositionSize + 1) / 2;

CUJU_FUNCTION_NO_INLINE inline void UpdateInGameCamera(World &world,
                                                       const SimulationContext &context) {
    state::MatchState &match = world.serialized.match;
    const int first = world.FirstTeamID(), second = 1 - first;
    float fieldOfView = 0.5f + DEFAULT_CAMERA_FIELD_OF_VIEW * 0.5f;
    float zoom = DEFAULT_CAMERA_ZOOM;
    float height = DEFAULT_CAMERA_HEIGHT * 1.5f;

    float playerBias = 0.6f;
    const state::SpatialState &designatedSpatialState =
        Humanoid(world, match.designatedPossessionPlayer).spatialState;
    Vector3 ballPosition = BallPredict(match.ball, 0) * (1.0f - playerBias) +
                           designatedSpatialState.position * playerBias;
    ballPosition += designatedSpatialState.directionVector * 1.0f;
    ballPosition += Vector3(((TeamByID(world, first).fadingTeamPossessionAmount - 1.0f) *
                                 -TeamDynamicSide(world, first) +
                             (TeamByID(world, second).fadingTeamPossessionAmount - 1.0f) *
                                 -TeamDynamicSide(world, second)) *
                                4.0f,
                            0, 0);
    ballPosition.coordinates[2] *= 0.1f;

    float maximumWidth =
        (float)((double)(PITCH_HALF_LENGTH * 0.84f) * (1.0 / (double)(zoom + 0.01f)));
    float maximumHeight =
        (float)((double)(PITCH_HALF_WIDTH * 0.60f) * (1.0 / (double)(zoom + 0.01f)) *
                (double)(height * 0.75f + 0.25f));
    if (fabsf(ballPosition.coordinates[0]) > maximumWidth)
        ballPosition.coordinates[0] = maximumWidth * SignSide(ballPosition.coordinates[0]);
    if (fabsf(ballPosition.coordinates[1]) > maximumHeight)
        ballPosition.coordinates[1] = maximumHeight * SignSide(ballPosition.coordinates[1]);

    // GRF: the arguments evaluate right to left, so the shake draws y before x.
    float randomY = RandomInRange(world.serialized.rng, -0.1f, 0.1f);
    float randomX = RandomInRange(world.serialized.rng, -0.1f, 0.1f);
    Vector3 shudder =
        Vector3(randomX, randomY, 0) * (match.ball.momentum.GetLength() * 0.8f + 6.0f);
    shudder *= 0.2f;
    auto &cameraPosition = match.cameraPosition;
    Vector3 add =
        ballPosition + shudder * ((float)cameraPosition.count / (float)cameraPositionSize);
    if (cameraPosition.count == (int)cameraPositionSize) {
        for (int slot = 0; slot + 1 < cameraPosition.count; slot++)
            cameraPosition[slot] = cameraPosition[slot + 1];
        cameraPosition[cameraPosition.count - 1] = add;
    } else {
        cameraPosition.push_back(add);
    }

    Vector3 average;
    float count = 0;
    const int positionCount = cameraPosition.count;
    const float *table = context.cameraWeights
                             ? context.cameraWeights + positionCount * (positionCount - 1) / 2
                             : nullptr;
    for (int index = 0; index < positionCount; index++) {
        float weight = table ? table[index] : CameraWeight(index, positionCount);
        average += cameraPosition[index] * weight;
        count += weight;
    }
    average /= Vector3(count);

    float angleFactor = 1.0f - DEFAULT_CAMERA_ANGLE_FACTOR * 0.4f;
    if (!match.goalScored || (match.goalScored && match.goalScoredTimer < 1000)) {
        zoom = (0.6f + zoom * 1.0f) * (1.0f / fieldOfView);
        height = 4.0f + height * 10;
        float distanceRotation = average.coordinates[1] / 800.0f;
        match.cameraOrientation.SetAngleAxis(distanceRotation + (0.42f - height * 0.01f) * PI,
                                             Vector3(1, 0, 0));
        match.cameraNodeOrientation.SetAngleAxis((-average.coordinates[0] / PITCH_HALF_LENGTH) *
                                                     (1.0f - angleFactor) * 0.25f * PI * 1.24f,
                                                 Vector3(0, 0, 1));
        match.cameraNodePosition =
            average * Vector3(1.0f * (1.0f - DEFAULT_CAMERA_ANGLE_FACTOR * 0.2f) *
                                  (1.0f - DEFAULT_CAMERA_ZOOM * 0.3f),
                              0.9f - DEFAULT_CAMERA_ZOOM * 0.3f, 0.2f) +
            Vector3(
                0,
                -41.4f - (DEFAULT_CAMERA_FIELD_OF_VIEW * 3.7f) + libm::powf(height, 1.2f) * 0.46f,
                10.0f + height) *
                zoom;
        match.cameraFieldOfView =
            (fieldOfView * 28.0f) - (match.cameraNodePosition.coordinates[1] / 30.0f);
        match.cameraNearCap = match.cameraNodePosition.coordinates[2];
        match.cameraFarCap = 200;
    } else {
        // `lastGoalScorer` is a player index, or -1 without a scorer.
        Vector3 targetPosition = BallPredict(match.ball, 0).Get2D();
        if (match.lastGoalScorer >= 0)
            targetPosition = Humanoid(world, match.lastGoalScorer).spatialState.position;
        float rotation = (float)match.goalScoredTimer * 0.0005f;
        match.cameraOrientation.SetAngleAxis(0.45f * PI, Vector3(1, 0, 0));
        match.cameraNodeOrientation.SetAngleAxis(rotation, Vector3(0, 0, 1));
        match.cameraNodePosition =
            targetPosition + Vector3(0, -1, 0).GetRotated2D(rotation) * 15.0f + Vector3(0, 0, 3);
        match.cameraFieldOfView = 35.0f;
        match.cameraNearCap = 1;
        match.cameraFarCap = 220;
    }
}

// The camera works on the unmirrored pitch, so team 1 is mirrored around its update.
CUJU_FUNCTION_NO_INLINE inline void UpdateCamera(World &world, const SimulationContext &context) {
    CUJU_PROFILE_SCOPE(UPDATE_CAMERA);
    state::MatchState &match = world.serialized.match;
    if (match.autoUpdateInGameCamera) {
        MirrorMatch(world, false, true, false);
        UpdateInGameCamera(world, context);
        MirrorMatch(world, false, true, false);
    }
    unsigned int zoomTime = 2000;
    unsigned int startTime = 0;
    if (match.actualTimeMilliseconds < zoomTime + startTime) {
        Quaternion initialOrientation(0, 0, 0, 1);
        initialOrientation.SetAngleAxis(0.0f * PI, Vector3(1, 0, 0));
        Quaternion headingOrientation(0, 0, 0, 1);
        initialOrientation = headingOrientation * initialOrientation;
        Vector3 initialPosition = Vector3(0.0f, 0.0f, 60.0);
        int elapsedTime =
            FloatToInt32(clamp((float)(match.actualTimeMilliseconds - startTime), 0, zoomTime));
        float bias = elapsedTime / (float)(zoomTime);
        bias *= PI;
        bias = libm::sinf(bias - 0.5f * PI) * -0.5f + 0.5f;
        match.cameraOrientation =
            match.cameraOrientation.GetSphericallyInterpolated(bias, Quaternion(0, 0, 0, 1));
        match.cameraNodeOrientation =
            match.cameraNodeOrientation.GetSphericallyInterpolated(bias, initialOrientation);
        match.cameraNodePosition =
            match.cameraNodePosition * (1.0f - bias) + initialPosition * bias;
        match.cameraFieldOfView = match.cameraFieldOfView * (1.0f - bias) + 40 * bias;
        match.cameraNearCap = match.cameraNearCap * (1.0f - bias) + 2.0f * bias;
    }
}

}  // namespace simulation
}  // namespace cuju

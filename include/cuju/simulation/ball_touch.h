// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// Touches of the ball and the touch and possession records of the teams.
#pragma once

#include "../profiling.h"

#include "../ball.h"
#include "humanoid.h"
#include "player_possession.h"

namespace cuju {
namespace simulation {

CUJU_FUNCTION inline void BallRecalculatePrediction(World &world) {
    BallMaterialize(world.serialized.match.ball);
    BallCalculatePrediction(world.serialized.match.ball, world.serialized.match.ballIsInGoal);
}

CUJU_FUNCTION_NO_INLINE inline void BallProcessSerialized(state::BallSerialized &ball,
                                                          bool ballIsInGoal) {
    CUJU_PROFILE_SCOPE(BALL_PROCESS);
    BallMaterialize(ball);
    BallSpatialInformation spatialInformation = BallCalculatePrediction(ball, ballIsInGoal);
    ball.momentum = spatialInformation.momentum;
    ball.rotationPerMillisecond = spatialInformation.rotationPerMillisecond;
    ball.positionBuffer = BallPredict(ball.predictions, 10);
    ball.orientationBuffer = ball.orientationPrediction;
    auto &history = ball.ballPositionHistory;
    if (history.count < (int)BALL_HISTORY_SIZE) {
        history.push_back(ball.positionBuffer);
    } else {
        // The lanes split the shift. Each reads its entries before the barrier and writes them
        // after it, so no lane reads an entry that another lane wrote.
        const int count = history.count;
        if (WARP_LANES == 1) {
            for (int entry = 1; entry < count; entry++) history[entry - 1] = history[entry];
        } else {
            Vector3 temporary[(BALL_HISTORY_SIZE + 31) / 32];
            int slot = 0;
            for (int entry = SplitLane() + 1; entry < count; entry += WARP_LANES)
                temporary[slot++] = history[entry];
            WarpSync();
            slot = 0;
            for (int entry = SplitLane() + 1; entry < count; entry += WARP_LANES)
                history[entry - 1] = temporary[slot++];
            WarpSync();
        }
        history[count - 1] = ball.positionBuffer;
    }
}

CUJU_FUNCTION_NO_INLINE inline void TeamUpdatePossessionStatistics(World &world,
                                                                   const SimulationContext &context,
                                                                   int teamID) {
    // Each player writes only its own fields and reads no team field that this function writes, so
    // the lanes split the players.
    WarpFor(world.serialized.match.playerCount, [&](int player) {
        if (world.playerTeamID[player] == teamID &&
            world.serialized.match.players[player].base.isActive)
            PlayerUpdatePossessionStatistics(world, *context.staticData, player);
    });
    state::TeamState &team = TeamByID(world, teamID);
    team.hasPossession = false;
    team.timeNeededToGetToBallMilliseconds = 100000;
    for (int player = 0; player < world.serialized.match.playerCount; player++) {
        if (world.playerTeamID[player] != teamID ||
            !world.serialized.match.players[player].base.isActive)
            continue;
        const state::PlayerState &playerState = world.serialized.match.players[player];
        if (playerState.hasPossession) team.hasPossession = true;
        if ((int)playerState.timeNeededToGetToBallMilliseconds <
            team.timeNeededToGetToBallMilliseconds)
            team.timeNeededToGetToBallMilliseconds =
                (int)playerState.timeNeededToGetToBallMilliseconds;
    }
}

CUJU_FUNCTION_NO_INLINE inline void BallTouch(World &world, const SimulationContext &context,
                                              const Vector3 &target) {
    CUJU_PROFILE_SCOPE(BALL_TOUCH);
    state::BallSerialized &ball = world.serialized.match.ball;
    ball.validPredictions = 0;
    if (ball.positionBuffer.coordinates[2] < 0.11f) ball.positionBuffer.coordinates[2] = 0.11f;
    ball.momentum = target;
    // GRF: the predictions run once for the momentum and once for the touch. Each run changes the
    // count of valid cached predictions, so the results depend on both.
    BallRecalculatePrediction(world);
    BallRecalculatePrediction(world);
    if (world.serialized.match.mentalImages.count > 0) {
        state::MentalImageState &mentalImage = world.serialized.match.mentalImages[0];
        mentalImage.ballPredictions.count = BALL_PREDICTION_SPAN_MILLISECONDS / 10;
        for (int index = 0; index < (int)(BALL_PREDICTION_SPAN_MILLISECONDS / 10); index++)
            mentalImage.ballPredictions[index] = ball.predictions[index];
        mentalImage.lazyFlip = ball.lazyFlip;
    }
    TeamUpdatePossessionStatistics(world, context, world.FirstTeamID());
    TeamUpdatePossessionStatistics(world, context, 1 - world.FirstTeamID());
}

CUJU_FUNCTION inline void BallSetRotation(World &world, float x, float y, float z,
                                          float bias = 1.0f) {
    Quaternion rotationX;
    rotationX.SetAngleAxis(clamp(x * 0.001f, -PI * 0.49f, PI * 0.49f), Vector3(-1, 0, 0));
    Quaternion rotationY;
    rotationY.SetAngleAxis(clamp(y * 0.001f, -PI * 0.49f, PI * 0.49f), Vector3(0, 1, 0));
    Quaternion rotationZ;
    rotationZ.SetAngleAxis(clamp(z * 0.001f, -PI * 0.49f, PI * 0.49f), Vector3(0, 0, 1));
    Quaternion temporaryRotationPerMillisecond = rotationX * rotationY * rotationZ;
    state::BallSerialized &ball = world.serialized.match.ball;
    ball.rotationPerMillisecond = ball.rotationPerMillisecond.GetSphericallyInterpolated(
        bias, temporaryRotationPerMillisecond);
    BallRecalculatePrediction(world);
}

CUJU_FUNCTION inline void BallSetPosition(World &world, const Vector3 &target) {
    state::BallSerialized &ball = world.serialized.match.ball;
    ball.validPredictions = 0;
    ball.positionBuffer = target;
    ball.momentum = Vector3(0);
    BallSetRotation(world, 0, 0, 0, 1.0f);
    ball.ballPositionHistory.count = 0;
}

// The offside records of a touch, defined in `referee.h`.
CUJU_FUNCTION inline bool RefereeBallTouched(World &world, const SimulationContext &context);

CUJU_FUNCTION inline bool TeamSetLastTouchPlayer(World &world, const SimulationContext &context,
                                                 int player, int touchType) {
    const int teamID = world.playerTeamID[player];
    TeamByID(world, teamID).lastTouchPlayer = player;
    state::PlayerState &playerState = world.serialized.match.players[player];
    playerState.base.lastTouchTimeMilliseconds = world.serialized.match.actualTimeMilliseconds;
    playerState.base.lastTouchType = touchType;
    world.serialized.match.lastTouchTeamIDs[touchType] = teamID;
    world.serialized.match.lastTouchTeamID = teamID;
    return RefereeBallTouched(world, context);
}

}  // namespace simulation
}  // namespace cuju

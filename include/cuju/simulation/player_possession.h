// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// How long a player needs to reach the ball, and whether the player has the ball.
#pragma once

#include "../game_definitions.h"
#include "../possession.h"
#include "access.h"
#include "enumerations.h"
#include "static_data.h"
#include "world.h"

namespace cuju {
namespace simulation {

CUJU_FUNCTION inline float GetMaximumVelocity(const StaticData &staticData, int player) {
    return SPRINT_VELOCITY * (0.9f + staticData.data[player].velocityStatistic * 0.1f);
}

CUJU_FUNCTION_NO_INLINE inline bool HasPossession(const World &world, int player) {
    const state::BallSerialized &ball = world.serialized.match.ball;
    const state::SpatialState &spatialState = Humanoid(world, player).spatialState;
    Vector3 playerMovement = spatialState.movement;
    if ((double)(spatialState.position - BallPredict(ball, 0)).GetLength() > 5.0) return false;
    if ((double)fabsf(BallPredict(ball, 0).coordinates[2]) > 0.5) return false;
    bool distanceMatches = true;
    float radius = 1.0f;
    Vector3 center =
        spatialState.position + spatialState.movement * 0.05f + spatialState.directionVector * 0.1f;
    if ((BallPredict(ball, 0).Get2D() - center).GetLength() > radius) distanceMatches = false;
    bool movementMatches = true;
    Vector3 ballMovement = (BallPredict(ball, 10) - BallPredict(ball, 0)) * 100.0f;
    if ((ballMovement - playerMovement).GetLength() > 6.0f) movementMatches = false;
    return distanceMatches && movementMatches;
}

CUJU_FUNCTION inline bool IsPassOrShot(int functionType) {
    return functionType == e_FunctionType_ShortPass || functionType == e_FunctionType_LongPass ||
           functionType == e_FunctionType_HighPass || functionType == e_FunctionType_Shot;
}

// Runs on one lane.
CUJU_FUNCTION_NO_INLINE inline void PlayerUpdatePossessionStatistics(World &world,
                                                                     const StaticData &staticData,
                                                                     int player) {
    state::PlayerState &playerState = world.serialized.match.players[player];
    const HumanoidState &humanoid = playerState.base.humanoid;
    const state::BallSerialized &ball = world.serialized.match.ball;
    const state::TeamState &team = TeamOfPlayer(world, player);
    const state::TeamState &opponentTeam = OpponentTeamOfPlayer(world, player);
    const Vector3 position = humanoid.spatialState.position;
    const Vector3 movement = humanoid.spatialState.movement;
    const float maximumVelocity = GetMaximumVelocity(staticData, player);

    playerState.timeNeededToGetToBallPreviousMilliseconds =
        playerState.timeNeededToGetToBallMilliseconds;

    uint32_t startTimeMilliseconds = 0;
    const bool touchPending =
        humanoid.currentAnimation.frameNumber < humanoid.currentAnimation.touchFrame;
    if (IsPassOrShot(humanoid.currentAnimation.functionType) && !touchPending)
        startTimeMilliseconds = 500;

    bool precise = team.designatedTeamPossessionPlayer == player;
    PossessionScanResult scan = PossessionScan(ball.predictions, ball.lazyFlip, position, movement,
                                               maximumVelocity, precise, startTimeMilliseconds);
    playerState.timeNeededToGetToBallMilliseconds = scan.timeNeededToGetToBallMilliseconds;
    playerState.timeNeededToGetToBallOptimisticMilliseconds =
        scan.timeNeededToGetToBallOptimisticMilliseconds;

    const bool touchAnimation = humanoid.currentAnimation.touchFrame != -1;
    if (touchAnimation && touchPending) {
        uint32_t animationTimeToBallMilliseconds =
            (uint32_t)((humanoid.currentAnimation.touchFrame -
                        humanoid.currentAnimation.frameNumber) *
                       10);
        if (animationTimeToBallMilliseconds < playerState.timeNeededToGetToBallMilliseconds)
            playerState.timeNeededToGetToBallMilliseconds = animationTimeToBallMilliseconds;
        playerState.timeNeededToGetToBallOptimisticMilliseconds =
            playerState.timeNeededToGetToBallMilliseconds;
    }

    if (playerState.timeNeededToGetToBallMilliseconds < DEFAULT_TOUCH_OFFSET_MILLISECONDS) {
        // Exact: `DEFAULT_TOUCH_OFFSET_MILLISECONDS * 0.001` is a double that rounds to float
        // before it scales the movement.
        playerState.timeNeededToGetToBallMilliseconds = FloatToUnsigned32(
            NormalizedClamp(
                ((position + movement * (float)(DEFAULT_TOUCH_OFFSET_MILLISECONDS * 0.001)) -
                 BallPredict(ball, DEFAULT_TOUCH_OFFSET_MILLISECONDS).Get2D())
                    .GetLength(),
                0.0f, 0.6f) *
            DEFAULT_TOUCH_OFFSET_MILLISECONDS);
        playerState.timeNeededToGetToBallOptimisticMilliseconds =
            playerState.timeNeededToGetToBallMilliseconds;
    }

    if (IsPassOrShot(humanoid.currentAnimation.functionType) && !touchPending) {
        playerState.hasPossession = false;
    } else {
        playerState.hasPossession = HasPossession(world, player);
    }

    playerState.hasBestPossession =
        playerState.hasPossession && opponentTeam.timeNeededToGetToBallMilliseconds >
                                         (int32_t)playerState.timeNeededToGetToBallMilliseconds;
    playerState.hasUniquePossession = playerState.hasPossession && !opponentTeam.hasPossession;

    if (world.serialized.match.ballRetainer == player) {
        playerState.timeNeededToGetToBallMilliseconds = 1;
        playerState.timeNeededToGetToBallOptimisticMilliseconds = 1;
        playerState.desiredTimeToBallMilliseconds =
            (int32_t)playerState.timeNeededToGetToBallMilliseconds;
        playerState.hasPossession = true;
        playerState.hasBestPossession = true;
        playerState.hasUniquePossession = true;
    } else if (world.serialized.match.ballRetainer != -1) {
        playerState.hasPossession = false;
        playerState.hasBestPossession = false;
        playerState.hasUniquePossession = false;
    }
}

}  // namespace simulation
}  // namespace cuju

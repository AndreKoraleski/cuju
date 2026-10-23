// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The side of a team, its active players, and the directions of passes and shots.
// Exact: the pass and shot math depends on its operation order and the double steps it keeps.
#pragma once

#include <vector>

#include "../libm/powf.h"
#include "humanoid.h"

namespace cuju {
namespace simulation {

// The sign of the x coordinate of the goal that the team defends, in the current frame.
CUJU_FUNCTION inline int TeamDynamicSide(const World &world, int teamID) {
    return world.flags.teamSide[teamID];
}

CUJU_FUNCTION inline PlayerList TeamActivePlayers(const World &world, int teamID) {
    PlayerList players;
    for (int player = 0; player < world.serialized.match.playerCount; player++)
        if (world.playerTeamID[player] == teamID &&
            world.serialized.match.players[player].base.isActive)
            players.push_back(player);
    return players;
}

CUJU_FUNCTION_NO_INLINE inline void GetAutoPass(int passType, const Vector3 &vector,
                                                Vector3 &resultingDirection,
                                                float &resultingPower) {
    float heightOffset = 0.11f;
    float powerFactor = 1.8f;
    float distanceExponent = 1.4f;
    if (passType == e_FunctionType_HighPass) {
        heightOffset = 0.45f - NormalizedClamp(vector.GetLength(), 0.0f, 60.0f) * 0.15f;
        powerFactor = 1.15f;
        distanceExponent = 1.4f;
    }
    resultingDirection = (vector.GetNormalized(0) + Vector3(0, 0, heightOffset)).GetNormalized(0);
    resultingPower =
        libm::powf(NormalizedClamp(vector.GetLength(), 0.0f, 60.0f), distanceExponent) *
        powerFactor;
}

CUJU_FUNCTION_NO_INLINE inline void GetPass(const World &world, int player, int passType,
                                            const Vector3 &inputDirection, float inputPower,
                                            float autoDirectionBias, float autoPowerBias,
                                            Vector3 &resultingDirection, float &resultingPower,
                                            int &targetPlayer, int forcedTargetPlayer = -1) {
    const state::SpatialState &spatialState = Humanoid(world, player).spatialState;
    const state::AnimationState &currentAnimation = Humanoid(world, player).currentAnimation;
    const int teamID = world.playerTeamID[player];

    bool fullAutoDirection = false;
    bool fullAutoPower = false;
    if (ExternalControllerActive(world, player)) {
        fullAutoDirection = true;
        fullAutoPower = true;
    }
    float adaptedAutoDirectionBias = autoDirectionBias;
    float adaptedAutoPowerBias = autoPowerBias;

    Vector3 playerPosition = spatialState.position + spatialState.movement.GetNormalized(0) * 0.2f;
    if (currentAnimation.touchFrame != -1) playerPosition = currentAnimation.touchPosition.Get2D();
    Vector3 manualTarget =
        playerPosition + inputDirection * clamp(inputPower * 60.0f, 1.0f, 100.0f);

    PlayerList players = TeamActivePlayers(world, teamID);
    if (players.size() < 2) {
        resultingDirection = spatialState.directionVector;
        resultingPower = 1.0f;
        targetPlayer = player;
        fullAutoDirection = false;
        fullAutoPower = false;
        adaptedAutoDirectionBias = 0.0f;
        adaptedAutoPowerBias = 0.0f;
    }

    int bestTargetPlayer = player;
    Vector3 autoTarget = manualTarget;
    if (forcedTargetPlayer >= 0) {
        bestTargetPlayer = forcedTargetPlayer;
        const state::SpatialState &forcedState = Humanoid(world, forcedTargetPlayer).spatialState;
        float passDuration = 0.3f + (forcedState.position - playerPosition).GetLength() * 0.05f;
        passDuration = libm::powf(clamp(passDuration, 0.0f, 1.0f), 0.7f) * 0.7f;
        autoTarget = forcedState.position + forcedState.movement * passDuration;
    } else {
        float bestRating = 10000;
        autoTarget = playerPosition;
        for (int teammate : players) {
            if (teammate == player) continue;
            const state::SpatialState &teammateState = Humanoid(world, teammate).spatialState;
            float passDuration =
                0.3f + (teammateState.position - playerPosition).GetLength() * 0.05f;
            passDuration = libm::powf(clamp(passDuration, 0.0f, 1.0f), 0.7f) * 0.7f;
            Vector3 targetPosition = teammateState.position + teammateState.movement * passDuration;
            float distanceRating =
                libm::powf(
                    NormalizedClamp((targetPosition - manualTarget).GetLength(), 0.0f, 70.0f),
                    0.8f) *
                0.8f;
            float angleRating =
                fabsf(
                    (targetPosition - playerPosition).GetNormalized(0).GetAngle2D(inputDirection) /
                    (1.0f * PI)) *
                1.0f;
            if (distanceRating + angleRating < bestRating) {
                bestRating = distanceRating + angleRating;
                bestTargetPlayer = teammate;
                autoTarget = targetPosition;
            }
        }
    }
    targetPlayer = bestTargetPlayer;

    Vector3 autoTargetRelative = autoTarget - playerPosition;
    Vector3 manualTargetRelative = manualTarget - playerPosition;

    if (forcedTargetPlayer >= 0 || (fullAutoDirection && fullAutoPower)) {
        adaptedAutoDirectionBias = 1.0;
        adaptedAutoPowerBias = 1.0;
    } else {
        float maximumAllowedDistance = 50.0f;
        // Exact: the subtraction from 1 runs in double and rounds to float once.
        float distanceFactor =
            (float)(1.0 - (double)libm::powf(
                              clamp((autoTargetRelative - manualTargetRelative).GetLength() /
                                        maximumAllowedDistance,
                                    0.0f, 1.0f),
                              1.5f));
        float proximityBonus = libm::powf(
            1.0f - NormalizedClamp((playerPosition - autoTarget).GetLength(), 0.0f, 12.0f), 0.5f);
        if (fullAutoDirection) {
            adaptedAutoDirectionBias = 1.0f;
        } else {
            adaptedAutoDirectionBias *= distanceFactor;
            adaptedAutoDirectionBias =
                libm::powf(adaptedAutoDirectionBias, 1.0f - proximityBonus * 0.9f);
        }
        if (fullAutoPower) {
            adaptedAutoPowerBias = 1.0f;
        } else {
            adaptedAutoPowerBias *= distanceFactor;
            adaptedAutoPowerBias = clamp(
                (float)((double)adaptedAutoPowerBias * (1.0 + (double)proximityBonus)), 0.0f, 1.0f);
        }
    }

    Vector3 offset;
    if (passType == e_FunctionType_LongPass) {
        float targetDistance = autoTargetRelative.GetLength() * adaptedAutoDirectionBias +
                               manualTargetRelative.GetLength() * (1.0f - adaptedAutoDirectionBias);
        offset = Vector3(-TeamDynamicSide(world, teamID) * targetDistance * 0.2f, 0, 0);
        autoTargetRelative += offset;
        manualTargetRelative += offset;
    }
    Vector3 resultingTargetRelative =
        (autoTargetRelative.GetNormalized(0) * adaptedAutoDirectionBias +
         manualTargetRelative.GetNormalized(0) * (1.0f - adaptedAutoDirectionBias))
            .GetNormalized(manualTarget);
    resultingTargetRelative *=
        float(autoTargetRelative.GetLength() * adaptedAutoPowerBias +
              manualTargetRelative.GetLength() * (1.0f - adaptedAutoPowerBias));
    GetAutoPass(passType, resultingTargetRelative, resultingDirection, resultingPower);
}

CUJU_FUNCTION_NO_INLINE inline Vector3 GetShotDirection(const World &world, int player,
                                                        const Vector3 &inputDirection,
                                                        float autoDirectionBias) {
    const state::SpatialState &spatialState = Humanoid(world, player).spatialState;
    const int side = TeamDynamicSide(world, world.playerTeamID[player]);
    Vector3 manualDirection = inputDirection;
    Vector3 goalPosition = Vector3(side * -PITCH_HALF_LENGTH, 0, 0);
    Vector3 toGoal =
        (goalPosition - (spatialState.position + spatialState.movement * 0.12f)).GetNormalized(0);
    float relativeAngle = toGoal.GetAngle2D(inputDirection);
    float sideFactor = clamp((relativeAngle / PI) / 0.5f, -1.0f, 1.0f);
    sideFactor = libm::powf(fabsf(sideFactor), 0.7f) * SignSide(sideFactor);
    goalPosition.coordinates[1] = sideFactor * GOAL_HALF_WIDTH * 0.9f * side;
    Vector3 autoDirection =
        (goalPosition - (spatialState.position + spatialState.movement * 0.12f)).GetNormalized(0);
    return (manualDirection * (1.0f - autoDirectionBias) + autoDirection * autoDirectionBias)
        .GetNormalized(inputDirection);
}

}  // namespace simulation
}  // namespace cuju

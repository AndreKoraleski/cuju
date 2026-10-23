// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The touches of a humanoid. A touch vector is the momentum of the ball after it, in m/s.
// Exact: the results depend on the number and the order of the RNG draws.
#pragma once

#include <stdexcept>

#include "../libm/powf.h"
#include "../rng.h"
#include "human_controller.h"
#include "humanoid.h"
#include "player_possession.h"
#include "select_animation.h"

namespace cuju {
namespace simulation {

CUJU_FUNCTION inline bool ControllerFloatVelocity(World &world, int player, float &velocity) {
    if (ExternalControllerActive(world, player)) {
        velocity = HumanGetFloatVelocity(world, player);
        return true;
    }
    velocity = world.serialized.match.players[player].base.controller.lastDesiredVelocity;
    return true;
}

// Three errors of a touch from 0 to 1. `ballMovementFactor` is the share of the incoming ball
// movement that the touch keeps.
CUJU_FUNCTION_NO_INLINE inline void GetDifficultyFactors(World &world,
                                                         const SimulationContext &context,
                                                         int player, const Vector3 &positionOffset,
                                                         float &distanceFactor, float &heightFactor,
                                                         float &ballMovementFactor) {
    const state::BallSerialized &ball = world.serialized.match.ball;
    const state::SpatialState &spatialState = Humanoid(world, player).spatialState;
    distanceFactor = 0.0f;
    heightFactor = 0.0f;
    ballMovementFactor = 0.0f;
    float positionOffsetPenalty = NormalizedClamp(positionOffset.GetLength(), 0.0f, 0.1f) * 2.0f;
    float ballBodyVelocityPenalty =
        libm::powf(
            NormalizedClamp((spatialState.movement - ball.momentum).GetLength(), 10.0f, 50.0f),
            1.5f) *
        5.0f;
    float normalizedDistance =
        NormalizedClamp(((spatialState.position + spatialState.directionVector * 0.2f) -
                         BallPredict(ball, 0).Get2D())
                            .GetLength(),
                        0.7f, 1.3f);
    // Exact: the square is one multiplication.
    float fartherAwayPenalty = normalizedDistance * normalizedDistance * 2.0f;

    distanceFactor += positionOffsetPenalty * 2.0f;
    distanceFactor += ballBodyVelocityPenalty;
    distanceFactor += fartherAwayPenalty * 4.0f;
    heightFactor += positionOffsetPenalty * 0.5f;
    heightFactor += ballBodyVelocityPenalty * 2.0f;
    heightFactor += fartherAwayPenalty;

    const int teamID = world.playerTeamID[player];
    if (world.serialized.match.lastTouchTeamID != teamID) {
        int lastTouchPlayer = TeamByID(world, 1 - teamID).lastTouchPlayer;
        if (lastTouchPlayer >= 0) {
            int decay = FloatToInt32(
                1000 - GetStatistic(world, context, player, e_Statistic_PhysicalReaction) * 500);
            float lastTouchBiasPenalty =
                libm::powf(GetLastTouchBias(world, lastTouchPlayer, decay), 0.6f) * 5.0f;
            distanceFactor += lastTouchBiasPenalty;
            heightFactor += lastTouchBiasPenalty;
            ballMovementFactor += lastTouchBiasPenalty * 0.1f;
        }
    }
    ballMovementFactor = clamp(ballMovementFactor, 0.0f, 0.9f);

    float skillPenaltyMultiplier =
        (1.0f - GetStatistic(world, context, player, e_Statistic_TechnicalBallControl) * 0.5f) *
        RandomInRange(world.serialized.rng, 0.5f, 1.0f);
    distanceFactor *= skillPenaltyMultiplier;
    heightFactor *= skillPenaltyMultiplier;
    ballMovementFactor *= skillPenaltyMultiplier;
    distanceFactor = clamp(distanceFactor, 0.0f, 1.0f);
    heightFactor = clamp(heightFactor, 0.0f, 1.0f);
    ballMovementFactor = clamp(ballMovementFactor, 0.0f, 1.0f);
}

CUJU_FUNCTION_NO_INLINE inline bool GetBallControlVector(
    World &world, const SimulationContext &context, int player, const Vector3 &nextStartPosition,
    float nextStartAngle, float nextBodyAngle, const Vector3 &outgoingMovement,
    const state::AnimationState &currentAnimation, int frameNumber,
    const state::SpatialState &spatialState, float &xRotation, float &yRotation,
    Vector3 &touchVector, float additionalFrontOfFootDistance = 0.0f) {
    const state::BallSerialized &ball = world.serialized.match.ball;
    const state::PlayerState &playerState = world.serialized.match.players[player];
    const animation::Animation &animationData =
        context.animations->animations[currentAnimation.animation];
    const state::PlayerCommand &originatingCommand = currentAnimation.originatingCommand;
    auto Statistic = [&](int statistic) { return GetStatistic(world, context, player, statistic); };

    float physicsBias = 0.7f;
    if (!playerState.hasPossession) physicsBias = 0.9f;
    if (FloatToVelocityCategory(animationData.outgoingVelocity) == e_Velocity_Idle)
        physicsBias = 1.0f;

    float explosivenessFactor = 0.6f;
    float maximumOverdriveMetersPerSecond = 1.0f * explosivenessFactor;
    if (originatingCommand.desiredVelocityFloat < DRIBBLE_WALK_SWITCH)
        maximumOverdriveMetersPerSecond = 0.0f;
    if (originatingCommand.desiredVelocityFloat > WALK_SPRINT_SWITCH)
        maximumOverdriveMetersPerSecond = 2.0f * explosivenessFactor;
    float currentDirectionFactor =
        spatialState.directionVector.GetDotProduct(originatingCommand.desiredDirection) * 0.5f +
        0.5f;
    float nextDirectionFactor = Vector3(0, -1, 0)
                                        .GetRotated2D(nextStartAngle)
                                        .GetDotProduct(originatingCommand.desiredDirection) *
                                    0.5f +
                                0.5f;
    float dotFactor = clamp(MinimumFloat(currentDirectionFactor, nextDirectionFactor), 0.0f, 1.0f);
    maximumOverdriveMetersPerSecond *= dotFactor;
    if (FloatToVelocityCategory(animationData.outgoingVelocity) == e_Velocity_Idle)
        maximumOverdriveMetersPerSecond = 0.0f;

    float originatingBias = 0.7f;
    float controllerVelocity;
    if (!ControllerFloatVelocity(world, player, controllerVelocity)) return false;
    Vector3 desiredMovement = originatingCommand.desiredDirection *
                              (originatingCommand.desiredVelocityFloat * originatingBias +
                               controllerVelocity * (1.0f - originatingBias));
    if (desiredMovement.GetLength() > WALK_SPRINT_SWITCH)
        desiredMovement = desiredMovement.GetNormalized(0) *
                          StretchSprintTo(desiredMovement.GetLength(), SPRINT_VELOCITY,
                                          GetMaximumVelocity(*context.staticData, player));
    float velocity = clamp(desiredMovement.GetLength(), outgoingMovement.GetLength(),
                           outgoingMovement.GetLength() + maximumOverdriveMetersPerSecond);
    desiredMovement =
        desiredMovement.GetNormalized(Vector3(0, -1, 0).GetRotated2D(nextStartAngle)) * velocity;
    Vector3 physicsMovement = Vector3(0, -1, 0).GetRotated2D(nextStartAngle) * velocity;

    float desiredVelocity = desiredMovement.GetLength();
    float physicsVelocity = physicsMovement.GetLength();

    Vector3 relativeFrontOfFootOffset =
        GetFrontOfFootOffsetRelative(physicsVelocity, nextBodyAngle - spatialState.angle.value(),
                                     BallPredict(ball, 0).coordinates[2]);
    float annoyanceVelocityFactor = Curve(
        NormalizedClamp(animationData.outgoingVelocity, IDLE_VELOCITY, SPRINT_VELOCITY), 0.7f);
    float opponentAnnoyanceFactor =
        (1.0f - NormalizedClamp(ClosestOpponentDistance(world, player), 0.5f, 1.5f)) *
        (1.0f - (Statistic(e_Statistic_MentalCalmness) * 0.5f +
                 Statistic(e_Statistic_PhysicalBalance) * 0.3f)) *
        annoyanceVelocityFactor;
    Vector3 frontOfFootOffset = Vector3(0, -1, 0).GetRotated2D(nextBodyAngle) *
                                (relativeFrontOfFootOffset.GetLength() +
                                 additionalFrontOfFootDistance + opponentAnnoyanceFactor * 3.0f);
    float heightFrontOfFootFactor =
        NormalizedClamp(BallPredict(ball, 0).coordinates[2], 0.5f, 1.0f) * 0.5f;
    frontOfFootOffset += relativeFrontOfFootOffset * heightFrontOfFootFactor * 0.5f +
                         spatialState.bodyDirectionVector * heightFrontOfFootFactor * 0.5f;

    Vector3 desiredPlannedBallPosition = nextStartPosition;
    Vector3 physicsPlannedBallPosition = nextStartPosition;
    float desiredVelocityFraction = NormalizedClamp(desiredVelocity, 0.0f, SPRINT_VELOCITY);
    float desiredDelayTime = desiredVelocityFraction * desiredVelocityFraction * 0.60f + 0.25f;
    float physicsVelocityFraction = NormalizedClamp(physicsVelocity, 0.0f, SPRINT_VELOCITY);
    float physicsDelayTime = physicsVelocityFraction * physicsVelocityFraction * 0.60f + 0.25f;
    desiredPlannedBallPosition += desiredMovement * desiredDelayTime + frontOfFootOffset;
    physicsPlannedBallPosition += physicsMovement * physicsDelayTime + frontOfFootOffset;

    Vector3 plannedBallPosition = physicsPlannedBallPosition * physicsBias +
                                  desiredPlannedBallPosition * (1.0f - physicsBias);
    Vector3 toPlannedBall = plannedBallPosition - BallPredict(ball, 0).Get2D();

    float timeToGo = (((animationData.frameCount - 1) - frameNumber) * 10) * 0.001f;
    timeToGo += physicsDelayTime * physicsBias + desiredDelayTime * (1.0f - physicsBias);
    timeToGo += DEFAULT_TOUCH_OFFSET_MILLISECONDS * 0.001f;

    float divisor = timeToGo * (0.38f + 0.02f * Statistic(e_Statistic_TechnicalDribble));
    divisor *= 1.1f;
    float power = libm::powf((toPlannedBall.GetLength() / divisor), 0.7f);
    Vector3 direction = toPlannedBall.GetNormalized(Vector3(0, -1, 0).GetRotated2D(nextBodyAngle));
    float height = clamp(0.1f + 1.5f * libm::powf(power / 10.0f, 1.6f), 0.0f, 1.5f);

    float powerMultiplier = 1.2f - (Statistic(e_Statistic_TechnicalBallControl) * 0.03f);
    float velocityBias = NormalizedClamp(velocity, WALK_VELOCITY, SPRINT_VELOCITY - 0.8f);
    powerMultiplier = 1.0f * (1.0f - velocityBias) + powerMultiplier * velocityBias;

    touchVector = direction * power * powerMultiplier + Vector3(0, 0, height);

    float backspinFactor = 30.0f;
    float touchVelocity = touchVector.GetLength();
    float velocityFactor = libm::powf(NormalizedClamp(touchVector.GetLength(), 0.0f, 10.0f), 0.7f);
    Vector3 spinDirection =
        (frontOfFootOffset.GetNormalized(0) * 0.6f + touchVector.GetNormalized(0) * 0.4f)
            .GetNormalized(0);
    xRotation = spinDirection.coordinates[1] *
                ((2 * PI * touchVelocity) - (backspinFactor * velocityFactor));
    yRotation = spinDirection.coordinates[0] *
                ((2 * PI * touchVelocity) - (backspinFactor * velocityFactor));
    return true;
}

CUJU_FUNCTION_NO_INLINE inline bool GetTrapVector(
    World &world, const SimulationContext &context, int player, const Vector3 &nextStartPosition,
    float nextStartAngle, float nextBodyAngle, const Vector3 &outgoingMovement,
    const state::AnimationState &currentAnimation, int frameNumber,
    const state::SpatialState &spatialState, const Vector3 &positionOffset, float &xRotation,
    float &yRotation, Vector3 &result) {
    float distanceFactor, heightFactor, ballMovementFactor;
    GetDifficultyFactors(world, context, player, positionOffset, distanceFactor, heightFactor,
                         ballMovementFactor);
    Vector3 ballControl;
    if (!GetBallControlVector(world, context, player, nextStartPosition, nextStartAngle,
                              nextBodyAngle, outgoingMovement, currentAnimation, frameNumber,
                              spatialState, xRotation, yRotation, ballControl, distanceFactor))
        return false;
    const Vector3 ballMovement = world.serialized.match.ball.momentum;
    ballControl.coordinates[2] +=
        ballMovement.coordinates[2] * heightFactor * 1.0f + heightFactor * 4.0f;
    result = ballControl * (1.0f - ballMovementFactor) + ballMovement * ballMovementFactor;
    return true;
}

CUJU_FUNCTION_NO_INLINE inline Vector3 GetShotVector(
    World &world, const SimulationContext &context, int player,
    const state::AnimationState &currentAnimation, const state::SpatialState &spatialState,
    const Vector3 &positionOffset, float &xRotation, float &yRotation, float &zRotation) {
    const animation::Animation &animationData =
        context.animations->animations[currentAnimation.animation];
    const AnimationDerived &derivedAnimation = context.derived[currentAnimation.animation];
    const state::PlayerCommand &originatingCommand = currentAnimation.originatingCommand;
    const Vector3 ballMovement = world.serialized.match.ball.momentum;
    auto Statistic = [&](int statistic) { return GetStatistic(world, context, player, statistic); };
    uint8_t *rng = world.serialized.rng;

    Vector3 touchMovement =
        CalculateMovementAtFrame(animationData.positions, currentAnimation.frameNumber, 0)
            .GetRotated2D(spatialState.angle.value());
    Vector3 touchDirection = touchMovement.GetNormalized(spatialState.directionVector);
    float touchVelocity = touchMovement.GetLength();
    if (FloatToVelocityCategory(touchVelocity) == e_Velocity_Idle)
        touchDirection = spatialState.directionVector;

    float idealAngle = 0.2f * PI;
    float negativeOffsetAngle = fabsf(touchDirection.GetAngle2D(
        originatingCommand.touchInformation.desiredDirection.GetRotated2D(-idealAngle)));
    float positiveOffsetAngle = fabsf(touchDirection.GetAngle2D(
        originatingCommand.touchInformation.desiredDirection.GetRotated2D(idealAngle)));
    float angle =
        (positiveOffsetAngle < negativeOffsetAngle) ? positiveOffsetAngle : negativeOffsetAngle;
    float directionFactor = clamp(angle / PI, 0.0f, 1.0f);
    directionFactor = Curve(1.0f - directionFactor, 0.8f);

    float playerDirectionDesiredDirectionPowerFactor = libm::powf(directionFactor, 0.8f);
    float playerVelocityPowerFactor =
        libm::powf(NormalizedClamp(touchVelocity, 0.0f, SPRINT_VELOCITY), 0.3f);
    float positionOffsetPowerFactor =
        1.0f - NormalizedClamp(positionOffset.GetLength(), 0.0f, 0.1f);
    float powerFactor = 1.0f * (0.7f + playerDirectionDesiredDirectionPowerFactor * 0.3f) *
                        (0.7f + playerVelocityPowerFactor * 0.3f) *
                        (0.4f + positionOffsetPowerFactor * 0.6f);
    powerFactor = clamp(powerFactor, 0.0f, 1.0f);
    float adaptedDesiredPower =
        45.0f * (0.7f + libm::powf(originatingCommand.touchInformation.desiredPower, 0.5f) * 0.3f);
    float animationMaximumPowerFactor = derivedAnimation.touchMaximumPowerFactor;
    if (animationMaximumPowerFactor == 0.0f) animationMaximumPowerFactor = 1.0f;
    float power = clamp(powerFactor * adaptedDesiredPower, 0.0f,
                        (32.0f + Statistic(e_Statistic_PhysicalShotPower) * 13.0f) *
                            (0.2f + animationMaximumPowerFactor * 0.8f));
    float playerMovementBallMovementPowerFactor = (touchMovement - ballMovement).GetLength();
    playerMovementBallMovementPowerFactor =
        NormalizedClamp(playerMovementBallMovementPowerFactor, 0.0f, 10.0f);
    power *= 1.0f + playerMovementBallMovementPowerFactor * 0.2f;

    float playerDirectionDesiredDirectionEasinessFactor = libm::powf(directionFactor, 0.8f);
    float playerVelocityEasinessFactor =
        1.0f - NormalizedClamp(fabsf(touchVelocity - DRIBBLE_VELOCITY), 0.0f, 4.0f);
    float positionOffsetEasinessFactor =
        1.0f - NormalizedClamp(positionOffset.GetLength(), 0.0f, 0.1f);
    float playerMovementBallMovementEasinessFactor =
        1.0f - playerMovementBallMovementPowerFactor *
                   (0.5f - Statistic(e_Statistic_TechnicalVolley) * 0.3f);
    float powerEasinessFactor = 1.0f - NormalizedClamp(power, 30.0f, 100.0f);
    powerEasinessFactor = Curve(powerEasinessFactor, 1.0f);
    float easinessFactor = 1.0f * (0.4f + playerDirectionDesiredDirectionEasinessFactor * 0.6f) *
                           (0.7f + playerVelocityEasinessFactor * 0.3f) *
                           (0.3f + positionOffsetEasinessFactor * 0.7f) *
                           (0.6f + playerMovementBallMovementEasinessFactor * 0.4f) *
                           (0.6f + powerEasinessFactor * 0.4f);
    float difficultyFactor = clamp(1.0f - easinessFactor, 0.0f, 1.0f);

    float desiredHeight = 0.05f;
    Vector3 desiredShot = (originatingCommand.touchInformation.desiredDirection.Get2D() +
                           Vector3(0, 0, desiredHeight))
                              .GetNormalized() *
                          power;
    Vector3 worstCaseDirection = originatingCommand.touchInformation.desiredDirection.Get2D();
    float laggyDirectionBias = difficultyFactor * 0.8f;
    worstCaseDirection =
        (touchDirection * laggyDirectionBias + worstCaseDirection * (1.0f - laggyDirectionBias))
            .GetNormalized(0);
    // GRF: the arguments evaluate right to left, so the components draw z, then y, then x.
    float randomZ = RandomInRange(rng, -1, 1);
    float randomY = RandomInRange(rng, -1, 1);
    float randomX = RandomInRange(rng, -1, 1);
    worstCaseDirection =
        worstCaseDirection + (Vector3(randomX, randomY, randomZ) * 0.5f * difficultyFactor);
    worstCaseDirection.Normalize();
    float worstCaseHeight = Curve(libm::powf(difficultyFactor, 0.7f), 0.7f) * 0.7f;
    float worstCasePower = power * (1.0f - libm::powf(difficultyFactor, 0.7f) * 0.5f);
    Vector3 worstCaseShot =
        (worstCaseDirection + Vector3(0, 0, worstCaseHeight)).GetNormalized() * worstCasePower;

    float worstCaseFactor = RandomInRange(rng, 0.0f, 1.0f);
    worstCaseFactor = libm::powf(worstCaseFactor, Statistic(e_Statistic_TechnicalShot) * 0.7f);
    Vector3 shot = desiredShot * (1.0f - worstCaseFactor) + worstCaseShot * worstCaseFactor;

    float randomCurveFactor = 0.3f + worstCaseFactor * 0.7f;
    float plannedCurveFactor = 0.7f;
    xRotation = -originatingCommand.touchInformation.desiredDirection.coordinates[1] * 20.0f +
                (RandomInRange(rng, -20, 20) * randomCurveFactor);
    yRotation = -originatingCommand.touchInformation.desiredDirection.coordinates[0] * 20.0f +
                (RandomInRange(rng, -20, 20) * randomCurveFactor);
    float bodyTouchAngle = spatialState.bodyDirectionVector.GetAngle2D(shot) / PI;
    if (fabsf(bodyTouchAngle) > 0.5f)
        bodyTouchAngle = (1.0f - fabsf(bodyTouchAngle)) * SignSide(bodyTouchAngle);
    bodyTouchAngle *= 2.0f;
    float amount = bodyTouchAngle * 0.25f;
    shot.Rotate2D(amount * (0.4f + 0.6f * NormalizedClamp(shot.GetLength(), 0.0f, 70.0f)));
    zRotation = amount * -420 + (RandomInRange(rng, -20, 20) * plannedCurveFactor);
    return shot;
}

CUJU_FUNCTION_NO_INLINE inline Vector3 GetBestPossibleTouch(World &world,
                                                            const SimulationContext &context,
                                                            int player, const Vector3 &desiredTouch,
                                                            int functionType) {
    HumanoidState &humanoid = world.serialized.match.players[player].base.humanoid;
    const state::AnimationState &currentAnimation = humanoid.currentAnimation;
    const AnimationDerived &derivedAnimation = context.derived[currentAnimation.animation];
    const Vector3 ballMovement = world.serialized.match.ball.momentum;
    const float maximumPowerShortPass = 30.0f, maximumPowerHighPass = 42.0f;
    float maximumPowerBase = maximumPowerShortPass;
    if (functionType == e_FunctionType_HighPass) maximumPowerBase = maximumPowerHighPass;
    Vector3 resultTouch = desiredTouch;

    float maximumPowerFactor = derivedAnimation.touchMaximumPowerFactor;
    if (maximumPowerFactor == 0.0f) maximumPowerFactor = 1.0f;
    maximumPowerFactor = maximumPowerFactor * 0.7f + 0.3f;
    float maximumPower =
        maximumPowerBase * maximumPowerFactor *
        (1.0f - clamp(humanoid.decayingPositionOffset.GetLength() * 2.5f, 0.0f, 0.25f));
    maximumPower += ballMovement.GetLength() * 0.5f;
    if (resultTouch.GetLength() > maximumPower) {
        float missingPower = resultTouch.GetLength() - maximumPower;
        resultTouch = resultTouch.GetNormalized(0) * maximumPower;
        resultTouch.coordinates[2] += clamp(missingPower, 0.0f, 10.0f) * 0.25f;
    }

    float difficultyFactor = derivedAnimation.touchDifficultyFactor;
    if (functionType == e_FunctionType_ShortPass || functionType == e_FunctionType_LongPass)
        difficultyFactor *=
            (1.0f - GetStatistic(world, context, player, e_Statistic_TechnicalShortPass) * 0.5f);
    if (functionType == e_FunctionType_HighPass)
        difficultyFactor *=
            (1.0f - GetStatistic(world, context, player, e_Statistic_TechnicalHighPass) * 0.5f);

    float distanceFactor, heightFactor, ballMovementFactor;
    GetDifficultyFactors(world, context, player, humanoid.decayingPositionOffset, distanceFactor,
                         heightFactor, ballMovementFactor);
    float randomRotation = distanceFactor * 0.15f + heightFactor * 0.15f +
                           ballMovementFactor * 0.3f + difficultyFactor * 0.5f;
    Vector3 animationBallDirection = derivedAnimation.ballDirection.GetRotated2D(
        humanoid.startAngle.value() + currentAnimation.rotationSmuggleOffset.value());
    if (animationBallDirection.GetLength() > 0.01f) {
        float bias = clamp(randomRotation * 1.5f, 0.0f, 1.0f);
        Vector3 nativeTouch =
            animationBallDirection.GetNormalized(resultTouch).Get2D() * resultTouch.GetLength() +
            resultTouch * Vector3(0, 0, 1);
        resultTouch = resultTouch * (1.0f - bias) + nativeTouch * bias;
    } else {
        float cappedRotation = (0.5f < randomRotation) ? 0.5f : randomRotation;
        float rotation =
            RandomInRange(world.serialized.rng, -0.5f * PI, 0.5f * PI) * cappedRotation;
        resultTouch.Rotate2D(rotation);
    }
    resultTouch *= 1.0f - distanceFactor * 0.3f;
    resultTouch.coordinates[2] += distanceFactor * 1.5f;
    resultTouch.coordinates[2] +=
        ballMovement.coordinates[2] * heightFactor * 0.5f + heightFactor * 1.0f;
    resultTouch = resultTouch * (1.0f - ballMovementFactor) + ballMovement * ballMovementFactor;
    resultTouch.coordinates[2] +=
        difficultyFactor * 5.0f * RandomInRange(world.serialized.rng, 0.2f, 1.0f);
    return resultTouch;
}

}  // namespace simulation
}  // namespace cuju

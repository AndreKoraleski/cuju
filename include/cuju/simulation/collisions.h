// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The collisions of the ball with the players and of the players with each other.
// Exact: the geometries of a humanoid come in pre-order over its nodes, and sums keep that order.
#pragma once

#include "../profiling.h"

#include <vector>

#include "../animation/skeleton.h"
#include "../rng.h"
#include "ai_functions.h"
#include "ball_touch.h"
#include "humanoid.h"
#include "select_animation.h"

namespace cuju {
namespace simulation {

// GRF: before the first touch, the team processed first counts as the last to touch.
CUJU_FUNCTION inline float MatchLastTouchBias(const World &world, int decayMilliseconds) {
    int team = world.serialized.match.lastTouchTeamID != -1 ? world.serialized.match.lastTouchTeamID
                                                            : world.FirstTeamID();
    return TeamLastTouchBias(world, team, decayMilliseconds);
}

CUJU_FUNCTION_NO_INLINE inline bool CheckBallCollisions(World &world,
                                                        const SimulationContext &context) {
    CUJU_PROFILE_SCOPE(CHECK_BALL_COLLISIONS);
    state::MatchState &match = world.serialized.match;
    if (match.actualTimeMilliseconds <=
        (uint64_t)(uint32_t)(match.lastBodyBallCollisionTimeMilliseconds + 150))
        return true;

    PlayerList players = TeamActivePlayers(world, world.FirstTeamID());
    for (int player : TeamActivePlayers(world, 1 - world.FirstTeamID())) players.push_back(player);

    Vector3 bounceVector;
    float bias = 0.0;
    int bounceCount = 0;
    for (int player : players) {
        const int teamID = world.playerTeamID[player];
        const int touchTimeThresholdMilliseconds = 200;
        float opponentLastTouchBias =
            TeamLastTouchBias(world, 1 - teamID, touchTimeThresholdMilliseconds);
        float lastTouchBias = GetLastTouchBias(world, player, touchTimeThresholdMilliseconds);
        if (!(lastTouchBias <= 0.01f && opponentLastTouchBias > 0.01f)) continue;

        state::PlayerState &playerState = match.players[player];
        const HumanoidState &humanoid = playerState.base.humanoid;
        const int functionType = humanoid.currentAnimation.functionType;
        bool collisionAnimation =
            functionType == e_FunctionType_Movement || functionType == e_FunctionType_Trip ||
            functionType == e_FunctionType_Sliding || functionType == e_FunctionType_Interfere ||
            functionType == e_FunctionType_Deflect;
        bool onlyWhenDirectionChangedUnexpectedly =
            functionType == e_FunctionType_Interfere || functionType == e_FunctionType_Deflect;
        bool directionChangedUnexpectedly = false;
        if (onlyWhenDirectionChangedUnexpectedly) {
            int imageIndex =
                MentalImageIndex(world, GetReactionTimeMilliseconds(world, context, player) +
                                            humanoid.currentAnimation.frameNumber * 10);
            float unexpectedDistance =
                (MentalBallPrediction(world, imageIndex, 1000) - BallPredict(match.ball, 1000))
                    .GetLength();
            if (unexpectedDistance > 0.5f) directionChangedUnexpectedly = true;
        }
        if (!(collisionAnimation && !playerState.hasUniquePossession &&
              onlyWhenDirectionChangedUnexpectedly == directionChangedUnexpectedly))
            continue;

        float boundingBoxSizeOffset = -0.1f;
        if (!playerState.hasPossession)
            boundingBoxSizeOffset += 0.03f;
        else
            boundingBoxSizeOffset -= 0.03f;
        if (functionType == e_FunctionType_Sliding || functionType == e_FunctionType_Interfere)
            boundingBoxSizeOffset += 0.1f;
        if (functionType == e_FunctionType_Deflect) boundingBoxSizeOffset += 0.2f;

        if (!(((humanoid.spatialState.position + Vector3(0, 0, 0.8f)) - BallPredict(match.ball, 0))
                  .GetLength() < 2.5f))
            continue;

        const animation::Skeleton &skeleton = context.skeletons[player];
        animation::Derived derived[state::HUMANOID_NODES];
        animation::DeriveNodes(skeleton, humanoid.nodes, derived);
        for (size_t node = 0; node < skeleton.nodes.size(); node++) {
            for (const animation::SkeletonGeometry &geometry : skeleton.nodes[node].geometries) {
                animation::WorldGeometry worldGeometry =
                    animation::GeometryWorld(geometry, derived[node]);
                float ballRadius = 0.11f + boundingBoxSizeOffset;
                if (!worldGeometry.boundingBox.Intersects(BallPredict(match.ball, 0), ballRadius))
                    continue;
                if (player == TeamByID(world, teamID).designatedTeamPossessionPlayer &&
                    MatchLastTouchBias(world, 200) < 0.01f) {
                    playerState.triggerControlledBallCollision = true;
                } else {
                    float movementBias = opponentLastTouchBias * 0.8f + 0.2f;
                    bounceVector += (BallPredict(match.ball, 0) - worldGeometry.position)
                                            .GetNormalized(Vector3(0)) *
                                        movementBias +
                                    humanoid.spatialState.movement * (1.0f - movementBias);
                    bounceCount++;
                    if (!TeamSetLastTouchPlayer(world, context, player, e_TouchType_Accidental))
                        return false;
                    Vector3 boxCenter = worldGeometry.boundingBox.Center();
                    bias +=
                        (1.0f -
                         clamp(((BallPredict(match.ball, 0) - boxCenter).GetLength() - ballRadius) /
                                   worldGeometry.boundingBox.Radius(),
                               0.0f, 1.0f)) *
                            0.9f +
                        0.1f;
                }
            }
        }
    }

    if (bias > 0.0f) {
        bounceVector /= Vector3(bounceCount * 1.0f);
        bounceVector.coordinates[2] *= 0.6f;
        bounceVector.Normalize();
        Vector3 currentMovement = match.ball.momentum;
        Vector3 fullCollisionVector = (bounceVector * 6.0f) +
                                      (bounceVector * currentMovement.GetLength() * 0.6f) +
                                      (currentMovement * -0.2f);
        bias = clamp(bias, 0.0f, 1.0f);
        bias = bias * 0.5f + 0.5f;
        Vector3 resultVector = fullCollisionVector * bias + currentMovement * (1.0f - bias);
        if (resultVector.GetLength() > currentMovement.GetLength())
            resultVector = resultVector.GetNormalized(0) * currentMovement.GetLength();
        resultVector *= 0.7f;
        BallTouch(world, context, resultVector);
        // GRF: the arguments evaluate right to left, so the spin draws z, then y, then x.
        float randomZ = RandomInRange(world.serialized.rng, -30, 30);
        float randomY = RandomInRange(world.serialized.rng, -30, 30);
        float randomX = RandomInRange(world.serialized.rng, -30, 30);
        BallSetRotation(world, randomX, randomY, randomZ, 0.5f * bias);
        match.lastBodyBallCollisionTimeMilliseconds = (uint32_t)match.actualTimeMilliseconds;
    }
    return true;
}

// The position in the fixed frame of the pitch, in which team 0 starts on side -1.
CUJU_FUNCTION inline Vector3 PitchPosition(const World &world, int player) {
    Vector3 position = Humanoid(world, player).spatialState.position;
    const int teamID = world.playerTeamID[player];
    const int side = world.flags.teamSide[teamID];
    bool onOriginalSide = teamID == 0 ? (side == -1) : (side == 1);
    if (!onOriginalSide) position.Mirror();
    return position;
}

CUJU_FUNCTION_NO_INLINE inline void RefereeTripNotice(World &world, int trippedPlayer,
                                                      int trippingPlayer, int tackleType) {
    state::MatchState &match = world.serialized.match;
    state::RefereeState &referee = match.referee;
    if (referee.buffer.active) return;
    const state::HumanoidState &trippedHumanoid = Humanoid(world, trippedPlayer),
                               &trippingHumanoid = Humanoid(world, trippingPlayer);
    const int trippedTeam = world.playerTeamID[trippedPlayer],
              trippingTeam = world.playerTeamID[trippingPlayer];
    if (tackleType == 2) {
        if ((double)TeamByID(world, trippedTeam).fadingTeamPossessionAmount > 1.1 &&
            (trippingHumanoid.currentAnimation.functionType == e_FunctionType_Interfere ||
             trippingHumanoid.currentAnimation.functionType == e_FunctionType_Sliding) &&
            (double)(trippedHumanoid.spatialState.position - BallPredict(match.ball, 0).Get2D())
                    .GetLength() < 2.0 &&
            trippingTeam != trippedTeam) {
            referee.foul.foulType = 1;
            referee.foul.advantage = true;
            referee.foul.foulPlayer = trippingPlayer;
            referee.foul.foulVictim = trippedPlayer;
            referee.foul.foulTime = match.actualTimeMilliseconds;
            referee.foul.foulPosition = PitchPosition(world, trippedPlayer);
            referee.foul.hasBeenProcessed = false;
        }
    } else if (tackleType == 3 &&
               (trippingPlayer != referee.foul.foulPlayer || referee.foul.foulType == 0)) {
        if (match.actualTimeMilliseconds -
                    match.players[trippingPlayer].base.lastTouchTimeMilliseconds >
                600 &&
            trippingHumanoid.currentAnimation.functionType == e_FunctionType_Sliding &&
            trippingTeam != trippedTeam &&
            (double)(BallPredict(match.ball, 0) - trippedHumanoid.spatialState.position)
                    .GetLength() < 8.0) {
            float severity = 1.0;
            if (trippingHumanoid.currentAnimation.touchFrame != -1) {
                // Exact: the frame difference divides as a double, `clamp` takes floats, and the
                // power runs in double.
                const int touchFrame = trippingHumanoid.currentAnimation.touchFrame,
                          currentFrame = trippingHumanoid.currentAnimation.frameNumber;
                float timing = clamp(
                    (float)(fabs((double)(touchFrame - currentFrame)) / touchFrame), 0.0, 1.0);
                severity = (float)(libm::pow((double)timing, 0.7) * 0.5);
                severity = (float)(severity +
                                   NormalizedClamp((BallPredict(match.ball, 0) -
                                                    trippingHumanoid.currentAnimation.touchPosition)
                                                       .GetLength(),
                                                   0.0, 2.0) *
                                       0.5);
            }
            // GRF: adds `dot * 0.5 + 0.5` to the severity as one term.
            severity =
                (float)(severity +
                        ((trippedHumanoid.spatialState.position -
                          trippingHumanoid.spatialState.position)
                                 .GetNormalized(0)
                                 .GetDotProduct(trippedHumanoid.spatialState.directionVector) *
                             0.5 +
                         0.5));
            if ((double)severity > 1.0) {
                referee.foul.foulType = 1;
                referee.foul.advantage = false;
                referee.foul.foulPlayer = trippingPlayer;
                referee.foul.foulVictim = trippedPlayer;
                referee.foul.foulTime = match.actualTimeMilliseconds;
                referee.foul.foulPosition = PitchPosition(world, trippedPlayer);
                referee.foul.hasBeenProcessed = false;
                if ((double)severity > 1.4) referee.foul.foulType = 2;
                if ((double)severity > 2.0) referee.foul.foulType = 3;
            }
        }
    }
}

CUJU_FUNCTION_NO_INLINE inline void TripMe(World &world, const SimulationContext &context,
                                           int player, const Vector3 &tripVector, int tripType) {
    if (world.serialized.match.ballRetainer == player) return;
    state::HumanoidState &humanoid = world.serialized.match.players[player].base.humanoid;
    const animation::Animation &playingAnimation =
        context.animations->animations[humanoid.currentAnimation.animation];
    if (playingAnimation.incomingSpecialState != animation::STRING_EMPTY ||
        playingAnimation.outgoingSpecialState != animation::STRING_EMPTY)
        return;
    if (humanoid.interruptAnimation == e_InterruptAnimation_None &&
        (humanoid.currentAnimation.functionType != e_FunctionType_Trip ||
         (context.derived[humanoid.currentAnimation.animation].tripTypeIsOne && tripType > 1)) &&
        humanoid.currentAnimation.functionType != e_FunctionType_Sliding) {
        humanoid.interruptAnimation = e_InterruptAnimation_Trip;
        humanoid.tripDirection = tripVector;
        humanoid.tripType = tripType;
    }
}

// A soft push from a player close by.
struct PlayerBounce {
    int opponent;
    float force;
};
using PlayerBounceList = InlineVector<PlayerBounce, state::MAXIMUM_PLAYERS>;

CUJU_FUNCTION inline bool BoundingBoxesIntersect(const animation::AxisAlignedBoundingBox &box,
                                                 const animation::AxisAlignedBoundingBox &other) {
    return (other.minimumCorner.coordinates[0] < box.maximumCorner.coordinates[0] &&
            other.minimumCorner.coordinates[1] < box.maximumCorner.coordinates[1] &&
            other.minimumCorner.coordinates[2] < box.maximumCorner.coordinates[2]) &&
           (other.maximumCorner.coordinates[0] > box.minimumCorner.coordinates[0] &&
            other.maximumCorner.coordinates[1] > box.minimumCorner.coordinates[1] &&
            other.maximumCorner.coordinates[2] > box.minimumCorner.coordinates[2]);
}

struct PlayerGeometry {
    const animation::SkeletonGeometry *skeletonGeometry;
    animation::WorldGeometry worldGeometry;
};
using PlayerGeometryList =
    InlineVector<PlayerGeometry, state::HUMANOID_NODES * animation::MAXIMUM_GEOMETRIES_PER_NODE>;
CUJU_FUNCTION_NO_INLINE inline void PlayerGeometries(const World &world,
                                                     const SimulationContext &context, int player,
                                                     PlayerGeometryList &out) {
    const animation::Skeleton &skeleton = context.skeletons[player];
    animation::Derived derived[state::HUMANOID_NODES];
    animation::DeriveNodes(skeleton, Humanoid(world, player).nodes, derived);
    out.clear();
    for (size_t node = 0; node < skeleton.nodes.size(); node++)
        for (const animation::SkeletonGeometry &geometry : skeleton.nodes[node].geometries)
            out.push_back(
                PlayerGeometry{&geometry, animation::GeometryWorld(geometry, derived[node])});
}

CUJU_FUNCTION_NO_INLINE inline void CheckHumanoidCollision(World &world,
                                                           const SimulationContext &context,
                                                           int firstPlayer, int secondPlayer,
                                                           PlayerBounceList &firstBounces,
                                                           PlayerBounceList &secondBounces) {
    state::MatchState &match = world.serialized.match;
    const float distanceFactor = 0.72f;
    const float bouncePlayerRadius = 0.5f * distanceFactor;
    const float similarPlayerRadius = 0.8f * distanceFactor;
    const float similarExponent = 0.2f;
    const float similarForceFactor = 0.25f;
    auto spatialStateOf = [&](int player) -> const state::SpatialState & {
        return Humanoid(world, player).spatialState;
    };
    auto statistic = [&](int player, int statisticIndex) {
        return GetStatistic(world, context, player, statisticIndex);
    };
    auto functionTypeOf = [&](int player) {
        return Humanoid(world, player).currentAnimation.functionType;
    };
    auto touchPending = [&](int player) {
        return Humanoid(world, player).currentAnimation.frameNumber <
               Humanoid(world, player).currentAnimation.touchFrame;
    };
    auto designatedTeamPlayer = [&](int player) {
        return TeamByID(world, world.playerTeamID[player]).designatedTeamPossessionPlayer;
    };

    Vector3 firstPosition = spatialStateOf(firstPlayer).position;
    Vector3 secondPosition = spatialStateOf(secondPlayer).position;
    float distance = (firstPosition - secondPosition).GetLength();

    float bounceBias = 0.0f;
    Vector3 bounceVector;
    float firstBackFacing = 0.5f;
    float secondBackFacing = 0.5f;

    if (distance < bouncePlayerRadius * 2.0f ||
        distance < (bouncePlayerRadius + similarPlayerRadius) * 2.0f) {
        bounceVector = (firstPosition - secondPosition).GetNormalized(Vector3(0, -1, 0));
        Vector3 firstFacing = spatialStateOf(firstPlayer)
                                  .directionVector.GetRotated2D(
                                      spatialStateOf(firstPlayer).relativeBodyAngle.value() * 0.7f);
        Vector3 secondFacing =
            spatialStateOf(secondPlayer)
                .directionVector.GetRotated2D(
                    spatialStateOf(secondPlayer).relativeBodyAngle.value() * 0.7f);
        firstBackFacing = clamp(firstFacing.GetDotProduct(bounceVector) * 0.5f + 0.5f, 0.0f, 1.0f);
        secondBackFacing =
            clamp(secondFacing.GetDotProduct(-bounceVector) * 0.5f + 0.5f, 0.0f, 1.0f);

        auto ballDistanceDifference = [&]() {
            float firstBallDistance =
                (BallPredict(match.ball, 10).Get2D() - spatialStateOf(firstPlayer).position)
                    .GetLength();
            float secondBallDistance =
                (BallPredict(match.ball, 10).Get2D() - spatialStateOf(secondPlayer).position)
                    .GetLength();
            float secondCapped = MinimumFloat(secondBallDistance, 1.2f),
                  firstCapped = MinimumFloat(firstBallDistance, 1.2f);
            return clamp(secondCapped - firstCapped, -0.6f, 0.6f) * 1.0f;
        };

        if (distance < bouncePlayerRadius * 2.0f) {
            bounceBias += firstBackFacing * 0.8f;
            bounceBias -= secondBackFacing * 0.8f;
            float firstVelocity = PlayerFloatVelocity(world, firstPlayer);
            float secondVelocity = PlayerFloatVelocity(world, secondPlayer);
            bounceBias -=
                clamp(((firstVelocity - secondVelocity) / SPRINT_VELOCITY) * 0.2f, -0.2f, 0.2f);
            if (touchPending(firstPlayer) &&
                functionTypeOf(firstPlayer) == e_FunctionType_Interfere)
                bounceBias +=
                    0.1f + 0.4f * statistic(firstPlayer, e_Statistic_TechnicalStandingTackle);
            if (touchPending(firstPlayer) && functionTypeOf(firstPlayer) == e_FunctionType_Sliding)
                bounceBias +=
                    0.1f + 0.4f * statistic(firstPlayer, e_Statistic_TechnicalSlidingTackle);
            if (touchPending(secondPlayer) &&
                functionTypeOf(secondPlayer) == e_FunctionType_Interfere)
                bounceBias -=
                    0.1f + 0.4f * statistic(secondPlayer, e_Statistic_TechnicalStandingTackle);
            if (touchPending(secondPlayer) &&
                functionTypeOf(secondPlayer) == e_FunctionType_Sliding)
                bounceBias -=
                    0.1f + 0.4f * statistic(secondPlayer, e_Statistic_TechnicalSlidingTackle);
            if (firstPlayer == match.designatedPossessionPlayer) bounceBias += 0.4f;
            if (secondPlayer == match.designatedPossessionPlayer) bounceBias -= 0.4f;
            if (firstPlayer == designatedTeamPlayer(firstPlayer) &&
                secondPlayer == designatedTeamPlayer(secondPlayer))
                bounceBias += ballDistanceDifference();
            bounceBias += statistic(firstPlayer, e_Statistic_PhysicalBalance) * 1.0f;
            bounceBias -= statistic(secondPlayer, e_Statistic_PhysicalBalance) * 1.0f;
            bounceBias = clamp(bounceBias, -1.0f, 1.0f);
            bounceBias *= 0.5f;
            float normalizedBounceBias = bounceBias * 0.5f + 0.5f;
            Vector3 firstOffset = (firstPosition - secondPosition).GetNormalized(0) *
                                  (bouncePlayerRadius - distance * 0.5f) *
                                  (1.0f - normalizedBounceBias) * 2.0f;
            Vector3 secondOffset = (secondPosition - firstPosition).GetNormalized(0) *
                                   (bouncePlayerRadius - distance * 0.5f) * normalizedBounceBias *
                                   2.0f;
            if (match.designatedPossessionPlayer == secondPlayer &&
                match.players[secondPlayer].hasPossession) {
                Vector3 leftPoint =
                    secondPosition +
                    spatialStateOf(secondPlayer).directionVector.GetRotated2D(0.3f * PI) *
                        bouncePlayerRadius * 2;
                Vector3 rightPoint =
                    secondPosition +
                    spatialStateOf(secondPlayer).directionVector.GetRotated2D(-0.3f * PI) *
                        bouncePlayerRadius * 2;
                Vector3 nearSide = (firstPosition - leftPoint).GetLength() <
                                           (firstPosition - rightPoint).GetLength()
                                       ? leftPoint
                                       : rightPoint;
                firstOffset += GetNormalizedMaximum(nearSide - firstPosition, 0.01f) *
                               statistic(firstPlayer, e_Statistic_PhysicalBalance) * 0.3f;
            } else if (match.designatedPossessionPlayer == firstPlayer &&
                       match.players[firstPlayer].hasPossession) {
                Vector3 leftPoint =
                    firstPosition +
                    spatialStateOf(firstPlayer).directionVector.GetRotated2D(0.3f * PI) *
                        bouncePlayerRadius * 2;
                Vector3 rightPoint =
                    firstPosition +
                    spatialStateOf(firstPlayer).directionVector.GetRotated2D(-0.3f * PI) *
                        bouncePlayerRadius * 2;
                Vector3 nearSide = (secondPosition - leftPoint).GetLength() <
                                           (secondPosition - rightPoint).GetLength()
                                       ? leftPoint
                                       : rightPoint;
                secondOffset += GetNormalizedMaximum(nearSide - secondPosition, 0.01f) *
                                statistic(secondPlayer, e_Statistic_PhysicalBalance) * 0.3f;
            }
            NormalizeMaximum(firstOffset, SPRINT_VELOCITY * 0.01f);
            NormalizeMaximum(secondOffset, SPRINT_VELOCITY * 0.01f);
            OffsetPosition(world, firstPlayer, firstOffset);
            OffsetPosition(world, secondPlayer, secondOffset);
        }

        float similarBias = 0.0f;
        if (similarForceFactor > 0.0f &&
            distance < (bouncePlayerRadius + similarPlayerRadius) * 2.0f) {
            float gap = distance - bouncePlayerRadius * 2.0f;
            // Exact: the comparison keeps the operand order of `std::max(0.0f, gap)`.
            float shellDistance = (0.0f < gap) ? gap : 0.0f;
            similarBias += firstBackFacing * 0.8f;
            similarBias -= secondBackFacing * 0.8f;
            float firstVelocity = PlayerFloatVelocity(world, firstPlayer);
            float secondVelocity = PlayerFloatVelocity(world, secondPlayer);
            similarBias -=
                clamp(((firstVelocity - secondVelocity) / SPRINT_VELOCITY) * 0.2f, -0.2f, 0.2f);
            if (firstPlayer == match.designatedPossessionPlayer) similarBias += 0.6f;
            if (secondPlayer == match.designatedPossessionPlayer) similarBias -= 0.6f;
            if (firstPlayer == designatedTeamPlayer(firstPlayer) &&
                secondPlayer == designatedTeamPlayer(secondPlayer))
                similarBias += ballDistanceDifference();
            similarBias += statistic(firstPlayer, e_Statistic_PhysicalBalance) * 1.0f;
            similarBias -= statistic(secondPlayer, e_Statistic_PhysicalBalance) * 1.0f;
            similarBias = clamp(similarBias, -1.0f, 1.0f);
            similarBias *= 0.9f;
            float similarForce =
                clamp(1.0f - (shellDistance / (similarPlayerRadius * 2.0f)), 0.0f, 1.0f);
            similarForce = libm::powf(similarForce, similarExponent);
            similarForce *= similarForceFactor;
            float normalizedSimilarBias = similarBias * 0.5f + 0.5f;
            firstBounces.push_back({secondPlayer, similarForce * (1.0f - normalizedSimilarBias)});
            secondBounces.push_back({firstPlayer, similarForce * normalizedSimilarBias});
        }

        if (distance < bouncePlayerRadius * 2.0f) {
            float firstSensitivity = 0.0f, secondSensitivity = 0.0f;
            firstSensitivity += (1.0f - firstBackFacing) * 1.0f;
            secondSensitivity += (1.0f - secondBackFacing) * 1.0f;
            float firstVelocity = PlayerFloatVelocity(world, firstPlayer);
            float secondVelocity = PlayerFloatVelocity(world, secondPlayer);
            firstSensitivity +=
                NormalizedClamp(firstVelocity, IDLE_VELOCITY, SPRINT_VELOCITY) * 1.0f;
            secondSensitivity +=
                NormalizedClamp(secondVelocity, IDLE_VELOCITY, SPRINT_VELOCITY) * 1.0f;
            if (match.players[firstPlayer].hasBestPossession) firstSensitivity += 1.0f;
            if (match.players[secondPlayer].hasBestPossession) secondSensitivity += 1.0f;
            float balanceWeight = 3.0f;
            firstSensitivity +=
                (1.0f - statistic(firstPlayer, e_Statistic_PhysicalBalance) * 1.0f) * balanceWeight;
            secondSensitivity +=
                (1.0f - statistic(secondPlayer, e_Statistic_PhysicalBalance) * 1.0f) *
                balanceWeight;
            firstSensitivity +=
                clamp(Humanoid(world, firstPlayer).decayingPositionOffset.GetLength() * 10.0f, 0.0f,
                      1.0f);
            secondSensitivity +=
                clamp(Humanoid(world, secondPlayer).decayingPositionOffset.GetLength() * 10.0f,
                      0.0f, 1.0f);
            float penetrationWeight = 6.0f;
            float penetration = ((spatialStateOf(firstPlayer).position +
                                  spatialStateOf(firstPlayer).movement * 0.03f) -
                                 (spatialStateOf(secondPlayer).position +
                                  spatialStateOf(secondPlayer).movement * 0.03f))
                                    .GetLength();
            firstSensitivity +=
                libm::powf(1.0f - NormalizedClamp(penetration, 0.0f, bouncePlayerRadius * 2.0f),
                           0.4f) *
                penetrationWeight;
            secondSensitivity +=
                libm::powf(1.0f - NormalizedClamp(penetration, 0.0f, bouncePlayerRadius * 2.0f),
                           0.4f) *
                penetrationWeight;
            float firstBallDistance =
                (BallPredict(match.ball, 10).Get2D() - spatialStateOf(firstPlayer).position)
                    .GetLength();
            float secondBallDistance =
                (BallPredict(match.ball, 10).Get2D() - spatialStateOf(secondPlayer).position)
                    .GetLength();
            firstSensitivity += 1.0f - NormalizedClamp(firstBallDistance, 0.0f, 0.7f);
            secondSensitivity += 1.0f - NormalizedClamp(secondBallDistance, 0.0f, 0.7f);
            firstSensitivity /= 5.0f + balanceWeight + penetrationWeight;
            secondSensitivity /= 5.0f + balanceWeight + penetrationWeight;
            const float tripTypeZeroThreshold = 0.38f, tripTypeOneThreshold = 0.48f,
                        tripTypeTwoThreshold = 0.58f;
            if (firstSensitivity > tripTypeZeroThreshold) {
                int tripType = 0;
                if (firstSensitivity > tripTypeOneThreshold) tripType = 1;
                if (firstSensitivity > tripTypeTwoThreshold) tripType = 2;
                if (tripType > 0) {
                    TripMe(world, context, firstPlayer,
                           (spatialStateOf(firstPlayer).movement * 0.1f +
                            spatialStateOf(secondPlayer).movement * 0.06f + bounceVector * 1.0f)
                               .GetNormalized(bounceVector),
                           tripType);
                    RefereeTripNotice(world, firstPlayer, secondPlayer, tripType);
                }
            }
            if (secondSensitivity > tripTypeZeroThreshold) {
                int tripType = 0;
                if (secondSensitivity > tripTypeOneThreshold) tripType = 1;
                if (secondSensitivity > tripTypeTwoThreshold) tripType = 2;
                if (tripType > 0) {
                    TripMe(world, context, secondPlayer,
                           (spatialStateOf(secondPlayer).movement * 0.1f +
                            spatialStateOf(firstPlayer).movement * 0.06f - bounceVector * 1.0f)
                               .GetNormalized(-bounceVector),
                           tripType);
                    RefereeTripNotice(world, secondPlayer, firstPlayer, tripType);
                }
            }
        }
    }

    auto frameNumberOf = [&](int player) {
        return Humanoid(world, player).currentAnimation.frameNumber;
    };
    auto frameCountOf = [&](int player) {
        return context.animations->animations[Humanoid(world, player).currentAnimation.animation]
            .frameCount;
    };
    int tackle = 0;
    if ((functionTypeOf(firstPlayer) == e_FunctionType_Sliding ||
         functionTypeOf(firstPlayer) == e_FunctionType_Interfere) &&
        frameNumberOf(firstPlayer) > 5 && frameNumberOf(firstPlayer) < 28)
        tackle += 1;
    if ((functionTypeOf(secondPlayer) == e_FunctionType_Sliding ||
         functionTypeOf(secondPlayer) == e_FunctionType_Interfere) &&
        frameNumberOf(secondPlayer) > 5 && frameNumberOf(secondPlayer) < 28)
        tackle += 2;
    if (distance < 2.0f && tackle > 0 && tackle < 3) {
        const int tackler = tackle == 1 ? firstPlayer : secondPlayer,
                  victim = tackle == 1 ? secondPlayer : firstPlayer;
        PlayerGeometryList tacklerGeometries, victimGeometries;
        PlayerGeometries(world, context, tackler, tacklerGeometries);
        PlayerGeometries(world, context, victim, victimGeometries);
        for (auto &tacklerGeometry : tacklerGeometries) {
            animation::AxisAlignedBoundingBox tacklerBox =
                tacklerGeometry.worldGeometry.boundingBox;
            tacklerBox.minimumCorner += 0.1f;
            tacklerBox.maximumCorner -= Vector3(0.1f);
            for (auto &victimGeometry : victimGeometries) {
                if (victimGeometry.skeletonGeometry->footOrLowerLeg) {
                    if (BoundingBoxesIntersect(tacklerBox,
                                               victimGeometry.worldGeometry.boundingBox)) {
                        if (frameNumberOf(tackler) > 10 &&
                            frameNumberOf(tackler) < frameCountOf(tackler) - 6) {
                            Vector3 tripVector = spatialStateOf(victim).directionVector;
                            int tripType = 3;
                            if (functionTypeOf(tackler) == e_FunctionType_Interfere) tripType = 1;
                            TripMe(world, context, victim, tripVector, tripType);
                            RefereeTripNotice(world, victim, tackler, tripType);
                        }
                        break;
                    }
                }
            }
        }
    }
}

CUJU_FUNCTION_NO_INLINE inline bool CheckHumanoidCollisions(World &world,
                                                            const SimulationContext &context) {
    CUJU_PROFILE_SCOPE(CHECK_HUMANOID_COLLISIONS);
    PlayerList players = TeamActivePlayers(world, world.FirstTeamID());
    for (int player : TeamActivePlayers(world, 1 - world.FirstTeamID())) players.push_back(player);
    PlayerBounceList playerBounces[state::MAXIMUM_PLAYERS];
    for (size_t firstIndex = 0; firstIndex + 1 < players.size(); firstIndex++)
        for (size_t secondIndex = firstIndex + 1; secondIndex < players.size(); secondIndex++)
            CheckHumanoidCollision(world, context, players[firstIndex], players[secondIndex],
                                   playerBounces[firstIndex], playerBounces[secondIndex]);
    for (size_t firstIndex = 0; firstIndex < players.size(); firstIndex++) {
        float totalForce = 0.0f;
        for (const PlayerBounce &bounce : playerBounces[firstIndex]) totalForce += bounce.force;
        if (totalForce > 0.0f) {
            Vector3 bounceVector;
            for (const PlayerBounce &bounce : playerBounces[firstIndex])
                bounceVector += (Humanoid(world, bounce.opponent).spatialState.movement -
                                 Humanoid(world, players[firstIndex]).spatialState.movement) *
                                bounce.force * (bounce.force / totalForce);
            OffsetPosition(world, players[firstIndex], bounceVector * 0.01f * 1.0f);
        }
    }
    return true;
}

}  // namespace simulation
}  // namespace cuju

// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// Command queue of a player each tick, from the built-in AI or from the input of its agent.
// Exact: an expression with a double literal runs in double, such as `3.7 * panic`.
#pragma once

#include "../profiling.h"

#include <cstdlib>
#include <vector>

#include "../ai_time.h"
#include "../libm/powf.h"
#include "ai_functions.h"
#include "human_controller.h"
#include "officials.h"
#include "player_controller.h"
#include "select_animation.h"
#include "team_process.h"
#include "team_select.h"

namespace cuju {
namespace simulation {

CUJU_FUNCTION inline float GetMindSet(int role) {
    float mindSet = 0.5f;
    if (role == e_PlayerRole_Goalkeeper) mindSet = 0.0f;
    if (role == e_PlayerRole_CentreBack) mindSet = 0.0f;
    if (role == e_PlayerRole_LeftBack || role == e_PlayerRole_RightBack) mindSet = 0.25f;
    if (role == e_PlayerRole_DefensiveMidfielder) mindSet = 0.25f;
    if (role == e_PlayerRole_LeftMidfielder || role == e_PlayerRole_CentralMidfielder ||
        role == e_PlayerRole_RightMidfielder)
        mindSet = 0.5f;
    if (role == e_PlayerRole_AttackingMidfielder) mindSet = 0.75f;
    if (role == e_PlayerRole_CentreForward) mindSet = 1.0f;
    return mindSet;
}

CUJU_FUNCTION inline float ClampVelocity(float velocity) {
    if (velocity < 0) return 0;
    if (velocity > SPRINT_VELOCITY) return SPRINT_VELOCITY;
    return velocity;
}

// Exact: the operand orders of `std::max` and `std::min`, which decide the result for NaN.
CUJU_FUNCTION inline float StandardMaximumFloat(float first, float second) {
    return (first < second) ? second : first;
}
CUJU_FUNCTION inline float StandardMinimumFloat(float first, float second) {
    return (second < first) ? second : first;
}

struct Line {
    Vector3 points[2];
    CUJU_FUNCTION Line() {}
    CUJU_FUNCTION Line(const Vector3 &start, const Vector3 &end) {
        points[0] = start, points[1] = end;
    }
    CUJU_FUNCTION float GetLength() const { return (points[0] - points[1]).GetLength(); }
    CUJU_FUNCTION float GetClosestToPoint(const Vector3 &point) const {
        if (points[0] == points[1]) return 0.0f;
        float lineDistance = (points[1] - points[0]).GetLength();
        if (lineDistance < 0.000001f) return 0.0f;
        return ((point.coordinates[0] - points[0].coordinates[0]) *
                    (points[1].coordinates[0] - points[0].coordinates[0]) +
                (point.coordinates[1] - points[0].coordinates[1]) *
                    (points[1].coordinates[1] - points[0].coordinates[1])) /
               (lineDistance * lineDistance);
    }
    CUJU_FUNCTION float GetDistanceToPoint(const Vector3 &point, float &fraction) const {
        fraction = GetClosestToPoint(point);
        Vector3 intersect;
        intersect.coordinates[0] = points[0].coordinates[0] +
                                   fraction * (points[1].coordinates[0] - points[0].coordinates[0]);
        intersect.coordinates[1] = points[0].coordinates[1] +
                                   fraction * (points[1].coordinates[1] - points[0].coordinates[1]);
        return (intersect - point).GetLength();
    }
    CUJU_FUNCTION Vector3 GetIntersectionPoint(const Line &line, float &fraction) const {
        const Vector3 &lineStart = line.points[0], &lineEnd = line.points[1];
        float divisor = (lineEnd.coordinates[1] - lineStart.coordinates[1]) *
                            (points[1].coordinates[0] - points[0].coordinates[0]) -
                        (lineEnd.coordinates[0] - lineStart.coordinates[0]) *
                            (points[1].coordinates[1] - points[0].coordinates[1]);
        if (divisor == 0.0f) return points[0];
        fraction = ((lineEnd.coordinates[0] - lineStart.coordinates[0]) *
                        (points[0].coordinates[1] - lineStart.coordinates[1]) -
                    (lineEnd.coordinates[1] - lineStart.coordinates[1]) *
                        (points[0].coordinates[0] - lineStart.coordinates[0])) /
                   (divisor);
        return points[0] + (points[1] - points[0]) * fraction;
    }
    CUJU_FUNCTION Vector3 GetIntersectionPoint(const Line &line) const {
        float fraction = 0.0f;
        return GetIntersectionPoint(line, fraction);
    }
};

CUJU_FUNCTION inline Vector3 GetClamped2D(const Vector3 &self, const Vector3 &first,
                                          const Vector3 &second) {
    Vector3 result = self;
    float firstToSecond = second.GetAngle2D(first);
    int direction = SignSide(firstToSecond);
    float firstToSelf = self.GetAngle2D(first);
    float secondToSelf = self.GetAngle2D(second);
    if (SignSide(firstToSelf) != direction || SignSide(secondToSelf) == direction) {
        if (fabsf(firstToSelf) < fabsf(secondToSelf))
            result = first;
        else
            result = second;
    }
    return result;
}

CUJU_FUNCTION_NO_INLINE inline void QuantizeDirection(Vector3 &inputDirection, float bias) {
    Vector3 normalizedInputDirection = inputDirection.GetNormalized(0);
    const int directions = 8;
    float angle = normalizedInputDirection.GetAngle2D();
    angle /= PI * 2.0f;
    angle = roundf(angle * directions);
    angle /= directions;
    angle *= PI * 2.0f;
    inputDirection =
        (normalizedInputDirection * (1.0f - bias) + (Vector3(1, 0, 0).GetRotated2D(angle) * bias))
            .GetNormalized(normalizedInputDirection) *
        inputDirection.GetLength();
}
constexpr float QUANTIZED_DIRECTION_BIAS = 0.0f;

CUJU_FUNCTION inline bool NeedDefendingMovement(int mySide, const Vector3 &position,
                                                const Vector3 &target) {
    float howDeepIsTarget =
        StandardMaximumFloat((target.coordinates[0] - position.coordinates[0]) * -mySide, 0.0f);
    float howWideIsTarget = fabsf(target.coordinates[1] - position.coordinates[1]);
    howDeepIsTarget -= 0.5f;
    return howWideIsTarget > howDeepIsTarget * 0.8f;
}

// GRF: a player missing from the mental image gets its first entry.
CUJU_FUNCTION_NO_INLINE inline state::PlayerImage MentalPlayerImage(const World &world,
                                                                    int mentalImageIndex,
                                                                    int player) {
    const state::MentalImageState &mentalImage =
        world.serialized.match.mentalImages[mentalImageIndex];
    const int timeStampNegative = (int)(world.serialized.match.actualTimeMilliseconds -
                                        (uint64_t)mentalImage.timeStampMilliseconds);
    for (int imageIndex = 0; imageIndex < mentalImage.players.count; imageIndex++) {
        const state::PlayerImage &image = mentalImage.players[imageIndex];
        if (image.player != player) continue;
        state::PlayerImage newImage = image;
        const state::SpatialState &spatialState = Humanoid(world, image.player).spatialState;
        Vector3 extrapolation = image.movement * (float)timeStampNegative * 0.001f;
        newImage.position = image.position + extrapolation;
        newImage.position = newImage.position.EnforceMaximumDeviation(
            spatialState.position, mentalImage.maximumDistanceDeviation);
        newImage.movement = newImage.movement.EnforceMaximumDeviation(
            spatialState.movement, mentalImage.maximumMovementDeviation);
        return newImage;
    }
    return mentalImage.players[0];
}

CUJU_FUNCTION_NO_INLINE inline float PlayerAverageVelocity(const World &world, int player,
                                                           float timePeriodSeconds) {
    const auto &history = world.serialized.match.players[player].base.positionHistoryPerSecond;
    unsigned int logSize = history.count;
    if (logSize == 0) return 0;
    Vector3 previousPosition;
    float totalDistance = 0;
    unsigned int count = 0;
    while (count <= (unsigned int)timePeriodSeconds) {
        Vector3 position = history[logSize - 1 - count];
        if (count > 0) totalDistance += (position - previousPosition).GetLength();
        count++;
        if (logSize - count == 0) break;
        previousPosition = position;
    }
    return totalDistance / timePeriodSeconds;
}

CUJU_FUNCTION inline bool AllowLastDitch(const World &world, int player,
                                         bool includingPossessionAmount = true) {
    const state::PlayerState &playerState = world.serialized.match.players[player];
    if (includingPossessionAmount &&
        TeamByID(world, world.playerTeamID[player]).teamPossessionAmount < 1.0f)
        return true;
    return (float)playerState.timeNeededToGetToBallOptimisticMilliseconds * 1.7f + 800 <
           (float)playerState.timeNeededToGetToBallMilliseconds;
}

CUJU_FUNCTION inline float AveragePossessionSide(const World &world,
                                                 unsigned int timeMilliseconds) {
    const auto &history = world.serialized.match.possessionSideHistory;
    float total = 0;
    unsigned int count = 0;
    for (int entry = history.count - 1; entry >= 0 && count <= timeMilliseconds / 10; entry--) {
        total += history[entry];
        count++;
    }
    if (count > 0) total /= (float)count;
    return total;
}

CUJU_HOST_DEVICE float PropertiesGetReal(const state::PropertiesState &properties, int id,
                                         float defaultValue = 0) {
    return ((properties.present >> id) & 1) ? properties.value[id] : defaultValue;
}

// GRF: the same for every team, so the state does not hold these two.
constexpr float DRIBBLE_OFFENSIVENESS = 0.5f;
constexpr float DRIBBLE_CENTER_MAGNET = 0.72f;

CUJU_FUNCTION_NO_INLINE inline float Mixup(float base, int id, int role) {
    using namespace state;
    float value = -100.0f;
    if (role == e_PlayerRole_CentreBack) {
        if (id == OFFENSE_WIDTH) value = 0.2f;
    }
    if (role == e_PlayerRole_LeftBack || role == e_PlayerRole_RightBack) {
        if (id == DEFENSE_OWN_HALF) value = -0.075f;
        if (id == OFFENSE_WIDTH) value = 0.2f;
        if (id == OFFENSE_OWN_HALF) value = -0.1f;
    }
    if (role == e_PlayerRole_LeftMidfielder || role == e_PlayerRole_RightMidfielder) {
        if (id == DEFENSE_OWN_HALF) value = -0.05f;
        if (id == OFFENSE_OWN_HALF) value = -0.1f;
    }
    if (role == e_PlayerRole_AttackingMidfielder) {
        if (id == DEFENSE_DEPTH) value = 0.125f;
    }
    if (role == e_PlayerRole_CentreForward) {
        if (id == DEFENSE_DEPTH) value = 0.125f;
    }
    if (value > -100.0f) return clamp(base + value, 0.0f, 1.0f);
    return base;
}

CUJU_FUNCTION_NO_INLINE inline Vector3 GetAdaptedFormationPosition(
    const World &world, const SimulationContext &context, int player, float backXBound,
    float frontXBound, float lowYBound, float highYBound, float xFocus, float xFocusStrength,
    float yFocus, float yFocusStrength, const Vector3 &microFocus, float microFocusStrength,
    float midfieldFocus, float midfieldFocusStrength, bool useDynamicFormationPosition) {
    int side = TeamDynamicSide(world, world.playerTeamID[player]);
    Vector3 position = useDynamicFormationPosition
                           ? world.serialized.match.players[player].dynamicFormationEntry.position
                           : Formation(world, context, player).position;
    if (midfieldFocusStrength > 0.0f) {
        float midfieldPositionFactor = midfieldFocus * 2.0f - 1.0f;
        float stretchBias = clamp(1.0f - fabsf(position.coordinates[0] * 1.2f), 0.0f, 1.0f);
        stretchBias = Curve(stretchBias, 1.0f);
        stretchBias *= midfieldFocusStrength;
        position.coordinates[0] =
            position.coordinates[0] * (1.0f - stretchBias) + midfieldPositionFactor * stretchBias;
    }
    float xLength = frontXBound - backXBound;
    position.coordinates[0] = backXBound + (position.coordinates[0] * 0.5f + 0.5f) * xLength;
    float yLength = highYBound - lowYBound;
    position.coordinates[1] = lowYBound + (position.coordinates[1] * -side * 0.5f + 0.5f) * yLength;
    Vector3 purePosition = position;
    if (xFocusStrength > 0.0f) {
        float bias =
            1.0f - clamp(fabsf(xFocus - position.coordinates[0]) / fabsf(backXBound - frontXBound),
                         0.0f, 1.0f);
        bias = -libm::cosf(bias * PI) * 0.5f + 0.5f;
        // Exact: the power is in double.
        bias = (float)libm::pow((double)bias, 0.8);
        bias *= xFocusStrength;
        position.coordinates[0] = position.coordinates[0] * (1.0f - bias) + xFocus * bias;
    }
    if (yFocusStrength > 0.0f) {
        float distance = clamp(
            fabsf(yFocus - position.coordinates[1]) / fabsf(highYBound - lowYBound), 0.0f, 1.0f);
        float bias = 1.0f - distance;
        bias *= 0.2f + 0.8f * fabsf(yFocus) / PITCH_HALF_WIDTH;
        bias *= yFocusStrength;
        position.coordinates[1] = position.coordinates[1] * (1.0f - bias) + yFocus * bias;
    }
    if (microFocusStrength > 0.0f) {
        float homogeneousYInfluenceBias = 0.2f;
        float homogeneousYPositionBias = 0.4f;
        float staticFocusDistance =
            ((microFocus - purePosition) * Vector3(1, 1.0f - homogeneousYInfluenceBias, 0))
                .GetLength() /
            50.0f;
        float focusDistance = staticFocusDistance;
        if (focusDistance < 1.0f) {
            float microFocusBias = 1.0f;
            microFocusBias *= 1.0f - focusDistance;
            microFocusBias = Curve(microFocusBias, 0.3f);
            float peakLocation = 0.15f, peakWidth = 0.25f, peakHeight = 0.1f;
            microFocusBias +=
                (1.0f - NormalizedClamp(fabsf(focusDistance - peakLocation), 0.0f, peakWidth)) *
                peakHeight;
            microFocusBias = clamp(microFocusBias, 0.0f, 1.0f);
            microFocusBias *= microFocusStrength;
            Vector3 microFocusPosition =
                microFocus * Vector3(1, 1.0f - homogeneousYPositionBias, 1) +
                position * Vector3(0, homogeneousYPositionBias, 0);
            position = position * (1.0f - microFocusBias) + microFocusPosition * (microFocusBias);
        }
    }
    return position;
}

CUJU_FUNCTION_NO_INLINE inline Vector3 TeamAdaptedFormationPosition(
    const World &world, const SimulationContext &context, int player,
    bool useDynamicFormationPosition) {
    const int teamID = world.playerTeamID[player];
    const int side = TeamDynamicSide(world, teamID);
    const state::TeamAIState &teamAI = TeamByID(world, teamID).teamController;
    const state::PropertiesState &tactics = teamAI.liveTeamTactics;
    const state::MatchState &match = world.serialized.match;
    int role = useDynamicFormationPosition ? match.players[player].dynamicFormationEntry.role
                                           : Formation(world, context, player).role;
    Vector3 focalPoint = Humanoid(world, match.designatedPossessionPlayer).spatialState.position;
    float urgencyBias =
        1.0f -
        NormalizedClamp((focalPoint - Humanoid(world, player).spatialState.position).GetLength(),
                        2.0f, 30.0f);
    float ballX = BallAveragePosition(world, FloatToUnsigned32(3500 * (1.0f - urgencyBias * 0.7f)))
                      .coordinates[0];
    float ballY = BallAveragePosition(world, FloatToUnsigned32(4000 * (1.0f - urgencyBias * 0.5f)))
                      .coordinates[1];

#define CUJU_TACTIC(name) Mixup(PropertiesGetReal(tactics, name), name, role)
    float offenseDepthFactor = CUJU_TACTIC(state::OFFENSE_DEPTH);
    float defenseDepthFactor = CUJU_TACTIC(state::DEFENSE_DEPTH);
    float offenseWidthFactor = CUJU_TACTIC(state::OFFENSE_WIDTH);
    float defenseWidthFactor = CUJU_TACTIC(state::DEFENSE_WIDTH);
    float offenseOwnHalfFactor = CUJU_TACTIC(state::OFFENSE_OWN_HALF);
    float defenseOwnHalfFactor = CUJU_TACTIC(state::DEFENSE_OWN_HALF);
    float offenseMidfieldFocus = CUJU_TACTIC(state::OFFENSE_MIDFIELD_FOCUS);
    float defenseMidfieldFocus = CUJU_TACTIC(state::DEFENSE_MIDFIELD_FOCUS);
    float offenseMidfieldFocusStrength = CUJU_TACTIC(state::OFFENSE_MIDFIELD_FOCUS_STRENGTH);
    float defenseMidfieldFocusStrength = CUJU_TACTIC(state::DEFENSE_MIDFIELD_FOCUS_STRENGTH);
    float offenseSideFocusStrength = CUJU_TACTIC(state::OFFENSE_SIDE_FOCUS_STRENGTH);
    float defenseSideFocusStrength = CUJU_TACTIC(state::DEFENSE_SIDE_FOCUS_STRENGTH);
    float offenseMicroFocusStrength = CUJU_TACTIC(state::OFFENSE_MICRO_FOCUS_STRENGTH);
    float defenseMicroFocusStrength = CUJU_TACTIC(state::DEFENSE_MICRO_FOCUS_STRENGTH);
#undef CUJU_TACTIC

    const double mind = GetMindSet(role);
    offenseSideFocusStrength =
        (float)((double)offenseSideFocusStrength + (-0.5 + mind) * (double)0.2f);
    defenseSideFocusStrength =
        (float)((double)defenseSideFocusStrength + (0.5 - mind) * (double)0.2f);
    const float offensivenessBias = teamAI.offensivenessBias;
    offenseSideFocusStrength =
        clamp(offenseSideFocusStrength + -0.3f + (offensivenessBias) * 0.3f, 0.0f, 1.0f);
    defenseSideFocusStrength =
        clamp(defenseSideFocusStrength + -0.3f + (1.0f - offensivenessBias) * 0.3f, 0.0f, 1.0f);

    float possessionAmountBias =
        NormalizedClamp(teamAI.fadingTeamPossessionAmount - 0.5f, 0.3f, 0.7f);
    float ballBias = NormalizedClamp(((ballX / PITCH_HALF_LENGTH) * -side), -0.7f, 0.7f);
    float ballBiasBias = 1.0f - fabsf(possessionAmountBias * 2.0f - 1.0f);
    ballBiasBias *= 0.6f;
    float possessionBias = possessionAmountBias * (1.0f - ballBiasBias) + ballBias * ballBiasBias;
    possessionBias = clamp(possessionBias + (offensivenessBias - 0.5f) * 0.3f, 0.0f, 1.0f);

    focalPoint =
        (BallAveragePosition(world, 3000).Get2D() * 1.0f + focalPoint * 0.0f) * possessionBias +
        (BallAveragePosition(world, 2000).Get2D() * 0.5f + focalPoint * 0.5f) *
            (1.0f - possessionBias);

    float adaptedDepth = teamAI.depth * (offenseDepthFactor * possessionBias +
                                         defenseDepthFactor * (1.0f - possessionBias));
    float adaptedWidth = teamAI.width * (offenseWidthFactor * possessionBias +
                                         defenseWidthFactor * (1.0f - possessionBias));
    float offsetX = PITCH_HALF_LENGTH * side *
                    ((offenseOwnHalfFactor * 2.0f - 1.0f) * possessionBias +
                     (defenseOwnHalfFactor * 2.0f - 1.0f) * (1.0f - possessionBias));
    float sideFocusStrength = (offenseSideFocusStrength * possessionBias +
                               defenseSideFocusStrength * (1.0f - possessionBias));
    float sideFocus = possessionBias * 2.0f - 1.0f;
    float sideX = 0.2f * sideFocus * -side * PITCH_HALF_LENGTH +
                  0.8f * -AveragePossessionSide(world, 6000) * PITCH_HALF_LENGTH;
    float centerX =
        clamp((ballX * (1.0f - sideFocusStrength) + sideX * sideFocusStrength) + offsetX,
              -PITCH_HALF_LENGTH, PITCH_HALF_LENGTH);
    float centerY = clamp(ballY, -PITCH_HALF_WIDTH, PITCH_HALF_WIDTH);
    float adaptCenterToFitDepthBias = 0.95f;
    float adaptCenterToFitWidthBias = 0.9f;
    centerX *= ((1.0f - adaptedDepth) / 1.0f) * adaptCenterToFitDepthBias +
               (1.0f - adaptCenterToFitDepthBias);
    centerY *= ((1.0f - adaptedWidth) / 1.0f) * adaptCenterToFitWidthBias +
               (1.0f - adaptCenterToFitWidthBias);
    float backXBound = centerX - adaptedDepth * PITCH_HALF_LENGTH * -side;
    float frontXBound = centerX + adaptedDepth * PITCH_HALF_LENGTH * -side;
    float lowYBound = centerY - adaptedWidth * PITCH_HALF_WIDTH;
    float highYBound = centerY + adaptedWidth * PITCH_HALF_WIDTH;
    if (backXBound * side > teamAI.offsideTrapX * side)
        backXBound = clamp(teamAI.offsideTrapX, -PITCH_HALF_LENGTH, PITCH_HALF_LENGTH);
    float yFocus = ballY * 1.0f;
    float yFocusStrength = 0.5f * possessionBias + 0.2f * (1.0f - possessionBias);
    Vector3 microFocus = focalPoint;
    Vector3 defensiveFocusPosition = Vector3(
        clamp(microFocus.coordinates[0] + side * 2.0f, -PITCH_HALF_LENGTH, backXBound * side),
        microFocus.coordinates[1] * 0.9f, 0);
    microFocus = (Vector3(clamp(microFocus.coordinates[0] - side * 1.0f, -PITCH_HALF_LENGTH,
                                PITCH_HALF_LENGTH),
                          microFocus.coordinates[1] * 0.9f, 0)) *
                     possessionBias +
                 (defensiveFocusPosition) * (1.0f - possessionBias);
    float microFocusStrength = (offenseMicroFocusStrength * possessionBias +
                                defenseMicroFocusStrength * (1.0f - possessionBias));
    float microFocusSideBias = NormalizedClamp((ballX / PITCH_HALF_LENGTH) * -side, -0.7f, 0.7f);
    microFocusSideBias = microFocusSideBias * 0.7f + 0.3f;
    float autoMicroFocusStrength =
        (libm::powf(microFocusSideBias, 0.8f) * possessionBias +
         (libm::powf(1.0f - microFocusSideBias, 0.6f)) * (1.0f - possessionBias));
    microFocusStrength = microFocusStrength * (0.2f + 0.8f * autoMicroFocusStrength);
    float manualMidfieldFocus =
        (offenseMidfieldFocus * possessionBias + defenseMidfieldFocus * (1.0f - possessionBias));
    float autoMidfieldFocus = NormalizedClamp((ballX / PITCH_HALF_LENGTH) * -side, -0.8f, 0.8f);
    float midfieldFocus = manualMidfieldFocus * 0.7f + autoMidfieldFocus * 0.3f;
    float midfieldFocusStrength = (offenseMidfieldFocusStrength * possessionBias +
                                   defenseMidfieldFocusStrength * (1.0f - possessionBias));

    Vector3 desiredPosition = GetAdaptedFormationPosition(
        world, context, player, backXBound, frontXBound, lowYBound, highYBound, 0.0f, 0.0f, yFocus,
        yFocusStrength, microFocus, microFocusStrength, midfieldFocus, midfieldFocusStrength,
        useDynamicFormationPosition);
    desiredPosition.coordinates[0] =
        clamp(desiredPosition.coordinates[0], -PITCH_HALF_LENGTH, PITCH_HALF_LENGTH);
    desiredPosition.coordinates[1] =
        clamp(desiredPosition.coordinates[1], -PITCH_HALF_WIDTH, PITCH_HALF_WIDTH);
    return desiredPosition;
}

CUJU_FUNCTION_NO_INLINE inline void TeamApplyOffsideTrap(const World &world, int teamID,
                                                         Vector3 &position) {
    const int side = TeamDynamicSide(world, teamID);
    const float offsideTrapX = TeamByID(world, teamID).teamController.offsideTrapX;
    float areaHalfLength = 2.0f;
    float absolutePositionX = position.coordinates[0] * side;
    float absoluteOffsideTrapX = offsideTrapX * side;
    if (absolutePositionX > absoluteOffsideTrapX - areaHalfLength) {
        float areaFront = absoluteOffsideTrapX - areaHalfLength;
        float positionFromAreaFront = absolutePositionX - areaFront;
        float positionFactor = positionFromAreaFront / (areaHalfLength * 2.0f);
        positionFactor = clamp(positionFactor, 0.0f, 1.0f);
        float absoluteResultX = areaFront + areaHalfLength * positionFactor;
        position.coordinates[0] = absoluteResultX * side;
    }
}

struct AIController {
    World &world;
    const SimulationContext &context;
    const int player;
    state::MatchState &match;
    state::PlayerState &playerState;
    // Null when an agent controls the player.
    state::AIControllerState *elizaState;
    state::PlayerControllerState &controller;
    // `side` is the sign of the x coordinate of the own goal.
    const int teamID, opponentID, side;
    int mentalImageIndex = 0;
    // In the order that the animation selection tries them.
    CommandQueue &queue;

    CUJU_FUNCTION AIController(World &world, const SimulationContext &context, int player,
                               CommandQueue &queue, state::PlayerControllerState &controller,
                               state::AIControllerState *elizaState)
        : world(world),
          context(context),
          player(player),
          match(world.serialized.match),
          playerState(world.serialized.match.players[player]),
          elizaState(elizaState),
          controller(controller),
          teamID(world.playerTeamID[player]),
          opponentID(1 - world.playerTeamID[player]),
          side(TeamDynamicSide(world, world.playerTeamID[player])),
          queue(queue) {}

    CUJU_FUNCTION const state::SpatialState &SpatialStateOf(int other) const {
        return Humanoid(world, other).spatialState;
    }
    CUJU_FUNCTION Vector3 Position(int other) const { return SpatialStateOf(other).position; }
    CUJU_FUNCTION Vector3 Movement(int other) const { return SpatialStateOf(other).movement; }
    CUJU_FUNCTION Vector3 Direction(int other) const {
        return SpatialStateOf(other).directionVector;
    }
    CUJU_FUNCTION Vector3 Position() const { return Position(player); }
    CUJU_FUNCTION Vector3 Movement() const { return Movement(player); }
    CUJU_FUNCTION Vector3 Direction() const { return Direction(player); }
    CUJU_FUNCTION float FloatVelocity() const { return PlayerFloatVelocity(world, player); }
    CUJU_FUNCTION float Statistic(int statistic) const {
        return GetStatistic(world, context, player, statistic);
    }
    CUJU_FUNCTION int StaticRole(int other) const { return Formation(world, context, other).role; }
    CUJU_FUNCTION int DynamicRole() const { return playerState.dynamicFormationEntry.role; }
    CUJU_FUNCTION const state::TeamState &Team() const { return TeamByID(world, teamID); }
    CUJU_FUNCTION const state::TeamState &OpponentTeam() const {
        return TeamByID(world, opponentID);
    }
    CUJU_FUNCTION const state::TeamAIState &TeamAI() const { return Team().teamController; }
    CUJU_FUNCTION Vector3 MentalBallPredictionAt(int timeMilliseconds) const {
        return MentalBallPrediction(world, mentalImageIndex, timeMilliseconds);
    }
    CUJU_FUNCTION Vector3 Ball(int timeMilliseconds) const {
        return BallPredict(match.ball, timeMilliseconds);
    }
    CUJU_FUNCTION float Random(float low, float high) {
        return RandomInRange(world.serialized.rng, low, high);
    }
    CUJU_FUNCTION static state::PlayerCommand DefaultCommand() { return DefaultPlayerCommand(); }

    CUJU_FUNCTION float CouldWinABallDuelLikeliness() const {
        int opponent = OpponentTeam().designatedTeamPossessionPlayer;
        Vector3 opponentPosition = Position(opponent) + Movement(opponent) * 0.1f;
        Vector3 meToOpponent = opponentPosition - (Position() + Movement() * 0.1f);
        Vector3 opponentToBall =
            MentalBallPredictionAt(100).Get2D() - (Position(opponent) + Movement(opponent) * 0.1f);
        float dot = meToOpponent.GetNormalized(0).GetDotProduct(opponentToBall.GetNormalized(0));
        dot = dot * 0.5f + 0.5f;
        return (1.0f - dot);
    }

    CUJU_FUNCTION_NO_INLINE void AddDefensiveComponent(Vector3 &desiredPosition, float bias,
                                                       int forcedOpponent = -1) const {
        int opponent = forcedOpponent < 0 ? playerState.manMarking : forcedOpponent;
        if (!(match.inPlay && !match.inSetPiece && StaticRole(player) != e_PlayerRole_Goalkeeper))
            return;
        if (opponent < 0) return;
        Vector3 defendPosition = desiredPosition;
        float possessionPlayerShootThreshold = 24.0f, genericOpponentShootThreshold = 8.0f;
        float minDistance = 0.4f, bufferDistance = 4.0f;
        state::PlayerImage opponentImage = MentalPlayerImage(
            world, MentalImageIndex(world, PlayerControllerReactionTime(world, context, player)),
            opponent);
        Vector3 opponentPosition = opponentImage.position + opponentImage.movement * 0.5f;
        float shootThreshold = genericOpponentShootThreshold;
        if (opponent == match.designatedPossessionPlayer)
            shootThreshold = possessionPlayerShootThreshold;
        Vector3 goalPosition = Vector3(PITCH_HALF_LENGTH * side, 0, 0);
        float opponentToGoalDistance = (goalPosition - opponentPosition).GetLength();
        float opponentToThresholdDistance =
            clamp(opponentToGoalDistance - shootThreshold, minDistance, PITCH_HALF_LENGTH);
        Vector3 shootingPoint =
            opponentPosition +
            (goalPosition - opponentPosition).GetNormalized(0) * opponentToThresholdDistance;
        const float trapX = TeamAI().offsideTrapX;
        if (shootingPoint.coordinates[0] * side > trapX * side) {
            Line opponentToGoalLine(opponentPosition, goalPosition);
            Line offsideLine(Vector3(trapX, -PITCH_HALF_WIDTH, 0),
                             Vector3(trapX, PITCH_HALF_WIDTH, 0));
            shootingPoint = opponentToGoalLine.GetIntersectionPoint(offsideLine);
            opponentToThresholdDistance = (shootingPoint - opponentPosition).GetLength();
        }
        Vector3 meToThreshold = shootingPoint - desiredPosition;
        float meToThresholdDistance = meToThreshold.GetLength();
        float slackedDistance =
            meToThresholdDistance - (opponentToThresholdDistance - bufferDistance);
        if (slackedDistance > 0.0f)
            defendPosition =
                desiredPosition + meToThreshold.GetNormalized(0) *
                                      clamp(slackedDistance, 0.0f, meToThresholdDistance);
        Vector3 actualPosition = Position() + Movement() * 0.14f;
        float actualToThresholdDistance = (shootingPoint - actualPosition).GetLength();
        float actualSlackedDistance = actualToThresholdDistance - opponentToThresholdDistance;
        if (actualSlackedDistance > 0.0f)
            defendPosition +=
                (goalPosition - defendPosition).GetNormalized(0) * actualSlackedDistance * 0.7f;
        desiredPosition = desiredPosition * (1.0f - bias) + defendPosition * bias;
    }

    CUJU_FUNCTION_NO_INLINE Vector3 GetDefendPosition(int opponent, float distance = 0.0f) const {
        state::PlayerImage opponentImage = MentalPlayerImage(world, mentalImageIndex, opponent);
        Vector3 goalPosition = Vector3(PITCH_HALF_LENGTH * side, 0, 0);
        Vector3 opponentPosition = opponentImage.position;
        Vector3 opponentToGoal = goalPosition - opponentPosition;
        Line opponentToGoalLine(opponentPosition, goalPosition);
        Line meToOpponentLine(Position(), opponentPosition);
        Vector3 midpoint = meToOpponentLine.points[0] * 0.5f + meToOpponentLine.points[1] * 0.5f;
        Line middleToOpponentPerpendicularLine(
            midpoint,
            midpoint +
                (meToOpponentLine.points[0] - midpoint).GetRotated2D((float)(0.5 * (double)PI)));
        float fraction = 0.0f;
        opponentToGoalLine.GetIntersectionPoint(middleToOpponentPerpendicularLine, fraction);
        fraction = clamp(fraction, 0.0f, 1.0f);
        Vector3 target = opponentToGoalLine.points[0] +
                         (opponentToGoalLine.points[1] - opponentToGoalLine.points[0]) * fraction;
        target += (opponentToGoal.GetNormalized(0) * SPRINT_VELOCITY * 0.1f) +
                  (opponentImage.movement * 0.14f) + (opponentToGoal.GetNormalized(0) * distance);
        return target;
    }

    CUJU_FUNCTION_NO_INLINE void KeeperDeflectCommand(bool onlyPickupAnimations) {
        if (StaticRole(player) != e_PlayerRole_Goalkeeper) return;
        if (GetDistance(Ball(400), Position()) > BALL_DISTANCE_OPTIMIZE_THRESHOLD + 10.0f) return;
        if (match.lastTouchTeamID == teamID && MatchLastTouchPlayer(world) != player &&
            match.lastTouchTeamIDs[e_TouchType_Intentional_Kicked] == teamID)
            return;
        if (match.ballRetainer >= 0) return;
        if (fabsf(Ball(160).coordinates[1]) > 20.05f) return;
        if ((double)(Ball(160).coordinates[0] * -side) > -(double)PITCH_HALF_LENGTH + 16.4) return;
        state::PlayerCommand command = DefaultCommand();
        command.desiredFunctionType = e_FunctionType_Deflect;
        command.useDesiredMovement = false;
        command.useDesiredLookAt = false;
        if (onlyPickupAnimations) command.onlyDeflectAnimationsThatPickupBall = true;
        queue.push_back(command);
    }

    CUJU_FUNCTION_NO_INLINE void SetPieceCommand() {
        state::PlayerCommand command = DefaultCommand();
        command.desiredFunctionType = e_FunctionType_Movement;
        command.useDesiredMovement = true;
        command.useDesiredLookAt = true;
        command.desiredDirection = Direction();
        command.desiredVelocityFloat = IDLE_VELOCITY;
        if (match.ballRetainer == player)
            command.desiredLookAt = Position() + controller.inputDirection * 10.0f;
        else
            command.desiredLookAt = Ball(0).Get2D();
        queue.push_back(command);
    }

    CUJU_FUNCTION_NO_INLINE void BallControlCommand(bool idleTurnToOpponentGoal = false,
                                                    bool knockOn = false,
                                                    bool stickyRunDirection = false,
                                                    bool keepCurrentBodyDirection = false) {
        if (GetDistance(Ball(200), Position()) > BALL_DISTANCE_OPTIMIZE_THRESHOLD) return;
        if (match.ballRetainer >= 0) return;
        if (!playerState.hasPossession && !AllowLastDitch(world, player)) {
            float strictnessInverse = 3.0f;
            if (fabsf(match.ball.momentum.coordinates[2]) > 5.0f * strictnessInverse) return;
            if ((match.ball.momentum.Get2D() - Movement()).GetLength() > 5.0f * strictnessInverse)
                return;
        }
        if (match.designatedPossessionPlayer == player ||
            (Team().designatedTeamPossessionPlayer == player &&
             CouldWinABallDuelLikeliness() >= 0.25f)) {
            state::PlayerCommand command = DefaultCommand();
            command.desiredFunctionType = e_FunctionType_BallControl;
            command.useDesiredMovement = true;
            command.desiredDirection = controller.inputDirection;
            QuantizeDirection(command.desiredDirection, QUANTIZED_DIRECTION_BIAS);
            command.desiredVelocityFloat = controller.inputVelocityFloat;
            if (FloatToVelocityCategory(command.desiredVelocityFloat) == e_Velocity_Idle &&
                idleTurnToOpponentGoal)
                command.desiredDirection = Vector3(-side, 0, 0);
            if (knockOn) command.modifier |= e_PlayerCommandModifier_KnockOn;
            if (controller.hasPossession && controller.hasBestPossession &&
                command.desiredVelocityFloat > WALK_SPRINT_SWITCH &&
                FloatVelocity() > DRIBBLE_WALK_SWITCH && stickyRunDirection) {
                float angle = command.desiredDirection.GetAngle2D(Direction());
                if (fabsf(angle) > 0.125f * PI && fabsf(angle) < 0.7f * PI) {
                    if (angle > 0) command.desiredDirection = Direction().GetRotated2D(0.125f * PI);
                    if (angle < 0)
                        command.desiredDirection = Direction().GetRotated2D(0.125f * -PI);
                }
            }
            command.useDesiredLookAt = true;
            const int velocityCategory = SpatialStateOf(player).velocityCategory;
            if (keepCurrentBodyDirection &&
                (velocityCategory == e_Velocity_Walk || velocityCategory == e_Velocity_Dribble) &&
                FloatToVelocityCategory(command.desiredVelocityFloat) == e_Velocity_Dribble) {
                command.desiredVelocityFloat = WALK_VELOCITY;
                command.desiredLookAt = Position() + Direction() * 10.0f;
            } else {
                command.desiredLookAt =
                    Position() + Movement() * 0.2f + command.desiredDirection * 10.0f;
            }
            queue.push_back(command);
        }
    }

    CUJU_FUNCTION_NO_INLINE void TrapCommand(bool idleTurnToOpponentGoal = false,
                                             bool knockOn = false) {
        if (match.ball.momentum.Get2D().GetLength() < 2.0f) return;
        if (GetDistance(Ball(200), Position()) > BALL_DISTANCE_OPTIMIZE_THRESHOLD) return;
        if (match.ballRetainer >= 0) return;
        if (!controller.hasPossession &&
            (match.designatedPossessionPlayer == player ||
             (Team().designatedTeamPossessionPlayer == player &&
              playerState.timeNeededToGetToBallOptimisticMilliseconds < 1000 &&
              controller.opponentTimeNeededToGetToBall > 400 &&
              !controller.opponentTeamHasPossession && CouldWinABallDuelLikeliness() >= 0.5f))) {
            state::PlayerCommand command = DefaultCommand();
            command.desiredFunctionType = e_FunctionType_Trap;
            command.useDesiredMovement = true;
            command.desiredDirection = controller.inputDirection;
            QuantizeDirection(command.desiredDirection, QUANTIZED_DIRECTION_BIAS);
            command.desiredVelocityFloat = controller.inputVelocityFloat;
            if (StaticRole(player) == e_PlayerRole_Goalkeeper &&
                !ExternalControllerActive(world, player)) {
                command.desiredDirection = Vector3(-side, 0, 0);
                command.desiredVelocityFloat = IDLE_VELOCITY;
            }
            if (FloatToVelocityCategory(command.desiredVelocityFloat) == e_Velocity_Idle &&
                idleTurnToOpponentGoal)
                command.desiredDirection = Vector3(-side, 0, 0);
            if (knockOn) command.modifier |= e_PlayerCommandModifier_KnockOn;
            command.useDesiredLookAt = true;
            command.desiredLookAt =
                Position() + Movement() * 0.3f + command.desiredDirection * 10.0f;
            queue.push_back(command);
        }
    }

    CUJU_FUNCTION_NO_INLINE void InterfereCommand(bool byAnyMeans) {
        if (GetDistance(Ball(200), Position()) > BALL_DISTANCE_OPTIMIZE_THRESHOLD) return;
        if (match.ballRetainer >= 0) return;
        if (controller.teamHasBestPossession) return;
        if (!byAnyMeans && CouldWinABallDuelLikeliness() < 0.2f) return;
        state::PlayerCommand command = DefaultCommand();
        command.desiredFunctionType = e_FunctionType_Interfere;
        command.useDesiredMovement = true;
        command.strictMovement = byAnyMeans ? e_StrictMovement_False : e_StrictMovement_True;
        command.desiredDirection = controller.inputDirection;
        command.desiredVelocityFloat = controller.inputVelocityFloat;
        queue.push_back(command);
    }

    CUJU_FUNCTION_NO_INLINE void SlidingCommand() {
        if (HumanGamerCount(Team()) != 0) return;
        if (match.ballRetainer >= 0) return;
        if (CouldWinABallDuelLikeliness() < 0.7f) return;
        if (!(!controller.teamHasBestPossession && controller.possessionAmount < 0.6f &&
              match.designatedPossessionPlayer != player && controller.opponentTeamHasPossession))
            return;
        const int opponent = controller.opponentPlayer;
        Vector3 ballPosition = MentalBallPrediction(world, MentalImageIndex(world, 20), 200);
        Vector3 playerPosition = Position() + Movement() * (float)0.2;
        float ballDistance = (playerPosition - ballPosition).GetLength();
        const state::AnimationState &opponentAnimation = Humanoid(world, opponent).currentAnimation;
        if ((ballDistance > 0.7f && ballDistance < 1.6f &&
             controller.opponentTimeNeededToGetToBall > 260) ||
            (ballDistance > 0.6f && ballDistance < 1.8f &&
             opponentAnimation.functionType == e_FunctionType_Shot &&
             opponentAnimation.frameNumber < opponentAnimation.touchFrame)) {
            state::PlayerCommand command = DefaultCommand();
            command.desiredFunctionType = e_FunctionType_Sliding;
            command.useDesiredMovement = true;
            command.desiredDirection =
                ((ballPosition.Get2D() + Movement(opponent) * 0.2f) - playerPosition)
                    .GetNormalized(Vector3(-side, 0, 0));
            command.desiredVelocityFloat = SPRINT_VELOCITY;
            command.useDesiredLookAt = true;
            command.desiredLookAt =
                Position() + Movement() * 0.1f + command.desiredDirection * 10.0f;
            queue.push_back(command);
        }
    }

    CUJU_FUNCTION_NO_INLINE unsigned int BallControlMovement(float desiredVelocityFloat,
                                                             Vector3 &bestDirection,
                                                             float &bestVelocityFloat,
                                                             Vector3 &bestLookAt) const {
        unsigned int desiredTimeToBallMilliseconds = 250 + DEFAULT_TOUCH_OFFSET_MILLISECONDS;
        Vector3 toBallMovement =
            MentalBallPredictionAt((int)desiredTimeToBallMilliseconds).Get2D() - Position();
        float toBallDistance = toBallMovement.GetLength();
        float autoDirectionBias = 1.0f;
        if (toBallDistance < 0.4f)
            autoDirectionBias = libm::powf(NormalizedClamp(toBallDistance, 0.2f, 0.4f), 0.5f);
        Vector3 autoDirection = toBallMovement.GetNormalized(Direction());
        Vector3 manualDirection = Direction();
        bestDirection =
            autoDirection * autoDirectionBias + manualDirection * (1.0f - autoDirectionBias);
        bestDirection.Normalize(Direction());
        Vector3 bestLookDirection = bestDirection;
        bestVelocityFloat = toBallDistance * DISTANCE_TO_VELOCITY_MULTIPLIER;
        // GRF: a velocity not below the dribble velocity, NaN included, takes the clamp.
        if (!(bestVelocityFloat < DRIBBLE_VELOCITY)) {
            bestVelocityFloat =
                clamp(desiredVelocityFloat, bestVelocityFloat, bestVelocityFloat + 8.0f);
            if (RangeVelocity(bestVelocityFloat) < bestVelocityFloat)
                bestVelocityFloat =
                    bestVelocityFloat * 0.9f + RangeVelocity(bestVelocityFloat) * 0.1f;
            if (RangeVelocity(bestVelocityFloat) > bestVelocityFloat)
                bestVelocityFloat =
                    bestVelocityFloat * 0.1f + RangeVelocity(bestVelocityFloat) * 0.9f;
        }
        bestLookAt = Position() + bestLookDirection * 10.0f;
        return playerState.timeNeededToGetToBallMilliseconds;
    }

    CUJU_FUNCTION_NO_INLINE unsigned int ToBallMovement(const Vector3 &desiredDirection,
                                                        float desiredVelocityFloat,
                                                        Vector3 &bestDirection,
                                                        float &bestVelocityFloat,
                                                        Vector3 &bestLookAt, float haste) const {
        const Vector3 playerPosition = Position();
        const float maximumVelocity = GetMaximumVelocity(*context.staticData, player);
        float movementWeight = 1.0f, timeWeight = 0.0f, perpendicularWeight = 0.1f,
              previousTargetWeight = 0.0f;
        Vector3 adaptedDesiredDirection = desiredDirection;

        Line ballLine(MentalBallPredictionAt(0).Get2D(), MentalBallPredictionAt(1000).Get2D());
        float fraction = 0;
        ballLine.GetDistanceToPoint(playerPosition, fraction);
        Vector3 playerBallShortestTargetPosition =
            ballLine.points[0] + (ballLine.points[1] - ballLine.points[0]) * fraction;

        Vector3 ballSpaceDirection =
            (MentalBallPredictionAt(0).Get2D() - MentalBallPredictionAt(1000).Get2D())
                .GetNormalized(adaptedDesiredDirection);
        float toBallSpaceAngle = ballSpaceDirection.GetAngle2D();
        Vector3 adaptedDesiredDirectionBallSpace =
            adaptedDesiredDirection.GetRotated2D(-toBallSpaceAngle);
        const int directions = 16;
        float angle = adaptedDesiredDirectionBallSpace.GetAngle2D();
        angle /= PI * 2.0f;
        angle = roundf(angle * directions);
        angle /= directions;
        angle *= PI * 2.0f;
        float bias = 1.0f;
        adaptedDesiredDirectionBallSpace = (adaptedDesiredDirectionBallSpace * (1.0f - bias) +
                                            (Vector3(1, 0, 0).GetRotated2D(angle) * bias))
                                               .GetNormalized(adaptedDesiredDirectionBallSpace);
        adaptedDesiredDirection = adaptedDesiredDirectionBallSpace.GetRotated2D(toBallSpaceAngle);
        Vector3 desiredMovement = adaptedDesiredDirection * desiredVelocityFloat;

        if (haste > 0.0f) {
            movementWeight = 0.0f;
            timeWeight = 1.0f;
            perpendicularWeight = 0.0f;
        }

        struct PossibleChoice {
            float rating = -10000.0f;
            int timeNeededMilliseconds = -10;
            int timeMilliseconds = -10;
        } bestChoice;

        const uint32_t timeToBallMilliseconds = playerState.timeNeededToGetToBallMilliseconds;
        unsigned int startTimeMilliseconds = FloatToUnsigned32(clamp(
            (float)timeToBallMilliseconds, 40, (float)(BALL_PREDICTION_SPAN_MILLISECONDS - 10)));
        Vector3 ballPrediction = MentalBallPredictionAt((int)startTimeMilliseconds);
        if (fabsf(ballPrediction.coordinates[0]) > PITCH_HALF_LENGTH - 0.2f ||
            fabsf(ballPrediction.coordinates[1]) > PITCH_HALF_WIDTH - 0.2f)
            startTimeMilliseconds = 80;
        int previousDesiredTargetTimeMilliseconds = playerState.desiredTimeToBallMilliseconds;
        int timeNeededToGetToBallMilliseconds = (int)timeToBallMilliseconds;
        Vector3 ballMovementRough = (MentalBallPredictionAt((int)(timeToBallMilliseconds + 10)) -
                                     MentalBallPredictionAt((int)timeToBallMilliseconds))
                                        .Get2D() *
                                    100.0f;
        Vector3 ballDirectionRough = ballMovementRough.GetNormalized(adaptedDesiredDirection);

        unsigned int timeStep = 1;
        // Speed: the time needed at a grid time depends on that time alone, so a miss computes the
        // next 32 grid times across the lanes.
        uint32_t usualTimes[32], optimisticTimes[32];
        unsigned int firstCachedTimeMilliseconds = 0, cachedCount = 0;
#if !defined(__CUDA_ARCH__)
        // Speed: the targets share the walk of the player when it moves, as in `PossessionScan`.
        TimeNeededWalk walk(playerPosition, Movement(), maximumVelocity);
#endif
        auto TimeNeededAt = [&](unsigned int queryTimeMilliseconds) {
            if (queryTimeMilliseconds < firstCachedTimeMilliseconds ||
                queryTimeMilliseconds >= firstCachedTimeMilliseconds + 10 * cachedCount) {
                firstCachedTimeMilliseconds = queryTimeMilliseconds;
                cachedCount = WARP_LANES;
                uint64_t packedTimes[32];
                WarpMap<uint64_t>((int)cachedCount,
                                  [&](int lane) {
                                      const unsigned int laneTimeMilliseconds =
                                          queryTimeMilliseconds + 10 * (unsigned int)lane;
                                      if (laneTimeMilliseconds >= BALL_PREDICTION_SPAN_MILLISECONDS)
                                          return (uint64_t)0;
                                      const Vector3 target =
                                          MentalBallPredictionAt((int)laneTimeMilliseconds).Get2D();
#if !defined(__CUDA_ARCH__)
                                      TimeNeeded result =
                                          walk.shared
                                              ? walk.Time(playerPosition, target, maximumVelocity,
                                                          true, laneTimeMilliseconds)
                                              : GetTimeNeededForDistanceMilliseconds(
                                                    playerPosition, Movement(), target,
                                                    maximumVelocity, true, laneTimeMilliseconds);
#else
                                      TimeNeeded result = GetTimeNeededForDistanceMilliseconds(
                                          playerPosition, Movement(), target, maximumVelocity, true,
                                          laneTimeMilliseconds);
#endif
                                      return ((uint64_t)result.usualMilliseconds << 32) |
                                             result.optimisticMilliseconds;
                                  },
                                  packedTimes);
                for (int lane = 0; lane < (int)cachedCount; lane++) {
                    usualTimes[lane] = (uint32_t)(packedTimes[lane] >> 32);
                    optimisticTimes[lane] = (uint32_t)packedTimes[lane];
                }
            }
            TimeNeeded result;
            result.usualMilliseconds =
                usualTimes[(queryTimeMilliseconds - firstCachedTimeMilliseconds) / 10];
            result.optimisticMilliseconds =
                optimisticTimes[(queryTimeMilliseconds - firstCachedTimeMilliseconds) / 10];
            return result;
        };
        for (unsigned int timeMilliseconds = startTimeMilliseconds;
             timeMilliseconds < BALL_PREDICTION_SPAN_MILLISECONDS;
             timeMilliseconds += 10 * timeStep) {
            bool forced = false;
            Vector3 targetPosition = MentalBallPredictionAt((int)timeMilliseconds);
            if (fabsf(targetPosition.coordinates[0]) > PITCH_HALF_LENGTH - 0.2f ||
                fabsf(targetPosition.coordinates[1]) > PITCH_HALF_WIDTH - 0.2f)
                forced = true;
            if (!forced)
                if (targetPosition.coordinates[2] >= 1.0f) continue;
            targetPosition = targetPosition.Get2D();
            TimeNeeded timeNeeded = TimeNeededAt(timeMilliseconds);
            unsigned int timeNeededMilliseconds = timeNeeded.usualMilliseconds;
            int nextTimeStep =
                FloatToInt32(
                    roundf((float)(int)(timeNeededMilliseconds - timeMilliseconds) * 0.05f)) -
                5;
            nextTimeStep = nextTimeStep < 1 ? 1 : nextTimeStep;
            nextTimeStep = 10 < nextTimeStep ? 10 : nextTimeStep;
            timeStep = (unsigned int)nextTimeStep;

            if (timeNeededMilliseconds <= timeMilliseconds) {
                float dot = (targetPosition - playerPosition)
                                .GetNormalized(0)
                                .GetDotProduct(ballDirectionRough);
                float justInTimeFactor =
                    clamp((float)timeToBallMilliseconds / (float)timeMilliseconds, 0.0f, 1.0f);
                if (justInTimeFactor < 0.35f) forced = true;
                if (dot > 0.0f) forced = true;
                float targetDistance = (targetPosition - playerPosition).GetLength();
                float targetVelocity = clamp(targetDistance * DISTANCE_TO_VELOCITY_MULTIPLIER,
                                             IDLE_VELOCITY, SPRINT_VELOCITY);
                Vector3 targetMovement =
                    (targetPosition - playerPosition).GetNormalized(0) * targetVelocity;
                float movementRating =
                    1.0f - NormalizedClamp((desiredMovement - targetMovement).GetLength(), 0.0f,
                                           SPRINT_VELOCITY);
                float directionRating =
                    1.0f -
                    NormalizedClamp(fabsf(adaptedDesiredDirection.GetAngle2D(
                                        targetMovement.GetNormalized(adaptedDesiredDirection))),
                                    0.0f, 0.5f * PI);
                movementRating = movementRating * 0.4f + directionRating * 0.6f;
                float timeRating = 1.0f - NormalizedClamp((float)timeMilliseconds, 0, 5000);
                float perpendicularRating =
                    1.0f -
                    NormalizedClamp((playerBallShortestTargetPosition - targetPosition).GetLength(),
                                    0, 2.0f * SPRINT_VELOCITY);
                float previousTargetRating = 0.0f;
                if (previousDesiredTargetTimeMilliseconds >= timeNeededToGetToBallMilliseconds)
                    previousTargetRating = 1.0f - abs(previousDesiredTargetTimeMilliseconds -
                                                      timeNeededToGetToBallMilliseconds) /
                                                      500.0f;
                float rating = movementRating * movementWeight + timeRating * timeWeight +
                               perpendicularRating * perpendicularWeight +
                               previousTargetRating * previousTargetWeight;
                if (rating > bestChoice.rating || (forced && bestChoice.timeMilliseconds == -10)) {
                    bestChoice.rating = rating;
                    bestChoice.timeMilliseconds = (int)timeMilliseconds;
                    bestChoice.timeNeededMilliseconds = (int)timeNeededMilliseconds;
                }
            }
            if (forced) {
                if (bestChoice.timeMilliseconds == -10) {
                    bestChoice.timeMilliseconds = (int)timeMilliseconds;
                    bestChoice.timeNeededMilliseconds = (int)timeMilliseconds;
                }
                break;
            }
        }
        if (bestChoice.timeMilliseconds == -10) {
            bestChoice.timeMilliseconds = BALL_PREDICTION_SPAN_MILLISECONDS - 10;
            bestChoice.timeNeededMilliseconds = BALL_PREDICTION_SPAN_MILLISECONDS - 10;
        }

        Vector3 targetPosition = MentalBallPredictionAt(bestChoice.timeMilliseconds).Get2D();
        bestDirection = (targetPosition - playerPosition).GetNormalized(Direction());
        float bestVelocityTimeBased =
            clamp(((float)bestChoice.timeNeededMilliseconds / (float)bestChoice.timeMilliseconds) *
                      maximumVelocity,
                  IDLE_VELOCITY, SPRINT_VELOCITY);
        bestVelocityFloat = bestVelocityTimeBased;
        if (bestChoice.timeMilliseconds > 250)
            bestVelocityFloat = bestVelocityFloat * 0.97f + FloatVelocity() * 0.03f;

        Vector3 lookAtBall = (MentalBallPredictionAt(20).Get2D() - playerPosition)
                                 .GetNormalized(adaptedDesiredDirection);
        Vector3 lookAtDirection = bestDirection;
        Vector3 lookAtDesiredDirection;
        if (bestChoice.timeMilliseconds < 100) {
            lookAtDesiredDirection = adaptedDesiredDirection;
        } else if (FloatToVelocityCategory(bestVelocityFloat) != e_Velocity_Idle) {
            lookAtDesiredDirection =
                GetClamped2D(adaptedDesiredDirection, lookAtBall, lookAtDirection);
        } else {
            lookAtDesiredDirection = adaptedDesiredDirection;
        }
        bestLookAt = playerPosition + lookAtBall.GetRotated2D(clamp(
                                          lookAtDesiredDirection.GetAngle2D(lookAtBall) * 1.0f,
                                          -0.25f * PI, 0.25f * PI)) *
                                          10.0f;
        return (unsigned int)bestChoice.timeMilliseconds;
    }

    CUJU_FUNCTION_NO_INLINE void MovementCommand(bool forceMagnet, bool extraHaste) {
        const int defaultLookAtTimeMilliseconds = 40;
        Vector3 quantizedInputDirection = controller.inputDirection;
        QuantizeDirection(quantizedInputDirection, QUANTIZED_DIRECTION_BIAS);
        state::PlayerCommand command = DefaultCommand();
        command.desiredFunctionType = e_FunctionType_Movement;
        command.useDesiredMovement = true;
        command.useDesiredLookAt = true;

        Vector3 manualDirection = quantizedInputDirection;
        float manualVelocityFloat = controller.inputVelocityFloat;
        Vector3 defaultLookDirection = manualDirection;
        float desiredVelocityFactor =
            1.0f - libm::powf(NormalizedClamp(StandardMinimumFloat(controller.inputVelocityFloat,
                                                                   FloatVelocity()),
                                              IDLE_DRIBBLE_SWITCH, SPRINT_VELOCITY),
                              0.5f) *
                       0.3f;
        Vector3 lookFocusPosition = MentalBallPredictionAt(defaultLookAtTimeMilliseconds).Get2D();
        lookFocusPosition += Direction() * 0.5f;
        float toFocusAngle = (lookFocusPosition - Position())
                                 .GetNormalized(manualDirection)
                                 .GetAngle2D(manualDirection);
        defaultLookDirection =
            manualDirection.GetRotated2D(toFocusAngle * libm::powf(desiredVelocityFactor, 0.7f));

        command.desiredDirection = manualDirection;
        command.desiredVelocityFloat = manualVelocityFloat;
        Vector3 autoDirection = command.desiredDirection;
        float autoVelocityFloat = command.desiredVelocityFloat;
        Vector3 autoLookDirection = defaultLookDirection;
        float adaptedPossessionAmount = controller.possessionAmount;
        playerState.desiredTimeToBallMilliseconds = 0;
        float autoBias = 0.0f;
        float inputDirectionIsOwnHalfFactor = NormalizedClamp(
            controller.inputDirection.GetDotProduct(Vector3(side, 0, 0)), -1.0f, 1.0f);
        const int teamDesignated = Team().designatedTeamPossessionPlayer;
        const float possessionAmount = controller.possessionAmount;

        if (match.ballRetainer == player) {
            autoBias = 0.0f;
        } else if (forceMagnet || match.designatedPossessionPlayer == player ||
                   ((GetLastTouchBias(world, player, 2000) > 0.01f && possessionAmount > 0.5f) &&
                    teamDesignated == player) ||
                   ((!controller.opponentTeamHasPossession && possessionAmount > 0.5f) &&
                    teamDesignated == player) ||
                   (possessionAmount > 0.99f && teamDesignated == player) ||
                   (controller.hasBestPossession && teamDesignated == player)) {
            if (controller.hasBestPossession) {
                Vector3 autoLookAt;
                playerState.desiredTimeToBallMilliseconds = (int)BallControlMovement(
                    controller.inputVelocityFloat, autoDirection, autoVelocityFloat, autoLookAt);
                autoLookDirection = (autoLookAt - Position()).GetNormalized(0);
                autoBias = 1.0f;
            } else {
                float haste = 0.0f;
                if (extraHaste || forceMagnet) {
                    haste = 1.0f;
                } else {
                    float thresholdPossessionAmountForHaste = 1.1f;
                    if (adaptedPossessionAmount < thresholdPossessionAmountForHaste) haste = 1.0f;
                }
                Vector3 autoLookAt;
                playerState.desiredTimeToBallMilliseconds =
                    (int)ToBallMovement(quantizedInputDirection, controller.inputVelocityFloat,
                                        autoDirection, autoVelocityFloat, autoLookAt, haste);
                autoLookDirection = (autoLookAt - Position()).GetNormalized(0);
                autoBias = 1.0f;
                if (match.designatedPossessionPlayer != player) {
                    float sameDirectionFactor =
                        1.0f -
                        clamp(fabsf(autoDirection.GetAngle2D(manualDirection)) / PI, 0.0f, 1.0f);
                    sameDirectionFactor = sameDirectionFactor * 0.5f + 0.5f;
                    autoBias = 0.0f;
                    if (GetLastTouchBias(world, player, 2000) > 0.01f) {
                        if (playerState.timeNeededToGetToBallMilliseconds < 1700)
                            autoBias = libm::powf(GetLastTouchBias(world, player, 2000), 0.4f) *
                                       libm::powf(sameDirectionFactor, 0.5f);
                    }
                    if (!controller.opponentTeamHasPossession) {
                        if (playerState.base.externalController >= 0) {
                            float switchBias = GetLastSwitchBias(world, controller);
                            float magnetBias = StandardMaximumFloat(
                                Curve(sameDirectionFactor, 0.8f), libm::powf(switchBias, 0.5f));
                            magnetBias *= 1.0f - inputDirectionIsOwnHalfFactor;
                            autoBias = clamp(
                                libm::powf(
                                    (StandardMaximumFloat(possessionAmount, 0.5f) - 0.5f) * 2.0f,
                                    0.5f) *
                                    magnetBias,
                                autoBias, 1.0f);
                        }
                    }
                    if (forceMagnet) autoBias = 1.0f;
                }
            }
        } else if (world.playerTeamID[match.designatedPossessionPlayer] != teamID) {
            if (teamDesignated == player) {
                state::PlayerImage opponentImage =
                    MentalPlayerImage(world, mentalImageIndex, controller.opponentPlayer);
                Vector3 opponentPosition = opponentImage.position + opponentImage.movement * 0.14f +
                                           opponentImage.directionVector * 0.6f;
                Vector3 opponentToGoalDirection =
                    (Vector3(PITCH_HALF_LENGTH * side, 0, 0) - opponentPosition).GetNormalized(0);
                float actionRadius = 5.0f;
                Vector3 focusPosition =
                    opponentPosition + (opponentToGoalDirection * actionRadius * 0.7f);
                float actionBias =
                    1.0f - Curve(NormalizedClamp((Position() - focusPosition).GetLength(), 0.0f,
                                                 actionRadius),
                                 0.7f);
                if (ExternalControllerActive(world, player))
                    autoBias = actionBias * 0.0f + GetLastSwitchBias(world, controller) * 0.3f;
                else
                    autoBias = actionBias * 0.0f;
                autoBias *= 1.0f - inputDirectionIsOwnHalfFactor;
                if (autoBias > 0.0f) {
                    float playerOpponentDistance = (opponentPosition - Position()).GetLength();
                    float manMarkingBias = libm::powf(
                        1.0f - Curve(NormalizedClamp(playerOpponentDistance, 0.0f, 7.0f), 0.7f),
                        1.2f);
                    Vector3 manMarkingDirection =
                        opponentImage.movement.GetNormalized(controller.inputDirection);
                    float manMarkingVelocityFloat = VelocityCategoryToFloat(opponentImage.velocity);
                    Vector3 manMarkingLookDirection =
                        (MentalBallPredictionAt(defaultLookAtTimeMilliseconds).Get2D() - Position())
                            .GetNormalized(0);
                    Vector3 huntTarget =
                        opponentPosition + opponentToGoalDirection * playerOpponentDistance * 0.3f;
                    Vector3 huntMovement = huntTarget - Position();
                    Vector3 huntDirection = huntMovement.GetNormalized(controller.inputDirection);
                    float huntVelocityFloat =
                        clamp(huntMovement.GetLength() * DISTANCE_TO_VELOCITY_MULTIPLIER,
                              IDLE_VELOCITY, SPRINT_VELOCITY);
                    Vector3 huntLookDirection =
                        (MentalBallPredictionAt(defaultLookAtTimeMilliseconds).Get2D() - Position())
                            .GetNormalized(0);
                    autoDirection = (manMarkingDirection * manMarkingBias +
                                     huntDirection * (1.0f - manMarkingBias))
                                        .GetNormalized(controller.inputDirection);
                    autoVelocityFloat = clamp(manMarkingVelocityFloat * manMarkingBias +
                                                  huntVelocityFloat * (1.0f - manMarkingBias),
                                              IDLE_VELOCITY, SPRINT_VELOCITY);
                    autoLookDirection = ((manMarkingLookDirection * manMarkingBias +
                                          huntLookDirection * (1.0f - manMarkingBias)))
                                            .GetNormalized(0);
                }
            }
        }

        Vector3 autoMovement = autoDirection * autoVelocityFloat;
        Vector3 manualMovement = manualDirection * manualVelocityFloat;
        Vector3 resultingMovement = manualMovement * (1.0f - autoBias) + autoMovement * autoBias;
        command.desiredDirection = resultingMovement.GetNormalized(quantizedInputDirection);
        command.desiredVelocityFloat =
            clamp(resultingMovement.GetLength(), IDLE_VELOCITY, SPRINT_VELOCITY);
        if (command.desiredVelocityFloat < IDLE_DRIBBLE_SWITCH)
            command.desiredDirection = autoLookDirection;
        Vector3 resultLookDirection = autoLookDirection;
        command.desiredLookAt = Position() + resultLookDirection * 10.0f;
        queue.push_back(command);
    }

    CUJU_FUNCTION_NO_INLINE float GetLazyVelocity(float desiredVelocityFloat) const {
        float adaptedDesiredVelocityFloat = desiredVelocityFloat;
        if (adaptedDesiredVelocityFloat > SPRINT_VELOCITY)
            adaptedDesiredVelocityFloat =
                SPRINT_VELOCITY + (adaptedDesiredVelocityFloat - SPRINT_VELOCITY) * 0.1f;
        const float fatigueInverse = playerState.base.fatigueFactorInverse;
        float startLazinessDistance = 20.0f * (fatigueInverse * 0.8f + 0.2f);
        float endLazinessDistance = 65.0f * (fatigueInverse * 0.5f + 0.5f);
        Vector3 opponentPosition = Position(OpponentTeam().designatedTeamPossessionPlayer);
        float actionDistance = (Position() - opponentPosition).GetLength();
        float teamPossession = clamp(controller.fadingTeamPossessionAmount - 0.5f, 0.0f, 1.0f);
        float mindSet = GetMindSet(DynamicRole());
        float lazinessByRole = mindSet + teamPossession * (1.0f - mindSet * 2.0f);
        float lazinessByPosition =
            NormalizedClamp(actionDistance, startLazinessDistance, endLazinessDistance);
        float lazyFactor = lazinessByPosition * (0.5f + lazinessByRole * 0.5f);
        float resultingVelocityFloat = adaptedDesiredVelocityFloat * (1.0f - lazyFactor);
        bool clampToDribble = desiredVelocityFloat >= DRIBBLE_VELOCITY;
        if (clampToDribble && resultingVelocityFloat < DRIBBLE_VELOCITY)
            resultingVelocityFloat = DRIBBLE_VELOCITY;
        float breathLeftFactor = 1.0f - NormalizedClamp(PlayerAverageVelocity(world, player, 10),
                                                        IDLE_VELOCITY, SPRINT_VELOCITY);
        float workRate = Statistic(e_Statistic_MentalWorkRate);
        breathLeftFactor = libm::powf(breathLeftFactor, 0.8f - workRate * 0.2f);
        breathLeftFactor = clamp(breathLeftFactor * 1.2f, 0.0f, 1.0f);
        breathLeftFactor = breathLeftFactor * lazyFactor + 1.0f * (1.0f - lazyFactor);
        resultingVelocityFloat =
            StandardMinimumFloat(resultingVelocityFloat, SPRINT_VELOCITY * breathLeftFactor);
        return resultingVelocityFloat;
    }

    CUJU_FUNCTION_NO_INLINE Vector3 GetSupportPositionForceField(const Vector3 &basePosition,
                                                                 bool makeRun) const {
        const int designatedPlayer = Team().designatedTeamPossessionPlayer;
        Vector3 currentPosition = Position() + Movement() * 0.1f;
        Vector3 mainManPosition = Position(designatedPlayer) + Movement(designatedPlayer) * 0.1f;
        ForceField forceField;
        float dynamicMindSet = GetMindSet(DynamicRole());
        bool forceNoOffside = true;
        float basePositionWeight = 0.7f, overallWeight = 1.0f;
        float opponentRepelWeight = 0.3f * overallWeight;
        float teammateRepelWeight = 0.4f * overallWeight;
        float ballRepelWeight = 1.0f * overallWeight;
        float runWeight = 1.0f * overallWeight;
        float flockToPossessionPlayerWeight = 0.45f * overallWeight;
        float webScale = 0.75f;
        switch (DynamicRole()) {
            case e_PlayerRole_CentreBack:
            case e_PlayerRole_LeftBack:
            case e_PlayerRole_RightBack:
                opponentRepelWeight *= 2.2f;
                break;
            case e_PlayerRole_DefensiveMidfielder:
                opponentRepelWeight *= 2.0f;
                break;
            case e_PlayerRole_CentralMidfielder:
            case e_PlayerRole_LeftMidfielder:
            case e_PlayerRole_RightMidfielder:
                opponentRepelWeight *= 1.6f;
                break;
            case e_PlayerRole_AttackingMidfielder:
                opponentRepelWeight *= 1.2f;
                break;
            case e_PlayerRole_CentreForward:
                opponentRepelWeight *= 1.0f;
                break;
            default:
                break;
        }
        float offsideX = GetOffsideLine(world, mentalImageIndex, opponentID, 240);
        float adaptedMakeRun = makeRun;
        {
            ForceSpot spot;
            spot.origin = basePosition;
            if (player == TeamAI().forwardSupportPlayer) {
                spot.origin.coordinates[0] += -side * (0.3f + 0.7f * dynamicMindSet) * 12.0f;
            } else {
                float amount = 22.0f;
                float laneY = -SignSide(mainManPosition.coordinates[1]) * 8.0f;
                amount *=
                    Curve(1.0f - NormalizedClamp(fabsf(laneY - currentPosition.coordinates[1]),
                                                 0.0f, 30.0f),
                          1.0f);
                float delta = -side * libm::powf(dynamicMindSet, 1.5f) * amount;
                spot.origin.coordinates[0] += delta;
            }
            spot.magnetType = e_MagnetType_Attract;
            spot.decayType = e_DecayType_Constant;
            spot.power = 1.0f * basePositionWeight;
            spot.power *= 0.3f + 0.7f * NormalizedClamp((spot.origin - currentPosition).GetLength(),
                                                        0.0f, 20.0f);
            forceField.push_back(spot);
        }
        if (adaptedMakeRun) {
            ForceSpot spot;
            spot.origin =
                Vector3(-side * PITCH_HALF_LENGTH, currentPosition.coordinates[1] * 0.5f, 0.0f);
            spot.magnetType = e_MagnetType_Attract;
            spot.decayType = e_DecayType_Constant;
            spot.power = 2.0f * runWeight;
            forceField.push_back(spot);
        }
        PlayerList opponents = GetClosestPlayers(
            world, context, opponentID, mainManPosition * 0.3f + currentPosition + 0.7f, false, 3);
        for (int opponent : opponents) {
            state::PlayerImage opponentImage = MentalPlayerImage(world, mentalImageIndex, opponent);
            ForceSpot spot;
            Vector3 opponentPosition = opponentImage.position + opponentImage.movement * 0.1f;
            spot.origin =
                opponentPosition + (opponentPosition - mainManPosition).GetNormalized(0) * 2.0f;
            spot.magnetType = e_MagnetType_Repel;
            spot.decayType = e_DecayType_Variable;
            spot.power = 1.0f * opponentRepelWeight;
            spot.scale = 5.0f;
            if (adaptedMakeRun) {
                spot.scale = 2.0f;
                spot.power *= 0.5f;
            }
            spot.exponent = 0.7f;
            forceField.push_back(spot);
        }
        if (Team().fadingTeamPossessionAmount >= 1.02f) {
            PlayerList players =
                GetClosestPlayers(world, context, teamID, currentPosition, false, 6);
            for (int mate : players) {
                if (mate == player) continue;
                state::PlayerImage mateImage = MentalPlayerImage(world, mentalImageIndex, mate);
                ForceSpot spot;
                spot.origin = mateImage.position + mateImage.movement * 0.1f;
                spot.magnetType = e_MagnetType_Repel;
                spot.decayType = e_DecayType_Variable;
                spot.power = 1.0f * teammateRepelWeight;
                spot.scale = 14.0f * webScale;
                spot.exponent = 1.0f;
                forceField.push_back(spot);
            }
        }
        if (player != designatedPlayer && Team().fadingTeamPossessionAmount >= 1.06f) {
            ForceSpot spot;
            spot.magnetType = e_MagnetType_Repel;
            spot.decayType = e_DecayType_Variable;
            spot.power = 1.0f * ballRepelWeight;
            spot.scale = 2.0f;
            spot.exponent = 0.5f;
            for (int timeMilliseconds : {200, 350, 500, 650}) {
                spot.origin = MentalBallPredictionAt(timeMilliseconds).Get2D();
                forceField.push_back(spot);
            }
        }
        if (player != designatedPlayer) {
            ForceSpot spot;
            spot.origin = mainManPosition;
            spot.magnetType = e_MagnetType_Attract;
            spot.decayType = e_DecayType_Variable;
            spot.power = 1.0f * flockToPossessionPlayerWeight;
            spot.scale = 28.0f * webScale;
            spot.exponent = 1.0f;
            forceField.push_back(spot);
            spot.magnetType = e_MagnetType_Repel;
            spot.scale = 16.0f * webScale;
            forceField.push_back(spot);
        }
        Vector3 forceFieldPosition =
            currentPosition + GetForceFieldMovement(forceField, currentPosition, 7);
        float margin = 0.08f;
        if (forceNoOffside)
            if (forceFieldPosition.coordinates[0] * -side > (offsideX * -side) - margin)
                forceFieldPosition.coordinates[0] = offsideX - (margin * -side);
        forceFieldPosition.coordinates[0] =
            clamp(forceFieldPosition.coordinates[0], -PITCH_HALF_LENGTH, PITCH_HALF_LENGTH);
        forceFieldPosition.coordinates[1] =
            clamp(forceFieldPosition.coordinates[1], -PITCH_HALF_WIDTH, PITCH_HALF_WIDTH);
        return forceFieldPosition;
    }

    enum StrategyKind { e_StrategyKind_Defense, e_StrategyKind_Midfield, e_StrategyKind_Offense };
    CUJU_FUNCTION_NO_INLINE void FieldStrategyRequestInput(StrategyKind kind, Vector3 &direction,
                                                           float &velocity) const {
        static const float staticFactor[] = {1.0f, 0.9f, 0.8f};
        static const float attackLow[] = {0.2f, 0.1f, 0.1f}, attackHigh[] = {0.9f, 0.7f, 0.6f};
        static const float runThreshold[] = {0.0f, 0.9f, 0.7f};
        static const float defensiveBase[] = {1.9f, 1.5f, 1.3f};
        Vector3 staticDesiredPosition = TeamAdaptedFormationPosition(world, context, player, false);
        Vector3 dynamicDesiredPosition = TeamAdaptedFormationPosition(world, context, player, true);
        float actionDistance = NormalizedClamp(
            GetDistance(Position(), Position(match.designatedPossessionPlayer)), 15.0f, 20.0f);
        float staticPositionBias = Curve(staticFactor[kind] * actionDistance, 1.0f);
        Vector3 desiredPosition = staticDesiredPosition * staticPositionBias +
                                  dynamicDesiredPosition * (1.0f - staticPositionBias);
        {
            float attackBias =
                NormalizedClamp((controller.fadingTeamPossessionAmount - 0.5f) * 1.0f,
                                attackLow[kind], attackHigh[kind]);
            bool makeRun = false;
            if (kind != e_StrategyKind_Defense && attackBias > runThreshold[kind]) {
                if (TeamAI().endApplyAttackingRunMilliseconds > match.actualTimeMilliseconds &&
                    TeamAI().attackingRunPlayer == player)
                    makeRun = true;
            }
            Vector3 supportPosition = GetSupportPositionForceField(desiredPosition, makeRun);
            desiredPosition = desiredPosition * (1.0f - attackBias) + supportPosition * attackBias;
        }
        {
            float mindset = GetMindSet(DynamicRole());
            AddDefensiveComponent(desiredPosition,
                                  libm::powf(clamp(defensiveBase[kind] - mindset -
                                                       controller.fadingTeamPossessionAmount,
                                                   0.0f, 1.0f),
                                             0.7f));
            if (kind != e_StrategyKind_Offense)
                TeamApplyOffsideTrap(world, teamID, desiredPosition);
        }
        direction = (desiredPosition - Position()).GetNormalized(Direction());
        float desiredVelocity =
            (desiredPosition - Position()).GetLength() * DISTANCE_TO_VELOCITY_MULTIPLIER;
        desiredVelocity = GetLazyVelocity(desiredVelocity);
        desiredVelocity = clamp(desiredVelocity, 0, SPRINT_VELOCITY);
        velocity = desiredVelocity;
    }

    CUJU_FUNCTION_NO_INLINE void CalculateIfBallIsBoundForGoal() {
        elizaState->ballBoundForGoal = false;
        bool intersect = false;
        elizaState->ballBoundForGoalYCoordinate = 0;
        float panic = 1.02f + (1.0f - (Statistic(e_Statistic_MentalDefensivePositioning) * 0.6f +
                                       Statistic(e_Statistic_MentalVision) * 0.4f)) *
                                  0.5f;
        if (MentalBallPredictionAt(4000).coordinates[0] * side > PITCH_HALF_LENGTH &&
            (Position() - MentalBallPredictionAt(250)).GetLength() < 32.0f) {
            Line ballToGoal(MentalBallPredictionAt(0).Get2D(), MentalBallPredictionAt(800).Get2D());
            Line goalLine(Vector3(PITCH_HALF_LENGTH * side, -PITCH_HALF_WIDTH, 0),
                          Vector3(PITCH_HALF_LENGTH * side, PITCH_HALF_WIDTH, 0));
            Vector3 intersectPoint = ballToGoal.GetIntersectionPoint(goalLine).Get2D();
            intersect = !((double)fabsf(intersectPoint.coordinates[1]) > 3.7 * (double)panic);
            if (intersect) {
                elizaState->ballBoundForGoalYCoordinate = intersectPoint.coordinates[1];
                elizaState->ballBoundForGoal = true;
            }
        }
    }

    CUJU_FUNCTION_NO_INLINE void GoalieRequestInput(Vector3 &direction, float &velocity) {
        float lineDistance = 10.0f;
        Vector3 ballPosition =
            MentalBallPredictionAt(
                FloatToInt32(600 + (float)playerState.timeNeededToGetToBallMilliseconds * 0.2f))
                .Get2D();
        Vector3 targetPosition = Vector3((PITCH_HALF_LENGTH - lineDistance) * side, 0, 0);
        Vector3 goalPosition = Vector3(PITCH_HALF_LENGTH * side, 0, 0);
        float maximumVelocity = SPRINT_VELOCITY;
        if (ballPosition.coordinates[0] * side > 0) {
            CalculateIfBallIsBoundForGoal();
            if (!elizaState->ballBoundForGoal) {
                maximumVelocity = SPRINT_VELOCITY;
                Vector3 toFirstPost = Vector3(PITCH_HALF_LENGTH * side, 3.7f, 0) - ballPosition;
                Vector3 toSecondPost = Vector3(PITCH_HALF_LENGTH * side, -3.7f, 0) - ballPosition;
                float angle = toSecondPost.GetAngle2D(toFirstPost);
                Vector3 middle =
                    toFirstPost.GetRotated2D(angle * 0.5f).GetNormalized(Vector3(side, 0, 0));
                Line ballToGoal(ballPosition, ballPosition + middle);
                Line backLine(Vector3((PITCH_HALF_LENGTH - 0.7f) * side, -PITCH_HALF_WIDTH, 0),
                              Vector3((PITCH_HALF_LENGTH - 0.7f) * side, PITCH_HALF_WIDTH, 0));
                Vector3 intersect = ballToGoal.GetIntersectionPoint(backLine).Get2D();
                intersect.coordinates[1] = clamp(intersect.coordinates[1], -3.7f, 3.7f);
                ballToGoal.points[1] = intersect;
                float awayFromGoalOffsetMeters = 0.7f;
                float awayFromGoalBias = 0.3f;
                awayFromGoalBias *=
                    NormalizedClamp(controller.fadingTeamPossessionAmount, 1.0f, 1.5f);
                if (controller.fadingTeamPossessionAmount < 1.0f) {
                    const int opponent = OpponentTeam().designatedTeamPossessionPlayer;
                    Vector3 opponentPosition = Position(opponent) + Movement(opponent) * 0.32f;
                    if (match.players[opponent].hasPossession == false)
                        ballToGoal.points[0] = opponentPosition * 0.6f + ballPosition * 0.4f;
                    else
                        ballToGoal.points[0] = opponentPosition * 0.4f + ballPosition * 0.6f;
                    float shootThreshold = 20.0f;
                    float opponentToGoalDistance = (goalPosition - opponentPosition).GetLength();
                    float opponentToThresholdDistance =
                        clamp(opponentToGoalDistance -
                                  shootThreshold * NormalizedClamp(opponentToGoalDistance, 0.0f,
                                                                   shootThreshold * 2.0f),
                              0.0f, PITCH_HALF_LENGTH);
                    Vector3 shootingPoint =
                        opponentPosition + (goalPosition - opponentPosition).GetNormalized(0) *
                                               opponentToThresholdDistance;
                    int mate =
                        GetClosestPlayer(world, context, teamID, shootingPoint, false, player);
                    float mateToThresholdDistance = 99999;
                    if (mate >= 0) {
                        Vector3 matePosition = Position(mate) + Movement(mate) * 0.24f;
                        mateToThresholdDistance = (shootingPoint - matePosition).GetLength();
                    }
                    if (mateToThresholdDistance > opponentToThresholdDistance + 1.0f) {
                        awayFromGoalBias = 1.0f;
                        int opponentHelper = GetClosestPlayer(world, context, opponentID,
                                                              goalPosition, false, opponent);
                        if (opponentHelper >= 0) {
                            Vector3 opponentHelperPosition =
                                Position(opponentHelper) + Movement(opponentHelper) * 0.32f;
                            float helperShootThreshold = 24.0f;
                            float opponentHelperToGoalDistance =
                                (goalPosition - opponentHelperPosition).GetLength();
                            float opponentHelperToThresholdDistance =
                                clamp(opponentHelperToGoalDistance -
                                          helperShootThreshold *
                                              NormalizedClamp(opponentHelperToGoalDistance, 0.0f,
                                                              helperShootThreshold * 2.0f),
                                      0.0f, PITCH_HALF_LENGTH);
                            Vector3 helperShootingPoint =
                                opponentHelperPosition +
                                (goalPosition - opponentHelperPosition).GetNormalized(0) *
                                    opponentHelperToThresholdDistance;
                            int mateHelper = GetClosestPlayer(world, context, teamID,
                                                              helperShootingPoint, false, player);
                            float mateHelperToThresholdDistance = 99999;
                            if (mateHelper >= 0)
                                mateHelperToThresholdDistance =
                                    (helperShootingPoint -
                                     (Position(mateHelper) + Movement(mateHelper) * 0.24f))
                                        .GetLength();
                            float secondaryDistanceDifference = 0.0f;
                            if (mateHelperToThresholdDistance > opponentHelperToThresholdDistance)
                                secondaryDistanceDifference =
                                    NormalizedClamp(mateHelperToThresholdDistance -
                                                        opponentHelperToThresholdDistance,
                                                    0.0f, 2.0f);
                            float helperVersusPrimaryDistanceRatio =
                                1.0f - NormalizedClamp(opponentHelperToThresholdDistance /
                                                           (opponentToThresholdDistance + 0.0001f),
                                                       1.0f, 1.5f);
                            helperVersusPrimaryDistanceRatio *= 0.7f;
                            awayFromGoalBias = clamp(1.0f - (secondaryDistanceDifference *
                                                             helperVersusPrimaryDistanceRatio),
                                                     0.0f, 1.0f);
                        }
                    }
                }
                bool applyRushOut =
                    TeamAI().endApplyKeeperRushMilliseconds > match.actualTimeMilliseconds;
                if (applyRushOut) awayFromGoalBias = 1.0f;
                float distance = StandardMaximumFloat(ballToGoal.GetLength() - 0.5f, 0.0f);
                awayFromGoalOffsetMeters =
                    clamp(distance * awayFromGoalBias, awayFromGoalOffsetMeters, PITCH_HALF_LENGTH);
                targetPosition = ballToGoal.points[1] +
                                 (ballToGoal.points[0] - ballToGoal.points[1]).GetNormalized(0) *
                                     awayFromGoalOffsetMeters;
                float fraction = 0.0f;
                float distanceToBallToGoalLine =
                    ballToGoal.GetDistanceToPoint(Position(), fraction);
                if ((targetPosition - goalPosition).GetLength() <
                        (Position() - goalPosition).GetLength() &&
                    distanceToBallToGoalLine < 1.0f && fraction > 0.0f)
                    maximumVelocity = WALK_VELOCITY;
                targetPosition.coordinates[0] =
                    clamp(targetPosition.coordinates[0], -PITCH_HALF_LENGTH + 0.2f,
                          PITCH_HALF_LENGTH - 0.2f);
            } else {
                maximumVelocity = SPRINT_VELOCITY;
                Line ballToGoal;
                ballToGoal.points[0] = MentalBallPredictionAt(10).Get2D();
                float minimumGoalLineDistance = 0.4f;
                Vector3 ballOverGoalLinePosition =
                    Vector3(PITCH_HALF_LENGTH * side, elizaState->ballBoundForGoalYCoordinate, 0);
                ballOverGoalLinePosition +=
                    (ballToGoal.points[0] - ballOverGoalLinePosition).GetNormalized(0) *
                    minimumGoalLineDistance;
                ballToGoal.points[1] = ballOverGoalLinePosition;
                float fraction = 0.0f;
                ballToGoal.GetDistanceToPoint(Position() + Movement() * 0.05f, fraction);
                float fractionAfterOneSecond = 0.0f;
                ballToGoal.GetDistanceToPoint(MentalBallPredictionAt(1010).Get2D(),
                                              fractionAfterOneSecond);
                bool keeperRunsToGoal = false;
                if ((double)fractionAfterOneSecond > 1e-4) {
                    float timeToReachKeeper = fraction / fractionAfterOneSecond;
                    Vector3 ballPositionAtKeeper =
                        MentalBallPredictionAt(FloatToInt32(10 + 1000 * timeToReachKeeper));
                    if ((double)ballPositionAtKeeper.coordinates[2] > 2.5) keeperRunsToGoal = true;
                }
                fraction = clamp(fraction, 0.0f, 1.0f);
                if (keeperRunsToGoal) {
                    targetPosition = ballOverGoalLinePosition;
                } else {
                    targetPosition = ballToGoal.points[0] +
                                     (ballToGoal.points[1] - ballToGoal.points[0]) * fraction;
                    targetPosition.coordinates[2] = 0.0f;
                    targetPosition.coordinates[0] =
                        clamp(targetPosition.coordinates[0], -PITCH_HALF_LENGTH + 0.2f,
                              PITCH_HALF_LENGTH - 0.2f);
                }
            }
        }
        direction = targetPosition - Position();
        velocity = clamp(direction.GetLength() * DISTANCE_TO_VELOCITY_MULTIPLIER, IDLE_VELOCITY,
                         maximumVelocity);
        direction.Normalize(Direction());
    }

    CUJU_FUNCTION_NO_INLINE float PassingOdds(const Vector3 &target, int passType,
                                              const ImageList &opponents,
                                              float ballVelocityMultiplier = 1.0f) const {
        float secondScale = 1.0f;
        Vector3 origin = Position() + Movement() * 0.12f;
        Line line(origin, target);
        float danger = 0.0f;
        for (const PlayerImagePosition &opponentImage : opponents) {
            Vector3 opponentPosition = opponentImage.position + opponentImage.movement * 0.2f;
            float fraction = 0.0f;
            float opponentDistance = line.GetDistanceToPoint(opponentPosition, fraction);
            if (fraction >= 0.0f && fraction <= 1.0f + 0.2f) {
                if ((passType == e_FunctionType_HighPass &&
                     (fraction < 0.2f || fraction > 0.65f)) ||
                    passType != e_FunctionType_HighPass) {
                    float clampedFraction = clamp(fraction, 0.0f, 1.0f);
                    Vector3 intersect =
                        origin * (1.0f - clampedFraction) + target * clampedFraction;
                    float opponentToIntersectSeconds = (opponentDistance + 1.0f) / SPRINT_VELOCITY;
                    Vector3 originToBallPosition = (intersect - origin);
                    float penaltyTime =
                        (passType == e_FunctionType_HighPass && fraction > 0.5f) ? 2.5f : 0.0f;
                    float ballToIntersectSeconds =
                        0.7f + originToBallPosition.GetLength() * fraction * 0.03f + penaltyTime;
                    ballToIntersectSeconds *= 1.0f / ballVelocityMultiplier;
                    danger += clamp(
                        ballToIntersectSeconds - opponentToIntersectSeconds + (secondScale * 0.5f),
                        0.0f, secondScale);
                }
            }
        }
        if (passType == e_FunctionType_HighPass) danger += 0.4f;
        danger = NormalizedClamp(danger, 0.0f, secondScale);
        return 1.0f - danger;
    }
    CUJU_FUNCTION float PassingOddsTo(int target, int passType, const ImageList &opponents) const {
        float initialTargetDistance = (Position(target) - Position()).GetLength();
        if (passType == e_FunctionType_HighPass && initialTargetDistance < 10.0f) return 0.0f;
        float estimatedTimeSeconds = 0.7f + initialTargetDistance * 0.03f;
        Vector3 targetPosition =
            Position(target) + Movement(target) * clamp(estimatedTimeSeconds, 0.0f, 0.5f);
        if (passType == e_FunctionType_LongPass)
            targetPosition += Vector3(-side * initialTargetDistance * 0.2f, 0, 0);
        return PassingOdds(targetPosition, passType, opponents);
    }

    CUJU_FUNCTION_NO_INLINE void AddPass(int target, int passType) {
        state::PlayerCommand command = DefaultCommand();
        command.desiredFunctionType = passType;
        command.useDesiredMovement = false;
        command.useDesiredLookAt = false;
        command.touchInformation.targetPlayer = -1;
        command.touchInformation.forcedTargetPlayer = target;
        command.touchInformation.inputDirection = Vector3(0);
        command.touchInformation.inputPower = 0;
        command.touchInformation.autoDirectionBias = 1.0f;
        command.touchInformation.autoPowerBias = 1.0f;
        state::TouchInformation &touch = command.touchInformation;
        GetPass(world, player, passType, touch.inputDirection, touch.inputPower,
                touch.autoDirectionBias, touch.autoPowerBias, touch.desiredDirection,
                touch.desiredPower, touch.targetPlayer, touch.forcedTargetPlayer);
        queue.push_back(command);
    }

    CUJU_FUNCTION_NO_INLINE void AddPanicPass() {
        int ySide = SignSide(Direction().coordinates[1]);
        Vector3 sensibleAwayDirection = ((Direction() * Vector3(0.8f, 1.0f, 0.0f)).GetNormalized() +
                                         Vector3(-side * 0.7f, ySide * 0.5f, 0))
                                            .GetNormalized(0) +
                                        Vector3(0, 0, 0.3f);
        sensibleAwayDirection.Normalize(Direction());
        state::PlayerCommand command = DefaultCommand();
        command.useDesiredMovement = false;
        command.useDesiredLookAt = false;
        command.touchInformation.inputDirection = sensibleAwayDirection;
        command.touchInformation.desiredDirection = sensibleAwayDirection;
        command.touchInformation.autoDirectionBias = 0.0f;
        command.touchInformation.autoPowerBias = 0.0f;
        command.desiredFunctionType = e_FunctionType_HighPass;
        command.touchInformation.inputPower = 0.7f;
        command.touchInformation.desiredPower = 0.7f;
        queue.push_back(command);
        command.desiredFunctionType = e_FunctionType_Shot;
        command.touchInformation.inputPower = 0.6f;
        command.touchInformation.desiredPower = 0.6f;
        queue.push_back(command);
        command.desiredFunctionType = e_FunctionType_LongPass;
        command.touchInformation.inputPower = 0.8f;
        command.touchInformation.desiredPower = 0.8f;
        queue.push_back(command);
    }

    CUJU_FUNCTION_NO_INLINE void BestDribbleMovement(Vector3 &desiredDirection,
                                                     float &desiredVelocity) const {
        Vector3 myPosition = Position(), myMovement = Movement();
        float offenseFactor =
            0.7f + DRIBBLE_OFFENSIVENESS * 0.05f + GetMindSet(DynamicRole()) * 0.05f;
        float powerMultiplier = 1.0f;
        float futureSeconds = 0.25f;
        PlayerList opponents = GetClosestPlayers(world, context, opponentID, myPosition, false, 5);
        InlineVector<state::PlayerImage, state::MAXIMUM_PLAYERS> opponentPlayerImages;
        for (int opponent : opponents)
            opponentPlayerImages.push_back(MentalPlayerImage(world, mentalImageIndex, opponent));
        float nearBackline =
            NormalizedClamp(fabsf(myPosition.coordinates[0]) / PITCH_HALF_LENGTH, 0.0f, 1.0f);
        // Exact: equals GRF's `powf(x, 2.0f)`.
        float centerModifierInverse = 1.0f - nearBackline * nearBackline;
        centerModifierInverse *= 0.5f;
        Vector3 opponentGoalPosition = Vector3(
            -side * PITCH_HALF_LENGTH,
            myPosition.coordinates[1] * (1.0f - DRIBBLE_CENTER_MAGNET) * centerModifierInverse, 0);
        ForceField forceField;
        for (const state::PlayerImage &opponentImage : opponentPlayerImages) {
            ForceSpot spot;
            spot.origin = opponentImage.position + opponentImage.movement * futureSeconds;
            spot.magnetType = e_MagnetType_Repel;
            spot.decayType = e_DecayType_Variable;
            spot.power = 2.0f * powerMultiplier;
            spot.scale = 10.0f;
            spot.exponent = 1.0f;
            forceField.push_back(spot);
        }
        {
            ForceSpot spot;
            spot.origin =
                Vector3(myPosition.coordinates[0],
                        (PITCH_HALF_WIDTH + 5.0f) * SignSide(myPosition.coordinates[1]), 0);
            spot.magnetType = e_MagnetType_Repel;
            spot.decayType = e_DecayType_Variable;
            spot.power = 4.0f * powerMultiplier;
            spot.scale = 20.0f;
            spot.exponent = 0.7f;
            forceField.push_back(spot);
            spot.origin = Vector3((PITCH_HALF_LENGTH + 5.0f) * SignSide(myPosition.coordinates[0]),
                                  myPosition.coordinates[1], 0);
            forceField.push_back(spot);
        }
        {
            ForceSpot spot;
            spot.origin = opponentGoalPosition;
            spot.magnetType = e_MagnetType_Attract;
            spot.decayType = e_DecayType_Constant;
            spot.power = offenseFactor * powerMultiplier;
            forceField.push_back(spot);
        }
        Vector3 forceFieldMovement =
            GetForceFieldMovement(forceField, myPosition + myMovement * futureSeconds, 1.0f);
        desiredDirection = forceFieldMovement.GetNormalized(Direction());
        desiredVelocity = clamp(forceFieldMovement.GetLength() * DISTANCE_TO_VELOCITY_MULTIPLIER,
                                IDLE_VELOCITY, SPRINT_VELOCITY);
        desiredVelocity = RangeVelocity(desiredVelocity);
    }

    CUJU_FUNCTION_NO_INLINE void GetOnTheBallCommands(Vector3 &rawInputDirection,
                                                      float &rawInputVelocityFloat) {
        float movementDifference =
            NormalizedClamp((match.ball.momentum - Movement()).GetLength(), 0.0f, 10.0f);
        float oneTouchIsHard = movementDifference - Statistic(e_Statistic_TechnicalShortPass) *
                                                        movementDifference * 0.8f;
        ImageList opponentPlayerImages;
        MentalTeamPlayerImages(world, mentalImageIndex, opponentID, opponentPlayerImages);
        float possessionTimeFactor =
            NormalizedClamp((float)playerState.possessionDurationMilliseconds, 0, 5000);
        // Exact: equals GRF's `powf(x, 2.0f)`.
        float longPossessionFactor = possessionTimeFactor * possessionTimeFactor;
        const float mind = GetMindSet(DynamicRole());
        float forwardSpaceWeight = 0.4f, spaceWeight = 0.3f;
        float forwardWeight = 2.0f + mind * 6.0f;
        float totalTacticalWeight = forwardSpaceWeight + spaceWeight + forwardWeight;
        float tacticalImprovementThreshold = 0.06f * (1.0f - mind);
        // Exact: equals GRF's `powf(x, 2.0f)`.
        float tacticalDifferenceWeight = 1.0f + (mind * mind) * 10.0f;
        float passWeight = 1.0f;
        float passMinimum = 0.2f * (1.0f - mind) - longPossessionFactor * 0.1f;
        float totalPassWeight = tacticalDifferenceWeight + passWeight;
        float passThreshold = 0.1f - longPossessionFactor * 0.05f;
        const state::TacticalPlayerSituation &situation = playerState.tacticalSituation;
        float tacticalRating = situation.forwardSpaceRating * forwardSpaceWeight +
                               situation.spaceRating * spaceWeight +
                               situation.forwardRating * forwardWeight;
        tacticalRating /= totalTacticalWeight;

        PlayerList mates = TeamActivePlayers(world, teamID);
        float bestTotalRating = 0.0f;
        int bestPlayer = -1;
        float bestPassRating = 0.0f;
        int bestPassType = e_FunctionType_ShortPass;
        for (int mate : mates) {
            if (mate == player) continue;
            const state::TacticalPlayerSituation &mateSituation =
                match.players[mate].tacticalSituation;
            float mateTacticalRating = mateSituation.forwardSpaceRating * forwardSpaceWeight +
                                       mateSituation.spaceRating * spaceWeight +
                                       mateSituation.forwardRating * forwardWeight;
            mateTacticalRating /= totalTacticalWeight;
            if (StaticRole(mate) == e_PlayerRole_Goalkeeper) mateTacticalRating *= 0.7f;
            if (mateTacticalRating > tacticalRating + tacticalImprovementThreshold) {
                float tacticalDifferenceRating = mateTacticalRating - tacticalRating;
                float passingOddsShort =
                    PassingOddsTo(mate, e_FunctionType_ShortPass, opponentPlayerImages);
                float passingOddsLong =
                    PassingOddsTo(mate, e_FunctionType_LongPass, opponentPlayerImages);
                float passingOddsHigh =
                    PassingOddsTo(mate, e_FunctionType_HighPass, opponentPlayerImages);
                float passRating;
                int passType;
                if (passingOddsShort >= passingOddsLong && passingOddsShort >= passingOddsHigh) {
                    passRating = passingOddsShort;
                    passType = e_FunctionType_ShortPass;
                } else if (passingOddsLong >= passingOddsHigh) {
                    passRating = passingOddsLong;
                    passType = e_FunctionType_LongPass;
                } else {
                    passRating = passingOddsHigh;
                    passType = e_FunctionType_HighPass;
                }
                float totalRating = tacticalDifferenceRating * tacticalDifferenceWeight +
                                    passRating * passWeight - oneTouchIsHard;
                totalRating /= totalPassWeight;
                if (totalRating > bestTotalRating && totalRating > passThreshold &&
                    passRating > passMinimum) {
                    bestTotalRating = totalRating;
                    bestPlayer = mate;
                    bestPassRating = passRating;
                    bestPassType = passType;
                }
            }
        }

        if (mind < 0.25f) {
            float panicProneness = 1.0f - mind * 2.0f;
            float goalCloseness =
                1.0f -
                NormalizedClamp((Position() - Vector3(PITCH_HALF_LENGTH * side, 0, 0)).GetLength(),
                                2.0f, 16.0f);
            if (DynamicRole() != e_PlayerRole_Goalkeeper) {
                if ((bestPlayer < 0 || bestPassRating < panicProneness * goalCloseness) &&
                    controller.possessionAmount < 0.9f + panicProneness * goalCloseness * 0.8f)
                    AddPanicPass();
            } else {
                if (controller.possessionAmount < 3.0f) AddPanicPass();
            }
        }
        if (bestPlayer >= 0) AddPass(bestPlayer, bestPassType);

        float goalDistance = NormalizedClamp(
            (Vector3(PITCH_HALF_LENGTH * -side, 0, 0) - Position()).GetLength(), 0.0f, 32.0f);
        float idealShotPositionFactor =
            1.0f - NormalizedClamp(
                       (Vector3((PITCH_HALF_LENGTH - 7.0f) * -side, 0, 0) - Position()).GetLength(),
                       0.0f, 16.0f);
        idealShotPositionFactor = Curve(idealShotPositionFactor, 1.0f);
        if (idealShotPositionFactor > 0.1f) {
            float lowPostOdds = PassingOdds(Vector3((PITCH_HALF_LENGTH + 1.0f) * -side, -3.6f, 0),
                                            e_FunctionType_Shot, opponentPlayerImages, 3.0f);
            float centerOdds = PassingOdds(Vector3((PITCH_HALF_LENGTH + 1.0f) * -side, 0.0f, 0),
                                           e_FunctionType_Shot, opponentPlayerImages, 3.0f);
            float highPostOdds = PassingOdds(Vector3((PITCH_HALF_LENGTH + 1.0f) * -side, 3.6f, 0),
                                             e_FunctionType_Shot, opponentPlayerImages, 3.0f);
            float odds = centerOdds, y = 0.0f;
            if (lowPostOdds > odds) {
                odds = lowPostOdds;
                y = -3.5f;
            }
            if (highPostOdds > odds) {
                odds = highPostOdds;
                y = 3.5f;
            }
            odds = libm::powf(odds, 0.5f);
            if (odds + Random(0.0f, 0.5f) > 0.5f) {
                state::PlayerCommand command = DefaultCommand();
                command.desiredFunctionType = e_FunctionType_Shot;
                command.useDesiredMovement = false;
                command.useDesiredLookAt = false;
                command.desiredVelocityFloat = rawInputVelocityFloat;
                const float shot = Statistic(e_Statistic_TechnicalShot);
                float randomY = y + Random(-1.0f + shot, 1.0f - shot);
                command.touchInformation.desiredDirection =
                    (Vector3((PITCH_HALF_LENGTH + 1.0f) * -side, randomY, 0) -
                     (Position() + Movement() * 0.2f))
                        .GetNormalized(Vector3(-side, 0, 0));
                command.touchInformation.desiredDirection =
                    (command.touchInformation.desiredDirection * 0.7f +
                     -Direction() * (FloatVelocity() / SPRINT_VELOCITY) * 0.3f)
                        .GetNormalized();
                command.touchInformation.autoDirectionBias = 1.0f;
                command.touchInformation.desiredPower = Random(0.7f * (0.6f + goalDistance * 0.4f),
                                                               1.0f * (0.6f + goalDistance * 0.4f));
                queue.push_back(command);
            }
        }
        BestDribbleMovement(rawInputDirection, rawInputVelocityFloat);
    }

    CUJU_FUNCTION_NO_INLINE void AddCelebration() {
        int xSide = (Ball(0).Get2D().coordinates[0] > 0) ? 1 : -1;
        int ySide = side;
        const int lastTouchPlayer = Team().lastTouchPlayer;
        if (lastTouchPlayer >= 0) ySide = (Position(lastTouchPlayer).coordinates[1] > 0) ? 1 : -1;
        Vector3 celebrationPosition =
            Vector3(PITCH_HALF_LENGTH * xSide, PITCH_HALF_WIDTH * ySide, 0);
        Vector3 desiredDirection = (celebrationPosition - Position()).GetNormalized();
        float desiredVelocityFloat =
            ClampVelocity((celebrationPosition - Position()).GetLength() / 4.0f);
        Vector3 desiredLookAt = Position() + Direction() * 1000;
        int celebrationType = 1;
        if (match.lastGoalTeam != TeamSlot(world, teamID)) {
            celebrationType = 2;
            desiredVelocityFloat = IDLE_VELOCITY;
        }
        int madeGoal = 1;
        if (celebrationType == 1) {
            if (lastTouchPlayer == player) {
                madeGoal = 2;
                desiredVelocityFloat = SPRINT_VELOCITY;
            } else if (lastTouchPlayer >= 0) {
                if ((Position(lastTouchPlayer) - Position()).GetLength() < 20) {
                    celebrationPosition = Position(lastTouchPlayer);
                    desiredDirection =
                        ((Position(lastTouchPlayer) * 0.5f + celebrationPosition * 0.5f) -
                         Position())
                            .GetNormalized();
                    desiredVelocityFloat = ClampVelocity((
                        float)((double)(Position(lastTouchPlayer) - Position()).GetLength() / 2.0));
                } else {
                    desiredVelocityFloat = IDLE_VELOCITY;
                }
                desiredLookAt = Position() + Direction(lastTouchPlayer) * 100;
            }
        }
        const uint64_t stopTime = match.referee.buffer.stopTime;
        if (match.actualTimeMilliseconds - stopTime > 2000 &&
            match.actualTimeMilliseconds - stopTime < 4000) {
            state::PlayerCommand command = DefaultCommand();
            command.desiredFunctionType = e_FunctionType_Special;
            command.useSpecialVariable1 = true;
            command.specialVariable1 = celebrationType;
            command.useSpecialVariable2 = true;
            command.specialVariable2 = madeGoal;
            command.useDesiredMovement = false;
            command.useDesiredLookAt = false;
            queue.push_back(command);
        }
        state::PlayerCommand command = DefaultCommand();
        command.desiredFunctionType = e_FunctionType_Movement;
        command.useDesiredMovement = true;
        command.desiredDirection = desiredDirection;
        command.desiredVelocityFloat = desiredVelocityFloat;
        command.useDesiredLookAt = true;
        command.desiredLookAt = desiredLookAt;
        queue.push_back(command);
    }

    CUJU_FUNCTION_NO_INLINE void SetPieceCommands() {
        const int setPieceType = TeamAI().setPieceType;
        state::PlayerCommand actionCommand = DefaultCommand();
        if (setPieceType == e_GameMode_Penalty) {
            actionCommand.desiredFunctionType = e_FunctionType_Shot;
            actionCommand.useDesiredMovement = false;
            actionCommand.useDesiredLookAt = false;
            float randomY = Random(-5, 5);
            actionCommand.touchInformation.desiredDirection =
                (Vector3(-side * PITCH_HALF_LENGTH, randomY, 0) - Position())
                    .GetNormalized(Vector3(-side, 0, 0));
            actionCommand.touchInformation.desiredPower = Random(0.4f, 1.0f);
            queue.push_back(actionCommand);
        } else {
            actionCommand.desiredFunctionType = e_FunctionType_ShortPass;
            actionCommand.useDesiredMovement = false;
            actionCommand.useDesiredLookAt = false;
            actionCommand.touchInformation.inputDirection = Direction();
            actionCommand.touchInformation.inputPower = 0.5f;
            actionCommand.touchInformation.autoDirectionBias = 1.0f;
            actionCommand.touchInformation.autoPowerBias = 1.0f;
            actionCommand.touchInformation.forcedTargetPlayer = -1;
            Vector3 desiredTargetPosition;
            bool doCommand = true;
            if (setPieceType == e_GameMode_GoalKick) {
                if (Random(0.0f, 1.0f) > 0.4f && HumanGamerCount(Team()) == 0) {
                    actionCommand.desiredFunctionType = e_FunctionType_HighPass;
                    float randomY = Random(-PITCH_HALF_WIDTH, PITCH_HALF_WIDTH);
                    desiredTargetPosition =
                        Vector3((PITCH_HALF_LENGTH * -side) * 0.2f, randomY, 0.0f);
                } else {
                    actionCommand.desiredFunctionType = e_FunctionType_ShortPass;
                    float randomY = Random(-PITCH_HALF_WIDTH, PITCH_HALF_WIDTH);
                    desiredTargetPosition =
                        Vector3(Position().coordinates[0] * 0.9f, randomY, 0.0f);
                }
            } else if (setPieceType == e_GameMode_KickOff) {
                actionCommand.desiredFunctionType = e_FunctionType_ShortPass;
                desiredTargetPosition = Position() + Direction() * 1.0f;
            } else if (setPieceType == e_GameMode_FreeKick) {
                if (Random(0.0f, 1.0f) > 0.5f) {
                    actionCommand.desiredFunctionType = e_FunctionType_HighPass;
                    float randomY = Random(-10.0f, 10.0f);
                    desiredTargetPosition = Vector3(PITCH_HALF_LENGTH * -side, randomY, 0.0f);
                } else {
                    actionCommand.desiredFunctionType = e_FunctionType_ShortPass;
                    float randomY = Random(-10.0f, 10.0f);
                    desiredTargetPosition = Position() + Vector3(-side * 10.0f, randomY, 0.0f);
                }
            } else if (setPieceType == e_GameMode_Corner) {
                if (Random(0.0f, 1.0f) > 0.3f) {
                    actionCommand.desiredFunctionType = e_FunctionType_HighPass;
                    // GRF: arguments evaluate right to left, so randomY draws first.
                    float randomY = Random(-10.0f, 10.0f);
                    float randomX = Random(0.0f, 0.12f);
                    desiredTargetPosition =
                        Vector3((PITCH_HALF_LENGTH * -side) * (0.99f - randomX), randomY, 0.0f);
                } else {
                    actionCommand.desiredFunctionType = e_FunctionType_ShortPass;
                    desiredTargetPosition = Vector3((PITCH_HALF_LENGTH * -side) * 0.8f,
                                                    Position().coordinates[1] * 0.8f, 0.0f);
                }
            } else if (setPieceType == e_GameMode_ThrowIn) {
                actionCommand.desiredFunctionType = e_FunctionType_ShortPass;
                desiredTargetPosition = Position();
            } else if (match.ballRetainer == player) {
                actionCommand.desiredFunctionType = e_FunctionType_HighPass;
                float randomY = Random(-PITCH_HALF_WIDTH, PITCH_HALF_WIDTH);
                desiredTargetPosition = Vector3(PITCH_HALF_LENGTH * side, randomY, 0.0f);
                int targetPlayer =
                    GetClosestPlayer(world, context, teamID, desiredTargetPosition, false, player);
                if (targetPlayer >= 0) {
                    int closestOpponent = GetClosestPlayer(world, context, opponentID,
                                                           Position(targetPlayer), false, -1);
                    if (closestOpponent >= 0) {
                        if (((Position(closestOpponent) + Movement(closestOpponent) * 0.1f) -
                             Position(targetPlayer))
                                    .GetLength() > 10.0f ||
                            playerState.possessionDurationMilliseconds > 4000)
                            actionCommand.touchInformation.forcedTargetPlayer = targetPlayer;
                        else
                            doCommand = false;
                    }
                }
            }
            if (doCommand) {
                state::TouchInformation &touch = actionCommand.touchInformation;
                if (touch.forcedTargetPlayer < 0)
                    touch.forcedTargetPlayer = GetClosestPlayer(
                        world, context, teamID, desiredTargetPosition, false, player);
                GetPass(world, player, actionCommand.desiredFunctionType, touch.inputDirection,
                        touch.inputPower, touch.autoDirectionBias, touch.autoPowerBias,
                        touch.desiredDirection, touch.desiredPower, touch.targetPlayer,
                        touch.forcedTargetPlayer);
                queue.push_back(actionCommand);
            }
        }

        if (match.ballRetainer != player) {
            state::PlayerCommand command = DefaultCommand();
            command.desiredFunctionType = e_FunctionType_Movement;
            command.useDesiredMovement = true;
            command.useDesiredLookAt = true;
            BallControlMovement(WALK_VELOCITY, command.desiredDirection,
                                command.desiredVelocityFloat, command.desiredLookAt);
            queue.push_back(command);
        } else {
            state::PlayerCommand command = DefaultCommand();
            command.desiredFunctionType = e_FunctionType_Movement;
            command.useDesiredMovement = true;
            command.desiredDirection =
                (actionCommand.touchInformation.desiredDirection.GetLength() > 0.0f)
                    ? actionCommand.touchInformation.desiredDirection.Get2D().GetNormalized(
                          Direction())
                    : (Vector3(0) - Position()).GetNormalized(Direction());
            command.desiredVelocityFloat = IDLE_VELOCITY;
            command.useDesiredLookAt = true;
            command.desiredLookAt = Position() + command.desiredDirection * 10.0f;
            queue.push_back(command);
        }
    }

    CUJU_FUNCTION_NO_INLINE bool RequestCommand() {
        if (match.designatedPossessionPlayer < 0)
            return Unsupported("the AI controller finds no designated possession player");
        if (Team().designatedTeamPossessionPlayer < 0 ||
            OpponentTeam().designatedTeamPossessionPlayer < 0)
            return Unsupported("the AI controller finds no designated team possession player");
        mentalImageIndex = MentalImageIndex(world, controller.mentalImageTime);
        controller.lastSwitchTimeDurationMilliseconds = 0;
        controller.lastSwitchTimeMilliseconds = 0;
        playerState.desiredTimeToBallMilliseconds = 0;
        ControllerCalculateSituation(world, player, controller);
        controller.opponentPlayer = TeamBestPossessionPlayer(world, opponentID);
        controller.timeNeededMilliseconds = (float)playerState.timeNeededToGetToBallMilliseconds;

        Vector3 rawInputDirection = Direction();
        float rawInputVelocityFloat = IDLE_VELOCITY;
        Vector3 manualMovementDirection = Direction();
        float manualMovementVelocityFloat = IDLE_VELOCITY;
        bool manualMovement = false;
        bool extraHaste = false;
        // GRF: laziness is fixed at creation from the player's own entry, so a keeper's sending-off
        // that hands the entry to another player leaves it alone.
        const bool lazyPlayer = context.staticData->formation[player].lazy;
        const int role = StaticRole(player);
        const state::RefereeBufferState &refereeBuffer = match.referee.buffer;

        if (!match.inPlay && match.goalScored) {
            AddCelebration();
            return true;
        } else if (!match.inPlay && refereeBuffer.active &&
                   (match.referee.foul.foulType == 2 || match.referee.foul.foulType == 3) &&
                   refereeBuffer.stopTime < match.actualTimeMilliseconds - 1000 &&
                   refereeBuffer.prepareTime > match.actualTimeMilliseconds) {
            state::PlayerCommand command = DefaultCommand();
            command.desiredFunctionType = e_FunctionType_Movement;
            command.useDesiredMovement = true;
            command.useDesiredLookAt = true;
            command.desiredDirection = Direction();
            command.desiredVelocityFloat = IDLE_VELOCITY;
            command.desiredLookAt = match.officials[0].humanoid.spatialState.position;
            queue.push_back(command);
            return true;
        } else if (!match.inPlay || lazyPlayer) {
            state::PlayerCommand command = DefaultCommand();
            command.desiredFunctionType = e_FunctionType_Movement;
            command.useDesiredMovement = true;
            command.desiredDirection =
                (Direction() * 0.6f + (Vector3(0) - Position()).GetNormalized(Direction()) * 0.4f)
                    .GetNormalized(Direction());
            command.desiredVelocityFloat =
                ClampVelocity(FloatVelocity() * 0.95f - Random(0.0f, 3.2f));
            command.useDesiredLookAt = true;
            command.desiredLookAt =
                Position() +
                (Ball(0).Get2D() - Position()).GetNormalized(command.desiredDirection) * 10.0f;
            queue.push_back(command);
            return true;
        } else if ((match.inSetPiece && TeamAI().taker == player) || match.ballRetainer == player) {
            SetPieceCommands();
            return true;
        } else if (match.inPlay && !match.inSetPiece && match.ballRetainer != player &&
                   match.designatedPossessionPlayer != player && role != e_PlayerRole_Goalkeeper) {
            if (role == e_PlayerRole_LeftBack || role == e_PlayerRole_CentreBack ||
                role == e_PlayerRole_RightBack)
                FieldStrategyRequestInput(e_StrategyKind_Defense, rawInputDirection,
                                          rawInputVelocityFloat);
            else if (role == e_PlayerRole_DefensiveMidfielder ||
                     role == e_PlayerRole_LeftMidfielder ||
                     role == e_PlayerRole_CentralMidfielder ||
                     role == e_PlayerRole_RightMidfielder ||
                     role == e_PlayerRole_AttackingMidfielder)
                FieldStrategyRequestInput(e_StrategyKind_Midfield, rawInputDirection,
                                          rawInputVelocityFloat);
            else if (role == e_PlayerRole_CentreForward)
                FieldStrategyRequestInput(e_StrategyKind_Offense, rawInputDirection,
                                          rawInputVelocityFloat);
        } else if (match.inPlay && !match.inSetPiece &&
                   match.designatedPossessionPlayer == player &&
                   (HumanGamerCount(Team()) == 0 || (!ExternalControllerActive(world, player) &&
                                                     playerState.base.externalController >= 0)) &&
                   role != e_PlayerRole_Goalkeeper) {
            if (playerState.timeNeededToGetToBallMilliseconds < 1000)
                GetOnTheBallCommands(rawInputDirection, rawInputVelocityFloat);
            extraHaste = false;
        } else if (match.inPlay && !match.inSetPiece && role == e_PlayerRole_Goalkeeper) {
            CalculateIfBallIsBoundForGoal();
            bool boundForGoal = elizaState->ballBoundForGoal;
            if (boundForGoal) manualMovement = true;
            if (player != match.designatedPossessionPlayer) manualMovement = true;
            GoalieRequestInput(manualMovementDirection, manualMovementVelocityFloat);
            if (!controller.hasUniquePossession || controller.possessionAmount < 3.4f) {
                bool onlyPickupAnimations = false;
                if (!boundForGoal && controller.possessionAmount > 1.3f)
                    onlyPickupAnimations = true;
                KeeperDeflectCommand(onlyPickupAnimations);
            }
            if (playerState.timeNeededToGetToBallMilliseconds < 1000 &&
                match.designatedPossessionPlayer == player)
                GetOnTheBallCommands(rawInputDirection, rawInputVelocityFloat);
        }

        bool forceMagnet = false;
        if (match.inPlay && !match.inSetPiece &&
            (TeamAI().endApplyTeamPressureMilliseconds > match.actualTimeMilliseconds &&
             TeamAI().teamPressurePlayer == player) &&
            !controller.teamHasBestPossession && match.designatedPossessionPlayer != player &&
            role != e_PlayerRole_Goalkeeper)
            forceMagnet = true;

        controller.inputDirection = rawInputDirection;
        controller.inputVelocityFloat = rawInputVelocityFloat;
        elizaState->lastDesiredDirection =
            rawInputDirection.GetLength() != 0 ? rawInputDirection : Direction();
        elizaState->lastDesiredVelocity = rawInputVelocityFloat;

        if (match.inPlay && !match.inSetPiece) {
            BallControlCommand();
            TrapCommand();
            InterfereCommand(false);
            SlidingCommand();
        }

        if (!manualMovement && match.inPlay && !match.inSetPiece) {
            const int opponent = OpponentTeam().designatedTeamPossessionPlayer;
            if (player != match.designatedPossessionPlayer) {
                float mindSet = GetMindSet(DynamicRole());
                float huntDistanceThreshold = 10.0f + (1.0f - mindSet) * 10.0f;
                huntDistanceThreshold *=
                    0.5f * playerState.base.fatigueFactorInverse +
                    0.5f * (1.0f - NormalizedClamp(PlayerAverageVelocity(world, player, 10),
                                                   IDLE_VELOCITY, SPRINT_VELOCITY));
                huntDistanceThreshold *= 0.3f + TeamDifficulty(world, teamID) * 0.7f;
                if (forceMagnet) {
                    controller.inputDirection =
                        Vector3(side * PITCH_HALF_LENGTH, 0, 0) - Position();
                    controller.inputVelocityFloat = IDLE_VELOCITY;
                    controller.inputDirection.Normalize(Direction());
                } else if (!controller.teamHasBestPossession && playerState.manMarking < 0 &&
                           ((Position(opponent) + Movement(opponent) * 0.12f) -
                            (Position() + Movement() * 0.04f))
                                   .GetLength() < huntDistanceThreshold) {
                    if (player == Team().designatedTeamPossessionPlayer &&
                        controller.possessionAmount > 0.8f) {
                        forceMagnet = true;
                        extraHaste = true;
                    }
                    int huntingPlayerCount = 2;
                    PlayerList closestPlayers = GetClosestPlayers(
                        world, context, teamID, Position(opponent) + Movement(opponent) * 0.1f,
                        false, huntingPlayerCount);
                    bool close = false;
                    for (int candidate : closestPlayers)
                        if (candidate == player) {
                            close = true;
                            break;
                        }
                    if (close) {
                        Vector3 defendPosition = GetDefendPosition(opponent);
                        if (NeedDefendingMovement(side, Position(), defendPosition)) {
                            controller.inputDirection =
                                (defendPosition - Position())
                                    .GetNormalized(controller.inputDirection);
                            controller.inputVelocityFloat =
                                clamp((defendPosition - Position()).GetLength() *
                                          DISTANCE_TO_VELOCITY_MULTIPLIER,
                                      0.0f, SPRINT_VELOCITY);
                            forceMagnet = true;
                        }
                    }
                }
            }
            controller.inputVelocityFloat = RangeVelocity(controller.inputVelocityFloat);
            MovementCommand(forceMagnet, extraHaste);
        } else {
            state::PlayerCommand command = DefaultCommand();
            command.desiredFunctionType = e_FunctionType_Movement;
            command.useDesiredMovement = true;
            command.desiredDirection = manualMovementDirection;
            command.desiredVelocityFloat = manualMovementVelocityFloat;
            command.useDesiredLookAt = true;
            command.desiredLookAt =
                Position() +
                (MentalBallPredictionAt(40).Get2D() - Position()).GetNormalized(0) * 10.0f;
            queue.push_back(command);
        }
        return true;
    }
};

CUJU_FUNCTION inline void TeamApplyTeamPressure(World &world, const SimulationContext &context,
                                                int teamID) {
    state::TeamAIState &teamAI = TeamByID(world, teamID).teamController;
    teamAI.endApplyTeamPressureMilliseconds = world.serialized.match.actualTimeMilliseconds + 500;
    int opponent = TeamBestPossessionPlayer(world, 1 - teamID);
    const state::SpatialState &opponentState = Humanoid(world, opponent).spatialState;
    Vector3 opponentPosition = opponentState.position + opponentState.movement * 0.24f;
    teamAI.teamPressurePlayer =
        GetClosestPlayer(world, context, teamID,
                         opponentPosition + Vector3(TeamDynamicSide(world, teamID) * 1.0f, 0, 0),
                         true, TeamGoalie(world, context, teamID));
    if (teamAI.teamPressurePlayer >= 0)
        world.serialized.match.players[teamAI.teamPressurePlayer].manMarking = opponent;
}
CUJU_FUNCTION inline void TeamApplyKeeperRush(World &world, int teamID) {
    TeamByID(world, teamID).teamController.endApplyKeeperRushMilliseconds =
        world.serialized.match.actualTimeMilliseconds + 300;
}

CUJU_FUNCTION_NO_INLINE inline bool HumanRequestCommand(World &world,
                                                        const SimulationContext &context,
                                                        int player, CommandQueue &queue) {
    CUJU_PROFILE_SCOPE(HUMAN_REQUEST_COMMAND);
    state::HumanControllerState &humanController = HumanGamerOf(world, player).controller;
    state::PlayerControllerState &controller = humanController.base;
    AIController eliza(world, context, player, queue, controller, nullptr);
    state::MatchState &match = world.serialized.match;
    state::PlayerState &playerState = match.players[player];
    const int teamID = world.playerTeamID[player];
    const state::TeamAIState &teamAI = TeamByID(world, teamID).teamController;
    auto Button = [&](int button) { return humanController.keyboard.buttonsPressed[button]; };
    auto ResetAction = [&]() {
        humanController.actionMode = 0;
        humanController.gaugeMilliseconds = 0;
        humanController.actionBufferTimeMilliseconds = 0;
    };

    eliza.mentalImageIndex = MentalImageIndex(world, controller.mentalImageTime);
    playerState.desiredTimeToBallMilliseconds = 0;
    controller.opponentPlayer = TeamBestPossessionPlayer(world, 1 - teamID);
    controller.timeNeededMilliseconds = (float)playerState.timeNeededToGetToBallMilliseconds;

    Vector3 rawInputDirection;
    float rawInputVelocityFloat = 0;
    HumanGetKeyboardInput(world, player, rawInputDirection, rawInputVelocityFloat);
    controller.inputDirection = rawInputDirection;
    controller.inputVelocityFloat = rawInputVelocityFloat;

    const state::AnimationState &currentAnimation = Humanoid(world, player).currentAnimation;
    const int functionType = currentAnimation.functionType;
    const bool touchPending = currentAnimation.frameNumber < currentAnimation.touchFrame;
    if (humanController.actionMode == 2 &&
        (functionType == e_FunctionType_ShortPass || functionType == e_FunctionType_LongPass ||
         functionType == e_FunctionType_HighPass || functionType == e_FunctionType_Shot) &&
        !touchPending)
        ResetAction();
    if (humanController.actionMode == 1 &&
        (functionType == e_FunctionType_Sliding || functionType == e_FunctionType_Interfere) &&
        !touchPending)
        ResetAction();
    if (humanController.actionMode == 2 && humanController.actionButton == e_ButtonFunction_Shot &&
        Button(e_ButtonFunction_ShortPass) && !match.inSetPiece)
        ResetAction();
    if (humanController.actionMode == 2 &&
        humanController.actionButton == e_ButtonFunction_HighPass &&
        Button(e_ButtonFunction_ShortPass) && !match.inSetPiece)
        ResetAction();
    if (humanController.actionMode == 2 && !match.inSetPiece &&
        (humanController.actionBufferTimeMilliseconds > 2000 ||
         playerState.timeNeededToGetToBallMilliseconds >
             playerState.timeNeededToGetToBallPreviousMilliseconds + 700 ||
         (currentAnimation.functionType == e_FunctionType_Interfere)))
        ResetAction();
    if (humanController.actionMode == 1 && humanController.actionBufferTimeMilliseconds > 1000)
        ResetAction();

    if (humanController.actionMode == 2) {
        if (!Button(humanController.actionButton) ||
            (Button(humanController.actionButton) && humanController.gaugeMilliseconds > 500) ||
            (!playerState.hasPossession && !match.inSetPiece &&
             humanController.actionBufferTimeMilliseconds > 0)) {
            int baseTimeMilliseconds = 60;
            float gaugeFactor = (float)(humanController.gaugeMilliseconds - baseTimeMilliseconds) *
                                (1.0f / float(1000 - baseTimeMilliseconds));
            gaugeFactor = clamp(gaugeFactor, 0.0f, 1.0f);
            auto Pass = [&](int passType, float exponent, float autoDirection, float autoPower) {
                state::PlayerCommand command = AIController::DefaultCommand();
                command.desiredFunctionType = passType;
                command.useDesiredMovement = false;
                command.useDesiredLookAt = false;
                float inputPower = clamp(libm::powf(gaugeFactor, exponent), 0.01f, 1.0f);
                state::TouchInformation &touch = command.touchInformation;
                touch.inputDirection = controller.inputDirection;
                touch.inputPower = inputPower;
                touch.autoDirectionBias = autoDirection;
                touch.autoPowerBias = autoPower;
                GetPass(world, player, passType, touch.inputDirection, touch.inputPower,
                        touch.autoDirectionBias, touch.autoPowerBias, touch.desiredDirection,
                        touch.desiredPower, touch.targetPlayer);
                queue.push_back(command);
            };
            // GRF: the exponents and biases are the default gameplay settings.
            if (match.inSetPiece && teamAI.taker == player &&
                teamAI.setPieceType == e_GameMode_KickOff) {
                state::PlayerCommand command = AIController::DefaultCommand();
                command.desiredFunctionType = e_FunctionType_ShortPass;
                state::TouchInformation &touch = command.touchInformation;
                touch.autoDirectionBias = 1.0f;
                touch.autoPowerBias = 1.0f;
                touch.inputDirection = eliza.Direction();
                touch.inputPower = 0.1f;
                Vector3 desiredTargetPosition = eliza.Position() + eliza.Direction() * 1.0f;
                touch.forcedTargetPlayer =
                    GetClosestPlayer(world, context, teamID, desiredTargetPosition, false, player);
                GetPass(world, player, command.desiredFunctionType, touch.inputDirection,
                        touch.inputPower, touch.autoDirectionBias, touch.autoPowerBias,
                        touch.desiredDirection, touch.desiredPower, touch.targetPlayer,
                        touch.forcedTargetPlayer);
                queue.push_back(command);
            } else if (humanController.actionButton == e_ButtonFunction_ShortPass) {
                Pass(e_FunctionType_ShortPass, 0.7f, 0.4f, 0.7f);
            } else if (humanController.actionButton == e_ButtonFunction_LongPass) {
                Pass(e_FunctionType_LongPass, 0.65f, 0.2f, 0.7f);
            } else if (humanController.actionButton == e_ButtonFunction_HighPass) {
                Pass(e_FunctionType_HighPass, 0.55f, 0.2f, 0.5f);
            } else if (humanController.actionButton == e_ButtonFunction_Shot) {
                state::PlayerCommand command = AIController::DefaultCommand();
                command.desiredFunctionType = e_FunctionType_Shot;
                command.useDesiredMovement = false;
                command.useDesiredLookAt = false;
                command.desiredVelocityFloat = controller.inputVelocityFloat;
                command.touchInformation.inputDirection = controller.inputDirection;
                command.touchInformation.autoDirectionBias = 1.0f;
                command.touchInformation.desiredDirection =
                    GetShotDirection(world, player, command.touchInformation.inputDirection,
                                     command.touchInformation.autoDirectionBias);
                command.touchInformation.desiredPower =
                    clamp(libm::powf(gaugeFactor, 0.6f), 0.01f, 1.0f);
                queue.push_back(command);
            }
        }
    } else if (humanController.actionMode == 1) {
        if (Button(humanController.actionButton)) {
            if (humanController.actionButton == e_ButtonFunction_Sliding) {
                state::PlayerCommand command = AIController::DefaultCommand();
                command.desiredFunctionType = e_FunctionType_Sliding;
                command.useDesiredMovement = true;
                command.desiredDirection = controller.inputDirection;
                command.desiredVelocityFloat = controller.inputVelocityFloat;
                command.useDesiredLookAt = true;
                command.desiredLookAt =
                    eliza.Position() + eliza.Movement() * 0.1f + command.desiredDirection * 10.0f;
                queue.push_back(command);
            }
            if (humanController.actionButton == e_ButtonFunction_TeamPressure)
                TeamApplyTeamPressure(world, context, teamID);
            if (humanController.actionButton == e_ButtonFunction_KeeperRush)
                TeamApplyKeeperRush(world, teamID);
        } else {
            humanController.actionMode = 0;
        }
    }

    if ((match.inSetPiece && teamAI.taker == player &&
         (humanController.actionMode != 2 ||
          (humanController.actionMode == 2 && Button(humanController.actionButton)) ||
          match.ballRetainer == player)) ||
        (match.inSetPiece && teamAI.taker != player && match.ballRetainer < 0)) {
        eliza.SetPieceCommand();
        return true;
    }

    Vector3 savedNonsteadyInputDirection = controller.inputDirection;
    controller.inputDirection = humanController.steadyDirection;

    if (match.inPlay && !match.inSetPiece) {
        bool idleTurnToOpponentGoal = false;
        bool knockOn = false;
        if (Button(e_ButtonFunction_Dribble)) idleTurnToOpponentGoal = true;
        if (Button(e_ButtonFunction_Dribble) && Button(e_ButtonFunction_Sprint)) knockOn = true;
        Vector3 savedInputDirection = controller.inputDirection;
        float savedInputVelocity = controller.inputVelocityFloat;
        if (humanController.actionMode == 2) {
            float dot = eliza.Direction().GetDotProduct(controller.inputDirection) * 0.5f + 0.5f;
            if (dot < 0) dot = 0;
            if (eliza.SpatialStateOf(player).velocityCategory != e_Velocity_Idle)
                controller.inputDirection =
                    (eliza.Direction() * 1.0f + controller.inputDirection * 0.0f)
                        .GetNormalized(controller.inputDirection);
            dot = libm::powf(dot, 1.5f);
            dot = dot * 0.8f + 0.2f;
            dot *= 0.9f;
            controller.inputVelocityFloat = eliza.FloatVelocity() * dot;
        }
        eliza.BallControlCommand(idleTurnToOpponentGoal, knockOn, true, false);
        eliza.TrapCommand(idleTurnToOpponentGoal, knockOn);
        if (humanController.actionMode == 2) {
            controller.inputDirection = savedInputDirection;
            controller.inputVelocityFloat = savedInputVelocity;
        }
        bool byAnyMeans = false;
        if (Button(e_ButtonFunction_Pressure)) byAnyMeans = true;
        eliza.InterfereCommand(byAnyMeans);
    }

    bool forceMagnet = false;
    bool extraHaste = false;
    if (humanController.actionMode != 2 && Button(e_ButtonFunction_Pressure)) {
        forceMagnet = true;
        extraHaste = true;
    }
    if (humanController.actionMode == 2) {
        forceMagnet = true;
        extraHaste = true;
    }
    eliza.MovementCommand(forceMagnet, extraHaste);

    if (!queue.empty()) {
        state::PlayerCommand &command = queue.back();
        if (!world.serialized.scenario.useMagnet) {
            command.desiredDirection = controller.inputDirection;
            command.desiredVelocityFloat = controller.inputVelocityFloat;
            if (command.desiredVelocityFloat < IDLE_DRIBBLE_SWITCH)
                command.desiredDirection =
                    (eliza.MentalBallPredictionAt(500).Get2D() - eliza.Position())
                        .GetNormalized(controller.inputDirection);
            command.desiredLookAt = eliza.MentalBallPredictionAt(500).Get2D();
        }
        if (match.inPlay && !match.inSetPiece && Button(e_ButtonFunction_Dribble) &&
            Button(e_ButtonFunction_Sprint)) {
            if (!controller.hasBestPossession) {
                command.desiredDirection = controller.inputDirection;
                command.desiredVelocityFloat = controller.inputVelocityFloat;
            }
        }
    }
    controller.inputDirection = savedNonsteadyInputDirection;
    return true;
}

CUJU_FUNCTION inline bool PlayerRequestCommand(World &world, const SimulationContext &context,
                                               int player, CommandQueue &queue);

CUJU_FUNCTION inline bool ElizaRequestCommand(World &world, const SimulationContext &context,
                                              int player, CommandQueue &queue) {
    CUJU_PROFILE_SCOPE(AI_CONTROLLER_REQUEST_COMMAND);
    state::AIControllerState &elizaState = world.serialized.match.players[player].base.controller;
    AIController eliza(world, context, player, queue, elizaState.base, &elizaState);
    return eliza.RequestCommand();
}

CUJU_FUNCTION inline bool PlayerRequestCommand(World &world, const SimulationContext &context,
                                               int player, CommandQueue &queue) {
    CUJU_PROFILE_SCOPE(REQUEST_COMMAND);
    if (ExternalControllerActive(world, player))
        return HumanRequestCommand(world, context, player, queue);
    return ElizaRequestCommand(world, context, player, queue);
}

}  // namespace simulation
}  // namespace cuju

// The team AI uses the controllers above, so its header comes last.
#include "team_ai.h"

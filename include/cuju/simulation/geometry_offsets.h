// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The pose offsets that the animation of a player applies on top of its frames.
#pragma once

#include "../profiling.h"

#include "humanoid.h"
#include "select_animation.h"

namespace cuju {
namespace simulation {

// The offset blends the body part toward `orientation`, with `bias` from 0 to 1 as its weight.
CUJU_FUNCTION inline void SetOffset(HumanoidState &humanoid, int bodyPart, float bias,
                                    const Quaternion &orientation) {
    humanoid.offsets.elements[bodyPart].bias = bias;
    humanoid.offsets.elements[bodyPart].orientation = orientation;
}

CUJU_FUNCTION_NO_INLINE inline int TeamBestPossessionPlayer(const World &world, int teamID) {
    int bestTimeMilliseconds = 10000000;
    int bestPlayer = -1;
    for (int player = 0; player < world.serialized.match.playerCount; player++) {
        if (world.playerTeamID[player] != teamID ||
            !world.serialized.match.players[player].base.isActive)
            continue;
        int timeMilliseconds =
            (int)world.serialized.match.players[player].timeNeededToGetToBallMilliseconds;
        if (timeMilliseconds < bestTimeMilliseconds) {
            bestTimeMilliseconds = timeMilliseconds;
            bestPlayer = player;
        }
    }
    return bestPlayer;
}

CUJU_FUNCTION inline Quaternion AngleAxis(float angle, const Vector3 &axis) {
    Quaternion quaternion;
    quaternion.SetAngleAxis(angle, axis);
    return quaternion;
}

// Runs on one lane.
CUJU_FUNCTION_NO_INLINE inline void CalculateGeometryOffsets(World &world,
                                                             const SimulationContext &context,
                                                             int player) {
    CUJU_PROFILE_SCOPE(CALCULATE_GEOMETRY_OFFSETS);
    (void)context;
    HumanoidState &humanoid = world.serialized.match.players[player].base.humanoid;
    const state::SpatialState &spatialState = humanoid.spatialState;
    const Quaternion identity(0, 0, 0, 1);
    for (int bodyPart :
         {e_BodyPart_Middle, e_BodyPart_Neck, e_BodyPart_LeftThigh, e_BodyPart_RightThigh,
          e_BodyPart_LeftKnee, e_BodyPart_RightKnee, e_BodyPart_LeftAnkle, e_BodyPart_RightAnkle,
          e_BodyPart_LeftShoulder, e_BodyPart_RightShoulder, e_BodyPart_LeftElbow,
          e_BodyPart_RightElbow, e_BodyPart_Body})
        SetOffset(humanoid, bodyPart, 0.0, identity);

    const float adaptLegsToTrueVelocityInfluence = 0.7f;
    const float adaptBodyToBallPositionInfluence = 0.5f;
    const float adaptArmsToOpponentInfluence = 0.9f;

    if (!(world.serialized.match.inPlay && world.serialized.match.ballRetainer != player)) return;
    if (humanoid.currentAnimation.functionType != e_FunctionType_Movement) return;

    {
        float actualVelocity = spatialState.actualMovement.GetLength();
        float animationVelocity = spatialState.animationMovement.GetLength();
        float velocityFactor = (actualVelocity + 0.2f) / (animationVelocity + 0.2f);
        float allowFasterFactor = 0.0f;
        if (spatialState.actualMovement.GetLength() > 7.0f) allowFasterFactor += 0.1;
        velocityFactor = clamp(velocityFactor, 0.1f, 1.0f + allowFasterFactor);
        float bendFactor = 0.8f;
        Quaternion defaultLeftHipOrientation =
            AngleAxis(-0.15f * PI * bendFactor, Vector3(1, 0.06, -0.12).GetNormalized());
        Quaternion defaultRightHipOrientation =
            AngleAxis(-0.15f * PI * bendFactor, Vector3(1, -0.06, 0.12).GetNormalized());
        Quaternion defaultKneeOrientation = AngleAxis(0.3f * PI * bendFactor, Vector3(1, 0, 0));
        Quaternion defaultAnkleOrientation = AngleAxis(-0.1f * PI * bendFactor, Vector3(1, 0, 0));
        float legBias = (1.0f - velocityFactor) * adaptLegsToTrueVelocityInfluence;
        SetOffset(humanoid, e_BodyPart_LeftThigh, legBias, defaultLeftHipOrientation);
        SetOffset(humanoid, e_BodyPart_RightThigh, legBias, defaultRightHipOrientation);
        SetOffset(humanoid, e_BodyPart_LeftKnee, legBias, defaultKneeOrientation);
        SetOffset(humanoid, e_BodyPart_RightKnee, legBias, defaultKneeOrientation);
        SetOffset(humanoid, e_BodyPart_LeftAnkle, legBias, defaultAnkleOrientation);
        SetOffset(humanoid, e_BodyPart_RightAnkle, legBias, defaultAnkleOrientation);
    }

    if (!world.serialized.match.players[player].hasPossession) {
        Vector3 toBall =
            (MentalBallPrediction(world, MentalImageIndex(world, humanoid.mentalImageTime), 10)
                 .Get2D() -
             spatialState.position)
                .GetNormalized(spatialState.directionVector);
        float angle =
            toBall.GetAngle2D(spatialState.relativeBodyDirectionVectorNonQuantized.GetRotated2D(
                spatialState.angle.value()));
        angle = clamp(fabsf(angle) - 0.05f * PI, 0.0f, PI) * SignSide(angle);
        float lookAtBallBias = libm::powf(1.0f - fabsf(angle / PI), 0.3f);
        lookAtBallBias = clamp(lookAtBallBias * adaptBodyToBallPositionInfluence, 0.0f, 1.0f);
        float ballDistance = GetDistance(BallPredict(world.serialized.match.ball, 100).Get2D(),
                                         spatialState.position);
        float ballProximityFalloff = 1.6f;
        if (ballDistance < ballProximityFalloff)
            lookAtBallBias *=
                NormalizedClamp(ballDistance, ballProximityFalloff * 0.4f, ballProximityFalloff);
        float velocityFraction =
            NormalizedClamp(spatialState.floatVelocity, IDLE_VELOCITY, SPRINT_VELOCITY - 0.5f);
        // Exact: the square is one multiplication.
        lookAtBallBias *= 1.0f - velocityFraction * velocityFraction * 0.3f;
        Quaternion middleOrientation = AngleAxis(angle, Vector3(0, 0, 1));
        middleOrientation.Normalize();
        SetOffset(humanoid, e_BodyPart_Middle, lookAtBallBias * 0.3f, middleOrientation);
        Quaternion headOrientation = AngleAxis(angle, Vector3(0, 0, 1));
        headOrientation.Normalize();
        SetOffset(humanoid, e_BodyPart_Neck, lookAtBallBias * 0.7f, headOrientation);
    }

    {
        const int teamID = world.playerTeamID[player];
        int opponent = TeamBestPossessionPlayer(world, 1 - teamID);
        // GRF: a team without an active player leaves a null pointer here.
        if (opponent < 0) CUJU_FATAL("a player has no active opponent");
        const state::SpatialState &opponentSpatialState = Humanoid(world, opponent).spatialState;
        if ((double)(opponentSpatialState.position - spatialState.position).GetLength() < 1.4 &&
            (double)opponentSpatialState.directionVector.GetDotProduct(
                spatialState.directionVector) > 0.0) {
            Vector3 baseRotationVector = Vector3(0, -1, 0)
                                             .GetRotated2D(spatialState.angle.value())
                                             .GetRotated2D(spatialState.relativeBodyAngle.value());
            Vector3 opponentVector = opponentSpatialState.position - spatialState.position;
            float angle =
                baseRotationVector.GetAngle2D(opponentVector.GetNormalized(baseRotationVector));
            // Exact: the angle bounds compare in double.
            const double absoluteAngle = fabsf(angle);
            const double doublePi = (double)PI;
            if (absoluteAngle > 0.6 * doublePi && absoluteAngle < 0.85 * doublePi) {
                Quaternion shoulder, elbow;
                if (angle > 0) {
                    shoulder = AngleAxis((float)(0.4 * doublePi), Vector3(0, 1, 0));
                    elbow = AngleAxis(0, Vector3(0, 1, 0));
                    SetOffset(humanoid, e_BodyPart_RightShoulder,
                              0.8f * adaptArmsToOpponentInfluence, shoulder.GetNormalized());
                    SetOffset(humanoid, e_BodyPart_RightElbow, 0.7f * adaptArmsToOpponentInfluence,
                              elbow.GetNormalized());
                } else {
                    shoulder = AngleAxis((float)(-0.4 * doublePi), Vector3(0, 1, 0));
                    elbow = AngleAxis(0, Vector3(0, 1, 0));
                    SetOffset(humanoid, e_BodyPart_LeftShoulder,
                              0.8f * adaptArmsToOpponentInfluence, shoulder.GetNormalized());
                    SetOffset(humanoid, e_BodyPart_LeftElbow, 0.7f * adaptArmsToOpponentInfluence,
                              elbow.GetNormalized());
                }
                Quaternion middleOrientation = AngleAxis(0.2f * PI, Vector3(1, 0, 0));
                SetOffset(humanoid, e_BodyPart_Middle, 0.3f * adaptArmsToOpponentInfluence,
                          middleOrientation.GetNormalized());
            }
            if (absoluteAngle > 0.2 * doublePi && absoluteAngle <= 0.6 * doublePi) {
                Quaternion shoulder, elbow;
                if (angle > 0) {
                    shoulder = AngleAxis((float)(0.4 * doublePi), Vector3(0, 1, 0));
                    elbow = AngleAxis((float)(-0.5 * doublePi), Vector3(1, 0, 0));
                    SetOffset(humanoid, e_BodyPart_RightShoulder,
                              0.7f * adaptArmsToOpponentInfluence, shoulder.GetNormalized());
                    SetOffset(humanoid, e_BodyPart_RightElbow, 0.8f * adaptArmsToOpponentInfluence,
                              elbow.GetNormalized());
                } else {
                    shoulder = AngleAxis((float)(-0.4 * doublePi), Vector3(0, 1, 0));
                    elbow = AngleAxis((float)(-0.5 * doublePi), Vector3(1, 0, 0));
                    SetOffset(humanoid, e_BodyPart_LeftShoulder,
                              0.7f * adaptArmsToOpponentInfluence, shoulder.GetNormalized());
                    SetOffset(humanoid, e_BodyPart_LeftElbow, 0.8f * adaptArmsToOpponentInfluence,
                              elbow.GetNormalized());
                }
            }
            // GRF: this bound is 0.2 radians, not `0.2 * PI`.
            if (absoluteAngle < 0.2) {
                Quaternion shoulder = AngleAxis((float)(-0.4 * doublePi), Vector3(1, 0, 0));
                Quaternion elbow = AngleAxis((float)(-0.2 * doublePi), Vector3(1, 0, 0));
                if (angle > 0) {
                    SetOffset(humanoid, e_BodyPart_RightShoulder,
                              0.6f * adaptArmsToOpponentInfluence, shoulder.GetNormalized());
                    SetOffset(humanoid, e_BodyPart_RightElbow, 0.8f * adaptArmsToOpponentInfluence,
                              elbow.GetNormalized());
                } else {
                    SetOffset(humanoid, e_BodyPart_LeftShoulder,
                              0.6f * adaptArmsToOpponentInfluence, shoulder.GetNormalized());
                    SetOffset(humanoid, e_BodyPart_LeftElbow, 0.8f * adaptArmsToOpponentInfluence,
                              elbow.GetNormalized());
                }
            }
        }
    }
}

}  // namespace simulation
}  // namespace cuju

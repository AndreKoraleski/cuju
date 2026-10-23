// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The spatial state of a humanoid and the player helpers that its tick reads.
#pragma once

#include "../profiling.h"

#include <functional>

#include "../animation/animation_data.h"
#include "../animation/animation_derived.h"
#include "../animation/skeleton.h"
#include "../animation/keyframes.h"
#include "../libm/pow.h"
#include "../game_definitions.h"
#include "../math_utilities.h"
#include "../x86.h"
#include "access.h"
#include "containers.h"
#include "enumerations.h"
#include "static_data.h"
#include "world.h"

namespace cuju {
namespace simulation {

using animation::AnimationCollection;

// Indexes the body parts of an animation and the node map of a skeleton. `e_BodyPart_Body` holds
// the orientation of the body and `e_BodyPart_Player` the position of the player.
enum BodyPart {
    e_BodyPart_Middle,
    e_BodyPart_Neck,
    e_BodyPart_LeftThigh,
    e_BodyPart_RightThigh,
    e_BodyPart_LeftKnee,
    e_BodyPart_RightKnee,
    e_BodyPart_LeftAnkle,
    e_BodyPart_RightAnkle,
    e_BodyPart_LeftShoulder,
    e_BodyPart_RightShoulder,
    e_BodyPart_LeftElbow,
    e_BodyPart_RightElbow,
    e_BodyPart_Body,
    e_BodyPart_Player
};
enum InterruptAnimation {
    e_InterruptAnimation_None,
    e_InterruptAnimation_Switch,
    e_InterruptAnimation_Sliding,
    e_InterruptAnimation_Bump,
    e_InterruptAnimation_Trip,
    e_InterruptAnimation_Cheat,
    e_InterruptAnimation_Cancel,
    e_InterruptAnimation_ReQueue
};

inline const char *&UnsupportedReason() {
    static thread_local const char *reason = "";
    return reason;
}
#if defined(__CUDACC__)
static __device__ const char *deviceUnsupportedReason;
#endif
// Records why the simulation reached a path it does not support. The caller returns the `false`
// it gives, and the world is then unspecified.
CUJU_FUNCTION inline bool Unsupported(const char *reason) {
#if !defined(__CUDA_ARCH__)
    UnsupportedReason() = reason;
#else
    deviceUnsupportedReason = reason;
#endif
    return false;
}

// A host replacement for the controller of a player. It fills the queue, applies the state
// changes of the controller and returns `false` if it cannot.
using RequestCommandFunction = std::function<bool(World &, int player, CommandQueue &)>;

struct SimulationContext {
    const StaticData *staticData;
    const AnimationCollection *animations;
    const animation::AnimationDerived *derived;
    // The skeletons of the players, then those of the officials.
    const animation::Skeleton *skeletons = nullptr;
    // A host source of commands in place of the controllers of the simulation, or null to run them.
    const RequestCommandFunction *requestCommand = nullptr;
    // `CameraWeight` laid out as `CAMERA_WEIGHTS` says, or null. Exact: the same function fills the
    // table, so it gives the bits of the computation.
    const float *cameraWeights = nullptr;
    // The animations that `CheckFunctionType` accepts for function type `f`, in increasing order,
    // from `functionTypeOffsets[f]` up to `functionTypeOffsets[f + 1]`.
    const int16_t *functionTypeAnimations = nullptr;
    const struct CrudeRecord *crude = nullptr;
    // `FUNCTION_TYPE_SLOTS + 1` entries.
    const int32_t *functionTypeOffsets = nullptr;
    // GRF: `physics_steps_per_frame`.
    int ticksPerStep = 10;
    // Null computes the values of the selection where they are read.
    const struct SelectionRecord *selection = nullptr;
    // `CalculateMovementAtFrame(positions, frame, 1)` of every frame of every animation with two
    // positions or more, from `SelectionRecord::smoothedMovementOffset` on, or null.
    const Vector3 *smoothedMovement = nullptr;
};

CUJU_FUNCTION inline int FormationSource(const World &world, int player) {
    const int formationOverride = world.formationOverride[player];
    return formationOverride ? formationOverride - 1 : player;
}
CUJU_FUNCTION inline const StaticFormation &Formation(const World &world,
                                                      const SimulationContext &context,
                                                      int player) {
    return context.staticData->formation[FormationSource(world, player)];
}

CUJU_FUNCTION inline void SetRadian(Radian &radian, float value) {
    radian.angle = value;
    radian.rotated = false;
}

CUJU_FUNCTION inline float FixAngle(float angle) {
    return ModulateIntoRange(-PI, PI, angle + 0.5f * PI);
}

CUJU_FUNCTION inline float RangeVelocity(float velocity) {
    float rangedVelocity = IDLE_VELOCITY;
    if (velocity >= IDLE_DRIBBLE_SWITCH && velocity < DRIBBLE_WALK_SWITCH)
        rangedVelocity = DRIBBLE_VELOCITY;
    else if (velocity >= DRIBBLE_WALK_SWITCH && velocity < WALK_SPRINT_SWITCH)
        rangedVelocity = WALK_VELOCITY;
    else if (velocity >= WALK_SPRINT_SWITCH)
        rangedVelocity = SPRINT_VELOCITY;
    return rangedVelocity;
}
CUJU_FUNCTION inline int FloatToVelocityCategory(float velocity) {
    float rangedVelocity = RangeVelocity(velocity);
    if (rangedVelocity == IDLE_VELOCITY) return e_Velocity_Idle;
    if (rangedVelocity == DRIBBLE_VELOCITY) return e_Velocity_Dribble;
    if (rangedVelocity == WALK_VELOCITY) return e_Velocity_Walk;
    if (rangedVelocity == SPRINT_VELOCITY) return e_Velocity_Sprint;
    return e_Velocity_Idle;
}

CUJU_FUNCTION inline float VelocityCategoryToFloat(int category) {
    switch (category) {
        case e_Velocity_Idle:
            return IDLE_VELOCITY;
        case e_Velocity_Dribble:
            return DRIBBLE_VELOCITY;
        case e_Velocity_Walk:
            return WALK_VELOCITY;
        case e_Velocity_Sprint:
            return SPRINT_VELOCITY;
    }
    return 0;
}

// The quantized velocity, not `SpatialState::floatVelocity`.
CUJU_FUNCTION inline float PlayerFloatVelocity(const World &world, int player) {
    return VelocityCategoryToFloat(Humanoid(world, player).spatialState.velocityCategory);
}

// The body directions allowed relative to the direction of movement, as `ALLOWED_BODY_DIRECTIONS`
// holds them. Exact: each angle is a multiple of pi computed in double and rounded to float.
CUJU_FUNCTION inline void AllowedBodyDirectionVectors(Vector3 out[5]) {
    out[0] = Vector3(0, -1, 0);
    out[1] = Vector3(0, -1, 0).GetRotated2D((float)(-0.25 * (double)PI));
    out[2] = Vector3(0, -1, 0).GetRotated2D((float)(0.25 * (double)PI));
    out[3] = Vector3(0, -1, 0).GetRotated2D((float)(-0.75 * (double)PI));
    out[4] = Vector3(0, -1, 0).GetRotated2D((float)(0.75 * (double)PI));
}

constexpr int NO_ALLOWED_BODY_DIRECTION = 5;

// The vectors of `AllowedBodyDirectionVectors`, then the zero vector, `NO_ALLOWED_BODY_DIRECTION`.
CUJU_TABLE_DEFINITION(Vector3, ALLOWED_BODY_DIRECTIONS,
                      {
                          {0x0p+0f, -0x1p+0f, 0x0p+0f},
                          {-0x1.6a09e6p-1f, -0x1.6a09e6p-1f, 0x0p+0f},
                          {0x1.6a09e6p-1f, -0x1.6a09e6p-1f, 0x0p+0f},
                          {-0x1.6a09e6p-1f, 0x1.6a09e6p-1f, 0x0p+0f},
                          {0x1.6a09e6p-1f, 0x1.6a09e6p-1f, 0x0p+0f},
                          {0x0p+0f, 0x0p+0f, 0x0p+0f},
                      });

// `vector.GetAngle2D(Vector3(0, -1, 0))` of each vector above. Exact: the first is -0, the last +0.
CUJU_TABLE_DEFINITION(float, ALLOWED_BODY_DIRECTION_ANGLES,
                      {-0x0p+0f, -0x1.921fb6p-1f, 0x1.921fb6p-1f, -0x1.2d97c8p+1f, 0x1.2d97c8p+1f,
                       0x0p+0f});

// A direction whose dot products all stay at -1 or below gives `NO_ALLOWED_BODY_DIRECTION`.
CUJU_FUNCTION_NO_INLINE inline int ForceIntoAllowedBodyDirectionIndex(const Vector3 &source) {
    float bestDotProduct = -1.0f;
    int best = NO_ALLOWED_BODY_DIRECTION;
    for (int index = 0; index < 5; index++) {
        float dotProduct = CUJU_TABLE(ALLOWED_BODY_DIRECTIONS)[index].GetDotProduct(source);
        if (dotProduct > bestDotProduct) {
            bestDotProduct = dotProduct;
            best = index;
        }
    }
    return best;
}

CUJU_FUNCTION inline Vector3 ForceIntoAllowedBodyDirectionVector(const Vector3 &source) {
    return CUJU_TABLE(ALLOWED_BODY_DIRECTIONS)[ForceIntoAllowedBodyDirectionIndex(source)];
}

// The movement in the ground plane in meters per second. `positions` holds two frames or more.
template <class Positions>
CUJU_FUNCTION_NO_INLINE inline Vector3 CalculateMovementAtFrame(const Positions &positions,
                                                                uint32_t frameNumber,
                                                                uint32_t smoothFrames) {
    if (frameNumber == positions.size() - 1)
        return (positions[positions.size() - 1] - positions[positions.size() - 2]).Get2D() * 100.0f;
    if (frameNumber == 0) return (positions[1] - positions[0]).Get2D() * 100.0f;
    Vector3 totalMovement;
    uint32_t count = 0;
    for (int frame = (int)(frameNumber - smoothFrames); frame <= (int)(frameNumber + smoothFrames);
         frame++) {
        if (frame > 0 && frame < (int)positions.size()) {
            totalMovement += (positions[frame] - positions[frame - 1]).Get2D() * 100.0f;
            count++;
        }
    }
    if (count > 0) totalMovement /= Vector3((float)count);
    return totalMovement;
}

CUJU_FUNCTION_NO_INLINE inline void CalculateSpatialState(const SimulationContext &context,
                                                          HumanoidState &humanoid) {
    CUJU_PROFILE_SCOPE(CALCULATE_SPATIAL_STATE);
    state::AnimationState &currentAnimation = humanoid.currentAnimation;
    const animation::Animation &animationData =
        context.animations->animations[currentAnimation.animation];
    state::SpatialState &spatialState = humanoid.spatialState;
    Vector3 position;
    if ((uint32_t)currentAnimation.positions.count > (uint32_t)currentAnimation.frameNumber) {
        position =
            humanoid.startPosition + currentAnimation.positions[currentAnimation.frameNumber] +
            currentAnimation.actionSmuggleOffset + currentAnimation.actionSmuggleSustainOffset +
            currentAnimation.movementSmuggleOffset;
    } else {
        Quaternion orientation;
        animation::GetKeyFrame(animationData, e_BodyPart_Player, currentAnimation.frameNumber,
                               orientation, position);
        position.coordinates[2] = 0.0f;
        position = humanoid.startPosition + position.GetRotated2D(humanoid.startAngle.value()) +
                   currentAnimation.actionSmuggleOffset +
                   currentAnimation.actionSmuggleSustainOffset +
                   currentAnimation.movementSmuggleOffset;
    }
    if (currentAnimation.frameNumber > 12) spatialState.foot = animationData.outgoingFoot;

    spatialState.actualMovement = (position - humanoid.previousPosition2D) * 100.0f;
    const float positionOffsetMovementIgnoreFactor = 0.5f;
    spatialState.physicsMovement =
        spatialState.actualMovement - (spatialState.actionSmuggleMovement) -
        (spatialState.movementSmuggleMovement) -
        (spatialState.positionOffsetMovement * positionOffsetMovementIgnoreFactor);
    spatialState.animationMovement = spatialState.physicsMovement;
    if (currentAnimation.positions.count > 0) {
        spatialState.animationMovement =
            CalculateMovementAtFrame(animationData.positions, currentAnimation.frameNumber, 1)
                .GetRotated2D(humanoid.startAngle.value());
    }
    spatialState.movement = spatialState.physicsMovement;

    Vector3 bodyPosition;
    Quaternion bodyOrientation;
    float z;
    animation::GetKeyFrameZ(animationData, e_BodyPart_Body, currentAnimation.frameNumber,
                            bodyOrientation, bodyPosition, z);

    Vector3 bodyDirectionVector = Vector3(0, -1, 0).GetRotated2D(
        z + humanoid.startAngle.value() + currentAnimation.rotationSmuggleOffset.value());

    spatialState.floatVelocity = spatialState.movement.GetLength();
    spatialState.velocityCategory = FloatToVelocityCategory(spatialState.floatVelocity);
    if (spatialState.velocityCategory != e_Velocity_Idle) {
        spatialState.directionVector = spatialState.movement.GetNormalized();
    } else {
        spatialState.directionVector = bodyDirectionVector;
    }
    spatialState.position = position;
    SetRadian(spatialState.angle,
              ModulateIntoRange(-PI, PI, FixAngle(spatialState.directionVector.GetAngle2D())));

    // Exact: the quantized direction is an allowed one, so the table gives the bits of its angle.
    int bodyDirection = 0;
    float nonQuantizedAngle = CUJU_TABLE(ALLOWED_BODY_DIRECTION_ANGLES)[0];
    if (spatialState.velocityCategory != e_Velocity_Idle) {
        Vector3 adaptedBodyDirectionVector =
            bodyDirectionVector.GetRotated2D(-spatialState.angle.value());
        float bodyAngleRelative = adaptedBodyDirectionVector.GetAngle2D(Vector3(0, -1, 0));
        // Without a limit the vector stays, and so does its angle.
        nonQuantizedAngle = bodyAngleRelative;
        if (spatialState.velocityCategory == e_Velocity_Sprint &&
            fabsf(bodyAngleRelative) >= 0.125f * PI) {
            adaptedBodyDirectionVector =
                Vector3(0, -1, 0).GetRotated2D(0.12f * PI * SignSide(bodyAngleRelative));
            nonQuantizedAngle = adaptedBodyDirectionVector.GetAngle2D(Vector3(0, -1, 0));
        } else if (spatialState.velocityCategory == e_Velocity_Walk &&
                   fabsf(bodyAngleRelative) >= 0.5f * PI) {
            adaptedBodyDirectionVector =
                Vector3(0, -1, 0).GetRotated2D(0.495f * PI * SignSide(bodyAngleRelative));
            nonQuantizedAngle = adaptedBodyDirectionVector.GetAngle2D(Vector3(0, -1, 0));
        }
        spatialState.relativeBodyDirectionVectorNonQuantized = adaptedBodyDirectionVector;
        bodyDirection = ForceIntoAllowedBodyDirectionIndex(adaptedBodyDirectionVector);
    } else {
        spatialState.relativeBodyDirectionVectorNonQuantized = Vector3(0, -1, 0);
    }
    spatialState.relativeBodyDirectionVector = CUJU_TABLE(ALLOWED_BODY_DIRECTIONS)[bodyDirection];
    SetRadian(spatialState.relativeBodyAngle,
              CUJU_TABLE(ALLOWED_BODY_DIRECTION_ANGLES)[bodyDirection]);
    SetRadian(spatialState.relativeBodyAngleNonQuantized, nonQuantizedAngle);
    spatialState.bodyDirectionVector =
        spatialState.relativeBodyDirectionVector.GetRotated2D(spatialState.angle.value());
    SetRadian(spatialState.bodyAngle,
              spatialState.bodyDirectionVector.GetAngle2D(Vector3(0, -1, 0)));

    humanoid.previousPosition2D = position;
}

// Exact: the operand order of `std::min`, so a NaN operand gives `first`.
CUJU_FUNCTION inline float MinimumFloat(float first, float second) {
    return (second < first) ? second : first;
}

CUJU_FUNCTION inline float TeamDifficulty(const World &world, int teamID) {
    return teamID == 0 ? world.serialized.scenario.leftTeamDifficulty
                       : world.serialized.scenario.rightTeamDifficulty;
}

CUJU_FUNCTION inline float GetStatistic(const World &world, const SimulationContext &context,
                                        int player, int statistic) {
    float multiplier = 0.3f + 0.7f * TeamDifficulty(world, world.playerTeamID[player]);
    multiplier *= 0.7f + 0.3f * world.serialized.match.players[player].base.fatigueFactorInverse;
    return context.staticData->data[player].statistics[statistic] * multiplier;
}

CUJU_FUNCTION inline bool ExternalControllerActive(const World &world, int player) {
    int externalController = world.serialized.match.players[player].base.externalController;
    if (externalController < 0) return false;
    // The gamers of the first team come first in the numbering of the external controllers.
    int firstTeamGamers = world.serialized.match.teams[0].humanGamers.count;
    const state::HumanGamerState &gamer =
        externalController < firstTeamGamers
            ? world.serialized.match.teams[0].humanGamers[externalController]
            : world.serialized.match.teams[1].humanGamers[externalController - firstTeamGamers];
    return !gamer.controller.keyboard.disabled;
}

CUJU_FUNCTION inline int GetReactionTimeMilliseconds(const World &world,
                                                     const SimulationContext &context, int player) {
    int reactionTimeMilliseconds = FloatToInt32(
        roundf(80.0f - GetStatistic(world, context, player, e_Statistic_PhysicalReaction) * 40.0f));
    if (ExternalControllerActive(world, player)) return reactionTimeMilliseconds;
    return FloatToInt32((float)reactionTimeMilliseconds +
                        (1.0f - TeamDifficulty(world, world.playerTeamID[player])) * 100);
}

CUJU_FUNCTION inline float GetLastTouchBias(const World &world, int player, int decayMilliseconds) {
    uint64_t adaptedTimeMilliseconds = world.serialized.match.actualTimeMilliseconds;
    if (decayMilliseconds > 0)
        return 1.0f -
               clamp(
                   (float)(adaptedTimeMilliseconds -
                           world.serialized.match.players[player].base.lastTouchTimeMilliseconds) /
                       (float)decayMilliseconds,
                   0.0f, 1.0f);
    return 0.0f;
}
CUJU_FUNCTION inline float TeamLastTouchBias(const World &world, int teamID,
                                             int decayMilliseconds) {
    int lastTouchPlayer = TeamByID(world, teamID).lastTouchPlayer;
    return lastTouchPlayer >= 0 ? GetLastTouchBias(world, lastTouchPlayer, decayMilliseconds)
                                : 0.0f;
}

CUJU_FUNCTION inline void OffsetPosition(World &world, int player, const Vector3 &offset) {
    state::HumanoidState &humanoid = world.serialized.match.players[player].base.humanoid;
    humanoid.nextStartPosition += offset;
    humanoid.startPosition += offset;
    humanoid.spatialState.position += offset;
    humanoid.spatialState.positionOffsetMovement += offset * 100.0f;
    humanoid.decayingPositionOffset += offset;
    if (humanoid.decayingPositionOffset.GetLength() > 0.1f)
        humanoid.decayingPositionOffset = humanoid.decayingPositionOffset.GetNormalized() * 0.1f;
    humanoid.currentAnimation.positionOffset += offset;
}

CUJU_FUNCTION inline int MentalImageIndex(const World &world, int historyMilliseconds) {
    int index = DoubleToInt32(round((double)((float)historyMilliseconds / 100.0)));
    if (index >= world.serialized.match.mentalImages.count)
        index = world.serialized.match.mentalImages.count - 1;
    if (index < 0) index = 0;
    return index;
}
CUJU_FUNCTION inline Vector3 MentalBallPrediction(const World &world, int imageIndex,
                                                  int timeMilliseconds) {
    const MentalImageState &mentalImage = world.serialized.match.mentalImages[imageIndex];
    // The offset from the stamp of the image wraps as an unsigned 32-bit value, so one outside the
    // prediction window, a negative one included, falls back to the last prediction.
    int index = (int)(uint32_t)((uint64_t)(int64_t)timeMilliseconds +
                                world.serialized.match.actualTimeMilliseconds -
                                (uint64_t)mentalImage.timeStampMilliseconds);
    if ((uint32_t)index >= BALL_PREDICTION_SPAN_MILLISECONDS)
        index = BALL_PREDICTION_SPAN_MILLISECONDS - 10;
    index = index / 10;
    if (index < 0) index = 0;
    Vector3 mentalResult = mentalImage.ballPredictions[index];
    if (mentalImage.lazyFlip) mentalResult.Mirror();
    Vector3 realResult = BallPredict(world.serialized.match.ball, timeMilliseconds);
    return mentalResult.EnforceMaximumDeviation(realResult, mentalImage.maximumDistanceDeviation);
}

}  // namespace simulation
}  // namespace cuju

// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The referee and the two linesmen, and the force fields and offside lines that the AI also uses.
#pragma once

#include "../profiling.h"

#include <vector>

#include "ai_functions.h"
#include "humanoid.h"
#include "humanoid_process.h"
#include "select_animation.h"

namespace cuju {
namespace simulation {

// Turns a distance in meters into a desired velocity in meters per second.
constexpr float DISTANCE_TO_VELOCITY_MULTIPLIER = 2.6f;
// Beyond this distance from the ball, a player skips the options that need the ball close.
constexpr float BALL_DISTANCE_OPTIMIZE_THRESHOLD = 10.0f;
enum e_MagnetType { e_MagnetType_Attract, e_MagnetType_Repel };
enum e_DecayType { e_DecayType_Constant, e_DecayType_Variable };

struct ForceSpot {
    Vector3 origin;
    int magnetType = e_MagnetType_Attract;
    int decayType = e_DecayType_Constant;
    float exponent = 1.0f;
    float power = 0.0f;
    float scale = 0.0f;
};
using ForceField = InlineVector<ForceSpot, MAXIMUM_FORCE_SPOTS>;

CUJU_FUNCTION_NO_INLINE inline Vector3 GetForceFieldMovement(const ForceField &forceField,
                                                             const Vector3 &currentPosition,
                                                             float attractorDampingDistance = 10) {
    Vector3 cumulativeVector;
    float cumulativeForce = 0.0f;
    // Exact: the lanes compute the terms of the spots, and the sums run in spot order as in the
    // sequential loop.
    struct Term {
        Vector3 contribution;
        float force;
        int32_t active;
    };
    auto term = [&](int spotIndex) {
        const ForceSpot &spot = forceField[spotIndex];
        Term spotTerm{Vector3(), 0.0f, 0};
        float distance = (spot.origin - currentPosition).GetLength();
        float intensity;
        if (spot.decayType == e_DecayType_Constant) {
            intensity = 1.0f;
        } else {
            intensity = clamp(1.0f - distance / spot.scale, 0.0f, 1.0f);
            if (spot.exponent != 1.0f) intensity = libm::powf(intensity, spot.exponent);
        }
        if (intensity > 0.0f) {
            Vector3 relativeOrigin = spot.origin - currentPosition;
            relativeOrigin.Normalize(0);
            if (spot.magnetType == e_MagnetType_Repel) {
                relativeOrigin = -relativeOrigin;
            } else if (distance < attractorDampingDistance) {
                relativeOrigin *= distance / attractorDampingDistance;
            }
            float force = spot.power * intensity;
            spotTerm.contribution = relativeOrigin * force;
            spotTerm.force = force;
            spotTerm.active = 1;
        }
        return spotTerm;
    };
    WarpPipeline<Term>(forceField.count, term, [&](int, const Term &spotTerm) {
        if (spotTerm.active) {
            cumulativeVector += spotTerm.contribution;
            cumulativeForce += spotTerm.force;
        }
    });
    if (cumulativeForce == 0.0f) return Vector3(0);
    return (cumulativeVector / cumulativeForce) * SPRINT_VELOCITY;
}

struct PlayerImagePosition {
    Vector3 position, movement;
    int role;
};
using ImageList = InlineVector<PlayerImagePosition, state::MAXIMUM_PLAYERS>;

// `result` keeps the order of the mental image. It is an out parameter because a returned list
// would be copied at the call.
CUJU_FUNCTION_NO_INLINE inline void MentalTeamPlayerImages(const World &world, int mentalImageIndex,
                                                           int teamID, ImageList &result) {
    const state::MentalImageState &mentalImage =
        world.serialized.match.mentalImages[mentalImageIndex];
    // Exact: the age is the 64-bit difference of the clocks truncated to `int`.
    const int imageAge = (int)(world.serialized.match.actualTimeMilliseconds -
                               (uint64_t)mentalImage.timeStampMilliseconds);
    result.clear();
    struct ImageEntry {
        Vector3 position, movement;
        int32_t role, valid;
    };
    auto image = [&](int entryIndex) {
        const state::PlayerImage &playerImage = mentalImage.players[entryIndex];
        ImageEntry entry{Vector3(), Vector3(), 0, 0};
        if (!world.serialized.match.players[playerImage.player].base.isActive ||
            world.playerTeamID[playerImage.player] != teamID)
            return entry;
        const state::SpatialState &spatialState = Humanoid(world, playerImage.player).spatialState;
        Vector3 extrapolation = playerImage.movement * (float)imageAge * 0.001f;
        Vector3 position = playerImage.position + extrapolation;
        entry.position = position.EnforceMaximumDeviation(spatialState.position,
                                                          mentalImage.maximumDistanceDeviation);
        entry.movement = playerImage.movement.EnforceMaximumDeviation(
            spatialState.movement, mentalImage.maximumMovementDeviation);
        entry.role = playerImage.role;
        entry.valid = 1;
        return entry;
    };
    WarpPipeline<ImageEntry>(mentalImage.players.count, image, [&](int, const ImageEntry &entry) {
        if (entry.valid)
            result.push_back(PlayerImagePosition{entry.position, entry.movement, entry.role});
    });
}

// Moves each image of `images` ahead along x by `futureSimulationMilliseconds`, in place.
CUJU_FUNCTION_NO_INLINE inline float OffsideLineFromImages(
    ImageList &images, const Vector3 &ballPosition, int side,
    unsigned int futureSimulationMilliseconds) {
    int ignoredOpponent = 0;
    Vector3 deepestOpponentPosition;
    for (size_t index = 0; index < images.size(); index++) {
        images[index].position.coordinates[0] +=
            images[index].movement.coordinates[0] * futureSimulationMilliseconds * 0.001f;
        if (images[index].position.coordinates[0] * side >
            images.at(ignoredOpponent).position.coordinates[0] * side)
            ignoredOpponent = (int)index;
    }
    for (size_t index = 0; index < images.size(); index++) {
        if (images[index].position.coordinates[0] * side >
                deepestOpponentPosition.coordinates[0] * side &&
            (int)index != ignoredOpponent)
            deepestOpponentPosition = images[index].position;
    }
    float offsideLine = deepestOpponentPosition.coordinates[0];
    if (ballPosition.coordinates[0] * side > offsideLine * side)
        offsideLine = ballPosition.coordinates[0];
    if (offsideLine * side < 0) offsideLine = 0;
    return clamp(offsideLine, -PITCH_HALF_LENGTH, PITCH_HALF_LENGTH);
}

CUJU_FUNCTION inline float GetOffsideLine(const World &world, int mentalImageIndex, int teamID,
                                          unsigned int futureSimulationMilliseconds = 0) {
    ImageList images;
    MentalTeamPlayerImages(world, mentalImageIndex, teamID, images);
    return OffsideLineFromImages(images, MentalBallPrediction(world, mentalImageIndex, 0),
                                 TeamDynamicSide(world, teamID), futureSimulationMilliseconds);
}

CUJU_FUNCTION_NO_INLINE inline Vector3 BallAveragePosition(const World &world,
                                                           unsigned int durationMilliseconds) {
    const state::BallSerialized &ball = world.serialized.match.ball;
    unsigned int total = 0;
    Vector3 average;
    for (int index = ball.ballPositionHistory.count - 1; index >= 0; index--) {
        Vector3 position = ball.ballPositionHistory[index];
        if (ball.lazyFlip) position.Mirror();
        average += position;
        total++;
        if (total * 10 > durationMilliseconds) break;
    }
    if (total > 0)
        average /= Vector3((float)total);
    else
        average = BallPredict(ball, 0);
    return average;
}

// Official 0 is the referee, officials 1 and 2 the linesmen.
CUJU_FUNCTION_NO_INLINE inline void RefereeRequestCommand(World &world, int official,
                                                          CommandQueue &queue) {
    CUJU_PROFILE_SCOPE(REFEREE_REQUEST_COMMAND);
    const state::MatchState &match = world.serialized.match;
    const state::SpatialState &spatialState = match.officials[official].humanoid.spatialState;
    const Vector3 position = spatialState.position;
    if (official == 0) {
        const state::RefereeState &referee = match.referee;
        if (referee.buffer.active && (referee.foul.foulType == 2 || referee.foul.foulType == 3) &&
            referee.buffer.prepareTime > match.actualTimeMilliseconds + 5000) {
            const Vector3 offenderPosition =
                Humanoid(world, referee.foul.foulPlayer).spatialState.position;
            Vector3 desiredPosition =
                offenderPosition + (position - offenderPosition).GetNormalized(0) * (float)2.0;
            if ((double)(position - desiredPosition).GetLength() > 2.0) {
                state::PlayerCommand command = DefaultPlayerCommand();
                command.desiredFunctionType = e_FunctionType_Movement;
                command.useDesiredMovement = true;
                command.useDesiredLookAt = true;
                command.desiredDirection =
                    (desiredPosition - position).GetNormalized(spatialState.directionVector);
                command.desiredVelocityFloat =
                    RangeVelocity((desiredPosition - position).GetLength() * 1.0f);
                command.desiredLookAt = offenderPosition;
                queue.push_back(command);
            } else {
                // The special animation with variable 3 shows the card.
                state::PlayerCommand command = DefaultPlayerCommand();
                command.desiredFunctionType = e_FunctionType_Special;
                command.useDesiredMovement = false;
                command.useDesiredLookAt = false;
                command.useSpecialVariable1 = true;
                command.specialVariable1 = 3;
                queue.push_back(command);
                state::PlayerCommand followCommand = DefaultPlayerCommand();
                followCommand.desiredFunctionType = e_FunctionType_Movement;
                followCommand.useDesiredMovement = true;
                followCommand.useDesiredLookAt = true;
                followCommand.desiredDirection =
                    (desiredPosition - position).GetNormalized(spatialState.directionVector);
                followCommand.desiredVelocityFloat = IDLE_VELOCITY;
                followCommand.desiredLookAt = offenderPosition;
                queue.push_back(followCommand);
            }
        } else {
            state::PlayerCommand command = DefaultPlayerCommand();
            command.desiredFunctionType = e_FunctionType_Movement;
            command.useDesiredMovement = true;
            command.useDesiredLookAt = true;
            ForceField forceField;
            {
                ForceSpot spot;
                spot.origin = BallAveragePosition(world, 2000).Get2D() * 0.6f;
                spot.magnetType = e_MagnetType_Attract;
                spot.decayType = e_DecayType_Constant;
                spot.power = 0.5f;
                forceField.push_back(spot);
            }
            {
                ForceSpot spot;
                spot.origin = BallPredict(match.ball, 200).Get2D();
                spot.magnetType = e_MagnetType_Repel;
                spot.decayType = e_DecayType_Variable;
                spot.power = 0.5f;
                spot.scale = 10.0f;
                forceField.push_back(spot);
            }
            for (int team : {world.FirstTeamID(), 1 - world.FirstTeamID()})
                for (int player : TeamActivePlayers(world, team)) {
                    const state::SpatialState &playerSpatialState =
                        Humanoid(world, player).spatialState;
                    ForceSpot spot;
                    spot.origin = playerSpatialState.position + playerSpatialState.movement * 0.4f;
                    spot.magnetType = e_MagnetType_Repel;
                    spot.decayType = e_DecayType_Variable;
                    spot.power = 0.5f;
                    spot.scale = 10.0f;
                    forceField.push_back(spot);
                }
            Vector3 desiredPosition = position + GetForceFieldMovement(forceField, position);
            command.desiredDirection =
                (desiredPosition - position).GetNormalized(spatialState.directionVector);
            command.desiredVelocityFloat = clamp(
                (desiredPosition - position).GetLength() * DISTANCE_TO_VELOCITY_MULTIPLIER * 0.5f,
                IDLE_VELOCITY, SPRINT_VELOCITY);
            command.desiredLookAt = BallPredict(match.ball, 60).Get2D();
            queue.push_back(command);
        }
    } else {
        state::PlayerCommand command = DefaultPlayerCommand();
        command.desiredFunctionType = e_FunctionType_Movement;
        command.useDesiredMovement = true;
        command.useDesiredLookAt = true;
        float offsideLine;
        Vector3 desiredPosition;
        const int latestImage = MentalImageIndex(world, 0);
        if (position.coordinates[1] < 0) {
            offsideLine = GetOffsideLine(world, latestImage, 1);
            desiredPosition = Vector3(offsideLine, -(PITCH_HALF_WIDTH + 0.8f), 0);
        } else {
            offsideLine = GetOffsideLine(world, latestImage, 0);
            desiredPosition = Vector3(offsideLine, PITCH_HALF_WIDTH + 0.8f, 0);
        }
        command.desiredDirection =
            (desiredPosition - position).GetNormalized(spatialState.directionVector);
        command.desiredVelocityFloat = RangeVelocity((desiredPosition - position).GetLength() *
                                                     DISTANCE_TO_VELOCITY_MULTIPLIER);
        command.desiredLookAt = Vector3(desiredPosition.coordinates[0], 0, 0);
        queue.push_back(command);
    }
}

CUJU_FUNCTION_NO_INLINE inline bool HumanoidBaseProcess(World &world,
                                                        const SimulationContext &context,
                                                        int official) {
    const int humanoidIndex = world.serialized.match.playerCount + official;
    HumanoidState &humanoid = world.serialized.match.officials[official].humanoid;
    state::AnimationState &currentAnimation = humanoid.currentAnimation;

    humanoid.decayingPositionOffset *= 0.95f;
    if ((double)humanoid.decayingPositionOffset.GetLength() < 0.005)
        humanoid.decayingPositionOffset.Set(0);
    humanoid.decayingDifficultyFactor =
        clamp(humanoid.decayingDifficultyFactor - 0.002f, 0.0f, 1.0f);

    CalculateSpatialState(context, humanoid);
    humanoid.spatialState.positionOffsetMovement = Vector3(0);
    currentAnimation.frameNumber++;
    humanoid.previousAnimationFrameNumber++;
    if (currentAnimation.frameNumber ==
            context.animations->animations[currentAnimation.animation].frameCount - 1 &&
        humanoid.interruptAnimation == e_InterruptAnimation_None)
        humanoid.interruptAnimation = e_InterruptAnimation_Switch;

    if (humanoid.interruptAnimation != e_InterruptAnimation_None) {
        CommandQueue commandQueue;
        if (humanoid.interruptAnimation == e_InterruptAnimation_Trip && humanoid.tripType != 0) {
            AddTripCommandToQueue(humanoid, commandQueue, humanoid.tripType);
            humanoid.tripType = 0;
            commandQueue.push_back(
                GetBasicMovementCommand(humanoid, humanoid.spatialState.floatVelocity));
        } else {
            RefereeRequestCommand(world, official, commandQueue);
        }
        bool found = false;
        for (const state::PlayerCommand &command : commandQueue) {
            Selector selector(world, context, humanoidIndex);
            SelectResult result =
                selector.BaseSelectAnimation(command, humanoid.interruptAnimation);
            if (result == UNSUPPORTED) return false;
            found = result == SELECT_TRUE;
            if (found) break;
        }
        if (found) {
            humanoid.startPosition = humanoid.spatialState.position;
            humanoid.startAngle = humanoid.spatialState.angle;
            {
                Selector selector(world, context, humanoidIndex);
                Vector3 nextStartPosition;
                float nextStartAngle;
                selector.CalculatePredictedSituation(nextStartPosition, nextStartAngle);
                humanoid.nextStartPosition = nextStartPosition;
                SetRadian(humanoid.nextStartAngle, nextStartAngle);
            }
            humanoid.animationApplyBuffer.animation = currentAnimation.animation;
            humanoid.animationApplyBuffer.smooth = true;
            humanoid.animationApplyBuffer.smoothFactor =
                (humanoid.interruptAnimation == e_InterruptAnimation_Switch) ? 0.6f : 1.0f;
            float animationDifference =
                context.derived[currentAnimation.animation].animationDifficultyFactor;
            if (animationDifference > humanoid.decayingDifficultyFactor)
                humanoid.decayingDifficultyFactor = animationDifference;
            if (humanoid.interruptAnimation == e_InterruptAnimation_ReQueue &&
                humanoid.previousAnimationFunctionType == currentAnimation.functionType)
                humanoid.requeueDelayFrames = 32;
        }
    }
    humanoid.requeueDelayFrames = (int)clamp((float)(humanoid.requeueDelayFrames - 1), 0, 10000);
    humanoid.interruptAnimation = e_InterruptAnimation_None;

    const animation::Animation &playingAnimation =
        context.animations->animations[currentAnimation.animation];
    float frameBias =
        (currentAnimation.frameNumber + 1) / (float)((playingAnimation.frameCount - 1) + 1);
    SetRadian(currentAnimation.rotationSmuggleOffset,
              currentAnimation.rotationSmuggle.begin.value() * (1.0f - frameBias) +
                  currentAnimation.rotationSmuggle.end.value() * frameBias);

    humanoid.animationApplyBuffer.frameNumber = currentAnimation.frameNumber;
    if ((uint32_t)currentAnimation.positions.count > (uint32_t)currentAnimation.frameNumber) {
        humanoid.animationApplyBuffer.position =
            humanoid.startPosition + currentAnimation.actionSmuggleOffset +
            currentAnimation.actionSmuggleSustainOffset + currentAnimation.movementSmuggleOffset +
            currentAnimation.positions[currentAnimation.frameNumber];
        SetRadian(humanoid.animationApplyBuffer.orientation,
                  humanoid.startAngle.value() + currentAnimation.rotationSmuggleOffset.value());
        humanoid.animationApplyBuffer.noPosition = true;
    } else {
        humanoid.animationApplyBuffer.position =
            humanoid.startPosition + currentAnimation.actionSmuggleOffset +
            currentAnimation.actionSmuggleSustainOffset + currentAnimation.movementSmuggleOffset;
        humanoid.animationApplyBuffer.orientation = humanoid.startAngle;
        humanoid.animationApplyBuffer.noPosition = false;
    }
    humanoid.animationApplyBuffer.offsets = humanoid.offsets;
    return true;
}

CUJU_FUNCTION inline bool OfficialsProcess(World &world, const SimulationContext &context) {
    CUJU_PROFILE_SCOPE(OFFICIALS_PROCESS);
    for (int official = 0; official < 3; official++)
        if (!HumanoidBaseProcess(world, context, official)) return false;
    return true;
}

}  // namespace simulation
}  // namespace cuju

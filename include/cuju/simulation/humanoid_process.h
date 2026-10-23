// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The tick of a player's humanoid, covering the next frame, the next animation and ball touches.
#pragma once

#include "../profiling.h"

#include <functional>
#include <stdexcept>
#include <vector>

#include "ai_functions.h"
#include "ball_touch.h"
#include "human_controller.h"
#include "team_select.h"
#include "humanoid.h"
#include "select_animation.h"
#include "touch.h"

namespace cuju {
namespace simulation {

CUJU_FUNCTION_NO_INLINE inline state::PlayerCommand DefaultPlayerCommand() {
    state::PlayerCommand command{};
    command.desiredFunctionType = e_FunctionType_Movement;
    command.useDesiredMovement = false;
    command.desiredVelocityFloat = IDLE_VELOCITY;
    command.strictMovement = e_StrictMovement_Dynamic;
    command.useDesiredLookAt = false;
    command.useTouchInformation = false;
    command.useTripType = false;
    command.useDesiredTripDirection = false;
    command.onlyDeflectAnimationsThatPickupBall = false;
    command.tripType = 1;
    command.useSpecialVariable1 = false;
    command.specialVariable1 = 0;
    command.useSpecialVariable2 = false;
    command.specialVariable2 = 0;
    command.modifier = 0;
    command.touchInformation.targetPlayer = -1;
    command.touchInformation.forcedTargetPlayer = -1;
    return command;
}

CUJU_FUNCTION inline state::PlayerCommand GetTripCommand(const HumanoidState &humanoid,
                                                         const Vector3 &tripVector, int tripType) {
    state::PlayerCommand command = DefaultPlayerCommand();
    command.desiredFunctionType = e_FunctionType_Trip;
    command.useDesiredMovement = false;
    command.useDesiredTripDirection = true;
    command.desiredTripDirection = tripVector;
    command.desiredVelocityFloat = humanoid.spatialState.floatVelocity;
    command.useTripType = true;
    command.tripType = tripType;
    return command;
}

CUJU_FUNCTION inline void AddTripCommandToQueue(const HumanoidState &humanoid, CommandQueue &queue,
                                                int tripType) {
    if (tripType == 1) {
        queue.push_back(GetTripCommand(humanoid, humanoid.tripDirection, tripType));
    } else {
        int otherTripType = 3;
        if (tripType == 3) otherTripType = 2;
        queue.push_back(GetTripCommand(humanoid, humanoid.tripDirection, tripType));
        queue.push_back(GetTripCommand(humanoid, humanoid.tripDirection, otherTripType));
        queue.push_back(GetTripCommand(humanoid, humanoid.tripDirection, 1));
    }
}

CUJU_FUNCTION inline state::PlayerCommand GetBasicMovementCommand(const HumanoidState &humanoid,
                                                                  float velocityFloat) {
    state::PlayerCommand command = DefaultPlayerCommand();
    command.desiredFunctionType = e_FunctionType_Movement;
    command.useDesiredMovement = true;
    command.useDesiredLookAt = true;
    command.desiredDirection = humanoid.spatialState.directionVector;
    command.desiredVelocityFloat = velocityFloat;
    command.desiredLookAt = humanoid.spatialState.position + command.desiredDirection * 10.0f;
    return command;
}

// Defined in `ai_controller.h`.
CUJU_FUNCTION inline bool PlayerRequestCommand(World &world, const SimulationContext &context,
                                               int player, CommandQueue &queue);

CUJU_FUNCTION_NO_INLINE inline bool HumanoidProcess(
    World &world, const SimulationContext &context, int player,
    const RequestCommandFunction *requestCommand = nullptr) {
    CUJU_PROFILE_SCOPE(HUMANOID_PROCESS);
    if (!requestCommand) requestCommand = context.requestCommand;
    state::PlayerState &playerState = world.serialized.match.players[player];
    HumanoidState &humanoid = playerState.base.humanoid;
    state::AnimationState &currentAnimation = humanoid.currentAnimation;
    const int teamID = world.playerTeamID[player];
    const state::TeamState &team = TeamByID(world, teamID);
    const state::BallSerialized &ball = world.serialized.match.ball;

    const int mentalImage = MentalImageIndex(world, humanoid.mentalImageTime);
    humanoid.decayingPositionOffset *= 0.95f;
    if ((double)humanoid.decayingPositionOffset.GetLength() < 0.005)
        humanoid.decayingPositionOffset.Set(0);
    humanoid.decayingDifficultyFactor =
        clamp(humanoid.decayingDifficultyFactor - 0.002f, 0.0f, 1.0f);

    bool teamTouchedLast = world.serialized.match.lastTouchTeamID == teamID;
    humanoid.mentalImageTime =
        teamTouchedLast ? 0 : GetReactionTimeMilliseconds(world, context, player);

    CalculateSpatialState(context, humanoid);
    humanoid.spatialState.positionOffsetMovement = Vector3(0);

    currentAnimation.frameNumber++;
    humanoid.previousAnimationFrameNumber++;

    {
        const animation::Animation &animationData =
            context.animations->animations[currentAnimation.animation];
        if (currentAnimation.frameNumber == animationData.frameCount - 1 &&
            humanoid.interruptAnimation == e_InterruptAnimation_None)
            humanoid.interruptAnimation = e_InterruptAnimation_Switch;

        bool mayReQueue = true;
        if (mayReQueue && humanoid.interruptAnimation != e_InterruptAnimation_None)
            mayReQueue = false;

        if (mayReQueue) {
            bool frameNumberPredicate = false;
            float actionDistance =
                ((humanoid.spatialState.position + humanoid.spatialState.movement * 0.1f) -
                 BallPredict(ball, 100).Get2D())
                    .GetLength();
            uint64_t isSecondTeam = TeamSlot(world, teamID) == 1 ? 1 : 0;
            uint64_t matchTimeMilliseconds = world.serialized.match.actualTimeMilliseconds;
            if (world.serialized.match.designatedPossessionPlayer == player &&
                actionDistance < 3.0f) {
                frameNumberPredicate = ((matchTimeMilliseconds + isSecondTeam * 10) % 20) == 0;
            } else if (world.serialized.match.designatedPossessionPlayer == player) {
                frameNumberPredicate = ((matchTimeMilliseconds + isSecondTeam * 10) % 30) == 0;
            } else if (team.designatedTeamPossessionPlayer == player) {
                frameNumberPredicate = ((matchTimeMilliseconds + isSecondTeam * 20) % 40) == 0;
            } else if (actionDistance < 5.0f) {
                frameNumberPredicate = ((matchTimeMilliseconds + isSecondTeam * 20) % 50) == 0;
            } else if (actionDistance < 10.0f) {
                frameNumberPredicate = ((matchTimeMilliseconds + isSecondTeam * 40) % 80) == 0;
            }
            if (!frameNumberPredicate) mayReQueue = false;
        }

        if (mayReQueue) {
            float ballDistance = (MentalBallPrediction(world, mentalImage, 500).Get2D() -
                                  humanoid.spatialState.position)
                                     .GetLength();
            const bool touchPending = currentAnimation.frameNumber < currentAnimation.touchFrame;
            if (((currentAnimation.functionType == e_FunctionType_Movement &&
                  !playerState.hasPossession && ballDistance < 16.0f) ||
                 (currentAnimation.functionType == e_FunctionType_Movement &&
                  playerState.hasPossession) ||
                 (currentAnimation.functionType == e_FunctionType_Trap && touchPending) ||
                 (currentAnimation.functionType == e_FunctionType_BallControl && touchPending)) &&
                (animationData.incomingSpecialState == animation::STRING_EMPTY) &&
                (animationData.outgoingSpecialState == animation::STRING_EMPTY)) {
                mayReQueue = true;
            } else {
                mayReQueue = false;
            }
        }
        if (mayReQueue) humanoid.interruptAnimation = e_InterruptAnimation_ReQueue;
    }

    if (humanoid.interruptAnimation != e_InterruptAnimation_None) {
        CommandQueue commandQueue;
        if (humanoid.interruptAnimation == e_InterruptAnimation_Trip && humanoid.tripType != 0) {
            AddTripCommandToQueue(humanoid, commandQueue, humanoid.tripType);
            humanoid.tripType = 0;
            commandQueue.push_back(
                GetBasicMovementCommand(humanoid, humanoid.spatialState.floatVelocity));
        } else {
#if defined(__CUDA_ARCH__)
            (void)requestCommand;
            commandQueue.clear();
            if (!PlayerRequestCommand(world, context, player, commandQueue)) return false;
#else
            if (requestCommand) {
                if (!(*requestCommand)(world, player, commandQueue))
                    return Unsupported("the command source gives no command");
            } else {
                commandQueue.clear();
                if (!PlayerRequestCommand(world, context, player, commandQueue)) return false;
            }
#endif
        }

        bool found = false;
        bool preferPassAndShot = false;
        for (const state::PlayerCommand &command : commandQueue) {
            if (command.desiredFunctionType == e_FunctionType_ShortPass ||
                command.desiredFunctionType == e_FunctionType_LongPass ||
                command.desiredFunctionType == e_FunctionType_HighPass ||
                command.desiredFunctionType == e_FunctionType_Shot)
                preferPassAndShot = true;
            Selector selector(world, context, player);
            SelectResult result =
                selector.SelectAnimation(command, humanoid.interruptAnimation, preferPassAndShot);
            if (result == UNSUPPORTED) return false;
            found = result == SELECT_TRUE;
            if (found) break;
        }

        if (humanoid.interruptAnimation == e_InterruptAnimation_Switch && !found)
            CUJU_FATAL("no animation applies to the humanoid");

        if (found) {
            humanoid.startPosition = humanoid.spatialState.position;
            humanoid.startAngle = humanoid.spatialState.angle;
            {
                Selector selector(world, context, player);
                Vector3 nextStartPosition;
                float nextStartAngle;
                selector.CalculatePredictedSituation(nextStartPosition, nextStartAngle);
                humanoid.nextStartPosition = nextStartPosition;
                SetRadian(humanoid.nextStartAngle, nextStartAngle);
            }
            humanoid.animationApplyBuffer.animation = currentAnimation.animation;
            humanoid.animationApplyBuffer.smooth = true;
            humanoid.animationApplyBuffer.smoothFactor =
                (humanoid.interruptAnimation == e_InterruptAnimation_Switch &&
                 humanoid.previousAnimationFunctionType == e_FunctionType_Movement &&
                 currentAnimation.functionType == e_FunctionType_Movement)
                    ? 0.0f
                    : 1.0f;
            if (currentAnimation.functionType == e_FunctionType_Shot)
                humanoid.animationApplyBuffer.smoothFactor = 0.8f;
            if (currentAnimation.functionType == e_FunctionType_ShortPass ||
                currentAnimation.functionType == e_FunctionType_HighPass)
                humanoid.animationApplyBuffer.smoothFactor = 0.8f;
            if (currentAnimation.functionType == e_FunctionType_Deflect ||
                currentAnimation.functionType == e_FunctionType_Sliding)
                humanoid.animationApplyBuffer.smoothFactor = 0.8f;
            if (currentAnimation.functionType == e_FunctionType_BallControl ||
                currentAnimation.functionType == e_FunctionType_Trap)
                humanoid.animationApplyBuffer.smoothFactor = 0.8f;

            float animationDifference =
                context.derived[currentAnimation.animation].animationDifficultyFactor;
            if (animationDifference > humanoid.decayingDifficultyFactor)
                humanoid.decayingDifficultyFactor = animationDifference;

            if (humanoid.interruptAnimation == e_InterruptAnimation_ReQueue &&
                humanoid.previousAnimationFunctionType == currentAnimation.functionType)
                humanoid.requeueDelayFrames = 22;
        }
    }

    humanoid.requeueDelayFrames =
        humanoid.requeueDelayFrames - 1 > 0 ? humanoid.requeueDelayFrames - 1 : 0;
    humanoid.interruptAnimation = e_InterruptAnimation_None;

    const animation::Animation &animationData =
        context.animations->animations[currentAnimation.animation];
    {
        float ballDistanceNow =
            (BallPredict(ball, 0).Get2D() - humanoid.spatialState.position).GetLength();
        float ballDistanceFuture =
            (BallPredict(ball, 200).Get2D() -
             (humanoid.spatialState.position + humanoid.spatialState.movement * 0.2f))
                .GetLength();
        float lastTouchBias = GetLastTouchBias(world, player, 1500);
        float opponentLastTouchBias = TeamLastTouchBias(world, 1 - teamID, 240);
        if (player == world.serialized.match.designatedPossessionPlayer &&
            ((lastTouchBias <= 0.01f && opponentLastTouchBias <= 0.01f &&
              currentAnimation.functionType == e_FunctionType_Movement && ballDistanceNow < 0.6f &&
              ballDistanceFuture > 0.65f && ballDistanceFuture > ballDistanceNow) ||
             (lastTouchBias <= 0.7f && playerState.hasPossession &&
              currentAnimation.functionType == e_FunctionType_Trip && ballDistanceNow < 0.4f)) &&
            BallPredict(ball, 0).coordinates[2] < 1.6f) {
            playerState.triggerControlledBallCollision = true;
        }
    }
    bool controlledBallCollision = playerState.triggerControlledBallCollision;
    if (controlledBallCollision) playerState.triggerControlledBallCollision = false;
    const AnimationDerived &derivedAnimation = context.derived[currentAnimation.animation];
    auto nextBodyAngleNow = [&]() {
        return humanoid.startAngle.value() + animationData.outgoingAngle.value() +
               animationData.outgoingBodyAngle.value() +
               currentAnimation.rotationSmuggle.end.value();
    };
    auto outgoingMovementNow = [&]() {
        PositionList positions;
        for (const Vector3 &position : currentAnimation.positions) positions.push_back(position);
        return Selector::CalculateOutgoingMovement(positions);
    };
    if (controlledBallCollision && currentAnimation.touchFrame == -1) {
        Vector3 currentBallMomentum = world.serialized.match.ball.momentum;
        float nextBodyAngle = nextBodyAngleNow();
        float xRotation = 0, yRotation = 0;
        Vector3 touchVector;
        if (!GetTrapVector(world, context, player, humanoid.nextStartPosition,
                           humanoid.nextStartAngle.value(), nextBodyAngle, outgoingMovementNow(),
                           currentAnimation, currentAnimation.frameNumber, humanoid.spatialState,
                           humanoid.decayingPositionOffset, xRotation, yRotation, touchVector))
            return false;
        if (currentAnimation.originatingCommand.modifier & e_PlayerCommandModifier_KnockOn)
            touchVector *= 1.35f;
        float bumpyRideBias = 0.0f;
        touchVector = touchVector * (1.0f - bumpyRideBias) + currentBallMomentum * bumpyRideBias;
        BallTouch(world, context, touchVector);
        BallSetRotation(world, xRotation, yRotation, 0, 0.2f * (1.0f - bumpyRideBias));
        if (!TeamSetLastTouchPlayer(world, context, player, derivedAnimation.touchType))
            return false;
    }

    if (currentAnimation.touchFrame == currentAnimation.frameNumber) {
        Vector3 desiredBallPosition;
        for (const animation::FootballKeyFrame &keyFrame : animationData.football)
            if (keyFrame.frame == currentAnimation.touchFrame)
                desiredBallPosition = keyFrame.position;
        float desiredBallHeight = desiredBallPosition.coordinates[2];
        float touchableDistance = 0.4f;
        float fullBallDistance =
            (BallPredict(world.serialized.match.ball, 0) -
             (currentAnimation.touchPosition + currentAnimation.positionOffset))
                .GetLength();
        if (animationData.incomingRetainState != animation::STRING_EMPTY) {
            fullBallDistance = 0.0f;
            touchableDistance = 1.0f;
        }
        float bumpyRideBias = fullBallDistance / touchableDistance;
        bumpyRideBias = clamp(bumpyRideBias - 0.001f, 0.0f, 1.0f);
        bumpyRideBias = Curve(bumpyRideBias, 1.0f);
        bumpyRideBias = Curve(bumpyRideBias, 0.5f);
        Vector3 currentBallMomentum = world.serialized.match.ball.momentum;

        if (fullBallDistance < touchableDistance &&
            fabsf(desiredBallHeight - BallPredict(world.serialized.match.ball, 0).coordinates[2]) <
                1.0f) {
            float nextBodyAngle = nextBodyAngleNow();
            const int functionType = currentAnimation.functionType;
            if (functionType == e_FunctionType_Trap ||
                (functionType == e_FunctionType_BallControl && !playerState.hasPossession)) {
                float xRotation = 0, yRotation = 0;
                Vector3 touchVector;
                if (!GetTrapVector(
                        world, context, player, humanoid.nextStartPosition,
                        humanoid.nextStartAngle.value(), nextBodyAngle, outgoingMovementNow(),
                        currentAnimation, currentAnimation.frameNumber, humanoid.spatialState,
                        humanoid.decayingPositionOffset, xRotation, yRotation, touchVector))
                    return false;
                if (currentAnimation.originatingCommand.modifier & e_PlayerCommandModifier_KnockOn)
                    touchVector *= 1.35f;
                touchVector =
                    touchVector * (1.0f - bumpyRideBias) + currentBallMomentum * bumpyRideBias;
                BallTouch(world, context, touchVector);
                BallSetRotation(world, xRotation, yRotation, 0, 0.5f * (1.0f - bumpyRideBias));
                if (!TeamSetLastTouchPlayer(world, context, player, derivedAnimation.touchType))
                    return false;
            } else if (functionType == e_FunctionType_BallControl) {
                float xRotation = 0, yRotation = 0;
                Vector3 touchVector;
                if (!GetBallControlVector(world, context, player, humanoid.nextStartPosition,
                                          humanoid.nextStartAngle.value(), nextBodyAngle,
                                          outgoingMovementNow(), currentAnimation,
                                          currentAnimation.frameNumber, humanoid.spatialState,
                                          xRotation, yRotation, touchVector))
                    return false;
                if (currentAnimation.originatingCommand.modifier & e_PlayerCommandModifier_KnockOn)
                    touchVector *= 1.35f;
                touchVector =
                    touchVector * (1.0f - bumpyRideBias) + currentBallMomentum * bumpyRideBias;
                BallTouch(world, context, touchVector);
                BallSetRotation(world, xRotation, yRotation, 0, 0.6f * (1.0f - bumpyRideBias));
                if (!TeamSetLastTouchPlayer(world, context, player, derivedAnimation.touchType))
                    return false;
            } else if (functionType == e_FunctionType_Interfere) {
                float xRotation = 0, yRotation = 0;
                Vector3 touchVector;
                if (!GetTrapVector(
                        world, context, player, humanoid.nextStartPosition,
                        humanoid.nextStartAngle.value(), nextBodyAngle, outgoingMovementNow(),
                        currentAnimation, currentAnimation.frameNumber, humanoid.spatialState,
                        humanoid.decayingPositionOffset, xRotation, yRotation, touchVector))
                    return false;
                touchVector = touchVector * 0.5f +
                              (BallPredict(world.serialized.match.ball, 0).Get2D() -
                               humanoid.spatialState.position)
                                      .GetNormalized() *
                                  4.0f +
                              Vector3(0, 0, RandomInRange(world.serialized.rng, 0.5f, 1.5f));
                touchVector =
                    touchVector * (1.0f - bumpyRideBias) + currentBallMomentum * bumpyRideBias;
                BallTouch(world, context, touchVector);
                BallSetRotation(world, xRotation, yRotation, 0.3f * (1.0f - bumpyRideBias));
                if (!TeamSetLastTouchPlayer(world, context, player, e_TouchType_Accidental))
                    return false;
            } else if (functionType == e_FunctionType_Sliding) {
                Vector3 touchVector = derivedAnimation.ballDirection.GetRotated2D(
                    humanoid.spatialState.angle.value());
                touchVector = touchVector * 6.0f + world.serialized.match.ball.momentum * -0.28f;
                touchVector += Vector3(0, 0, 6);
                touchVector =
                    touchVector * (1.0f - bumpyRideBias) + currentBallMomentum * bumpyRideBias;
                BallTouch(world, context, touchVector);
                if (!TeamSetLastTouchPlayer(world, context, player, e_TouchType_Accidental))
                    return false;
            } else if (functionType == e_FunctionType_ShortPass ||
                       functionType == e_FunctionType_LongPass ||
                       functionType == e_FunctionType_HighPass) {
                const state::TouchInformation &touchInformation =
                    currentAnimation.originatingCommand.touchInformation;
                Vector3 ballDirection = touchInformation.desiredDirection;
                float ballPower = touchInformation.desiredPower;
                int targetPlayer = touchInformation.targetPlayer;
                Vector3 inputDirection = touchInformation.inputDirection;
                if (ExternalControllerActive(world, player))
                    inputDirection = HumanGetDirection(world, player);
                Vector3 temporaryBallDirection = ballDirection;
                float temporaryBallPower = ballPower;
                int temporaryTargetPlayer = -1;
                GetPass(world, player, currentAnimation.originatingCommand.desiredFunctionType,
                        inputDirection, touchInformation.inputPower,
                        touchInformation.autoDirectionBias, touchInformation.autoPowerBias,
                        temporaryBallDirection, temporaryBallPower, temporaryTargetPlayer,
                        touchInformation.forcedTargetPlayer);
                float maximumDeviationAngle = 0.15f * PI;
                float angleDifference =
                    temporaryBallDirection.Get2D().GetAngle2D(ballDirection.Get2D());
                if (fabsf(angleDifference) <= maximumDeviationAngle) {
                    ballDirection = temporaryBallDirection;
                    ballPower = temporaryBallPower;
                    targetPlayer = temporaryTargetPlayer;
                } else if (fabsf(angleDifference) < 2.0f * maximumDeviationAngle) {
                    float clampedAngleDifference =
                        clamp(angleDifference, -maximumDeviationAngle, maximumDeviationAngle);
                    ballDirection = ballDirection.GetRotated2D(clampedAngleDifference);
                    if (temporaryTargetPlayer != targetPlayer) {
                        float refinedBias = NormalizedClamp(fabsf(clampedAngleDifference), 0.0f,
                                                            fabsf(angleDifference));
                        ballPower =
                            ballPower * (1.0f - refinedBias) + temporaryBallPower * refinedBias;
                        targetPlayer = temporaryTargetPlayer;
                    } else {
                        ballPower = temporaryBallPower;
                    }
                }
                if (targetPlayer >= 0 && !TeamSelectPlayer(world, context, teamID, targetPlayer))
                    return false;
                float zRotation = 0.0f;
                Vector3 touchVector = ballDirection * 36 * (ballPower + 0.3f);
                {
                    touchVector = GetBestPossibleTouch(world, context, player, touchVector,
                                                       currentAnimation.functionType);
                    float bodyTouchAngle =
                        humanoid.spatialState.bodyDirectionVector.GetAngle2D(touchVector) / PI;
                    if (fabsf(bodyTouchAngle) > 0.5f)
                        bodyTouchAngle = (1.0f - fabsf(bodyTouchAngle)) * SignSide(bodyTouchAngle);
                    bodyTouchAngle *= 2.0f;
                    float amount = bodyTouchAngle * 0.25f;
                    if (currentAnimation.functionType == e_FunctionType_HighPass) amount *= 0.2f;
                    touchVector.Rotate2D(
                        amount *
                        (0.4f + 0.6f * NormalizedClamp(touchVector.GetLength(), 0.0f, 70.0f)));
                    zRotation = amount * -340;
                }
                touchVector =
                    touchVector * (1.0f - bumpyRideBias) + currentBallMomentum * bumpyRideBias;
                BallTouch(world, context, touchVector);
                float forwardness = 3.5f;
                if (currentAnimation.functionType == e_FunctionType_HighPass) forwardness = -1.3f;
                float touchLength = clamp(touchVector.GetLength(), 0.0, 15.0);
                float xRotation =
                    touchVector.GetNormalized(0).coordinates[1] * (touchLength * forwardness);
                float yRotation =
                    touchVector.GetNormalized(0).coordinates[0] * (touchLength * forwardness);
                BallSetRotation(world, xRotation, yRotation, zRotation,
                                0.9f * (1.0f - bumpyRideBias));
                if (!TeamSetLastTouchPlayer(world, context, player, derivedAnimation.touchType))
                    return false;
            } else if (functionType == e_FunctionType_Shot) {
                const state::TouchInformation &touchInformation =
                    currentAnimation.originatingCommand.touchInformation;
                Vector3 inputDirection = touchInformation.inputDirection;
                if (ExternalControllerActive(world, player))
                    inputDirection = HumanGetDirection(world, player);
                // GRF: the result goes unused, as `GetShotVector` reads the direction of the
                // command, and the call has no side effects.
                (void)GetShotDirection(world, player, inputDirection,
                                       touchInformation.autoDirectionBias);
                float xRotation = 0, yRotation = 0, zRotation = 0;
                Vector3 touchVector =
                    GetShotVector(world, context, player, currentAnimation, humanoid.spatialState,
                                  humanoid.decayingPositionOffset, xRotation, yRotation, zRotation);
                touchVector =
                    touchVector * (1.0f - bumpyRideBias) + currentBallMomentum * bumpyRideBias;
                BallTouch(world, context, touchVector);
                BallSetRotation(world, xRotation, yRotation, zRotation,
                                0.7f * (1.0f - bumpyRideBias));
                if (!TeamSetLastTouchPlayer(world, context, player, derivedAnimation.touchType))
                    return false;
            } else if (functionType == e_FunctionType_Deflect) {
                bool canRetain = true;
                if (derivedAnimation.outgoingRetainRawValue == animation::STRING_EMPTY)
                    canRetain = false;
                if (world.serialized.match.ballRetainer != -1) canRetain = false;
                float velocityDifficulty = NormalizedClamp(
                    (world.serialized.match.ball.momentum - humanoid.spatialState.movement)
                        .GetLength(),
                    0.0f, 40.0f);
                float reactionDifficulty = 0.0f;
                int lastTouchPlayer = TeamByID(world, 1 - teamID).lastTouchPlayer;
                if (lastTouchPlayer >= 0)
                    reactionDifficulty = libm::powf(
                        GetLastTouchBias(
                            world, lastTouchPlayer,
                            FloatToInt32(1200 - GetStatistic(world, context, player,
                                                             e_Statistic_PhysicalReaction) *
                                                    400)),
                        0.6f);
                if ((1.0f - velocityDifficulty) * (1.0f - reactionDifficulty) < 0.3f)
                    canRetain = false;
                if (canRetain) {
                    world.serialized.match.ballRetainer = player;
                } else {
                    Vector3 currentBallMovement = world.serialized.match.ball.momentum.Get2D();
                    Vector3 playerMovement = humanoid.spatialState.movement;
                    float randomLateral = RandomInRange(world.serialized.rng, -1, 1);
                    Vector3 touchVector = (-currentBallMovement * 0.1f + playerMovement * 2.0f +
                                           Vector3(-TeamDynamicSide(world, teamID), 0, 0) * 4.0f +
                                           Vector3(0, randomLateral, 0))
                                              .GetNormalized(0) *
                                          (currentBallMovement.GetLength() * 0.3f +
                                           playerMovement.GetLength() * 2.5f);
                    touchVector.coordinates[2] += 1.2f;
                    touchVector =
                        touchVector * (1.0f - bumpyRideBias) + currentBallMomentum * bumpyRideBias;
                    BallTouch(world, context, touchVector);
                    BallSetRotation(world, 0, 0, 0, 0.2f * (1.0f - bumpyRideBias));
                }
                if (!TeamSetLastTouchPlayer(world, context, player, e_TouchType_Accidental))
                    return false;
            }
        }
    }
    if (world.serialized.match.ballRetainer == player) {
        if ((currentAnimation.touchFrame <= currentAnimation.frameNumber &&
             derivedAnimation.outgoingRetainRawValue != animation::STRING_EMPTY) ||
            (currentAnimation.touchFrame > currentAnimation.frameNumber &&
             animationData.incomingRetainState != animation::STRING_EMPTY) ||
            (derivedAnimation.incomingRetainRawValue != animation::STRING_EMPTY &&
             derivedAnimation.outgoingRetainRawValue != animation::STRING_EMPTY)) {
            int bodyPart = -1;
            if (derivedAnimation.outgoingRetainRawValue != animation::STRING_EMPTY)
                bodyPart = derivedAnimation.outgoingRetainBodyPart;
            if (bodyPart < 0 && derivedAnimation.incomingRetainRawValue != animation::STRING_EMPTY)
                bodyPart = derivedAnimation.incomingRetainBodyPart;
            if (bodyPart < 0) CUJU_FATAL("an animation that retains the ball names no body part");
            const animation::Skeleton &skeleton = context.skeletons[player];
            animation::Derived derivedNodes[state::HUMANOID_NODES];
            animation::DeriveNodes(skeleton, humanoid.nodes, derivedNodes);
            const animation::Derived &node = derivedNodes[skeleton.nodeMap[bodyPart]];
            BallTouch(world, context, Vector3(0));
            BallSetRotation(world, 0, 0, 0, 1.0f);
            BallSetPosition(world, node.position + node.rotation * Vector3(0, 0, -0.36f));
            if (!TeamSetLastTouchPlayer(world, context, player, e_TouchType_Intentional_NonKicked))
                return false;
        } else {
            world.serialized.match.ballRetainer = -1;
        }
    }

    const int effectiveFrameCount = animationData.frameCount - 1;

    // A smuggle spreads over the frames the offset to the position, movement or rotation that the
    // selection planned.
    if (currentAnimation.touchFrame != -1 &&
        currentAnimation.frameNumber <= currentAnimation.touchFrame) {
        float value =
            libm::cosf(
                (currentAnimation.frameNumber / (float)(currentAnimation.touchFrame + 1) - 0.5f) *
                PI * 2.0f) +
            1.0f;
        value = value * 0.1f + 0.9f;
        humanoid.spatialState.actionSmuggleMovement =
            (currentAnimation.actionSmuggle / (float)(currentAnimation.touchFrame + 1)) * value *
            100.0f;
        currentAnimation.actionSmuggleOffset +=
            humanoid.spatialState.actionSmuggleMovement / 100.0f;
    } else {
        humanoid.spatialState.actionSmuggleMovement = Vector3(0);
    }

    if (currentAnimation.touchFrame == -1 && currentAnimation.frameNumber <= effectiveFrameCount) {
        float value =
            libm::cosf((currentAnimation.frameNumber / (float)(effectiveFrameCount + 1) - 0.5f) *
                       PI * 2.0f) +
            1.0f;
        value = value * 0.1f + 0.9f;
        humanoid.spatialState.movementSmuggleMovement =
            (currentAnimation.movementSmuggle / (float)(effectiveFrameCount + 1)) * value * 100.0f;
        currentAnimation.movementSmuggleOffset +=
            humanoid.spatialState.movementSmuggleMovement / 100.0f;
    } else {
        humanoid.spatialState.movementSmuggleMovement = Vector3(0);
    }

    const int beginRotationFrameCount = 16;
    int capFrames = beginRotationFrameCount < effectiveFrameCount + 1 ? beginRotationFrameCount
                                                                      : effectiveFrameCount + 1;
    float cappedFrameBias =
        MinimumFloat(1.0f, (currentAnimation.frameNumber + 1) / (float)capFrames);
    float beginFrameBias = cappedFrameBias;
    float endFrameBias = cappedFrameBias;
    if (currentAnimation.touchFrame != -1) {
        int capTouch = beginRotationFrameCount < currentAnimation.touchFrame + 1
                           ? beginRotationFrameCount
                           : currentAnimation.touchFrame + 1;
        beginFrameBias = MinimumFloat(1.0f, (currentAnimation.frameNumber + 1) / (float)capTouch);
        if (currentAnimation.frameNumber > currentAnimation.touchFrame) {
            endFrameBias = (currentAnimation.frameNumber - currentAnimation.touchFrame) /
                           (float)(effectiveFrameCount - currentAnimation.touchFrame);
        } else {
            endFrameBias = 0.0f;
        }
    }
    SetRadian(currentAnimation.rotationSmuggleOffset,
              currentAnimation.rotationSmuggle.begin.value() * (1.0f - beginFrameBias) +
                  currentAnimation.rotationSmuggle.end.value() * endFrameBias);

    if (world.serialized.match.ballRetainer == player &&
        Formation(world, context, player).role == e_PlayerRole_Goalkeeper &&
        world.serialized.match.inSetPiece == false && world.serialized.match.inPlay == true) {
        const int side = TeamDynamicSide(world, teamID);
        const Vector3 ballNow = BallPredict(world.serialized.match.ball, 0);
        if (ballNow.coordinates[1] > 20.05f)
            OffsetPosition(
                world, player,
                Vector3(0, clamp(20.05f - ballNow.coordinates[1], -0.5f, 0.5f), 0) * 0.3f);
        if (ballNow.coordinates[1] < -20.05f)
            OffsetPosition(
                world, player,
                Vector3(0, clamp(-20.05f - ballNow.coordinates[1], -0.5f, 0.5f), 0) * 0.3f);
        if (ballNow.coordinates[0] * -side > -PITCH_HALF_LENGTH + 16.4f)
            OffsetPosition(
                world, player,
                Vector3(clamp((-PITCH_HALF_LENGTH + 16.4f) - ballNow.coordinates[0] * -side, -0.5f,
                              0.5f),
                        0, 0) *
                    -side * 0.3f);
        if (ballNow.coordinates[0] * -side < -PITCH_HALF_LENGTH + 0.1f)
            OffsetPosition(
                world, player,
                Vector3(clamp((-PITCH_HALF_LENGTH + 0.1f) - ballNow.coordinates[0] * -side, -0.5f,
                              0.5f),
                        0, 0) *
                    -side * 0.4f);
    }

    // A frame without a stored position leaves it to the animation, with `noPosition` false.
    humanoid.animationApplyBuffer.frameNumber = currentAnimation.frameNumber;
    if ((uint32_t)currentAnimation.positions.count > (uint32_t)currentAnimation.frameNumber) {
        humanoid.animationApplyBuffer.position =
            humanoid.startPosition + currentAnimation.positions[currentAnimation.frameNumber] +
            currentAnimation.actionSmuggleOffset + currentAnimation.actionSmuggleSustainOffset +
            currentAnimation.movementSmuggleOffset;
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

}  // namespace simulation
}  // namespace cuju

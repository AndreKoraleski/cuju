// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The reset of the world for a set piece and the placement of the players for it.
// Exact: the placement keeps every conversion between double and float of GRF's arithmetic.
#pragma once

#include <vector>

#include "select_animation.h"
#include "team_ai.h"

namespace cuju {
namespace simulation {

constexpr int e_Foot_Right = 1;

CUJU_FUNCTION inline bool LeftTeamOwnsBall(const World &world) {
    const state::ScenarioState &scenario = world.serialized.scenario;
    float leftDistance = 1000000, rightDistance = 1000000;
    for (int player = 0; player < scenario.leftTeamSize; player++)
        leftDistance = StandardMinimumFloat(
            leftDistance,
            (scenario.leftTeam[player].startPosition - scenario.ballPosition).GetLength());
    for (int player = 0; player < scenario.rightTeamSize; player++)
        rightDistance = StandardMinimumFloat(
            rightDistance,
            (scenario.rightTeam[player].startPosition - scenario.ballPosition).GetLength());
    return leftDistance < rightDistance;
}

// A humanoid index counts the players first, then the officials.
CUJU_FUNCTION inline HumanoidState &AnyHumanoid(World &world, int humanoidIndex) {
    const int playerCount = world.serialized.match.playerCount;
    return humanoidIndex < playerCount
               ? world.serialized.match.players[humanoidIndex].base.humanoid
               : world.serialized.match.officials[humanoidIndex - playerCount].humanoid;
}

CUJU_FUNCTION_NO_INLINE inline bool HumanoidResetPosition(World &world,
                                                          const SimulationContext &context,
                                                          int humanoidIndex,
                                                          const Vector3 &newPosition,
                                                          const Vector3 &focusPosition) {
    HumanoidState &humanoid = AnyHumanoid(world, humanoidIndex);
    state::SpatialState &spatialState = humanoid.spatialState;
    humanoid.startPosition = newPosition;
    SetRadian(
        humanoid.startAngle,
        FixAngle((focusPosition - newPosition).GetNormalized(Vector3(0, -1, 0)).GetAngle2D()));
    humanoid.nextStartPosition = humanoid.startPosition;
    humanoid.nextStartAngle = humanoid.startAngle;
    humanoid.previousPosition2D = humanoid.startPosition;
    spatialState.position = humanoid.startPosition;
    spatialState.angle = humanoid.startAngle;
    spatialState.directionVector = Vector3(0, -1, 0).GetRotated2D(humanoid.startAngle.value());
    spatialState.floatVelocity = 0;
    spatialState.actualMovement = Vector3(0);
    spatialState.physicsMovement = Vector3(0);
    spatialState.animationMovement = Vector3(0);
    spatialState.movement = Vector3(0);
    spatialState.actionSmuggleMovement = Vector3(0);
    spatialState.movementSmuggleMovement = Vector3(0);
    spatialState.positionOffsetMovement = Vector3(0);
    SetRadian(spatialState.relativeBodyAngleNonQuantized, 0);
    spatialState.velocityCategory = e_Velocity_Idle;
    spatialState.floatVelocity = 0;
    spatialState.movement = Vector3(0);
    spatialState.relativeBodyDirectionVector = Vector3(0, -1, 0);
    SetRadian(spatialState.relativeBodyAngle, 0);
    spatialState.bodyDirectionVector = Vector3(0, -1, 0);
    SetRadian(spatialState.bodyAngle, 0);
    spatialState.foot = e_Foot_Right;

    Selector selector(world, context, humanoidIndex);
    const int idleAnimationID = selector.GetIdleMovementAnimationID();
    const animation::Animation &idleAnimation = context.animations->animations[idleAnimationID];
    state::AnimationState &currentAnimation = humanoid.currentAnimation;
    currentAnimation.id = idleAnimationID;
    currentAnimation.animation = idleAnimationID;
    currentAnimation.positions.count = 0;
    for (const Vector3 &position : idleAnimation.positions)
        currentAnimation.positions.push_back(position);
    // GRF: the frame draws below the effective frame count, which is one less than the frame count.
    currentAnimation.frameNumber = FloatToInt32(
        RandomInRange(world.serialized.rng, 0, (float)(idleAnimation.frameCount - 1 - 1)));
    currentAnimation.touchFrame = -1;
    currentAnimation.originatingInterrupt = e_InterruptAnimation_None;
    currentAnimation.actionSmuggle = Vector3(0);
    currentAnimation.actionSmuggleOffset = Vector3(0);
    currentAnimation.actionSmuggleSustain = Vector3(0);
    currentAnimation.actionSmuggleSustainOffset = Vector3(0);
    currentAnimation.movementSmuggle = Vector3(0);
    currentAnimation.movementSmuggleOffset = Vector3(0);
    SetRadian(currentAnimation.rotationSmuggle.begin, 0);
    SetRadian(currentAnimation.rotationSmuggle.end, 0);
    SetRadian(currentAnimation.rotationSmuggleOffset, 0);
    currentAnimation.functionType = e_FunctionType_Movement;
    currentAnimation.incomingMovement = Vector3(0);
    currentAnimation.outgoingMovement = Vector3(0);
    currentAnimation.positionOffset = Vector3(0);
    humanoid.previousAnimationFrameNumber = 0;
    humanoid.previousAnimationFunctionType = e_FunctionType_Movement;
    humanoid.nodes[0].position = humanoid.startPosition;
    humanoid.animationApplyBuffer.animation = currentAnimation.animation;
    humanoid.animationApplyBuffer.smooth = false;
    humanoid.animationApplyBuffer.smoothFactor = 0.0f;
    humanoid.animationApplyBuffer.position = humanoid.startPosition;
    humanoid.animationApplyBuffer.orientation = humanoid.startAngle;
    // The offsets of the apply buffer keep their values.
    humanoid.interruptAnimation = e_InterruptAnimation_None;
    humanoid.tripType = 0;
    humanoid.decayingPositionOffset = Vector3(0);
    humanoid.decayingDifficultyFactor = 0.0f;
    humanoid.movementHistory.count = 0;
    humanoid.requeueDelayFrames = 0;
    humanoid.tripDirection = Vector3(0);
    const animation::Skeleton &skeleton = context.skeletons[humanoidIndex];
    for (int bodyPart = 0; bodyPart < state::BODY_PARTS - 1; bodyPart++)
        humanoid.nodes[skeleton.nodeMap[bodyPart]].rotation = Quaternion();
    return true;
}

// The throw-in taker's animation that holds the ball. It runs right after a reset of the
// position, so the movement similarity uses the idle predicates.
CUJU_FUNCTION_NO_INLINE inline bool HumanoidSelectRetainAnimation(World &world,
                                                                  const SimulationContext &context,
                                                                  int humanoidIndex) {
    HumanoidState &humanoid = AnyHumanoid(world, humanoidIndex);
    Selector selector(world, context, humanoidIndex);
    CrudeSelectionQuery query;
    query.byFunctionType = true;
    query.functionType = e_FunctionType_Movement;
    query.byIncomingVelocity = true;
    query.incomingVelocity = e_Velocity_Idle;
    query.byOutgoingVelocity = true;
    query.outgoingVelocity = e_Velocity_Idle;
    // The crude selection filters by the incoming retain state only.
    query.incomingRetainState = animation::STRING_RIGHT_ELBOW;
    AnimationList candidates;
    CrudeSelection(selector.SC(), query, candidates);
    if (candidates.empty()) return Unsupported("no animation retains the ball for the restart");
    selector.SetMovementSimilarityPredicate(Vector3(0, -1, 0), e_Velocity_Idle);
    selector.Sort<float>(candidates,
                         [&](int candidate) { return selector.MovementSimilarityKey(candidate); });
    SetRadian(humanoid.startAngle, FixAngle((Vector3(0) - humanoid.startPosition).GetAngle2D()));
    state::AnimationState &currentAnimation = humanoid.currentAnimation;
    currentAnimation.positions.count = 0;
    currentAnimation.animation = candidates[0];
    currentAnimation.id = candidates[0];
    currentAnimation.frameNumber = 0;
    currentAnimation.touchFrame = -1;
    currentAnimation.actionSmuggle = Vector3(0);
    currentAnimation.actionSmuggleOffset = Vector3(0);
    currentAnimation.actionSmuggleSustain = Vector3(0);
    currentAnimation.actionSmuggleSustainOffset = Vector3(0);
    currentAnimation.movementSmuggle = Vector3(0);
    currentAnimation.movementSmuggleOffset = Vector3(0);
    SetRadian(currentAnimation.rotationSmuggle.begin, 0);
    SetRadian(currentAnimation.rotationSmuggle.end, 0);
    SetRadian(currentAnimation.rotationSmuggleOffset, 0);
    currentAnimation.functionType = e_FunctionType_Movement;
    humanoid.animationApplyBuffer.animation = currentAnimation.animation;
    humanoid.animationApplyBuffer.smooth = false;
    humanoid.animationApplyBuffer.position = humanoid.startPosition;
    humanoid.animationApplyBuffer.orientation = humanoid.startAngle;
    world.serialized.match.ballRetainer = humanoidIndex;
    return true;
}

CUJU_FUNCTION_NO_INLINE inline bool PlayerBaseResetSituation(World &world,
                                                             const SimulationContext &context,
                                                             int humanoidIndex,
                                                             const Vector3 &focusPosition) {
    const int playerCount = world.serialized.match.playerCount;
    if (humanoidIndex < playerCount) {
        state::PlayerBaseState<state::AIControllerState> &base =
            world.serialized.match.players[humanoidIndex].base;
        base.positionHistoryPerSecond.count = 0;
        base.lastTouchTimeMilliseconds = 0;
        base.lastTouchType = e_TouchType_None;
        if (base.isActive) {
            base.humanoid.mentalImageTime = 0;
            Vector3 position = base.humanoid.spatialState.position;
            if (!HumanoidResetPosition(world, context, humanoidIndex, position, focusPosition))
                return false;
        }
        if (ExternalControllerActive(world, humanoidIndex)) {
            HumanControllerReset(world, base.externalController);
        } else {
            base.controller.lastDesiredDirection = Vector3(0);
            base.controller.lastDesiredVelocity = 0;
        }
    } else {
        state::PlayerBaseState<state::NoController> &base =
            world.serialized.match.officials[humanoidIndex - playerCount];
        base.positionHistoryPerSecond.count = 0;
        base.lastTouchTimeMilliseconds = 0;
        base.lastTouchType = e_TouchType_None;
        if (base.isActive) {
            base.humanoid.mentalImageTime = 0;
            Vector3 position = base.humanoid.spatialState.position;
            if (!HumanoidResetPosition(world, context, humanoidIndex, position, focusPosition))
                return false;
        }
        // The controller of an official has no state to reset.
    }
    return true;
}

CUJU_FUNCTION_NO_INLINE inline bool PlayerResetSituation(World &world,
                                                         const SimulationContext &context,
                                                         int player, const Vector3 &focusPosition) {
    if (!PlayerBaseResetSituation(world, context, player, focusPosition)) return false;
    state::PlayerState &playerState = world.serialized.match.players[player];
    playerState.hasPossession = false;
    playerState.hasBestPossession = false;
    playerState.hasUniquePossession = false;
    playerState.possessionDurationMilliseconds = 0;
    playerState.timeNeededToGetToBallMilliseconds = 1000;
    playerState.timeNeededToGetToBallOptimisticMilliseconds = 1000;
    playerState.desiredTimeToBallMilliseconds = 0;
    playerState.manMarking = -1;
    playerState.triggerControlledBallCollision = false;
    playerState.tacticalSituation.forwardSpaceRating = 0;
    playerState.tacticalSituation.toGoalSpaceRating = 0;
    playerState.tacticalSituation.spaceRating = 0;
    return true;
}

CUJU_FUNCTION_NO_INLINE inline void TeamAIReset(state::TeamAIState &teamAI) {
    teamAI.taker = -1;
    teamAI.setPieceType = e_GameMode_Normal;
    teamAI.offsideTrapX = 0;
    teamAI.endApplyAttackingRunMilliseconds = 0;
    teamAI.attackingRunPlayer = -1;
    teamAI.endApplyTeamPressureMilliseconds = 0;
    teamAI.teamPressurePlayer = -1;
    teamAI.endApplyKeeperRushMilliseconds = 0;
    teamAI.teamHasPossession = false;
    teamAI.teamHasUniquePossession = false;
    teamAI.opponentTeamHasPossession = false;
    teamAI.opponentTeamHasUniquePossession = false;
    teamAI.teamHasBestPossession = false;
    teamAI.teamPossessionAmount = 1.0f;
    teamAI.fadingTeamPossessionAmount = 1.0f;
    teamAI.timeNeededToGetToBall = 100;
    teamAI.opponentTimeNeededToGetToBall = 100;
}

CUJU_FUNCTION inline PlayerList TeamAllPlayers(const World &world, int teamID) {
    PlayerList players;
    for (int player = 0; player < world.serialized.match.playerCount; player++)
        if (world.playerTeamID[player] == teamID) players.push_back(player);
    return players;
}

CUJU_FUNCTION_NO_INLINE inline bool TeamResetSituation(World &world,
                                                       const SimulationContext &context, int teamID,
                                                       const Vector3 &focusPosition) {
    state::TeamState &team = TeamByID(world, teamID);
    team.timeNeededToGetToBallMilliseconds = 100;
    team.hasPossession = false;
    team.teamPossessionAmount = 1.0f;
    team.fadingTeamPossessionAmount = 1.0f;
    team.lastTouchPlayer = -1;
    PlayerList players = TeamAllPlayers(world, teamID);
    team.designatedTeamPossessionPlayer = players.at(0);
    for (int player : players)
        if (world.serialized.match.players[player].base.isActive &&
            !PlayerResetSituation(world, context, player, focusPosition))
            return false;
    TeamAIReset(team.teamController);
    return true;
}

CUJU_FUNCTION inline void BallResetSituation(state::BallSerialized &ball,
                                             const Vector3 &focusPosition) {
    // Materializing the ball keeps the prediction at 400 milliseconds.
    BallMaterialize(ball);
    ball.momentum = Vector3(0);
    ball.rotationPerMillisecond = Quaternion();
    for (unsigned int prediction = 0; prediction < BALL_PREDICTION_SPAN_MILLISECONDS / 10;
         prediction++)
        ball.predictions[prediction] = Vector3(focusPosition + Vector3(0, 0, 0.11f));
    ball.orientationPrediction = Quaternion();
    ball.ballPositionHistory.count = 0;
    ball.positionBuffer = Vector3(focusPosition + Vector3(0, 0, 0.11f));
    ball.validPredictions = 0;
    ball.orientationBuffer = Quaternion();
    ball.ballTouchesNet = false;
}

CUJU_FUNCTION_NO_INLINE inline bool MatchResetSituation(World &world,
                                                        const SimulationContext &context,
                                                        const Vector3 &focusPosition) {
    state::MatchState &match = world.serialized.match;
    match.cameraPosition.count = 0;
    match.ballRetainer = -1;
    match.ballIsInGoal = false;
    match.goalScored = false;
    match.mentalImages.count = 0;
    match.goalScored = false;
    match.ballIsInGoal = false;
    for (int32_t &teamID : match.lastTouchTeamIDs) teamID = -1;
    match.lastTouchTeamID = -1;
    match.lastGoalScorer = -1;
    match.bestPossessionTeam = -1;
    match.possessionSideHistory.count = 0;
    match.lastBodyBallCollisionTimeMilliseconds = 0;
    BallResetSituation(match.ball, focusPosition);
    const int firstTeamID = world.FirstTeamID();
    if (!TeamResetSituation(world, context, firstTeamID, focusPosition)) return false;
    if (!TeamResetSituation(world, context, 1 - firstTeamID, focusPosition)) return false;
    // The referee is the first official.
    return PlayerBaseResetSituation(world, context, match.playerCount + 0, focusPosition);
}

CUJU_FUNCTION_NO_INLINE inline bool TeamAIPrepareSetPiece(World &world,
                                                          const SimulationContext &context,
                                                          int teamID, int setPiece, int otherTeamID,
                                                          int kickoffTakerTeamID, int takerTeamID) {
    state::MatchState &match = world.serialized.match;
    state::TeamState &team = TeamByID(world, teamID);
    state::TeamAIState &teamAI = team.teamController;
    const int side = TeamDynamicSide(world, teamID);
    auto flatBallPosition = [&]() { return BallPredict(match.ball, 0).Get2D(); };
    auto resetPosition = [&](int player, const Vector3 &position, const Vector3 &focus) {
        return HumanoidResetPosition(world, context, player, position, focus);
    };
    teamAI.setPieceType = setPiece;
    if (setPiece == e_GameMode_Normal) return true;
    PlayerList players = TeamActivePlayers(world, teamID);
    for (size_t index = 0; index < players.size();) {
        Vector3 focus = flatBallPosition();
        if (Formation(world, context, players[index]).role == e_PlayerRole_Goalkeeper) {
            if (setPiece == e_GameMode_KickOff) {
                focus.coordinates[0] = 0;
                focus.coordinates[1] = 0;
            }
            if (!resetPosition(players[index],
                               Vector3((float)((double)(PITCH_HALF_LENGTH * side) * 0.98), 0, 0),
                               focus))
                return false;
            players.erase(players.begin() + index);
        } else {
            index++;
        }
    }
    const bool isTakerTeam = takerTeamID == teamID;
    // Exact: GRF passes a double here, which rounds to float.
    team.fadingTeamPossessionAmount = isTakerTeam ? 1.5f : 0.5f;
    const double doubleSide = side, halfWidth = PITCH_HALF_LENGTH, halfHeight = PITCH_HALF_WIDTH;
    auto toFloat = [](double value) { return (float)value; };

    switch (setPiece) {
        case e_GameMode_KickOff:
            for (size_t index = 0; index < players.size(); index++) {
                const Vector3 &formationPosition =
                    Formation(world, context, players[index]).position;
                Vector3 basePosition =
                    formationPosition * Vector3(toFloat((double)(-side * PITCH_HALF_LENGTH) * 0.6),
                                                toFloat((double)(-side * PITCH_HALF_WIDTH) * 0.6),
                                                0);
                basePosition.coordinates[1] += RandomInRange(world.serialized.rng, -2.0f, 2.0f);
                basePosition.coordinates[0] = toFloat(basePosition.coordinates[0] * 0.5);
                basePosition.coordinates[0] =
                    toFloat(basePosition.coordinates[0] + (halfWidth * 0.2) * doubleSide);
                if ((double)(basePosition.coordinates[0] * side) < 0.5)
                    basePosition.coordinates[0] = toFloat(0.5 * doubleSide);
                if ((double)basePosition.GetLength() < 9.4) {
                    basePosition.Normalize(Vector3(side, 0, 0));
                    basePosition *= 9.4f;
                }
                if (!resetPosition(players[index], basePosition, flatBallPosition())) return false;
                if (isTakerTeam) {
                    PlayerList result =
                        GetClosestPlayers(world, context, teamID, Vector3(0), false, 2);
                    for (size_t rank = 0; rank < result.size(); rank++)
                        if (!resetPosition(result[rank],
                                           Vector3(0, toFloat((double)rank * 1.4 * doubleSide), 0),
                                           flatBallPosition()))
                            return false;
                }
            }
            break;
        case e_GameMode_GoalKick:
            for (size_t index = 0; index < players.size(); index++) {
                float backXBound, frontXBound;
                if (isTakerTeam) {
                    backXBound = toFloat((double)(side * PITCH_HALF_LENGTH) * 0.5);
                    frontXBound = toFloat((double)(-side * PITCH_HALF_LENGTH) * 0.2);
                } else {
                    backXBound = toFloat((double)(side * PITCH_HALF_LENGTH) * 0.4);
                    frontXBound = toFloat((double)(-side * PITCH_HALF_LENGTH) * 0.1);
                }
                float lowYBound = toFloat(-halfHeight * 0.7),
                      highYBound = toFloat(halfHeight * 0.7);
                Vector3 basePosition = GetAdaptedFormationPosition(
                    world, context, players[index], backXBound, frontXBound, lowYBound, highYBound,
                    0, 0, 0, 0, Vector3(0), 0, 0, 0, false);
                if (!resetPosition(players[index], basePosition, flatBallPosition())) return false;
            }
            break;
        case e_GameMode_Corner:
            for (size_t index = 0; index < players.size(); index++) {
                float backXBound, frontXBound, xFocus, xFocusStrength, yFocus, yFocusStrength,
                    midfieldFocus, midfieldFocusStrength;
                Vector3 ballPosition = flatBallPosition();
                if (isTakerTeam) {
                    backXBound = toFloat((double)(-side * PITCH_HALF_LENGTH) * 0.2);
                    frontXBound = toFloat((double)(-side * PITCH_HALF_LENGTH) * 0.96);
                    xFocus = toFloat(frontXBound * 0.85);
                    xFocusStrength = 0.7f;
                    yFocus = ballPosition.coordinates[1] * 0.1f;
                    yFocusStrength = 0.7f;
                    midfieldFocus = 0.9f;
                    midfieldFocusStrength = 0.5f;
                } else {
                    backXBound = toFloat((double)(side * PITCH_HALF_LENGTH) * 0.98);
                    frontXBound = toFloat((double)(side * PITCH_HALF_LENGTH) * 0.5);
                    xFocus = toFloat(backXBound * 0.94);
                    xFocusStrength = 0.8f;
                    yFocus = ballPosition.coordinates[1] * 0.1f;
                    yFocusStrength = 0.9f;
                    midfieldFocus = 0.1f;
                    midfieldFocusStrength = 0.7f;
                }
                float lowYBound = toFloat(-halfHeight * 0.6),
                      highYBound = toFloat(halfHeight * 0.6);
                Vector3 basePosition = GetAdaptedFormationPosition(
                    world, context, players[index], backXBound, frontXBound, lowYBound, highYBound,
                    xFocus, xFocusStrength, yFocus, yFocusStrength,
                    Vector3(ballPosition.coordinates[0] * 0.95f, ballPosition.coordinates[1] * 0.1f,
                            0),
                    0.9f, midfieldFocus, midfieldFocusStrength, false);
                if (!resetPosition(players[index], basePosition, flatBallPosition())) return false;
            }
            break;
        case e_GameMode_ThrowIn:
            for (size_t index = 0; index < players.size(); index++) {
                float backXBound, frontXBound, xFocus, xFocusStrength, yFocus, yFocusStrength;
                Vector3 ballPosition = flatBallPosition();
                if (isTakerTeam) {
                    backXBound = clamp(ballPosition.coordinates[0] + 30 * side, -PITCH_HALF_LENGTH,
                                       PITCH_HALF_LENGTH);
                    frontXBound = clamp(ballPosition.coordinates[0] + 20 * -side,
                                        -PITCH_HALF_LENGTH, PITCH_HALF_LENGTH);
                    xFocus = clamp(ballPosition.coordinates[0] + 4 * -side, -PITCH_HALF_LENGTH,
                                   PITCH_HALF_LENGTH);
                    xFocusStrength = toFloat(0.4);
                    yFocus = toFloat(ballPosition.coordinates[1] * 0.996);
                    yFocusStrength = toFloat(0.6);
                } else {
                    backXBound = clamp(ballPosition.coordinates[0] + 30 * side, -PITCH_HALF_LENGTH,
                                       PITCH_HALF_LENGTH);
                    frontXBound = clamp(ballPosition.coordinates[0] + 15 * -side,
                                        -PITCH_HALF_LENGTH, PITCH_HALF_LENGTH);
                    xFocus = clamp(ballPosition.coordinates[0] + 16 * side, -PITCH_HALF_LENGTH,
                                   PITCH_HALF_LENGTH);
                    xFocusStrength = toFloat(0.2);
                    yFocus = toFloat(ballPosition.coordinates[1] * 0.95);
                    yFocusStrength = toFloat(0.5);
                }
                float lowYBound = -PITCH_HALF_WIDTH * 0.75f + ballPosition.coordinates[1] * 0.25f;
                float highYBound = PITCH_HALF_WIDTH * 0.75f + ballPosition.coordinates[1] * 0.25f;
                Vector3 basePosition = GetAdaptedFormationPosition(
                    world, context, players[index], backXBound, frontXBound, lowYBound, highYBound,
                    xFocus, xFocusStrength, yFocus, yFocusStrength, ballPosition, toFloat(0.7), 0,
                    0, false);
                if (!resetPosition(players[index], basePosition, flatBallPosition())) return false;
            }
            break;
        case e_GameMode_FreeKick: {
            for (size_t index = 0; index < players.size(); index++) {
                float backXBound, frontXBound, xFocus, xFocusStrength, yFocus, yFocusStrength;
                Vector3 ballPosition = flatBallPosition();
                const float sideHalfWidth = side * PITCH_HALF_LENGTH;
                if (isTakerTeam) {
                    float xOffset = toFloat(
                        (double)clamp((ballPosition.coordinates[0] * -side) / PITCH_HALF_LENGTH,
                                      toFloat(-1.0), toFloat(1.0)) *
                            0.5 +
                        0.5);
                    backXBound =
                        clamp(toFloat((double)sideHalfWidth * (0.7 - (double)xOffset * 0.7)),
                              -PITCH_HALF_LENGTH, PITCH_HALF_LENGTH);
                    frontXBound =
                        clamp(toFloat((double)sideHalfWidth * (-0.3 - (double)xOffset * 0.7)),
                              -PITCH_HALF_LENGTH, PITCH_HALF_LENGTH);
                    xFocus = clamp(ballPosition.coordinates[0] + 10 * -side, -PITCH_HALF_LENGTH,
                                   PITCH_HALF_LENGTH);
                    xFocusStrength = toFloat(0.5 + (double)xOffset * 0.2);
                    yFocus = toFloat(ballPosition.coordinates[1] * 0.4);
                    yFocusStrength = toFloat(0.6 + (double)xOffset * 0.2);
                } else {
                    float xOffset = toFloat(
                        (double)clamp((ballPosition.coordinates[0] * -side) / PITCH_HALF_LENGTH,
                                      toFloat(-1.0), toFloat(1.0)) *
                            0.5 +
                        0.5);
                    backXBound =
                        clamp(toFloat((double)sideHalfWidth * (1.0 - (double)xOffset * 0.6)),
                              -PITCH_HALF_LENGTH, PITCH_HALF_LENGTH);
                    frontXBound =
                        clamp(toFloat((double)sideHalfWidth * (0.5 - (double)xOffset * 0.8)),
                              -PITCH_HALF_LENGTH, PITCH_HALF_LENGTH);
                    xFocus = clamp(ballPosition.coordinates[0] + 20 * side, -PITCH_HALF_LENGTH,
                                   PITCH_HALF_LENGTH);
                    xFocusStrength = toFloat(0.6 - (double)xOffset * 0.4);
                    yFocus = toFloat(ballPosition.coordinates[1] * 0.2);
                    yFocusStrength = toFloat(0.8 - (double)xOffset * 0.4);
                }
                float lowYBound = toFloat(-halfHeight * 0.7),
                      highYBound = toFloat(halfHeight * 0.7);
                Vector3 basePosition = GetAdaptedFormationPosition(
                    world, context, players[index], backXBound, frontXBound, lowYBound, highYBound,
                    xFocus, xFocusStrength, yFocus, yFocusStrength, ballPosition, toFloat(0.4), 0,
                    0, false);
                if (!isTakerTeam) {
                    if ((double)(basePosition - flatBallPosition()).GetLength() < 9.15)
                        basePosition = flatBallPosition() +
                                       (basePosition - flatBallPosition()).GetNormalized() * 9.15f;
                }
                if (!resetPosition(players[index], basePosition, flatBallPosition())) return false;
            }
            if (!isTakerTeam &&
                (double)(flatBallPosition() - Vector3(side * PITCH_HALF_LENGTH, 0, 0)).GetLength() <
                    40.0) {
                PlayerList result =
                    GetClosestPlayers(world, context, teamID, flatBallPosition(), false, 3);
                for (size_t rank = 0; rank < result.size(); rank++) {
                    Vector3 toGoal = (Vector3(side * PITCH_HALF_LENGTH, 0, 0) - flatBallPosition())
                                         .GetNormalized(0);
                    toGoal += Vector3(0, toFloat(1.0 - (double)rank), 0) * toFloat(0.07);
                    toGoal.Normalize();
                    if (!resetPosition(result[rank], flatBallPosition() + toGoal * 9.15f,
                                       flatBallPosition()))
                        return false;
                }
            }
            break;
        }
        case e_GameMode_Penalty:
            for (size_t index = 0; index < players.size(); index++) {
                float backXBound, frontXBound, xFocus, xFocusStrength, yFocus, yFocusStrength;
                Vector3 ballPosition = flatBallPosition();
                if (isTakerTeam) {
                    backXBound = clamp(ballPosition.coordinates[0] + 50 * side, -PITCH_HALF_LENGTH,
                                       PITCH_HALF_LENGTH);
                    frontXBound = clamp(ballPosition.coordinates[0] + 10 * -side,
                                        -PITCH_HALF_LENGTH, PITCH_HALF_LENGTH);
                    xFocus = ballPosition.coordinates[0];
                    xFocusStrength = toFloat(0.6);
                    yFocus = toFloat(0.0);
                    yFocusStrength = toFloat(0.8);
                } else {
                    backXBound = clamp(ballPosition.coordinates[0] + 11 * side, -PITCH_HALF_LENGTH,
                                       PITCH_HALF_LENGTH);
                    frontXBound = clamp(ballPosition.coordinates[0] + 20 * -side,
                                        -PITCH_HALF_LENGTH, PITCH_HALF_LENGTH);
                    xFocus = ballPosition.coordinates[0];
                    xFocusStrength = toFloat(1.0);
                    yFocus = toFloat(0.0);
                    yFocusStrength = toFloat(1.0);
                }
                float lowYBound = toFloat(-halfHeight * 0.8),
                      highYBound = toFloat(halfHeight * 0.8);
                Vector3 basePosition = GetAdaptedFormationPosition(
                    world, context, players[index], backXBound, frontXBound, lowYBound, highYBound,
                    xFocus, xFocusStrength, yFocus, yFocusStrength, Vector3(0), 0, 0, 0, false);
                int penaltySide = (BallPredict(match.ball, 0).coordinates[0] < 0) ? -1 : 1;
                if ((double)(basePosition.coordinates[0] * penaltySide) > halfWidth - 16.5 - 0.5)
                    basePosition.coordinates[0] = toFloat((halfWidth - 16.5 - 0.5) * penaltySide);
                if ((double)(basePosition - flatBallPosition()).GetLength() < 9.15 + 0.5)
                    basePosition =
                        flatBallPosition() +
                        (basePosition - flatBallPosition()).GetNormalized() * toFloat(9.15 + 0.5);
                if (!resetPosition(players[index], basePosition, flatBallPosition())) return false;
            }
            break;
        default:
            for (size_t index = 0; index < players.size(); index++) {
                Vector3 basePosition =
                    Formation(world, context, players[index]).position *
                    Vector3(toFloat((double)(-side * PITCH_HALF_LENGTH) * 0.7),
                            toFloat((double)(-side * PITCH_HALF_WIDTH) * 0.7), 0);
                if (!resetPosition(players[index], basePosition, flatBallPosition())) return false;
            }
            break;
    }

    if (setPiece == e_GameMode_KickOff) {
        PlayerList formationPlayers = TeamAllPlayers(world, teamID);
        PlayerList playersToPosition = TeamAllPlayers(world, teamID);
        // GRF: after the start, a kickoff to the team that did not own the ball at the start swaps
        // the start positions of the two teams, if they are of equal size.
        if (match.matchTimeMilliseconds > 0 &&
            TeamAllPlayers(world, otherTeamID).size() == formationPlayers.size() &&
            (LeftTeamOwnsBall(world) ^ (kickoffTakerTeamID == 0)))
            formationPlayers = TeamAllPlayers(world, otherTeamID);
        for (size_t index = 0; index < formationPlayers.size(); index++) {
            if (!match.players[playersToPosition[index]].base.isActive) continue;
            Vector3 basePosition =
                Formation(world, context, formationPlayers[index]).startPosition *
                Vector3(-side * PITCH_HALF_LENGTH, -side * PITCH_HALF_WIDTH, 0);
            if (!resetPosition(playersToPosition[index], basePosition, flatBallPosition()))
                return false;
        }
    }

    if (isTakerTeam) {
        Vector3 ballPosition = flatBallPosition();
        PlayerList closest =
            GetClosestPlayers(world, context, teamID, flatBallPosition(), false, 2);
        if (closest.empty()) return Unsupported("the set piece has no taker");
        teamAI.taker = closest[0];
        if (setPiece == e_GameMode_KickOff) {
            // The taker of a kickoff keeps its formation position.
        } else if (setPiece == e_GameMode_ThrowIn) {
            if (!resetPosition(
                    closest[0],
                    ballPosition + flatBallPosition().GetNormalized(Vector3(0, -side, 0)) * 0.3f,
                    ballPosition))
                return false;
        } else if (setPiece == e_GameMode_FreeKick) {
            if (!resetPosition(teamAI.taker, ballPosition + Vector3(side, 0, 0) * 2.3f,
                               ballPosition))
                return false;
        } else {
            if (!resetPosition(
                    teamAI.taker,
                    ballPosition + flatBallPosition().GetNormalized(Vector3(0, -side, 0)) * 2.3f,
                    ballPosition))
                return false;
        }
        if (setPiece == e_GameMode_ThrowIn &&
            !HumanoidSelectRetainAnimation(world, context, teamAI.taker))
            return false;
        if (setPiece == e_GameMode_Penalty) {
            if (!resetPosition(teamAI.taker, ballPosition + Vector3(side, 0, 0) * 3.0f,
                               ballPosition))
                return false;
        }
    } else {
        teamAI.taker = -1;
    }
    return true;
}

CUJU_FUNCTION_NO_INLINE inline bool PlayerSendOff(World &world, const SimulationContext &context,
                                                  int player) {
    state::MatchState &match = world.serialized.match;
    state::PlayerState &playerState = match.players[player];
    const int teamID = world.playerTeamID[player];
    RandomInRange(world.serialized.rng, 0, 3);  // GRF: the referee's message draws once.
    auto currentPosition = [&]() { return playerState.base.humanoid.spatialState.position; };
    if (!PlayerResetSituation(world, context, player, currentPosition())) return false;
    if (playerState.base.externalController >= 0) {
        state::TeamState &team = TeamByID(world, teamID);
        for (int gamerIndex = 0; gamerIndex < team.humanGamers.count; gamerIndex++) {
            if (team.humanGamers[gamerIndex].selectedPlayer != player) continue;
            int somePlayer =
                GetClosestPlayer(world, context, teamID, currentPosition(), true, player, true);
            team.mainSelectedPlayer = somePlayer;
            HumanGamerSetSelectedPlayer(world, GlobalGamerIndex(world, teamID, gamerIndex),
                                        somePlayer);
        }
    }
    // GRF: the deactivation resets the player a second time.
    if (!PlayerResetSituation(world, context, player, currentPosition())) return false;
    playerState.base.isActive = false;
    playerState.base.externalController = -1;
    TeamByID(world, teamID).designatedTeamPossessionPlayer =
        GetClosestPlayer(world, context, teamID, BallPredict(match.ball, 0).Get2D(), false);
    if (Formation(world, context, player).role == e_PlayerRole_Goalkeeper) {
        int firstActive = -1;
        for (int candidate = 0; candidate < match.playerCount && firstActive < 0; candidate++)
            if (world.playerTeamID[candidate] == teamID && match.players[candidate].base.isActive)
                firstActive = candidate;
        if (firstActive < 0) CUJU_FATAL("the sending-off leaves no active player");
        world.formationOverride[firstActive] = (int8_t)(1 + FormationSource(world, player));
    }
    return true;
}

CUJU_FUNCTION_NO_INLINE inline bool RefereePrepareSetPiece(World &world,
                                                           const SimulationContext &context,
                                                           int setPiece) {
    state::RefereeState &referee = world.serialized.match.referee;
    state::RefereeBufferState &buffer = referee.buffer;
    if (setPiece == e_GameMode_FreeKick) {
        buffer.restartPosition.coordinates[0] =
            clamp(buffer.restartPosition.coordinates[0], (float)(-0.95 * PITCH_HALF_LENGTH),
                  (float)(0.95 * PITCH_HALF_LENGTH));
        buffer.restartPosition.coordinates[1] =
            clamp(buffer.restartPosition.coordinates[1], (float)(-0.95 * PITCH_HALF_WIDTH),
                  (float)(0.95 * PITCH_HALF_WIDTH));
    }
    if (!MatchResetSituation(world, context,
                             world.serialized.scenario.reverseTeamProcessing
                                 ? -buffer.restartPosition
                                 : buffer.restartPosition))
        return false;
    const int firstTeamID = world.FirstTeamID(), secondTeamID = 1 - firstTeamID;
    // The buffer stores the team of the set piece as its slot in the processing order.
    const int setPieceTeamID = buffer.setPieceTeam == 0 ? firstTeamID : secondTeamID;
    if (!TeamAIPrepareSetPiece(world, context, firstTeamID, setPiece, secondTeamID, setPieceTeamID,
                               buffer.teamID))
        return false;
    if (!TeamAIPrepareSetPiece(world, context, secondTeamID, setPiece, firstTeamID, setPieceTeamID,
                               buffer.teamID))
        return false;
    buffer.taker = TeamByID(world, buffer.teamID).teamController.taker;
    referee.offsidePlayers.count = 0;
    return true;
}

}  // namespace simulation
}  // namespace cuju

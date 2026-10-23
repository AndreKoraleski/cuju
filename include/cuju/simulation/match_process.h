// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// One tick of a match. Each step runs in the frame it expects, between two `MirrorMatch` calls.
#pragma once

#include "../profiling.h"

#include "../animation/apply.h"
#include "../ball.h"
#include "ai_functions.h"
#include "ball_touch.h"
#include "camera.h"
#include "collisions.h"
#include "geometry_offsets.h"
#include "officials.h"
#include "team_process.h"
#include "team_select.h"
#include "humanoid.h"

namespace cuju {
namespace simulation {

// Slot 0 holds the newest of at most three mental images.
CUJU_FUNCTION_NO_INLINE inline void PushMentalImage(World &world) {
    CUJU_PROFILE_SCOPE(PUSH_MENTAL_IMAGE);
    state::MatchState &match = world.serialized.match;
    // Speed: word copies across the lanes. A struct assignment unrolls into thousands of
    // instructions.
    int imageCount = match.mentalImages.count < 3 ? match.mentalImages.count + 1 : 3;
    for (int image = imageCount - 1; image > 0; image--) {
        WarpCopy(match.mentalImages[image], match.mentalImages[image - 1]);
        world.flags.mentalBallMirrored[image] = world.flags.mentalBallMirrored[image - 1];
    }
    state::MentalImageState &mentalImage = match.mentalImages[0];
    WarpReset(mentalImage);
    mentalImage.timeStampMilliseconds = (uint32_t)match.actualTimeMilliseconds;
    mentalImage.maximumDistanceDeviation = 2.5f;
    mentalImage.maximumMovementDeviation = WALK_VELOCITY;
    for (int slot = 0; slot < 2; slot++) {
        const int teamID = slot == 0 ? world.FirstTeamID() : 1 - world.FirstTeamID();
        for (int player : TeamActivePlayers(world, teamID)) {
            const state::SpatialState &spatialState = Humanoid(world, player).spatialState;
            state::PlayerImage playerImage{};
            playerImage.player = player;
            playerImage.position = spatialState.position;
            playerImage.directionVector = spatialState.directionVector;
            playerImage.velocity = spatialState.velocityCategory;
            playerImage.movement = spatialState.movement;
            playerImage.role = match.players[player].dynamicFormationEntry.role;
            mentalImage.players.push_back(playerImage);
        }
    }
    const int predictionCount = (int)(BALL_PREDICTION_SPAN_MILLISECONDS / 10);
    mentalImage.ballPredictions.count = predictionCount;
    WarpCopyWords(reinterpret_cast<uint32_t *>(&mentalImage.ballPredictions.elements[0]),
                  reinterpret_cast<const uint32_t *>(&match.ball.predictions[0]),
                  predictionCount * (int)(sizeof(Vector3) / 4));
    mentalImage.lazyFlip = match.ball.lazyFlip;
    world.flags.mentalBallMirrored[0] = false;
    match.mentalImages.count = imageCount;
}

CUJU_FUNCTION_NO_INLINE inline void CalculateBestPossessionTeamID(World &world) {
    state::MatchState &match = world.serialized.match;
    if (match.ballRetainer != -1) {
        match.bestPossessionTeam = TeamSlot(world, world.playerTeamID[match.ballRetainer]);
        return;
    }
    const int first = world.FirstTeamID();
    int firstTime = TeamByID(world, first).timeNeededToGetToBallMilliseconds;
    int secondTime = TeamByID(world, 1 - first).timeNeededToGetToBallMilliseconds;
    if (firstTime < secondTime)
        match.bestPossessionTeam = TeamSlot(world, first);
    else if (firstTime > secondTime)
        match.bestPossessionTeam = TeamSlot(world, 1 - first);
    else
        match.bestPossessionTeam = -1;
}

// Exact: the bounds of the parameters compare in double.
CUJU_FUNCTION_NO_INLINE inline bool TriangleIntersectsLine(const Vector3 vertices[3],
                                                           const Vector3 &normal,
                                                           const Vector3 &lineStart,
                                                           const Vector3 &lineEnd) {
    Vector3 firstEdge = vertices[1] - vertices[0];
    Vector3 secondEdge = vertices[2] - vertices[0];
    Vector3 direction = lineEnd - lineStart;
    Vector3 startOffset = lineStart - vertices[0];
    float numerator = -normal.GetDotProduct(startOffset);
    float denominator = normal.GetDotProduct(direction);
    if (fabsf(denominator) < 0.000001f) return false;
    float ratio = numerator / denominator;
    if ((double)ratio <= 0.0 || (double)ratio >= 1.0) return false;
    Vector3 intersectVector = lineStart + direction * ratio;
    float firstSquared = firstEdge.GetDotProduct(firstEdge);
    float edgesProduct = firstEdge.GetDotProduct(secondEdge);
    float secondSquared = secondEdge.GetDotProduct(secondEdge);
    Vector3 intersectionOffset = intersectVector - vertices[0];
    float offsetFirst = intersectionOffset.GetDotProduct(firstEdge);
    float offsetSecond = intersectionOffset.GetDotProduct(secondEdge);
    float determinant = edgesProduct * edgesProduct - firstSquared * secondSquared;
    float firstWeight = (edgesProduct * offsetSecond - secondSquared * offsetFirst) / determinant;
    if ((double)firstWeight < 0.0 || (double)firstWeight > 1.0) return false;
    float secondWeight = (edgesProduct * offsetFirst - firstSquared * offsetSecond) / determinant;
    if ((double)secondWeight < 0.0 || (double)(firstWeight + secondWeight) > 1.0) return false;
    return true;
}

CUJU_FUNCTION_NO_INLINE inline bool CheckForGoal(const World &world, int side,
                                                 const Vector3 &previousBallPosition) {
    const state::BallSerialized &ball = world.serialized.match.ball;
    if ((double)fabsf(BallPredict(ball, 10).coordinates[0]) < (double)PITCH_HALF_LENGTH - 1.0)
        return false;
    const Vector3 lineStart = previousBallPosition, lineEnd = BallPredict(ball, 0);
    const float x = (PITCH_HALF_LENGTH + LINE_HALF_WIDTH + 0.11f) * side;
    const Vector3 normal(-side, 0, 0);
    const Vector3 lowerTriangle[3] = {Vector3(x, 3.7f, 0), Vector3(x, -3.7f, 0),
                                      Vector3(x, 3.7f, 2.5f)};
    const Vector3 upperTriangle[3] = {Vector3(x, -3.7f, 0), Vector3(x, -3.7f, 2.5f),
                                      Vector3(x, 3.7f, 2.5f)};
    bool intersectsLower = TriangleIntersectsLine(lowerTriangle, normal, lineStart, lineEnd);
    bool intersectsUpper = TriangleIntersectsLine(upperTriangle, normal, lineStart, lineEnd);
    if ((double)fabsf(previousBallPosition.coordinates[1]) > 3.7 &&
        (double)fabsf(previousBallPosition.coordinates[0]) >
            (double)(PITCH_HALF_LENGTH - LINE_HALF_WIDTH) - 0.11)
        return false;
    return intersectsLower || intersectsUpper;
}

// Returns `false` when the tick ends early, while play waits for a restart, which only advances
// the clock, or on an unsupported path, which also sets `ok` to `false`.
CUJU_FUNCTION_NO_INLINE inline bool MatchProcess(World &world, const SimulationContext &context,
                                                 bool &ok) {
    CUJU_PROFILE_SCOPE(MATCH_PROCESS);
    state::MatchState &match = world.serialized.match;
    ok = true;
    const bool reverse = world.serialized.scenario.reverseTeamProcessing;
    const int first = world.FirstTeamID(), second = 1 - first;

    MirrorMatch(world, reverse, !reverse, reverse);
    if (match.inPlay && !CheckBallCollisions(world, context)) return ok = false;
    if (!RefereeProcess(world, context)) return ok = false;
    Vector3 previousBallPosition = BallPredict(match.ball, 0);
    MirrorMatch(world, reverse, !reverse, reverse);
    if (!match.inPlay && match.referee.buffer.prepareTime + 10 < match.actualTimeMilliseconds) {
        BumpActualTime(world, context, 10);
        return false;
    }
    MirrorMatch(world, false, false, reverse);
    BallProcessSerialized(match.ball, match.ballIsInGoal);
    MirrorMatch(world, false, false, reverse);

    if (match.mentalImages.count == 0 || match.actualTimeMilliseconds % 100 == 0)
        PushMentalImage(world);

    if (!TeamUpdateSwitch(world, context, first)) return ok = false;
    if (!TeamUpdateSwitch(world, context, second)) return ok = false;

    MirrorMatch(world, first == 1, first == 0, first == 1);
    if (!TeamProcess(world, context, first)) return ok = false;
    MirrorMatch(world, true, true, true);
    if (!TeamProcess(world, context, second)) return ok = false;
    MirrorMatch(world, first == 0, first == 1, first == 0);

    MirrorMatch(world, reverse, !reverse, reverse);
    if (!OfficialsProcess(world, context)) return ok = false;
    MirrorMatch(world, reverse, !reverse, reverse);

    MirrorMatch(world, first == 1, first == 0, first == 1);
    TeamUpdatePossessionStatistics(world, context, first);
    MirrorMatch(world, true, true, true);
    TeamUpdatePossessionStatistics(world, context, second);
    MirrorMatch(world, first == 0, first == 1, first == 0);

    CalculateBestPossessionTeamID(world);

    if (match.ballRetainer == -1) {
        if (match.bestPossessionTeam != -1) {
            const int bestTeamID = match.bestPossessionTeam == 0 ? first : second;
            int candidate = TeamByID(world, bestTeamID).designatedTeamPossessionPlayer;
            if (candidate != match.designatedPossessionPlayer) {
                uint32_t designatedTime = match.players[match.designatedPossessionPlayer]
                                              .timeNeededToGetToBallMilliseconds;
                uint32_t candidateTime = match.players[candidate].timeNeededToGetToBallMilliseconds;
                float timeRating = (float)(candidateTime + 10) / (float)(designatedTime + 10);
                if (timeRating < 0.85f) match.designatedPossessionPlayer = candidate;
            }
        } else {
            match.designatedPossessionPlayer =
                TeamByID(world, world.playerTeamID[match.designatedPossessionPlayer])
                    .designatedTeamPossessionPlayer;
        }
    } else {
        match.designatedPossessionPlayer = match.ballRetainer;
    }

    MirrorMatch(world, reverse, !reverse, reverse);
    if (!CheckHumanoidCollisions(world, context)) return ok = false;

    BumpActualTime(world, context, 10);

    bool firstTeamGoal = false, secondTeamGoal = false;
    if (match.inPlay) {
        firstTeamGoal = CheckForGoal(world, TeamDynamicSide(world, first), previousBallPosition);
        secondTeamGoal = CheckForGoal(world, TeamDynamicSide(world, second), previousBallPosition);
    }
    bool goal = firstTeamGoal | secondTeamGoal;
    match.ballIsInGoal |= goal;
    MirrorMatch(world, reverse, !reverse, reverse);
    if (match.inPlay) {
        if (goal) {
            // The ball entered one team's goal, so the other scores. `goalCount` is in processing
            // order.
            const int scoringTeam = firstTeamGoal ? second : first;
            match.goalCount[scoringTeam == first ? 0 : 1] += 1;
            match.goalScored = true;
            match.lastGoalTeam = TeamSlot(world, scoringTeam);
            if (!TeamUpdateTactics(world, scoringTeam)) return ok = false;
        }
        if (firstTeamGoal || secondTeamGoal) {
            const int goalTeam = match.lastGoalTeam == 0 ? first : second;
            bool ownGoal = true;
            if (match.lastTouchTeamIDs[e_TouchType_Intentional_Kicked] == goalTeam ||
                match.lastTouchTeamIDs[e_TouchType_Intentional_NonKicked] == goalTeam)
                ownGoal = false;
            match.lastGoalScorer =
                TeamByID(world, ownGoal ? 1 - goalTeam : goalTeam).lastTouchPlayer;
        }
    }

    if (match.inPlay && match.bestPossessionTeam != -1) {
        float sideValue = 0;
        sideValue +=
            (TeamByID(world, 0).fadingTeamPossessionAmount - 0.5f) * TeamDynamicSide(world, 0);
        sideValue +=
            (TeamByID(world, 1).fadingTeamPossessionAmount - 0.5f) * TeamDynamicSide(world, 1);
        auto &history = match.possessionSideHistory;
        if (history.count == 600) {
            for (int index = 0; index + 1 < history.count; index++)
                history[index] = history[index + 1];
            history[history.count - 1] = sideValue;
        } else {
            history.push_back(sideValue);
        }
    }

    state::RefereeState &referee = match.referee;
    if (referee.buffer.active && (referee.foul.foulType == 2 || referee.foul.foulType == 3) &&
        referee.buffer.stopTime < match.actualTimeMilliseconds - 1000) {
        auto setAutoUpdateInGameCamera = [&](bool autoUpdate) {
            if (autoUpdate != match.autoUpdateInGameCamera) {
                match.cameraPosition.count = 0;
                match.autoUpdateInGameCamera = autoUpdate;
            }
        };
        if (referee.buffer.prepareTime > match.actualTimeMilliseconds) {
            setAutoUpdateInGameCamera(false);
            Vector3 refereePosition = match.officials[0].humanoid.spatialState.position;
            if (reverse) refereePosition.Mirror();
            const Vector3 target = refereePosition + Vector3(0, 0, 0.8f);
            const float zoom = 1.5f;
            match.cameraOrientation.SetAngleAxis(0.4f * PI, Vector3(1, 0, 0));
            match.cameraNodeOrientation.SetAngleAxis(
                (float)((double)target.GetAngle2D() + 1.5 * (double)PI), Vector3(0, 0, 1));
            match.cameraNodePosition =
                target - target.Get2D().GetNormalized(Vector3(0, -1, 0)) * 10 * (1.0f / zoom) +
                Vector3(0, 0, 3);
            match.cameraFieldOfView = 60.0f;
            match.cameraNearCap = 1;
            match.cameraFarCap = 220;
            if (match.officials[0].humanoid.currentAnimation.functionType ==
                    e_FunctionType_Special &&
                match.referee.buffer.active) {
                match.referee.buffer.prepareTime = match.actualTimeMilliseconds + 1000;
                match.referee.buffer.startTime = match.referee.buffer.prepareTime + 2000;
            }
        } else {
            setAutoUpdateInGameCamera(true);
        }
    }
    return true;
}

// Runs on one lane.
CUJU_FUNCTION_NO_INLINE inline void PlayerPreparePutBuffers(World &world,
                                                            const SimulationContext &context,
                                                            int player) {
    CalculateGeometryOffsets(world, context, player);
    state::PlayerState &playerState = world.serialized.match.players[player];
    const state::TeamState &team = TeamByID(world, world.playerTeamID[player]);
    const bool externallyControlled = ExternalControllerActive(world, player);
    playerState.bufferNameCaptionShowCondition = externallyControlled;
    int humanGamerCount = 0;
    for (int gamerIndex = 0; gamerIndex < team.humanGamers.count; gamerIndex++)
        if (!team.humanGamers[gamerIndex].controller.keyboard.disabled) humanGamerCount++;
    if (humanGamerCount == 0)
        playerState.bufferNameCaptionShowCondition = team.designatedTeamPossessionPlayer == player;
    int color = 5;
    const state::HumanGamerState *gamer = nullptr;
    if (playerState.base.externalController >= 0) {
        int controllerIndex = playerState.base.externalController,
            firstTeamGamerCount = world.serialized.match.teams[0].humanGamers.count;
        gamer = controllerIndex < firstTeamGamerCount
                    ? &world.serialized.match.teams[0].humanGamers[controllerIndex]
                    : &world.serialized.match.teams[1]
                           .humanGamers[controllerIndex - firstTeamGamerCount];
        color = gamer->keyboardIndex % 6;
    }
    const float colors[6][3] = {{100, 140, 255}, {100, 255, 140}, {255, 110, 110},
                                {255, 255, 60},  {200, 80, 200},  {200, 200, 200}};
    playerState.bufferPlayerColor = Vector3(colors[color][0], colors[color][1], colors[color][2]);
    if (externallyControlled && gamer->controller.actionMode == 2)
        playerState.bufferPlayerColor =
            playerState.bufferPlayerColor *
            (libm::sinf(world.serialized.match.actualTimeMilliseconds * 0.02f) * 0.3f + 0.7f);
}

CUJU_FUNCTION inline void MatchPreparePutBuffers(World &world, const SimulationContext &context) {
    CUJU_PROFILE_SCOPE(PUT_BUFFERS);
    const int first = world.FirstTeamID(), second = 1 - first;
    // The lanes split a team. Each player writes only its own offsets and buffers, and reads only
    // the ball and the spatial state of the other team.
    MirrorMatch(world, false, false, first == 1);
    {
        PlayerList players = TeamActivePlayers(world, first);
        WarpFor(players.count,
                [&](int index) { PlayerPreparePutBuffers(world, context, players[index]); });
    }
    MirrorMatch(world, false, false, true);
    {
        PlayerList players = TeamActivePlayers(world, second);
        WarpFor(players.count,
                [&](int index) { PlayerPreparePutBuffers(world, context, players[index]); });
    }
    MirrorMatch(world, false, false, first == 0);
}

// The humanoids are independent, so one batch spreads every node across the lanes.
CUJU_FUNCTION inline bool MatchFetchPutBuffers(World &world, const SimulationContext &context) {
    const int first = world.FirstTeamID(), second = 1 - first;
    const animation::Skeleton *skeletons[animation::MAXIMUM_PUT_HUMANOIDS];
    state::HumanoidState *humanoids[animation::MAXIMUM_PUT_HUMANOIDS];
    int count = 0;
    for (int team : {first, second})
        for (int player : TeamActivePlayers(world, team)) {
            skeletons[count] = &context.skeletons[player];
            humanoids[count++] = &world.serialized.match.players[player].base.humanoid;
        }
    const int playerCount = world.serialized.match.playerCount;
    for (int official = 0; official < 3; official++) {
        skeletons[count] = &context.skeletons[playerCount + official];
        humanoids[count++] = &world.serialized.match.officials[official].humanoid;
    }
    animation::FetchPutBuffersBatch(*context.animations, skeletons, humanoids, count);
    return true;
}

CUJU_FUNCTION inline bool GameTaskProcessPhase(World &world, const SimulationContext &context) {
    CUJU_PROFILE_SCOPE(TICK);
    bool ok;
    bool process = MatchProcess(world, context, ok);
    if (!ok) return false;
    UpdateCamera(world, context);
    if (process) {
        MatchPreparePutBuffers(world, context);
        if (!MatchFetchPutBuffers(world, context)) return false;
    }
    return true;
}

}  // namespace simulation
}  // namespace cuju

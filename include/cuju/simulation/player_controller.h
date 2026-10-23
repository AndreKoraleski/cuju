// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The controllers of the players and the situation ratings that the player AI reads.
#pragma once

#include <vector>

#include "../libm/powf.h"
#include "ai_functions.h"
#include "human_controller.h"
#include "humanoid.h"
#include "officials.h"

namespace cuju {
namespace simulation {

// Reaction times are in milliseconds.
CUJU_FUNCTION inline int BaseReactionTime(const World &world, const SimulationContext &context,
                                          int player) {
    return FloatToInt32(
        roundf(80.0f - GetStatistic(world, context, player, e_Statistic_PhysicalReaction) * 40.0f));
}
CUJU_FUNCTION inline int PlayerControllerReactionTime(const World &world,
                                                      const SimulationContext &context,
                                                      int player) {
    int reactionTimeMilliseconds = BaseReactionTime(world, context, player);
    return FloatToInt32((float)reactionTimeMilliseconds +
                        (1.0f - TeamDifficulty(world, world.playerTeamID[player])) * 100);
}

// GRF: before any touch, the team processed first counts as the last to touch.
CUJU_FUNCTION inline int MatchLastTouchPlayer(const World &world) {
    int teamID = world.serialized.match.lastTouchTeamID != -1
                     ? world.serialized.match.lastTouchTeamID
                     : world.FirstTeamID();
    return TeamByID(world, teamID).lastTouchPlayer;
}

CUJU_FUNCTION inline void ElizaControllerProcess(World &world, const SimulationContext &context,
                                                 int player) {
    state::PlayerState &playerState = world.serialized.match.players[player];
    int reactionTimeMilliseconds = PlayerControllerReactionTime(world, context, player);
    if (MatchLastTouchPlayer(world) == player &&
        playerState.base.lastTouchType != e_TouchType_Accidental)
        reactionTimeMilliseconds = 0;
    playerState.base.controller.base.mentalImageTime = reactionTimeMilliseconds;
}

CUJU_FUNCTION inline state::PlayerControllerState &ActiveControllerBase(World &world, int player) {
    if (ExternalControllerActive(world, player)) return HumanGamerOf(world, player).controller.base;
    return world.serialized.match.players[player].base.controller.base;
}

CUJU_FUNCTION_NO_INLINE inline float CalculateFreeSpace(const World &world, int mentalImage,
                                                        int teamID, const Vector3 &focusPosition,
                                                        float safeDistance,
                                                        float futureTimeSeconds) {
    float currentSituation = 0.0f;
    ImageList opponents;
    MentalTeamPlayerImages(world, mentalImage, 1 - teamID, opponents);
    for (PlayerImagePosition &opponent : opponents) {
        if (opponent.role == e_PlayerRole_Goalkeeper) continue;
        opponent.position = opponent.position + opponent.movement * 0.2f;
        Vector3 toFocusMovement = (focusPosition - opponent.position).GetNormalized(0) *
                                  SPRINT_VELOCITY * clamp(futureTimeSeconds - 0.2f, 0.0f, 1000.0f);
        if (toFocusMovement.GetLength() > (focusPosition - opponent.position).GetLength())
            toFocusMovement = focusPosition - opponent.position;
        opponent.position += toFocusMovement;
        float situation =
            1.0f -
            clamp((opponent.position - focusPosition).GetLength(), 0, safeDistance) / safeDistance;
        currentSituation += situation;
    }
    return 1.0f - NormalizedClamp(currentSituation, 0.0f, 2.5f);
}

CUJU_FUNCTION_NO_INLINE inline void CalculateTacticalSituation(World &world, int player) {
    state::PlayerState &playerState = world.serialized.match.players[player];
    const state::SpatialState &spatialState = playerState.base.humanoid.spatialState;
    const int teamID = world.playerTeamID[player];
    const int side = TeamDynamicSide(world, teamID);
    const int mentalImage =
        MentalImageIndex(world, ActiveControllerBase(world, player).mentalImageTime);
    float timeSeconds = 0.5f;
    Vector3 checkPosition =
        spatialState.position + Vector3(-side, 0, 0) * SPRINT_VELOCITY * timeSeconds;
    playerState.tacticalSituation.forwardSpaceRating =
        CalculateFreeSpace(world, mentalImage, teamID, checkPosition, 5.0f, timeSeconds);
    timeSeconds = 0.1f;
    checkPosition = spatialState.position + spatialState.movement * timeSeconds;
    playerState.tacticalSituation.spaceRating =
        CalculateFreeSpace(world, mentalImage, teamID, checkPosition, 5.0f, timeSeconds);
    playerState.tacticalSituation.forwardRating =
        1.0f -
        clamp((Vector3(PITCH_HALF_LENGTH * -side, 0, 0) - spatialState.position).GetLength() /
                  (PITCH_HALF_LENGTH * 2.0f),
              0.0f, 1.0f);
    playerState.tacticalSituation.forwardRating =
        libm::powf(playerState.tacticalSituation.forwardRating, 1.5f);
}

CUJU_FUNCTION inline bool TeamHasUniquePossession(const World &world, int teamID) {
    return TeamByID(world, teamID).hasPossession && !TeamByID(world, 1 - teamID).hasPossession;
}

CUJU_FUNCTION_NO_INLINE inline void ControllerCalculateSituation(
    World &world, int player, state::PlayerControllerState &controller) {
    const state::MatchState &match = world.serialized.match;
    const state::PlayerState &playerState = match.players[player];
    const int teamID = world.playerTeamID[player];
    const state::TeamState &team = TeamByID(world, teamID),
                           &opponentTeam = TeamByID(world, 1 - teamID);
    controller.hasPossession = playerState.hasPossession;
    controller.hasUniquePossession = playerState.hasUniquePossession;
    controller.teamHasPossession = team.hasPossession;
    controller.teamHasUniquePossession = TeamHasUniquePossession(world, teamID);
    controller.opponentTeamHasPossession = opponentTeam.hasPossession;
    controller.opponentTeamHasUniquePossession = TeamHasUniquePossession(world, 1 - teamID);
    controller.hasBestChanceOfPossession = team.designatedTeamPossessionPlayer == player;
    controller.teamHasBestPossession = match.bestPossessionTeam == TeamSlot(world, teamID);
    controller.possessionAmount = (float)(opponentTeam.timeNeededToGetToBallMilliseconds + 200) /
                                  (float)(controller.timeNeededMilliseconds + 200);
    controller.teamPossessionAmount = team.teamPossessionAmount;
    controller.fadingTeamPossessionAmount = team.fadingTeamPossessionAmount;
    if (controller.hasBestChanceOfPossession) {
        float distance = NormalizedClamp(
            (BallPredict(match.ball, 300).Get2D() - playerState.base.humanoid.spatialState.position)
                .GetLength(),
            2.0f, 14.0f);
        // Exact: a product in place of GRF's power of 2.0f gives the same float for every input.
        float distanceBias = distance * distance;
        controller.fadingTeamPossessionAmount =
            controller.fadingTeamPossessionAmount * distanceBias +
            controller.teamPossessionAmount * (1.0f - distanceBias);
    }
    controller.opponentTimeNeededToGetToBall = opponentTeam.timeNeededToGetToBallMilliseconds;
    controller.hasBestPossession = controller.hasPossession && controller.possessionAmount >= 1.0f;
}

// Defined in `team_ai.h`. A `manualPlayer` of -1 lets the team AI choose the runner.
CUJU_FUNCTION inline void TeamApplyAttackingRun(World &world, const SimulationContext &context,
                                                int teamID, int manualPlayer = -1);

// Only for a player under external control.
CUJU_FUNCTION_NO_INLINE inline bool HumanControllerProcess(World &world,
                                                           const SimulationContext &context,
                                                           int player) {
    state::MatchState &match = world.serialized.match;
    state::HumanControllerState &humanController = HumanGamerOf(world, player).controller;
    state::PlayerControllerState &controller = humanController.base;
    const bool *buttons = humanController.keyboard.buttonsPressed;

    // GRF: a gamer's controller reacts with the base reaction time, without the team difficulty.
    int reactionTimeMilliseconds = BaseReactionTime(world, context, player);
    if (MatchLastTouchPlayer(world) == player &&
        match.players[player].base.lastTouchType != e_TouchType_Accidental)
        reactionTimeMilliseconds = 0;
    controller.mentalImageTime = reactionTimeMilliseconds;

    Vector3 currentDirection;
    float unusedVelocity = 0.0f;
    HumanGetKeyboardInput(world, player, currentDirection, unusedVelocity);
    humanController.previousDirection = currentDirection;
    // The steady direction follows the input at once, so its snapshot time is now.
    humanController.steadyDirection = currentDirection;
    humanController.lastSteadyDirectionSnapshotTimeMilliseconds =
        (int32_t)match.actualTimeMilliseconds;

    ControllerCalculateSituation(world, player, controller);

    const int teamID = world.playerTeamID[player];
    const state::TeamAIState &teamAI = TeamByID(world, teamID).teamController;
    if (humanController.actionMode == 0 && (!match.inSetPiece || teamAI.taker == player)) {
        float possessionContext = controller.possessionAmount - 1.0f;
        if (match.designatedPossessionPlayer == player) {
            possessionContext = 1.0f;
        } else {
            if (buttons[e_ButtonFunction_ShortPass]) possessionContext += 0.15f;
            if (buttons[e_ButtonFunction_LongPass]) possessionContext += 0.15f;
            if (buttons[e_ButtonFunction_Shot]) possessionContext += 0.15f;
            if (buttons[e_ButtonFunction_Pressure]) possessionContext -= 0.15f;
            if (buttons[e_ButtonFunction_Sliding]) possessionContext -= 0.15f;
            if (buttons[e_ButtonFunction_TeamPressure]) possessionContext -= 0.15f;
            if (BallPredict(match.ball, 0).coordinates[2] > 1.5f) possessionContext += 0.2f;
        }
        // GRF: every button's previous state reads as released, so a held button is a fresh press
        // on every tick.
        if (possessionContext < 0.0f) {
            const bool allow = !match.inSetPiece;
            if (buttons[e_ButtonFunction_Pressure] && allow) {
                humanController.actionMode = 1;
                humanController.actionButton = e_ButtonFunction_Pressure;
            }
            if (buttons[e_ButtonFunction_Sliding] && allow) {
                humanController.actionMode = 1;
                humanController.actionButton = e_ButtonFunction_Sliding;
            }
            if (buttons[e_ButtonFunction_TeamPressure] && allow) {
                humanController.actionMode = 1;
                humanController.actionButton = e_ButtonFunction_TeamPressure;
            }
            if (buttons[e_ButtonFunction_KeeperRush] && allow) {
                humanController.actionMode = 1;
                humanController.actionButton = e_ButtonFunction_KeeperRush;
            }
        } else {
            bool allowHighPass = true, allowShot = true;
            if (teamAI.taker == player && teamAI.setPieceType == e_GameMode_ThrowIn) {
                allowHighPass = false;
                allowShot = false;
            }
            if (buttons[e_ButtonFunction_ShortPass]) {
                humanController.actionMode = 2;
                humanController.actionButton = e_ButtonFunction_ShortPass;
            }
            if (buttons[e_ButtonFunction_LongPass]) {
                humanController.actionMode = 2;
                humanController.actionButton = e_ButtonFunction_LongPass;
            }
            if (buttons[e_ButtonFunction_HighPass] && allowHighPass) {
                humanController.actionMode = 2;
                humanController.actionButton = e_ButtonFunction_HighPass;
            }
            if (buttons[e_ButtonFunction_Shot] && allowShot) {
                humanController.actionMode = 2;
                humanController.actionButton = e_ButtonFunction_Shot;
            }
        }
    }
    if (humanController.actionMode == 2) {
        if (buttons[humanController.actionButton]) {
            humanController.gaugeMilliseconds += 10;
            humanController.gaugeMilliseconds =
                (int)clamp((float)humanController.gaugeMilliseconds, 10, 1000);
            humanController.actionBufferTimeMilliseconds = 0;
        } else {
            humanController.actionBufferTimeMilliseconds += 10;
        }
    }
    if (buttons[e_ButtonFunction_Switch] && controller.hasPossession)
        TeamApplyAttackingRun(world, context, teamID);
    return true;
}

}  // namespace simulation
}  // namespace cuju

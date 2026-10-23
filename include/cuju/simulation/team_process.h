// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// One tick of a team, its possession amounts, team AI, players, gamer switches, designated player.
#pragma once

#include "../profiling.h"

#include <map>
#include <vector>

#include "ai_functions.h"
#include "geometry_offsets.h"
#include "human_controller.h"
#include "humanoid_process.h"
#include "player_controller.h"
#include "team_select.h"

namespace cuju {
namespace simulation {

// Defined in `team_ai.h`, and `PlayerSendOff` in `reset.h`.
CUJU_FUNCTION inline bool TeamAIProcess(World &world, const SimulationContext &context, int teamID);
CUJU_FUNCTION inline bool TeamCalculateDynamicRoles(World &world, const SimulationContext &context,
                                                    int teamID);
CUJU_FUNCTION inline bool TeamCalculateManMarking(World &world, const SimulationContext &context,
                                                  int teamID);
CUJU_FUNCTION inline bool PlayerSendOff(World &world, const SimulationContext &context, int player);

CUJU_FUNCTION inline bool DynamicPlayerSelection(const World &world) {
    const state::ScenarioState &scenario = world.serialized.scenario;
    int left = 0, right = 0;
    for (int player = 0; player < scenario.leftTeamSize; player++)
        left += scenario.leftTeam[player].controllable ? 1 : 0;
    for (int player = 0; player < scenario.rightTeamSize; player++)
        right += scenario.rightTeam[player].controllable ? 1 : 0;
    return !((left == scenario.leftAgents || scenario.leftAgents == 0) &&
             (right == scenario.rightAgents || scenario.rightAgents == 0));
}

#if defined(__CUDACC__)
// nvcc sees a read of `sortedDistances` before a write, but the sort reads only slots it wrote.
#pragma nv_diag_suppress 549
#endif
CUJU_FUNCTION_NO_INLINE inline PlayerList GetClosestPlayers(
    const World &world, const SimulationContext &context, int teamID, const Vector3 &position,
    bool onlyAIControlled, unsigned int playerCount = 3, bool onlySelectable = false) {
    float sortedDistances[state::MAXIMUM_PLAYERS];
    int sortedPlayers[state::MAXIMUM_PLAYERS], count = 0;
    struct Candidate {
        float distance;
        int32_t eligible;
    };
    auto evaluate = [&](int player) {
        Candidate candidate{0.0f, 0};
        if (world.playerTeamID[player] != teamID ||
            !world.serialized.match.players[player].base.isActive)
            return candidate;
        candidate.distance = (Humanoid(world, player).spatialState.position - position).GetLength();
        candidate.eligible = (!onlyAIControlled || !ExternalControllerActive(world, player)) &&
                             (!onlySelectable || Formation(world, context, player).controllable);
        return candidate;
    };
    // Exact: a player goes after those at an equal distance, so ties keep state order.
    WarpPipeline<Candidate>(world.serialized.match.playerCount, evaluate,
                            [&](int player, const Candidate &candidate) {
                                if (!candidate.eligible) return;
                                const float distance = candidate.distance;
                                int slot = count++;
                                for (; slot > 0 && distance < sortedDistances[slot - 1]; slot--) {
                                    sortedDistances[slot] = sortedDistances[slot - 1];
                                    sortedPlayers[slot] = sortedPlayers[slot - 1];
                                }
                                sortedDistances[slot] = distance;
                                sortedPlayers[slot] = player;
                            });
    PlayerList result;
    for (unsigned int index = 0; index < playerCount && (int)index < count; index++)
        result.push_back(sortedPlayers[index]);
    return result;
}
#if defined(__CUDACC__)
#pragma nv_diag_default 549
#endif

CUJU_FUNCTION_NO_INLINE inline int GetBestSwitchTargetPlayer(const World &world,
                                                             const SimulationContext &context,
                                                             int teamID,
                                                             const Vector3 &desiredMovement) {
    (void)desiredMovement;
    const state::MatchState &match = world.serialized.match;
    const state::TeamState &team = TeamByID(world, teamID);
    const int side = TeamDynamicSide(world, teamID);
    Vector3 actionPosition =
        Humanoid(world, match.designatedPossessionPlayer).spatialState.position * 0.5f +
        BallPredict(match.ball, 100).Get2D() * 0.5f;
    Vector3 defensePosition =
        (actionPosition * Vector3(1.0f, 0.8f, 0.0f)) + Vector3(side * 4.0f, 0.0f, 0.0f);
    Vector3 offensePosition =
        (actionPosition * Vector3(1.0f, 0.8f, 0.0f)) + Vector3(-side * 8.0f, 0.0f, 0.0f);
    defensePosition =
        defensePosition * 0.8f +
        Humanoid(world, team.designatedTeamPossessionPlayer).spatialState.position * 0.2f;
    float offenseBias = team.fadingTeamPossessionAmount - 0.5f;
    offenseBias = offenseBias * 0.2f + clamp(team.teamPossessionAmount - 0.5f, 0.0f, 1.0f) * 0.8f;
    offenseBias = clamp((float)((double)(offenseBias - 0.5f) * 2.0 + 0.5), 0.0f, 1.0f);
    offenseBias = clamp(libm::powf(offenseBias, 1.5f), 0.0f, 1.0f);
    Vector3 resultingPosition =
        defensePosition * (1.0f - offenseBias) + offensePosition * offenseBias;

    PlayerList teamPlayers = GetClosestPlayers(world, context, teamID, resultingPosition, true, 8);
    PlayerList kept;
    for (int player : teamPlayers)
        if (Formation(world, context, player).role != e_PlayerRole_Goalkeeper)
            kept.push_back(player);
    if (kept.empty()) return -1;
    int bestPlayerIndex = 0;
    float tooLateDistance = 1.0f;
    int designated = match.designatedPossessionPlayer;
    if (world.playerTeamID[designated] != teamID) {
        const state::SpatialState &opponentState = Humanoid(world, designated).spatialState;
        Vector3 goalPosition = Vector3(side * PITCH_HALF_LENGTH, 0.0f, 0.0f);
        float opponentGoalDistance =
            (goalPosition - (opponentState.position + opponentState.movement * 0.5f)).GetLength();
        for (size_t index = 0; index < kept.size(); index++) {
            const state::SpatialState &mateState = Humanoid(world, kept[index]).spatialState;
            float mateGoalDistance =
                (goalPosition - (mateState.position + mateState.movement * 0.5f)).GetLength();
            if (mateGoalDistance < opponentGoalDistance + tooLateDistance +
                                       clamp(opponentGoalDistance * 0.1f, 0.0f, 3.0f)) {
                bestPlayerIndex = (int)index;
                break;
            }
        }
    }
    return kept.at(bestPlayerIndex);
}

CUJU_FUNCTION inline int TeamGoalie(const World &world, const SimulationContext &context,
                                    int teamID) {
    for (int player = 0; player < world.serialized.match.playerCount; player++)
        if (world.playerTeamID[player] == teamID &&
            world.serialized.match.players[player].base.isActive &&
            Formation(world, context, player).role == e_PlayerRole_Goalkeeper)
            return player;
    return -1;
}

CUJU_FUNCTION_NO_INLINE inline bool PlayerProcess(World &world, const SimulationContext &context,
                                                  int player) {
    CUJU_PROFILE_SCOPE(PLAYER_PROCESS);
    state::MatchState &match = world.serialized.match;
    state::PlayerState &playerState = match.players[player];
    if (!playerState.base.isActive) return true;
    playerState.desiredTimeToBallMilliseconds =
        (playerState.desiredTimeToBallMilliseconds - 10 < 0)
            ? 0
            : playerState.desiredTimeToBallMilliseconds - 10;
    if (ExternalControllerActive(world, player) && !HumanControllerProcess(world, context, player))
        return false;
    ElizaControllerProcess(world, context, player);
    if (match.inPlay) {
        if (match.actualTimeMilliseconds % 1000 == 0)
            playerState.base.positionHistoryPerSecond.push_back(
                playerState.base.humanoid.spatialState.position);
        if (playerState.hasPossession)
            playerState.possessionDurationMilliseconds += 10;
        else
            playerState.possessionDurationMilliseconds = 0;
        if ((match.actualTimeMilliseconds + (uint64_t)(context.staticData->stableID[player] * 10)) %
                100 ==
            0)
            CalculateTacticalSituation(world, player);
    }
    Vector3 positionBefore = playerState.base.humanoid.spatialState.position;
    if (!HumanoidProcess(world, context, player, context.requestCommand)) return false;
    if (match.inPlay) {
        Vector3 positionAfter = playerState.base.humanoid.spatialState.position;
        float distance = (positionAfter - positionBefore).GetLength();
        // GRF: the raw statistic, without the difficulty and fatigue scaling of `GetStatistic`.
        const float stamina =
            context.staticData->data[player].statistics[e_Statistic_PhysicalStamina];
        playerState.base.fatigueFactorInverse -= distance * 0.00003f * (2.0f - stamina) *
                                                 (1.0f / context.staticData->matchDurationFactor);
        playerState.base.fatigueFactorInverse =
            clamp(playerState.base.fatigueFactorInverse, 0.01f, 1.0f);
    }
    if (playerState.cards > 1 &&
        playerState.cardEffectiveTimeMilliseconds <= match.actualTimeMilliseconds &&
        TeamActivePlayers(world, world.playerTeamID[player]).size() > 1)
        return PlayerSendOff(world, context, player);
    return true;
}

CUJU_FUNCTION_NO_INLINE inline bool TeamProcess(World &world, const SimulationContext &context,
                                                int teamID) {
    CUJU_PROFILE_SCOPE(TEAM_PROCESS);
    state::MatchState &match = world.serialized.match;
    state::TeamState &team = TeamByID(world, teamID);
    const state::TeamState &opponentTeam = TeamByID(world, 1 - teamID);
    team.teamPossessionAmount = (float)(opponentTeam.timeNeededToGetToBallMilliseconds + 1500) /
                                (float)(team.timeNeededToGetToBallMilliseconds + 1500);
    float temporaryFading = team.fadingTeamPossessionAmount * 0.995f +
                            clamp(team.teamPossessionAmount, 0.5f, 1.5f) * 0.005f;
    team.fadingTeamPossessionAmount +=
        clamp(temporaryFading - team.fadingTeamPossessionAmount, -0.005f, 0.005f);
    if (!match.inPlay || match.inSetPiece || match.ballRetainer != -1) {
        float amount;
        if (match.ballRetainer != -1)
            amount = (world.playerTeamID[match.ballRetainer] == teamID) ? 1.5f : 0.5f;
        else
            amount = (match.bestPossessionTeam == TeamSlot(world, teamID)) ? 1.5f : 0.5f;
        team.fadingTeamPossessionAmount = team.teamPossessionAmount = amount;
    }

    TeamHumanGamersSelectAnyone(world, context, teamID);

    if (match.inPlay && !match.inSetPiece) {
        if (!TeamAIProcess(world, context, teamID)) return false;
        uint64_t teamOffset = teamID == (1 - world.FirstTeamID()) ? 200 : 0;
        if ((match.actualTimeMilliseconds + teamOffset) % 400 == 0 &&
            !TeamCalculateDynamicRoles(world, context, teamID))
            return false;
        if ((match.actualTimeMilliseconds + teamOffset + 100) % 400 == 0 &&
            !TeamCalculateManMarking(world, context, teamID))
            return false;
    }

    for (int player = 0; player < match.playerCount; player++)
        if (world.playerTeamID[player] == teamID && match.players[player].base.isActive &&
            !PlayerProcess(world, context, player))
            return false;

    if (match.inPlay) {
        for (int gamerIndex = 0; gamerIndex < team.humanGamers.count; gamerIndex++) {
            state::HumanGamerState &gamer = team.humanGamers[gamerIndex];
            int selectedPlayer = gamer.selectedPlayer;
            int bestPossessionPlayer = TeamBestPossessionPlayer(world, teamID);
            const bool *buttons = gamer.controller.keyboard.buttonsPressed;
            // GRF: every button's previous state reads as released, so a held switch is a fresh
            // press on every tick.
            const bool switchPressed = buttons[e_ButtonFunction_Switch] &&
                                       (!(selectedPlayer == bestPossessionPlayer &&
                                          selectedPlayer == team.designatedTeamPossessionPlayer) ||
                                        team.teamPossessionAmount < 1.0f);
            // GRF: reads the possession of a null player here and crashes.
            if (switchPressed && selectedPlayer < 0)
                return Unsupported("the switch of a gamer that controls no player");
            if (switchPressed && !match.players[selectedPlayer].hasUniquePossession) {
                int targetPlayer = -1;
                const bool bestTeam = match.bestPossessionTeam == TeamSlot(world, teamID);
                if (match.players[team.designatedTeamPossessionPlayer].base.externalController <
                        0 &&
                    bestTeam) {
                    targetPlayer = team.designatedTeamPossessionPlayer;
                } else if (match.players[bestPossessionPlayer].base.externalController < 0 &&
                           bestTeam) {
                    targetPlayer = bestPossessionPlayer;
                } else {
                    targetPlayer = GetBestSwitchTargetPlayer(
                        world, context, teamID, KeyboardDirection(gamer.controller.keyboard));
                    if (targetPlayer >= 0 &&
                        match.players[targetPlayer].base.externalController >= 0)
                        targetPlayer = -1;
                }
                if (targetPlayer == TeamGoalie(world, context, teamID)) targetPlayer = -1;
                if (targetPlayer >= 0 && Formation(world, context, targetPlayer).controllable) {
                    team.mainSelectedPlayer = targetPlayer;
                    HumanGamerSetSelectedPlayer(world, GlobalGamerIndex(world, teamID, gamerIndex),
                                                targetPlayer);
                }
            }
        }
    } else if (match.actualTimeMilliseconds >= 2000) {
        if (DynamicPlayerSelection(world))
            for (int gamerIndex = 0; gamerIndex < team.humanGamers.count; gamerIndex++)
                if (team.humanGamers[gamerIndex].selectedPlayer >= 0)
                    HumanGamerSetSelectedPlayer(world, GlobalGamerIndex(world, teamID, gamerIndex),
                                                -1);
        team.mainSelectedPlayer = -1;
    }

    int designatedPlayerTimeMilliseconds =
        (int)match.players[team.designatedTeamPossessionPlayer].timeNeededToGetToBallMilliseconds;
    int bestPlayer = TeamBestPossessionPlayer(world, teamID);
    int opponentTimeMilliseconds = opponentTeam.timeNeededToGetToBallMilliseconds;
    if (team.designatedTeamPossessionPlayer != bestPlayer) {
        int designated = team.designatedTeamPossessionPlayer;
        int bestPlayerTimeMilliseconds =
            (int)match.players[bestPlayer].timeNeededToGetToBallMilliseconds;
        float timeRating = (float)(bestPlayerTimeMilliseconds + 500) /
                           (float)(designatedPlayerTimeMilliseconds + 500);
        if (match.players[bestPlayer].hasPossession) timeRating *= 0.5f;
        if (match.players[designated].hasPossession) timeRating /= 0.5f;
        if (ExternalControllerActive(world, bestPlayer)) timeRating *= 0.8f;
        if (ExternalControllerActive(world, designated)) timeRating /= 0.8f;
        if (!ExternalControllerActive(world, bestPlayer) &&
            designatedPlayerTimeMilliseconds < opponentTimeMilliseconds - 100) {
            timeRating += 0.2f;
            timeRating *= 1.2f;
        }
        if (timeRating < 0.8f) team.designatedTeamPossessionPlayer = bestPlayer;
    }
    return true;
}

}  // namespace simulation
}  // namespace cuju

// The player AI needs the functions above. `ai_controller.h` ends by including `team_ai.h`.
#include "ai_controller.h"

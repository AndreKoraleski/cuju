// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The players that the gamers of a team control, and the reset of their controllers.
#pragma once

#include "../profiling.h"

#include "human_controller.h"
#include "humanoid.h"

namespace cuju {
namespace simulation {

// A global gamer index puts the gamers of the team processed first before the others. A player's
// `externalController` holds the global index of its gamer, or -1 without one.
CUJU_FUNCTION inline int GlobalGamerIndex(const World &world, int teamID, int gamer) {
    return TeamSlot(world, teamID) == 0 ? gamer
                                        : world.serialized.match.teams[0].humanGamers.count + gamer;
}
CUJU_FUNCTION inline state::HumanGamerState &GamerByGlobal(World &world, int gamerIndex) {
    int firstTeamGamerCount = world.serialized.match.teams[0].humanGamers.count;
    return gamerIndex < firstTeamGamerCount
               ? world.serialized.match.teams[0].humanGamers[gamerIndex]
               : world.serialized.match.teams[1].humanGamers[gamerIndex - firstTeamGamerCount];
}

// The state keeps a second copy of each keyboard, in global order, which must stay equal to the
// first. GRF: the sprint, dribble and defensive buttons are sticky and keep their state.
CUJU_FUNCTION inline void KeyboardReleaseNonStickyButtons(World &world, int gamerIndex) {
    for (state::KeyboardState *keyboard :
         {&GamerByGlobal(world, gamerIndex).controller.keyboard,
          &world.serialized.match.secondKeyboardCopies[gamerIndex]})
        for (int button :
             {e_ButtonFunction_LongPass, e_ButtonFunction_HighPass, e_ButtonFunction_ShortPass,
              e_ButtonFunction_Shot, e_ButtonFunction_Sliding, e_ButtonFunction_Switch})
            keyboard->buttonsPressed[button] = false;
}

CUJU_FUNCTION_NO_INLINE inline void HumanControllerReset(World &world, int gamerIndex) {
    state::HumanControllerState &controller = GamerByGlobal(world, gamerIndex).controller;
    controller.actionMode = 0;
    controller.gaugeMilliseconds = 0;
    controller.actionButton = e_ButtonFunction_ShortPass;
    controller.actionBufferTimeMilliseconds = 0;
    controller.base.lastSwitchTimeMilliseconds = (uint64_t)(int64_t)-10000;
    controller.base.lastSwitchTimeDurationMilliseconds = 300;
    controller.lastSteadyDirectionSnapshotTimeMilliseconds = 0;
    controller.steadyDirection = Vector3(0, -1, 0);
    controller.previousDirection = Vector3(0, -1, 0);
    controller.base.fadingTeamPossessionAmount = 1.0;
    KeyboardReleaseNonStickyButtons(world, gamerIndex);
}

CUJU_FUNCTION inline void HumanControllerSetPlayer(World &world, int gamerIndex, int player) {
    state::HumanControllerState &controller = GamerByGlobal(world, gamerIndex).controller;
    controller.base.lastSwitchTimeMilliseconds = world.serialized.match.actualTimeMilliseconds;
    controller.base.player = player;
    // The state stores a team as its slot, not its ID.
    controller.base.team = TeamSlot(world, world.playerTeamID[player]);
    controller.base.opponentTeam = TeamSlot(world, 1 - world.playerTeamID[player]);
}

// `gamerIndex` is global, or -1 to hand the player back to the AI.
CUJU_FUNCTION inline void SetExternalController(World &world, int player, int gamerIndex) {
    state::PlayerState &playerState = world.serialized.match.players[player];
    playerState.base.externalController = gamerIndex;
    if (gamerIndex >= 0) {
        HumanControllerReset(world, gamerIndex);
        HumanControllerSetPlayer(world, gamerIndex, player);
    } else {
        playerState.base.controller.lastDesiredDirection = Vector3(0);
        playerState.base.controller.lastDesiredVelocity = 0;
    }
}

CUJU_FUNCTION_NO_INLINE inline void HumanGamerSetSelectedPlayer(World &world, int gamerIndex,
                                                                int player) {
    state::HumanGamerState &gamer = GamerByGlobal(world, gamerIndex);
    if (player >= 0 && world.serialized.match.players[player].base.externalController >= 0) return;
    if (gamer.selectedPlayer >= 0) {
        if (gamer.selectedPlayer == player) return;
        SetExternalController(world, gamer.selectedPlayer, -1);
    }
    if (player >= 0) {
        gamer.selectedPlayer = player;
        SetExternalController(world, player, gamerIndex);
    } else {
        gamer.selectedPlayer = -1;
    }
}

CUJU_FUNCTION_NO_INLINE inline bool TeamSelectPlayer(World &world, const SimulationContext &context,
                                                     int teamID, int player) {
    state::TeamState &team = TeamByID(world, teamID);
    if (Formation(world, context, player).controllable) {
        team.mainSelectedPlayer = player;
        if (world.serialized.match.players[player].base.externalController < 0 &&
            team.humanGamers.count > 0) {
            auto &switchPriority = team.switchPriority;
            int gamer = switchPriority[0];
            HumanGamerSetSelectedPlayer(world, GlobalGamerIndex(world, teamID, gamer), player);
            for (int index = 0; index + 1 < switchPriority.count; index++)
                switchPriority[index] = switchPriority[index + 1];
            switchPriority[switchPriority.count - 1] = gamer;
        }
    }
    team.designatedTeamPossessionPlayer = player;
    return true;
}

// Exact: a tie keeps the player that comes first in the state.
CUJU_FUNCTION_NO_INLINE inline int GetClosestPlayer(const World &world,
                                                    const SimulationContext &context, int teamID,
                                                    const Vector3 &position, bool onlyAIControlled,
                                                    int except = -1, bool onlySelectable = false) {
    float closestDistance = 10000;
    int closestPlayer = -1;
    for (int player = 0; player < world.serialized.match.playerCount; player++) {
        if (world.playerTeamID[player] != teamID) continue;
        if (!world.serialized.match.players[player].base.isActive || player == except) continue;
        float distance = (Humanoid(world, player).spatialState.position - position).GetLength();
        if (distance < closestDistance) {
            if ((!onlyAIControlled || !ExternalControllerActive(world, player)) &&
                (!onlySelectable || Formation(world, context, player).controllable)) {
                closestDistance = distance;
                closestPlayer = player;
            }
        }
    }
    return closestPlayer;
}

// A time in milliseconds, not a distance, the shortest of the players that enabled gamers control,
// or 10000.
CUJU_FUNCTION inline int HumanControlledToBallDistance(const World &world, int teamID) {
    const state::TeamState &team = TeamByID(world, teamID);
    int timeToBall = 10000;
    for (int gamerIndex = 0; gamerIndex < team.humanGamers.count; gamerIndex++) {
        const state::HumanGamerState &gamer = team.humanGamers[gamerIndex];
        if (gamer.selectedPlayer >= 0 && !gamer.controller.keyboard.disabled) {
            int time = (int)world.serialized.match.players[gamer.selectedPlayer]
                           .timeNeededToGetToBallMilliseconds;
            timeToBall = (time < timeToBall) ? time : timeToBall;
        }
    }
    return timeToBall;
}

CUJU_FUNCTION_NO_INLINE inline bool TeamUpdateSwitch(World &world, const SimulationContext &context,
                                                     int teamID) {
    CUJU_PROFILE_SCOPE(TEAM_UPDATE_SWITCH);
    state::MatchState &match = world.serialized.match;
    state::TeamState &team = TeamByID(world, teamID);
    if (match.inPlay && team.humanGamers.count > 1) {
        auto &switchPriority = team.switchPriority;
        int myTurn = switchPriority[0];
        if (team.humanGamers[myTurn].selectedPlayer == match.designatedPossessionPlayer) {
            for (int index = 0; index + 1 < switchPriority.count; index++)
                switchPriority[index] = switchPriority[index + 1];
            switchPriority[switchPriority.count - 1] = myTurn;
        }
    }
    if (match.inPlay && team.humanGamers.count > 0) {
        int designated = team.designatedTeamPossessionPlayer;
        if (!ExternalControllerActive(world, designated) &&
            3 * (int)match.players[designated].timeNeededToGetToBallMilliseconds <
                HumanControlledToBallDistance(world, teamID) &&
            Formation(world, context, designated).role != e_PlayerRole_Goalkeeper) {
            if (!TeamSelectPlayer(world, context, teamID, designated)) return false;
        }
    }
    if (match.inPlay && team.humanGamers.count > 0) {
        int designated = team.designatedTeamPossessionPlayer;
        if (!ExternalControllerActive(world, designated) &&
            (match.players[designated].hasUniquePossession || match.inSetPiece)) {
            if (Formation(world, context, designated).role != e_PlayerRole_Goalkeeper) {
                if (!TeamSelectPlayer(world, context, teamID, designated)) return false;
            }
        }
    }
    return true;
}

CUJU_FUNCTION_NO_INLINE inline void TeamHumanGamersSelectAnyone(World &world,
                                                                const SimulationContext &context,
                                                                int teamID) {
    state::MatchState &match = world.serialized.match;
    if (!match.inPlay) return;
    state::TeamState &team = TeamByID(world, teamID);
    if (team.mainSelectedPlayer == -1)
        team.mainSelectedPlayer = GetClosestPlayer(
            world, context, teamID, BallPredict(match.ball, 0).Get2D(), true, -1, true);
    for (int gamerIndex = 0; gamerIndex < team.humanGamers.count; gamerIndex++) {
        if (team.humanGamers[gamerIndex].selectedPlayer == -1) {
            int player = GetClosestPlayer(world, context, teamID,
                                          BallPredict(match.ball, 0).Get2D(), true, -1, true);
            if (player >= 0)
                HumanGamerSetSelectedPlayer(world, GlobalGamerIndex(world, teamID, gamerIndex),
                                            player);
        }
    }
}

}  // namespace simulation
}  // namespace cuju

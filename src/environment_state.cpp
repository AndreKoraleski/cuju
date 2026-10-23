// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// Loads and saves the match and the whole state, in GRF's field order.
#include "cuju/state/environment_state.h"

namespace cuju {
namespace state {

static void ProcessMatch(Archive &archive, MatchState &match, int freeControllerCount) {
    PROCESS_FIELD(match.firstTeam, TYPE_INTEGER);
    PROCESS_FIELD(match.secondTeam, TYPE_INTEGER);
    archive.field = "match.mentalImages";
    ProcessVector(archive, match.mentalImages,
                  [&](MentalImageState &mentalImage) { mentalImage.Process(archive); });
    PROCESS_STRUCT(match.teams[0]);
    PROCESS_STRUCT(match.teams[1]);
    for (int player = 0; player < archive.Count(match.playerCount, MAXIMUM_PLAYERS); player++) {
        ArchiveScope scope(archive, "match.players", player);
        match.players[player].Process(archive);
    }
    PROCESS_FIELD(match.goalCount[0], TYPE_INTEGER);
    PROCESS_FIELD(match.goalCount[1], TYPE_INTEGER);
    PROCESS_FIELD(match.possessionLastMinute, TYPE_FLOAT);
    for (int official = 0; official < 3; official++) {
        ArchiveScope scope(archive, "match.officials", official);
        match.officials[official].Process(archive);
    }
    // GRF: the state stores the keyboard of every human gamer a second time, first team first.
    const int keyboards = match.teams[0].humanGamers.count + match.teams[1].humanGamers.count;
    for (int keyboard = 0; keyboard < archive.Count(keyboards, MAXIMUM_CONTROLLERS); keyboard++) {
        ArchiveScope scope(archive, "match.secondKeyboardCopies", keyboard);
        match.secondKeyboardCopies[keyboard].Process(archive);
    }
    match.freeControllerCount = freeControllerCount;
    for (int controller = 0; controller < freeControllerCount; controller++) {
        ArchiveScope scope(archive, "match.freeControllers", controller);
        match.freeControllers[controller].Process(archive);
    }
    PROCESS_STRUCT(match.ball);
    PROCESS_FIELD(match.matchTimeMilliseconds, TYPE_UNSIGNED_LONG);
    PROCESS_FIELD(match.actualTimeMilliseconds, TYPE_UNSIGNED_LONG);
    PROCESS_FIELD(match.goalScoredTimer, TYPE_UNSIGNED_LONG);
    PROCESS_FIELD(match.matchPhase, TYPE_MATCH_PHASE);
    PROCESS_FIELD(match.inPlay, TYPE_BOOLEAN);
    PROCESS_FIELD(match.inSetPiece, TYPE_BOOLEAN);
    PROCESS_FIELD(match.goalScored, TYPE_BOOLEAN);
    PROCESS_FIELD(match.ballIsInGoal, TYPE_BOOLEAN);
    PROCESS_POINTER_FIELD(match.lastGoalTeam, TEAM);
    PROCESS_POINTER_FIELD(match.lastGoalScorer, PLAYER);
    for (int entry = 0; entry < 4; entry++) {
        ArchiveScope scope(archive, "match.lastTouchTeamIDs", entry);
        archive.field = archive.layout ? "" : "lastTouchTeamIDs";
        archive.Process(match.lastTouchTeamIDs[entry], TYPE_INTEGER);
    }
    PROCESS_FIELD(match.lastTouchTeamID, TYPE_INTEGER);
    PROCESS_POINTER_FIELD(match.bestPossessionTeam, TEAM);
    PROCESS_POINTER_FIELD(match.designatedPossessionPlayer, PLAYER);
    PROCESS_POINTER_FIELD(match.ballRetainer, PLAYER);
    PROCESS_FIELD(match.possessionSideHistoryMaximumTimeMilliseconds, TYPE_UNSIGNED_INTEGER);
    archive.field = "match.possessionSideHistory";
    ProcessVector(archive, match.possessionSideHistory,
                  [&](float &side) { archive.Process(side, TYPE_FLOAT); });
    PROCESS_FIELD(match.autoUpdateInGameCamera, TYPE_BOOLEAN);
    PROCESS_FIELD(match.cameraOrientation, TYPE_QUATERNION);
    PROCESS_FIELD(match.cameraNodeOrientation, TYPE_QUATERNION);
    PROCESS_FIELD(match.cameraNodePosition, TYPE_VECTOR3);
    PROCESS_FIELD(match.cameraFieldOfView, TYPE_FLOAT);
    PROCESS_FIELD(match.cameraNearCap, TYPE_FLOAT);
    PROCESS_FIELD(match.cameraFarCap, TYPE_FLOAT);
    archive.field = "match.cameraPosition";
    ProcessVector(archive, match.cameraPosition,
                  [&](Vector3 &position) { archive.Process(position, TYPE_VECTOR3); });
    PROCESS_FIELD(match.lastBodyBallCollisionTimeMilliseconds, TYPE_UNSIGNED_INTEGER);
    PROCESS_STRUCT(match.referee);
}

void ProcessEnvironmentScenario(Archive &archive, EnvironmentStateFull &environment) {
    PROCESS_FIELD(environment.gameState, TYPE_GAME_STATE);
    PROCESS_FIELD(environment.waitingForGameCount, TYPE_INTEGER);
    for (size_t byte = 0; byte < sizeof(environment.rng); byte++) {
        ArchiveScope scope(archive, "environment.rng", (int)byte);
        archive.field = archive.layout ? "" : "rng";
        archive.Process(environment.rng[byte], TYPE_CHARACTER);
    }
    PROCESS_STRUCT(environment.scenario);
}

void ProcessEnvironmentState(Archive &archive, EnvironmentStateFull &environment,
                             int freeControllerCount) {
    ProcessEnvironmentScenario(archive, environment);
    // GRF: the match holds every player that it created for both teams, active or not.
    environment.match.playerCount =
        environment.scenario.leftTeamSize + environment.scenario.rightTeamSize;
    if (environment.match.playerCount > MAXIMUM_PLAYERS)
        throw std::runtime_error("the state holds more players than a match holds");
    PROCESS_FIELD(environment.step, TYPE_INTEGER);
    ProcessMatch(archive, environment.match, freeControllerCount);
}

}  // namespace state
}  // namespace cuju

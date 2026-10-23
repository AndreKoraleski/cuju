// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC
// SPDX-License-Identifier: Apache-2.0

// The episode logic of GRF's Python environment on top of the ticks of the simulation.
#pragma once

#include "../profiling.h"

#include <vector>

#include "match_process.h"

namespace cuju {
namespace simulation {

enum Action {
    e_Action_Idle = 0,
    e_Action_Left,
    e_Action_TopLeft,
    e_Action_Top,
    e_Action_TopRight,
    e_Action_Right,
    e_Action_BottomRight,
    e_Action_Bottom,
    e_Action_BottomLeft,
    e_Action_LongPass,
    e_Action_HighPass,
    e_Action_ShortPass,
    e_Action_Shot,
    e_Action_KeeperRush,
    e_Action_Sliding,
    e_Action_Pressure,
    e_Action_TeamPressure,
    e_Action_Switch,
    e_Action_Sprint,
    e_Action_Dribble,
    e_Action_ReleaseDirection,
    e_Action_ReleaseLongPass,
    e_Action_ReleaseHighPass,
    e_Action_ReleaseShortPass,
    e_Action_ReleaseShot,
    e_Action_ReleaseKeeperRush,
    e_Action_ReleaseSliding,
    e_Action_ReleasePressure,
    e_Action_ReleaseTeamPressure,
    e_Action_ReleaseSwitch,
    e_Action_ReleaseSprint,
    e_Action_ReleaseDribble,
    e_Action_BuiltinAI
};

// `actionSet` is 0 for `default` with 19 actions, 1 for `v2` with 20 and 2 for `full` with 33.
CUJU_FUNCTION inline int ActionSetSize(int actionSet) {
    return actionSet == 0 ? 19 : actionSet == 1 ? 20 : 33;
}
// GRF: each action set is a prefix of `full`, so `actionSet` does not change the result.
CUJU_FUNCTION_NO_INLINE inline int ActionFromSet(int actionSet, int index) {
    constexpr int fullSet[33] = {e_Action_Idle,
                                 e_Action_Left,
                                 e_Action_TopLeft,
                                 e_Action_Top,
                                 e_Action_TopRight,
                                 e_Action_Right,
                                 e_Action_BottomRight,
                                 e_Action_Bottom,
                                 e_Action_BottomLeft,
                                 e_Action_LongPass,
                                 e_Action_HighPass,
                                 e_Action_ShortPass,
                                 e_Action_Shot,
                                 e_Action_Sprint,
                                 e_Action_ReleaseDirection,
                                 e_Action_ReleaseSprint,
                                 e_Action_Sliding,
                                 e_Action_Dribble,
                                 e_Action_ReleaseDribble,
                                 e_Action_BuiltinAI,
                                 e_Action_KeeperRush,
                                 e_Action_Pressure,
                                 e_Action_TeamPressure,
                                 e_Action_Switch,
                                 e_Action_ReleaseLongPass,
                                 e_Action_ReleaseHighPass,
                                 e_Action_ReleaseShortPass,
                                 e_Action_ReleaseShot,
                                 e_Action_ReleaseKeeperRush,
                                 e_Action_ReleaseSliding,
                                 e_Action_ReleasePressure,
                                 e_Action_ReleaseTeamPressure,
                                 e_Action_ReleaseSwitch};
    (void)actionSet;
    return fullSet[index];
}

enum AxisNegation : int { NEGATE_X = 1, NEGATE_Y = 2 };

// The action that points the same way on a pitch whose x, y or both change sign.
CUJU_FUNCTION inline int NegateActionAxes(int action, bool negateX, bool negateY) {
    int horizontal, vertical;
    switch (action) {
        case e_Action_Left:
            horizontal = -1, vertical = 0;
            break;
        case e_Action_TopLeft:
            horizontal = -1, vertical = -1;
            break;
        case e_Action_Top:
            horizontal = 0, vertical = -1;
            break;
        case e_Action_TopRight:
            horizontal = 1, vertical = -1;
            break;
        case e_Action_Right:
            horizontal = 1, vertical = 0;
            break;
        case e_Action_BottomRight:
            horizontal = 1, vertical = 1;
            break;
        case e_Action_Bottom:
            horizontal = 0, vertical = 1;
            break;
        case e_Action_BottomLeft:
            horizontal = -1, vertical = 1;
            break;
        default:
            return action;
    }
    if (negateX) horizontal = -horizontal;
    if (negateY) vertical = -vertical;
    const int directions[3][3] = {{e_Action_TopLeft, e_Action_Top, e_Action_TopRight},
                                  {e_Action_Left, action, e_Action_Right},
                                  {e_Action_BottomLeft, e_Action_Bottom, e_Action_BottomRight}};
    return directions[vertical + 1][horizontal + 1];
}

constexpr int MAXIMUM_STICKY_ACTIONS = 13;
CUJU_FUNCTION inline int StickyActionCount(int actionSet) { return actionSet == 2 ? 13 : 10; }
CUJU_FUNCTION inline int StickyAction(int index) {
    constexpr int actions[MAXIMUM_STICKY_ACTIONS] = {
        e_Action_Left,        e_Action_TopLeft,     e_Action_Top,        e_Action_TopRight,
        e_Action_Right,       e_Action_BottomRight, e_Action_Bottom,     e_Action_BottomLeft,
        e_Action_Sprint,      e_Action_Dribble,     e_Action_KeeperRush, e_Action_Pressure,
        e_Action_TeamPressure};  // The last three are in `full` only.
    return actions[index];
}

// An agent acts through a keyboard controller. IDs 0 to 10 belong to the left team, 11 to 21 to
// the right team.
constexpr int MAXIMUM_TEAM_CONTROLLERS = 11;

// The state holds a gamer's keyboard twice, in the gamer and in the match, every other controller
// once among the free controllers, and some controllers not at all.
using KeyboardCopies = InlineVector<state::KeyboardState *, 2>;
CUJU_FUNCTION_NO_INLINE inline KeyboardCopies Keyboard(World &world, int controllerID) {
    state::MatchState &match = world.serialized.match;
    const int gamerCount = match.teams[0].humanGamers.count + match.teams[1].humanGamers.count;
    for (int gamer = 0; gamer < gamerCount; gamer++) {
        if (GamerByGlobal(world, gamer).keyboardIndex == controllerID) {
            KeyboardCopies copies;
            copies.push_back(&GamerByGlobal(world, gamer).controller.keyboard);
            copies.push_back(&match.secondKeyboardCopies[gamer]);
            return copies;
        }
    }
    // The free controllers are the controllers that no gamer owns, in ID order.
    int freeIndex = 0;
    for (int controller = 0; controller < controllerID; controller++) {
        bool owned = false;
        for (int gamer = 0; gamer < gamerCount; gamer++)
            owned |= GamerByGlobal(world, gamer).keyboardIndex == controller;
        if (!owned) freeIndex++;
    }
    KeyboardCopies copies;
    if (freeIndex < match.freeControllerCount) copies.push_back(&match.freeControllers[freeIndex]);
    return copies;
}

// `player` is the index of the agent within its team.
CUJU_FUNCTION_NO_INLINE inline void EnvironmentAction(World &world, int action, bool leftTeam,
                                                      int player) {
    const int controllerID = player + (leftTeam ? 0 : MAXIMUM_TEAM_CONTROLLERS);
    for (state::KeyboardState *keyboard : Keyboard(world, controllerID)) {
        // GRF: a state holds the direction of a keyboard times its mirror sign.
        auto setDirection = [&](float x, float y) {
            keyboard->direction = Vector3(x, y, 0) * keyboard->mirror;
        };
        auto setButton = [&](int button, bool pressed) {
            keyboard->buttonsPressed[button] = pressed;
        };
        keyboard->disabled = false;
        switch (action) {
            case e_Action_Idle:
                break;
            case e_Action_Left:
                setDirection(-1, 0);
                break;
            case e_Action_TopLeft:
                setDirection(-1, 1);
                break;
            case e_Action_Top:
                setDirection(0, 1);
                break;
            case e_Action_TopRight:
                setDirection(1, 1);
                break;
            case e_Action_Right:
                setDirection(1, 0);
                break;
            case e_Action_BottomRight:
                setDirection(1, -1);
                break;
            case e_Action_Bottom:
                setDirection(0, -1);
                break;
            case e_Action_BottomLeft:
                setDirection(-1, -1);
                break;
            case e_Action_LongPass:
                setButton(e_ButtonFunction_LongPass, true);
                break;
            case e_Action_HighPass:
                setButton(e_ButtonFunction_HighPass, true);
                break;
            case e_Action_ShortPass:
                setButton(e_ButtonFunction_ShortPass, true);
                break;
            case e_Action_Shot:
                setButton(e_ButtonFunction_Shot, true);
                break;
            case e_Action_KeeperRush:
                setButton(e_ButtonFunction_KeeperRush, true);
                break;
            case e_Action_Sliding:
                setButton(e_ButtonFunction_Sliding, true);
                break;
            case e_Action_Pressure:
                setButton(e_ButtonFunction_Pressure, true);
                break;
            case e_Action_TeamPressure:
                setButton(e_ButtonFunction_TeamPressure, true);
                break;
            case e_Action_Switch:
                setButton(e_ButtonFunction_Switch, true);
                break;
            case e_Action_Sprint:
                setButton(e_ButtonFunction_Sprint, true);
                break;
            case e_Action_Dribble:
                setButton(e_ButtonFunction_Dribble, true);
                break;
            case e_Action_ReleaseDirection:
                setDirection(0, 0);
                break;
            case e_Action_ReleaseLongPass:
                setButton(e_ButtonFunction_LongPass, false);
                break;
            case e_Action_ReleaseHighPass:
                setButton(e_ButtonFunction_HighPass, false);
                break;
            case e_Action_ReleaseShortPass:
                setButton(e_ButtonFunction_ShortPass, false);
                break;
            case e_Action_ReleaseShot:
                setButton(e_ButtonFunction_Shot, false);
                break;
            case e_Action_ReleaseKeeperRush:
                setButton(e_ButtonFunction_KeeperRush, false);
                break;
            case e_Action_ReleaseSliding:
                setButton(e_ButtonFunction_Sliding, false);
                break;
            case e_Action_ReleasePressure:
                setButton(e_ButtonFunction_Pressure, false);
                break;
            case e_Action_ReleaseTeamPressure:
                setButton(e_ButtonFunction_TeamPressure, false);
                break;
            case e_Action_ReleaseSwitch:
                setButton(e_ButtonFunction_Switch, false);
                break;
            case e_Action_ReleaseSprint:
                setButton(e_ButtonFunction_Sprint, false);
                break;
            case e_Action_ReleaseDribble:
                setButton(e_ButtonFunction_Dribble, false);
                break;
            case e_Action_BuiltinAI:
                // GRF: `AIControlledKeyboard::SetDisabled(true)` zeroes the direction, and a state
                // holds it times the mirror sign, so a right keyboard holds -0.
                keyboard->disabled = true;
                setDirection(0, 0);
                for (bool &button : keyboard->buttonsPressed) button = false;
                break;
        }
    }
}

// A controller that the state does not hold is at its defaults, no direction and no button pressed.
CUJU_FUNCTION inline bool StickyActionState(World &world, int action, bool leftTeam, int player) {
    KeyboardCopies copies = Keyboard(world, player + (leftTeam ? 0 : MAXIMUM_TEAM_CONTROLLERS));
    if (copies.count == 0) return false;
    const state::KeyboardState &keyboard = *copies[0];
    // Exact: the mirror sign is 1 or -1, so a second multiplication recovers the chosen direction.
    const Vector3 direction = keyboard.direction * keyboard.mirror;
    switch (action) {
        case e_Action_Left:
            return direction == Vector3(-1, 0, 0);
        case e_Action_TopLeft:
            return direction == Vector3(-1, 1, 0);
        case e_Action_Top:
            return direction == Vector3(0, 1, 0);
        case e_Action_TopRight:
            return direction == Vector3(1, 1, 0);
        case e_Action_Right:
            return direction == Vector3(1, 0, 0);
        case e_Action_BottomRight:
            return direction == Vector3(1, -1, 0);
        case e_Action_Bottom:
            return direction == Vector3(0, -1, 0);
        case e_Action_BottomLeft:
            return direction == Vector3(-1, -1, 0);
        case e_Action_KeeperRush:
            return keyboard.buttonsPressed[e_ButtonFunction_KeeperRush];
        case e_Action_Pressure:
            return keyboard.buttonsPressed[e_ButtonFunction_Pressure];
        case e_Action_TeamPressure:
            return keyboard.buttonsPressed[e_ButtonFunction_TeamPressure];
        case e_Action_Sprint:
            return keyboard.buttonsPressed[e_ButtonFunction_Sprint];
        case e_Action_Dribble:
            return keyboard.buttonsPressed[e_ButtonFunction_Dribble];
    }
    return false;
}

CUJU_FUNCTION_NO_INLINE inline bool EnvironmentStep(World &world,
                                                    const SimulationContext &context) {
    for (int tick = 0; tick < context.ticksPerStep; tick++)
        if (!GameTaskProcessPhase(world, context)) return false;
    if (world.serialized.match.inPlay) {
        world.serialized.step++;
        for (int controller = 0; controller < 2 * MAXIMUM_TEAM_CONTROLLERS; controller++)
            for (state::KeyboardState *keyboard : Keyboard(world, controller))
                for (int button : {e_ButtonFunction_LongPass, e_ButtonFunction_HighPass,
                                   e_ButtonFunction_ShortPass, e_ButtonFunction_Shot,
                                   e_ButtonFunction_Sliding, e_ButtonFunction_Switch})
                    keyboard->buttonsPressed[button] = false;
    }
    return true;
}

// GRF: the scales `X_FIELD_SCALE` and `Y_FIELD_SCALE` are double literals, so x and y divide in
// double, and z divides by the integer 1.
CUJU_FUNCTION inline float EnvironmentCoordinate(const Vector3 &vector, int index) {
    if (index == 0) return (float)((double)vector.coordinates[0] / 54.4);
    if (index == 1) return (float)((double)vector.coordinates[1] / -83.6);
    return vector.coordinates[2] / 1;
}

struct PlayerObservation {
    float position[2], direction[2];
    float tiredFactor;
    bool active, yellowCard;
    int role;
};
struct TeamObservation {
    int count = 0;
    PlayerObservation players[state::MAXIMUM_TEAM_PLAYERS];
    int designatedPlayer = -1;
};
// GRF's observation, seen from the left team. `team[0]` is the left team. Player indices count
// within a team, and -1 marks no player or no team.
struct Observation {
    float ball[3], ballDirection[3], ballRotation[3];
    TeamObservation team[2];
    int controlledPlayer[2][MAXIMUM_TEAM_CONTROLLERS];
    int gameMode;
    int score[2];
    int ballOwnedTeam, ballOwnedPlayer;
    int step;
    bool isInPlay;
};

// Speed: an out parameter, since a returned `Observation` is copied at every call site.
CUJU_FUNCTION_NO_INLINE inline void EnvironmentGetState(const World &world,
                                                        const SimulationContext &context,
                                                        Observation &observation) {
    CUJU_PROFILE_SCOPE(ENVIRONMENT_GET_STATE);
    const state::MatchState &match = world.serialized.match;
    // Speed: a loop over the words is shorter than a store per member. The prototype keeps the -1
    // of a team without a designated player.
    WarpReset(observation);
    auto setCoordinates = [](float *out, const Vector3 &vector) {
        for (int index = 0; index < 3; index++) out[index] = EnvironmentCoordinate(vector, index);
    };
    setCoordinates(observation.ball, BallAveragePosition(world, 5));
    float rotationX, rotationY, rotationZ;
    match.ball.rotationPerMillisecond.GetAngles(rotationX, rotationY, rotationZ);
    setCoordinates(observation.ballRotation,
                   Vector3(rotationX, rotationY, rotationZ) / (float)context.ticksPerStep);
    setCoordinates(observation.ballDirection, match.ball.momentum / (float)context.ticksPerStep);
    observation.ballOwnedPlayer = -1;
    observation.ballOwnedTeam = -1;
    observation.score[0] = MatchGoalCount(world, 0);
    observation.score[1] = MatchGoalCount(world, 1);
    // GRF: the step before the start at 2000 ms, which ends at 1900 ms, reports in play.
    observation.isInPlay = match.inPlay || match.actualTimeMilliseconds == 1900;
    observation.gameMode =
        match.inSetPiece ? match.referee.buffer.desiredSetPiece : e_GameMode_Normal;
    for (auto &teamControllers : observation.controlledPlayer)
        for (int &controlled : teamControllers) controlled = -1;
    const int firstTeamID = world.FirstTeamID();
    for (int teamID : {firstTeamID, 1 - firstTeamID}) {
        TeamObservation &teamObservation = observation.team[teamID];
        const state::TeamState &team = TeamByID(world, teamID);
        for (int player : TeamAllPlayers(world, teamID)) {
            const state::PlayerState &playerState = match.players[player];
            if (playerState.base.externalController >= 0) {
                int controller =
                    GamerByGlobal(const_cast<World &>(world), playerState.base.externalController)
                        .keyboardIndex;
                observation.controlledPlayer[teamID][controller % MAXIMUM_TEAM_CONTROLLERS] =
                    teamObservation.count;
            }
            Vector3 position = playerState.base.humanoid.spatialState.position;
            Vector3 movement = playerState.base.humanoid.spatialState.movement;
            if (teamID == 1) {
                position.Mirror();
                movement.Mirror();
            }
            PlayerObservation &playerObservation = teamObservation.players[teamObservation.count];
            playerObservation.position[0] = EnvironmentCoordinate(position, 0);
            playerObservation.position[1] = EnvironmentCoordinate(position, 1);
            Vector3 direction = movement / (float)context.ticksPerStep;
            playerObservation.direction[0] = EnvironmentCoordinate(direction, 0);
            playerObservation.direction[1] = EnvironmentCoordinate(direction, 1);
            playerObservation.tiredFactor = 1 - playerState.base.fatigueFactorInverse;
            playerObservation.yellowCard = playerState.cards > 0;
            playerObservation.active = playerState.base.isActive;
            playerObservation.role = Formation(world, context, player).role;
            if (playerState.hasPossession && match.lastTouchTeamID != -1 &&
                TeamByID(world, MatchLastTouchTeam(world)).lastTouchPlayer == player) {
                observation.ballOwnedPlayer = teamObservation.count;
                observation.ballOwnedTeam = match.lastTouchTeamID;
            }
            if (player == team.mainSelectedPlayer)
                teamObservation.designatedPlayer = teamObservation.count;
            teamObservation.count++;
        }
    }
    observation.step = world.serialized.step;
}

enum GameState {
    e_GameState_Created,
    e_GameState_Initiated,
    e_GameState_Running,
    e_GameState_Done
};

// `world` holds a copy of a template, the state after a reset of the same scenario, agents and
// `reverseTeamProcessing`. Resets with other seeds differ only in the RNG, the stored seed and
// the idle frames of the humanoids, so this reseeds and replays GRF's reset draws in their order.
CUJU_FUNCTION_NO_INLINE inline bool EnvironmentResetFromTemplate(World &world,
                                                                 const SimulationContext &context,
                                                                 uint32_t seed) {
    world.serialized.scenario.gameEngineRandomSeed = seed;
    Randomize(world, seed);
    const int playerCount = world.serialized.match.playerCount;
    for (int player = 0; player < playerCount; player++) {
        const int side = TeamDynamicSide(world, world.playerTeamID[player]);
        if (!HumanoidResetPosition(world, context, player, Vector3(0), Vector3(0))) return false;
        Vector3 position =
            Formation(world, context, player).position * 25 * Vector3(-side, -side, 0);
        if (!HumanoidResetPosition(world, context, player, position, Vector3(0))) return false;
    }
    RandomInRange(world.serialized.rng, 1, 4);  // The skin colour of the officials.
    for (int official = 0; official < 3; official++)
        for (int draw = 0; draw < 2; draw++)
            if (!HumanoidResetPosition(world, context, playerCount + official, Vector3(0),
                                       Vector3(0)))
                return false;
    const Vector3 placed[3] = {Vector3(10, -10, 0), Vector3(25, -36.5f, 0), Vector3(-25, 36.5f, 0)};
    for (int official = 0; official < 3; official++)
        if (!HumanoidResetPosition(world, context, playerCount + official, placed[official],
                                   Vector3(0)))
            return false;
    // GRF: the lighting draws six times and affects nothing else.
    RandomInRange(world.serialized.rng, -1.7f, 1.7f);
    RandomInRange(world.serialized.rng, -1.7f, 1.7f);
    RandomInRange(world.serialized.rng, 0, 1);
    for (int draw = 0; draw < 3; draw++)
        RandomInRange(world.serialized.rng, (float)-0.1, (float)0.1);
    return true;
}

CUJU_FUNCTION inline bool EnvironmentCoreAfterReset(World &world, const SimulationContext &context,
                                                    struct EnvironmentCore &core);

struct EnvironmentCore {
    int previousScoreDifference = 0;
    int previousGameMode = -1;
    int previousBallOwnedTeam = -1;
    Observation observation;
};

// `actions` holds one `Action` per agent, the left agents first. `reward` is the left team's
// change of score difference. `tickHook` runs after every tick and returns `false` to stop.
// GRF: after 20 steps in a set piece every agent short-passes, then the ball owner's agent runs
// to one side and back while the rest idle.
template <class TickHook>
CUJU_FUNCTION_NO_INLINE inline bool EnvironmentCoreStep(World &world,
                                                        const SimulationContext &context,
                                                        EnvironmentCore &core, const int *actions,
                                                        int &reward, bool &done,
                                                        TickHook tickHook) {
    const state::ScenarioState &scenario = world.serialized.scenario;
    size_t actionIndex = 0;
    for (bool leftTeam : {true, false}) {
        const int agents = leftTeam ? scenario.leftAgents : scenario.rightAgents;
        for (int agent = 0; agent < agents; agent++) {
            int playerAction = actions[actionIndex];
            if (world.serialized.waitingForGameCount == 20) {
                playerAction = e_Action_ShortPass;
            } else if (world.serialized.waitingForGameCount > 20) {
                playerAction = e_Action_Idle;
                const int controlled = core.observation.controlledPlayer[leftTeam ? 0 : 1][agent];
                if (core.observation.ballOwnedTeam != -1 &&
                    (core.observation.ballOwnedTeam ^ (int)leftTeam) &&
                    controlled == core.observation.ballOwnedPlayer) {
                    if ((world.serialized.waitingForGameCount < 30) != leftTeam)
                        playerAction = e_Action_Left;
                    else
                        playerAction = e_Action_Right;
                }
            }
            actionIndex++;
            EnvironmentAction(world, playerAction, leftTeam, agent);
        }
    }
    while (true) {
        for (int tick = 0; tick < context.ticksPerStep; tick++) {
            if (!GameTaskProcessPhase(world, context)) return false;
            if (!tickHook()) return false;
        }
        if (world.serialized.match.inPlay) {
            world.serialized.step++;
            for (int controller = 0; controller < 2 * MAXIMUM_TEAM_CONTROLLERS; controller++)
                for (state::KeyboardState *keyboard : Keyboard(world, controller))
                    for (int button : {e_ButtonFunction_LongPass, e_ButtonFunction_HighPass,
                                       e_ButtonFunction_ShortPass, e_ButtonFunction_Shot,
                                       e_ButtonFunction_Sliding, e_ButtonFunction_Switch})
                        keyboard->buttonsPressed[button] = false;
        }
        EnvironmentGetState(world, context, core.observation);
        if (core.observation.isInPlay) break;
    }
    const Observation &observation = core.observation;
    if (scenario.endEpisodeOnScore && (observation.score[0] > 0 || observation.score[1] > 0))
        world.serialized.gameState = e_GameState_Done;
    if (scenario.endEpisodeOnOutOfPlay && observation.gameMode != e_GameMode_Normal &&
        core.previousGameMode == e_GameMode_Normal)
        world.serialized.gameState = e_GameState_Done;
    core.previousGameMode = observation.gameMode;
    if (scenario.endEpisodeOnPossessionChange && observation.ballOwnedTeam != -1 &&
        core.previousBallOwnedTeam != -1 && observation.ballOwnedTeam != core.previousBallOwnedTeam)
        world.serialized.gameState = e_GameState_Done;
    if (observation.ballOwnedTeam != -1) core.previousBallOwnedTeam = observation.ballOwnedTeam;
    const int scoreDifference = observation.score[0] - observation.score[1];
    reward = scoreDifference - core.previousScoreDifference;
    core.previousScoreDifference = scoreDifference;
    if (observation.gameMode != e_GameMode_Normal)
        world.serialized.waitingForGameCount += 1;
    else
        world.serialized.waitingForGameCount = 0;
    if (observation.step >= scenario.gameDuration) world.serialized.gameState = e_GameState_Done;
    done = world.serialized.gameState == e_GameState_Done;
    return true;
}

CUJU_FUNCTION inline bool EnvironmentCoreAfterReset(World &world, const SimulationContext &context,
                                                    EnvironmentCore &core) {
    core = EnvironmentCore();
    while (true) {
        EnvironmentGetState(world, context, core.observation);
        if (core.observation.isInPlay) return true;
        if (!EnvironmentStep(world, context)) return false;
    }
}

}  // namespace simulation
}  // namespace cuju

// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The structures of a state, GRF's serialized world, and the codec that loads and saves them.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "../quaternion.h"
#include "../vector3.h"
#include "archive.h"

namespace cuju {
namespace state {

// Each `Process` visits the fields in the order of GRF's byte format, and the same code loads and
// saves, so a load then a save reproduces the input bytes. A pointer is an `int32_t` index, -1 for
// null. Players count the first team then the second, teams are 0 and 1, human gamers count the
// first team's then the second's, keyboards count in creation order, animations index the tables.

#ifndef CUJU_TEAM_PLAYERS
// Room for players in one team. `-DCUJU_TEAM_PLAYERS=5`, for 5 versus 5 only, shrinks every world.
#define CUJU_TEAM_PLAYERS 11
#endif
constexpr int MAXIMUM_TEAM_PLAYERS = CUJU_TEAM_PLAYERS;
// One per frame. GRF's animations have at most 306 frames, and the table load rejects more.
constexpr int MAXIMUM_ANIMATION_POSITIONS = 320;

// GRF: the type tags are mangled C++ type names, which GRF's states hold byte for byte.
#define TYPE_FLOAT "f"
#define TYPE_INTEGER "i"
#define TYPE_UNSIGNED_INTEGER "j"
#define TYPE_BOOLEAN "b"
#define TYPE_CHARACTER "c"
#define TYPE_UNSIGNED_LONG "m"
#define TYPE_VECTOR3 "N7blunted7Vector3E"
#define TYPE_QUATERNION "N7blunted10QuaternionE"
#define TYPE_RADIAN "N7blunted6radianE"
#define TYPE_GAME_STATE "9GameState"
#define TYPE_ROLE "12e_PlayerRole"
#define TYPE_VELOCITY "10e_Velocity"
#define TYPE_GAME_MODE "10e_GameMode"
#define TYPE_FUNCTION_TYPE "14e_FunctionType"
#define TYPE_INTERRUPT "15e_InterruptAnim"
#define TYPE_STRICT "16e_StrictMovement"
#define TYPE_TOUCH_TYPE "11e_TouchType"
#define TYPE_BUTTON "16e_ButtonFunction"
#define TYPE_MATCH_PHASE "12e_MatchPhase"
#define TYPE_LOCAL_MODE "N7blunted11e_LocalModeE"
#define TYPE_BODY_PART "N7blunted8BodyPartE"
#define TYPE_FOOT "N7blunted6e_FootE"

// Both macros read the enclosing function's `Archive` parameter, which must be named `archive`.
#define PROCESS_FIELD(x, tag) (archive.field = #x, archive.Process(x, tag))
// `kind` is a `PointerKind` without its `POINTER_` prefix.
#define PROCESS_POINTER_FIELD(x, kind) \
    (archive.field = #x, archive.ProcessPointer(x, POINTER_##kind, TYPE_INTEGER))

// GRF's radian. The true angle is `angle - PI` when `rotated`, and a load keeps the padding.
struct Radian {
    float angle;
    bool rotated;
    uint8_t padding[3];
    CUJU_FUNCTION float value() const { return rotated ? angle - PI : angle; }
};
static_assert(sizeof(Radian) == 8, "radian layout");

constexpr int BODY_PARTS = 14;
constexpr int HUMANOID_NODES = 14;

struct BiasedOffset {
    Quaternion orientation;
    float bias;
    void Process(Archive &archive) {
        PROCESS_FIELD(orientation, TYPE_QUATERNION);
        PROCESS_FIELD(bias, TYPE_FLOAT);
    }
};
struct BiasedOffsets {
    BiasedOffset elements[BODY_PARTS];
    void Process(Archive &archive) {
        for (int part = 0; part < BODY_PARTS; part++) {
            ArchiveScope scope(archive, "elements", part);
            elements[part].Process(archive);
        }
    }
};
struct MovementHistoryEntry {
    Vector3 position;
    Quaternion orientation;
    int32_t timeDifferenceMilliseconds;
    int32_t nodeName;
    void Process(Archive &archive) {
        PROCESS_FIELD(position, TYPE_VECTOR3);
        PROCESS_FIELD(orientation, TYPE_QUATERNION);
        PROCESS_FIELD(timeDifferenceMilliseconds, TYPE_INTEGER);
        PROCESS_FIELD(nodeName, TYPE_BODY_PART);
    }
};

struct NodeState {
    Vector3 position;
    Quaternion rotation;
    Vector3 scale;
    int32_t localMode;
    void Process(Archive &archive) {
        PROCESS_FIELD(position, TYPE_VECTOR3);
        PROCESS_FIELD(rotation, TYPE_QUATERNION);
        PROCESS_FIELD(scale, TYPE_VECTOR3);
        PROCESS_FIELD(localMode, TYPE_LOCAL_MODE);
    }
};

struct TouchInformation {
    Vector3 inputDirection;
    float inputPower, autoDirectionBias, autoPowerBias;
    Vector3 desiredDirection;
    float desiredPower;
    int32_t targetPlayer, forcedTargetPlayer;
    void Process(Archive &archive) {
        PROCESS_FIELD(inputDirection, TYPE_VECTOR3);
        PROCESS_FIELD(inputPower, TYPE_FLOAT);
        PROCESS_FIELD(autoDirectionBias, TYPE_FLOAT);
        PROCESS_FIELD(autoPowerBias, TYPE_FLOAT);
        PROCESS_FIELD(desiredDirection, TYPE_VECTOR3);
        PROCESS_FIELD(desiredPower, TYPE_FLOAT);
        PROCESS_POINTER_FIELD(targetPlayer, PLAYER);
        PROCESS_POINTER_FIELD(forcedTargetPlayer, PLAYER);
    }
};
struct PlayerCommand {
    int32_t desiredFunctionType;
    bool useDesiredMovement;
    Vector3 desiredDirection;
    int32_t strictMovement;
    float desiredVelocityFloat;
    bool useDesiredLookAt;
    Vector3 desiredLookAt;
    bool useTouchInformation;
    TouchInformation touchInformation;
    bool onlyDeflectAnimationsThatPickupBall, useTripType;
    int32_t tripType;
    bool useDesiredTripDirection;
    Vector3 desiredTripDirection;
    bool useSpecialVariable1;
    int32_t specialVariable1;
    bool useSpecialVariable2;
    int32_t specialVariable2;
    int32_t modifier;
    void Process(Archive &archive) {
        PROCESS_FIELD(desiredFunctionType, TYPE_FUNCTION_TYPE);
        PROCESS_FIELD(useDesiredMovement, TYPE_BOOLEAN);
        PROCESS_FIELD(desiredDirection, TYPE_VECTOR3);
        PROCESS_FIELD(strictMovement, TYPE_STRICT);
        PROCESS_FIELD(desiredVelocityFloat, TYPE_FLOAT);
        PROCESS_FIELD(useDesiredLookAt, TYPE_BOOLEAN);
        PROCESS_FIELD(desiredLookAt, TYPE_VECTOR3);
        PROCESS_FIELD(useTouchInformation, TYPE_BOOLEAN);
        PROCESS_STRUCT(touchInformation);
        PROCESS_FIELD(onlyDeflectAnimationsThatPickupBall, TYPE_BOOLEAN);
        PROCESS_FIELD(useTripType, TYPE_BOOLEAN);
        PROCESS_FIELD(tripType, TYPE_INTEGER);
        PROCESS_FIELD(useDesiredTripDirection, TYPE_BOOLEAN);
        PROCESS_FIELD(desiredTripDirection, TYPE_VECTOR3);
        PROCESS_FIELD(useSpecialVariable1, TYPE_BOOLEAN);
        PROCESS_FIELD(specialVariable1, TYPE_INTEGER);
        PROCESS_FIELD(useSpecialVariable2, TYPE_BOOLEAN);
        PROCESS_FIELD(specialVariable2, TYPE_INTEGER);
        PROCESS_FIELD(modifier, TYPE_INTEGER);
    }
};
struct FormationEntry {
    int32_t role;
    Vector3 position, startPosition;
    bool lazy, controllable;
    void Process(Archive &archive) {
        PROCESS_FIELD(role, TYPE_ROLE);
        PROCESS_FIELD(position, TYPE_VECTOR3);
        PROCESS_FIELD(startPosition, TYPE_VECTOR3);
        PROCESS_FIELD(lazy, TYPE_BOOLEAN);
        PROCESS_FIELD(controllable, TYPE_BOOLEAN);
    }
};
struct PlayerImage {
    Vector3 position, directionVector, movement;
    int32_t player;
    int32_t velocity, role;
    void Process(Archive &archive) {
        PROCESS_FIELD(position, TYPE_VECTOR3);
        PROCESS_FIELD(directionVector, TYPE_VECTOR3);
        PROCESS_FIELD(movement, TYPE_VECTOR3);
        PROCESS_POINTER_FIELD(player, PLAYER);
        PROCESS_FIELD(velocity, TYPE_VELOCITY);
        PROCESS_FIELD(role, TYPE_ROLE);
    }
};

struct RotationSmuggle {
    Radian begin, end;
    void Process(Archive &archive) {
        PROCESS_FIELD(begin, TYPE_RADIAN);
        PROCESS_FIELD(end, TYPE_RADIAN);
    }
};
struct AnimationState {
    int32_t animation;
    int32_t id, frameNumber, functionType, originatingInterrupt;
    Vector3 actionSmuggle, actionSmuggleOffset, actionSmuggleSustain, actionSmuggleSustainOffset,
        movementSmuggle, movementSmuggleOffset;
    RotationSmuggle rotationSmuggle;
    Radian rotationSmuggleOffset;
    int32_t touchFrame;
    Vector3 touchPosition, incomingMovement, outgoingMovement, positionOffset;
    PlayerCommand originatingCommand;
    FixedVector<Vector3, MAXIMUM_ANIMATION_POSITIONS> positions;
    void Process(Archive &archive) {
        PROCESS_POINTER_FIELD(animation, ANIMATION);
        PROCESS_FIELD(id, TYPE_INTEGER);
        PROCESS_FIELD(frameNumber, TYPE_INTEGER);
        PROCESS_FIELD(functionType, TYPE_FUNCTION_TYPE);
        PROCESS_FIELD(originatingInterrupt, TYPE_INTERRUPT);
        PROCESS_FIELD(actionSmuggle, TYPE_VECTOR3);
        PROCESS_FIELD(actionSmuggleOffset, TYPE_VECTOR3);
        PROCESS_FIELD(actionSmuggleSustain, TYPE_VECTOR3);
        PROCESS_FIELD(actionSmuggleSustainOffset, TYPE_VECTOR3);
        PROCESS_FIELD(movementSmuggle, TYPE_VECTOR3);
        PROCESS_FIELD(movementSmuggleOffset, TYPE_VECTOR3);
        PROCESS_STRUCT(rotationSmuggle);
        PROCESS_FIELD(rotationSmuggleOffset, TYPE_RADIAN);
        PROCESS_FIELD(touchFrame, TYPE_INTEGER);
        PROCESS_FIELD(touchPosition, TYPE_VECTOR3);
        PROCESS_FIELD(incomingMovement, TYPE_VECTOR3);
        PROCESS_FIELD(outgoingMovement, TYPE_VECTOR3);
        PROCESS_FIELD(positionOffset, TYPE_VECTOR3);
        PROCESS_STRUCT(originatingCommand);
        archive.field = "positions";
        ProcessVector(archive, positions,
                      [&](Vector3 &position) { archive.Process(position, TYPE_VECTOR3); });
    }
};
struct AnimationApplyBuffer {
    int32_t animation;
    int32_t frameNumber;
    bool smooth;
    float smoothFactor;
    bool noPosition;
    Vector3 position;
    Radian orientation;
    BiasedOffsets offsets;
    void Process(Archive &archive) {
        PROCESS_POINTER_FIELD(animation, ANIMATION);
        PROCESS_FIELD(frameNumber, TYPE_INTEGER);
        PROCESS_FIELD(smooth, TYPE_BOOLEAN);
        PROCESS_FIELD(smoothFactor, TYPE_FLOAT);
        PROCESS_FIELD(noPosition, TYPE_BOOLEAN);
        PROCESS_FIELD(position, TYPE_VECTOR3);
        PROCESS_FIELD(orientation, TYPE_RADIAN);
        PROCESS_STRUCT(offsets);
    }
};
struct SpatialState {
    Vector3 position;
    Radian angle;
    Vector3 directionVector;
    int32_t velocityCategory;
    float floatVelocity;
    Vector3 actualMovement, physicsMovement, animationMovement, movement, actionSmuggleMovement,
        movementSmuggleMovement, positionOffsetMovement;
    Radian bodyAngle;
    Vector3 bodyDirectionVector;
    Radian relativeBodyAngleNonQuantized, relativeBodyAngle;
    Vector3 relativeBodyDirectionVector, relativeBodyDirectionVectorNonQuantized;
    int32_t foot;
    void Process(Archive &archive) {
        PROCESS_FIELD(position, TYPE_VECTOR3);
        PROCESS_FIELD(angle, TYPE_RADIAN);
        PROCESS_FIELD(directionVector, TYPE_VECTOR3);
        PROCESS_FIELD(velocityCategory, TYPE_VELOCITY);
        PROCESS_FIELD(floatVelocity, TYPE_FLOAT);
        PROCESS_FIELD(actualMovement, TYPE_VECTOR3);
        PROCESS_FIELD(physicsMovement, TYPE_VECTOR3);
        PROCESS_FIELD(animationMovement, TYPE_VECTOR3);
        PROCESS_FIELD(movement, TYPE_VECTOR3);
        PROCESS_FIELD(actionSmuggleMovement, TYPE_VECTOR3);
        PROCESS_FIELD(movementSmuggleMovement, TYPE_VECTOR3);
        PROCESS_FIELD(positionOffsetMovement, TYPE_VECTOR3);
        PROCESS_FIELD(bodyAngle, TYPE_RADIAN);
        PROCESS_FIELD(bodyDirectionVector, TYPE_VECTOR3);
        PROCESS_FIELD(relativeBodyAngleNonQuantized, TYPE_RADIAN);
        PROCESS_FIELD(relativeBodyAngle, TYPE_RADIAN);
        PROCESS_FIELD(relativeBodyDirectionVector, TYPE_VECTOR3);
        PROCESS_FIELD(relativeBodyDirectionVectorNonQuantized, TYPE_VECTOR3);
        PROCESS_FIELD(foot, TYPE_FOOT);
    }
};
struct HumanoidState {
    NodeState nodes[HUMANOID_NODES];
    AnimationApplyBuffer animationApplyBuffer;
    BiasedOffsets offsets;
    AnimationState currentAnimation;
    int32_t previousAnimationFrameNumber, previousAnimationFunctionType;
    Vector3 startPosition;
    Radian startAngle;
    Vector3 nextStartPosition;
    Radian nextStartAngle;
    SpatialState spatialState;
    Vector3 previousPosition2D;
    int32_t interruptAnimation, requeueDelayFrames, tripType;
    Vector3 tripDirection, decayingPositionOffset;
    float decayingDifficultyFactor;
    FixedVector<MovementHistoryEntry, 32> movementHistory;
    int32_t mentalImageTime;
    void Process(Archive &archive) {
        for (int node = 0; node < HUMANOID_NODES; node++) {
            ArchiveScope scope(archive, "nodes", node);
            nodes[node].Process(archive);
        }
        PROCESS_STRUCT(animationApplyBuffer);
        PROCESS_STRUCT(offsets);
        PROCESS_STRUCT(currentAnimation);
        PROCESS_FIELD(previousAnimationFrameNumber, TYPE_INTEGER);
        PROCESS_FIELD(previousAnimationFunctionType, TYPE_FUNCTION_TYPE);
        PROCESS_FIELD(startPosition, TYPE_VECTOR3);
        PROCESS_FIELD(startAngle, TYPE_RADIAN);
        PROCESS_FIELD(nextStartPosition, TYPE_VECTOR3);
        PROCESS_FIELD(nextStartAngle, TYPE_RADIAN);
        PROCESS_STRUCT(spatialState);
        PROCESS_FIELD(previousPosition2D, TYPE_VECTOR3);
        PROCESS_FIELD(interruptAnimation, TYPE_INTERRUPT);
        PROCESS_FIELD(requeueDelayFrames, TYPE_INTEGER);
        PROCESS_FIELD(tripType, TYPE_INTEGER);
        PROCESS_FIELD(tripDirection, TYPE_VECTOR3);
        PROCESS_FIELD(decayingPositionOffset, TYPE_VECTOR3);
        PROCESS_FIELD(decayingDifficultyFactor, TYPE_FLOAT);
        archive.field = "movementHistory";
        ProcessVector(archive, movementHistory,
                      [&](MovementHistoryEntry &entry) { entry.Process(archive); });
        PROCESS_FIELD(mentalImageTime, TYPE_INTEGER);
    }
};

struct PlayerControllerState {
    int32_t player;
    Vector3 inputDirection;
    float inputVelocityFloat;
    int32_t team, opponentTeam;
    int32_t opponentPlayer;
    float timeNeededMilliseconds;
    int32_t mentalImageTime;
    uint64_t lastSwitchTimeMilliseconds;
    uint32_t lastSwitchTimeDurationMilliseconds;
    bool hasPossession, hasUniquePossession, teamHasPossession, teamHasUniquePossession,
        opponentTeamHasPossession, opponentTeamHasUniquePossession, hasBestPossession,
        teamHasBestPossession;
    float possessionAmount, teamPossessionAmount, fadingTeamPossessionAmount;
    int32_t opponentTimeNeededToGetToBall;
    bool hasBestChanceOfPossession;
    void Process(Archive &archive) {
        PROCESS_POINTER_FIELD(player, PLAYER);
        PROCESS_FIELD(inputDirection, TYPE_VECTOR3);
        PROCESS_FIELD(inputVelocityFloat, TYPE_FLOAT);
        PROCESS_POINTER_FIELD(team, TEAM);
        PROCESS_POINTER_FIELD(opponentTeam, TEAM);
        PROCESS_POINTER_FIELD(opponentPlayer, PLAYER);
        PROCESS_FIELD(timeNeededMilliseconds, TYPE_FLOAT);
        PROCESS_FIELD(mentalImageTime, TYPE_INTEGER);
        PROCESS_FIELD(lastSwitchTimeMilliseconds, TYPE_UNSIGNED_LONG);
        PROCESS_FIELD(lastSwitchTimeDurationMilliseconds, TYPE_UNSIGNED_INTEGER);
        PROCESS_FIELD(hasPossession, TYPE_BOOLEAN);
        PROCESS_FIELD(hasUniquePossession, TYPE_BOOLEAN);
        PROCESS_FIELD(teamHasPossession, TYPE_BOOLEAN);
        PROCESS_FIELD(teamHasUniquePossession, TYPE_BOOLEAN);
        PROCESS_FIELD(opponentTeamHasPossession, TYPE_BOOLEAN);
        PROCESS_FIELD(opponentTeamHasUniquePossession, TYPE_BOOLEAN);
        PROCESS_FIELD(hasBestPossession, TYPE_BOOLEAN);
        PROCESS_FIELD(teamHasBestPossession, TYPE_BOOLEAN);
        PROCESS_FIELD(possessionAmount, TYPE_FLOAT);
        PROCESS_FIELD(teamPossessionAmount, TYPE_FLOAT);
        PROCESS_FIELD(fadingTeamPossessionAmount, TYPE_FLOAT);
        PROCESS_FIELD(opponentTimeNeededToGetToBall, TYPE_INTEGER);
        PROCESS_FIELD(hasBestChanceOfPossession, TYPE_BOOLEAN);
    }
};
struct AIControllerState {
    PlayerControllerState base;
    bool ballBoundForGoal;
    float ballBoundForGoalYCoordinate;
    Vector3 lastDesiredDirection;
    float lastDesiredVelocity;
    void Process(Archive &archive) {
        PROCESS_STRUCT(base);
        PROCESS_FIELD(ballBoundForGoal, TYPE_BOOLEAN);
        PROCESS_FIELD(ballBoundForGoalYCoordinate, TYPE_FLOAT);
        PROCESS_FIELD(lastDesiredDirection, TYPE_VECTOR3);
        PROCESS_FIELD(lastDesiredVelocity, TYPE_FLOAT);
    }
};
struct KeyboardState {
    float mirror;
    // GRF: the state stores the direction times `mirror`, which is 1 or -1.
    Vector3 direction;
    bool disabled;
    bool buttonsPressed[11];
    void Process(Archive &archive) {
        PROCESS_FIELD(mirror, TYPE_FLOAT);
        PROCESS_FIELD(direction, TYPE_VECTOR3);
        PROCESS_FIELD(disabled, TYPE_BOOLEAN);
        for (int button = 0; button < 11; button++) {
            ArchiveScope scope(archive, "buttonsPressed", button);
            archive.field = archive.layout ? "" : "buttons_pressed";
            archive.Process(buttonsPressed[button], TYPE_BOOLEAN);
        }
    }
};
struct HumanControllerState {
    PlayerControllerState base;
    KeyboardState keyboard;
    int32_t actionMode, actionButton, actionBufferTimeMilliseconds, gaugeMilliseconds;
    Vector3 previousDirection, steadyDirection;
    int32_t lastSteadyDirectionSnapshotTimeMilliseconds;
    void Process(Archive &archive) {
        PROCESS_STRUCT(base);
        PROCESS_STRUCT(keyboard);
        PROCESS_FIELD(actionMode, TYPE_INTEGER);
        PROCESS_FIELD(actionButton, TYPE_BUTTON);
        PROCESS_FIELD(actionBufferTimeMilliseconds, TYPE_INTEGER);
        PROCESS_FIELD(gaugeMilliseconds, TYPE_INTEGER);
        PROCESS_FIELD(previousDirection, TYPE_VECTOR3);
        PROCESS_FIELD(steadyDirection, TYPE_VECTOR3);
        PROCESS_FIELD(lastSteadyDirectionSnapshotTimeMilliseconds, TYPE_INTEGER);
    }
};
struct HumanGamerState {
    int32_t selectedPlayer;
    int32_t team;
    int32_t keyboardIndex;
    HumanControllerState controller;
    void Process(Archive &archive) {
        PROCESS_POINTER_FIELD(selectedPlayer, PLAYER);
        PROCESS_POINTER_FIELD(team, TEAM);
        PROCESS_POINTER_FIELD(keyboardIndex, KEYBOARD);
        PROCESS_STRUCT(controller);
    }
};

template <class Controller>
struct PlayerBaseState {
    bool isActive;
    HumanoidState humanoid;
    Controller controller;
    int32_t externalController;
    uint64_t lastTouchTimeMilliseconds;
    int32_t lastTouchType;
    float fatigueFactorInverse;
    FixedVector<Vector3, 512> positionHistoryPerSecond;
    void Process(Archive &archive) {
        PROCESS_FIELD(isActive, TYPE_BOOLEAN);
        PROCESS_STRUCT(humanoid);
        // GRF stores the controller of an active person only. The layout covers every controller.
        if (isActive || archive.Covering()) PROCESS_STRUCT(controller);
        PROCESS_POINTER_FIELD(externalController, GAMER);
        PROCESS_FIELD(lastTouchTimeMilliseconds, TYPE_UNSIGNED_LONG);
        PROCESS_FIELD(lastTouchType, TYPE_TOUCH_TYPE);
        PROCESS_FIELD(fatigueFactorInverse, TYPE_FLOAT);
        archive.field = "positionHistoryPerSecond";
        ProcessVector(archive, positionHistoryPerSecond,
                      [&](Vector3 &position) { archive.Process(position, TYPE_VECTOR3); });
    }
};
struct NoController {
    void Process(Archive &) {}
};
struct TacticalPlayerSituation {
    float forwardSpaceRating, toGoalSpaceRating, spaceRating, forwardRating;
    void Process(Archive &archive) {
        PROCESS_FIELD(forwardSpaceRating, TYPE_FLOAT);
        PROCESS_FIELD(toGoalSpaceRating, TYPE_FLOAT);
        PROCESS_FIELD(spaceRating, TYPE_FLOAT);
        PROCESS_FIELD(forwardRating, TYPE_FLOAT);
    }
};
struct PlayerState {
    PlayerBaseState<AIControllerState> base;
    int32_t manMarking;
    FormationEntry dynamicFormationEntry;
    bool hasPossession, hasBestPossession, hasUniquePossession;
    int32_t possessionDurationMilliseconds;
    uint32_t timeNeededToGetToBallMilliseconds, timeNeededToGetToBallOptimisticMilliseconds,
        timeNeededToGetToBallPreviousMilliseconds;
    bool triggerControlledBallCollision;
    TacticalPlayerSituation tacticalSituation;
    bool bufferNameCaptionShowCondition;
    Vector3 bufferPlayerColor;
    int32_t desiredTimeToBallMilliseconds, cards;
    uint64_t cardEffectiveTimeMilliseconds;
    void Process(Archive &archive) {
        PROCESS_STRUCT(base);
        PROCESS_POINTER_FIELD(manMarking, PLAYER);
        PROCESS_STRUCT(dynamicFormationEntry);
        PROCESS_FIELD(hasPossession, TYPE_BOOLEAN);
        PROCESS_FIELD(hasBestPossession, TYPE_BOOLEAN);
        PROCESS_FIELD(hasUniquePossession, TYPE_BOOLEAN);
        PROCESS_FIELD(possessionDurationMilliseconds, TYPE_INTEGER);
        PROCESS_FIELD(timeNeededToGetToBallMilliseconds, TYPE_UNSIGNED_INTEGER);
        PROCESS_FIELD(timeNeededToGetToBallOptimisticMilliseconds, TYPE_UNSIGNED_INTEGER);
        PROCESS_FIELD(timeNeededToGetToBallPreviousMilliseconds, TYPE_UNSIGNED_INTEGER);
        PROCESS_FIELD(triggerControlledBallCollision, TYPE_BOOLEAN);
        PROCESS_STRUCT(tacticalSituation);
        PROCESS_FIELD(bufferNameCaptionShowCondition, TYPE_BOOLEAN);
        PROCESS_FIELD(bufferPlayerColor, TYPE_VECTOR3);
        PROCESS_FIELD(desiredTimeToBallMilliseconds, TYPE_INTEGER);
        PROCESS_FIELD(cards, TYPE_INTEGER);
        PROCESS_FIELD(cardEffectiveTimeMilliseconds, TYPE_UNSIGNED_LONG);
    }
};

// GRF stores the tactics as a map from 14 keys to `%f` texts. The codec keeps the float `atof` of
// each text, whose `%f` gives every six-decimal text back, so a save reproduces the bytes.
constexpr int TACTIC_KEYS = 14;
// GRF: in the alphabetical order of the keys, the order in which a state stores the map.
enum TacticID {
    DEFENSE_DEPTH,
    DEFENSE_MICRO_FOCUS_STRENGTH,
    DEFENSE_MIDFIELD_FOCUS,
    DEFENSE_MIDFIELD_FOCUS_STRENGTH,
    DEFENSE_OWN_HALF,
    DEFENSE_SIDE_FOCUS_STRENGTH,
    DEFENSE_WIDTH,
    OFFENSE_DEPTH,
    OFFENSE_MICRO_FOCUS_STRENGTH,
    OFFENSE_MIDFIELD_FOCUS,
    OFFENSE_MIDFIELD_FOCUS_STRENGTH,
    OFFENSE_OWN_HALF,
    OFFENSE_SIDE_FOCUS_STRENGTH,
    OFFENSE_WIDTH
};
inline const char *TacticKey(int tactic) {
    static const char *const keys[TACTIC_KEYS] = {
        "position_defense_depth_factor",           "position_defense_microfocus_strength",
        "position_defense_midfieldfocus",          "position_defense_midfieldfocus_strength",
        "position_defense_ownhalf_factor",         "position_defense_sidefocus_strength",
        "position_defense_width_factor",           "position_offense_depth_factor",
        "position_offense_microfocus_strength",    "position_offense_midfieldfocus",
        "position_offense_midfieldfocus_strength", "position_offense_ownhalf_factor",
        "position_offense_sidefocus_strength",     "position_offense_width_factor"};
    return keys[tactic];
}
struct PropertiesState {
    uint32_t present;
    float value[TACTIC_KEYS];
    void Process(Archive &archive) {
        archive.field = "properties";
        int32_t count = 0;
        for (int tactic = 0; tactic < TACTIC_KEYS; tactic++) count += (present >> tactic) & 1;
        archive.Process(count, TYPE_INTEGER);
        if (archive.load()) {
            if (count < 0 || count > TACTIC_KEYS)
                throw std::runtime_error("properties: unexpected size");
            present = 0;
            for (int entry = 0; entry < count; entry++) {
                std::string key, text;
                archive.ProcessString(key);
                archive.ProcessString(text);
                int id = 0;
                while (id < TACTIC_KEYS && key != TacticKey(id)) id++;
                if (id == TACTIC_KEYS) throw std::runtime_error("properties: unknown key " + key);
                char formatted[32];
                snprintf(formatted, sizeof(formatted), "%f", (double)(float)atof(text.c_str()));
                if (text != formatted)
                    throw std::runtime_error("properties: value does not round-trip: " + text);
                present |= 1u << id;
                value[id] = (float)atof(text.c_str());
            }
        } else {
            for (int id = 0; id < TACTIC_KEYS; id++) {
                if (!((present >> id) & 1)) continue;
                std::string key = TacticKey(id);
                char formatted[32];
                snprintf(formatted, sizeof(formatted), "%f", (double)value[id]);
                std::string text = formatted;
                archive.ProcessString(key);
                archive.ProcessString(text);
            }
        }
    }
};
struct TacticalOpponentInformation {
    int32_t player;
    float dangerFactor;
    void Process(Archive &archive) {
        PROCESS_POINTER_FIELD(player, PLAYER);
        PROCESS_FIELD(dangerFactor, TYPE_FLOAT);
    }
};
struct TeamAIState {
    int32_t taker;
    int32_t setPieceType;
    PropertiesState baseTeamTactics, liveTeamTactics;
    float offensivenessBias;
    bool teamHasPossession, teamHasUniquePossession, opponentTeamHasPossession,
        opponentTeamHasUniquePossession, teamHasBestPossession;
    float teamPossessionAmount, fadingTeamPossessionAmount;
    int32_t timeNeededToGetToBall, opponentTimeNeededToGetToBall;
    float depth, width, offsideTrapX;
    uint64_t endApplyAttackingRunMilliseconds;
    int32_t attackingRunPlayer;
    uint64_t endApplyTeamPressureMilliseconds;
    int32_t teamPressurePlayer;
    uint64_t endApplyKeeperRushMilliseconds;
    int32_t forwardSupportPlayer;
    FixedVector<TacticalOpponentInformation, MAXIMUM_TEAM_PLAYERS> tacticalOpponentInformation;
    void Process(Archive &archive) {
        PROCESS_POINTER_FIELD(taker, PLAYER);
        PROCESS_FIELD(setPieceType, TYPE_GAME_MODE);
        PROCESS_STRUCT(baseTeamTactics);
        PROCESS_STRUCT(liveTeamTactics);
        PROCESS_FIELD(offensivenessBias, TYPE_FLOAT);
        PROCESS_FIELD(teamHasPossession, TYPE_BOOLEAN);
        PROCESS_FIELD(teamHasUniquePossession, TYPE_BOOLEAN);
        PROCESS_FIELD(opponentTeamHasPossession, TYPE_BOOLEAN);
        PROCESS_FIELD(opponentTeamHasUniquePossession, TYPE_BOOLEAN);
        PROCESS_FIELD(teamHasBestPossession, TYPE_BOOLEAN);
        PROCESS_FIELD(teamPossessionAmount, TYPE_FLOAT);
        PROCESS_FIELD(fadingTeamPossessionAmount, TYPE_FLOAT);
        PROCESS_FIELD(timeNeededToGetToBall, TYPE_INTEGER);
        PROCESS_FIELD(opponentTimeNeededToGetToBall, TYPE_INTEGER);
        PROCESS_FIELD(depth, TYPE_FLOAT);
        PROCESS_FIELD(width, TYPE_FLOAT);
        PROCESS_FIELD(offsideTrapX, TYPE_FLOAT);
        PROCESS_FIELD(endApplyAttackingRunMilliseconds, TYPE_UNSIGNED_LONG);
        PROCESS_POINTER_FIELD(attackingRunPlayer, PLAYER);
        PROCESS_FIELD(endApplyTeamPressureMilliseconds, TYPE_UNSIGNED_LONG);
        PROCESS_POINTER_FIELD(teamPressurePlayer, PLAYER);
        PROCESS_FIELD(endApplyKeeperRushMilliseconds, TYPE_UNSIGNED_LONG);
        PROCESS_POINTER_FIELD(forwardSupportPlayer, PLAYER);
        archive.field = "tacticalOpponentInformation";
        ProcessVector(archive, tacticalOpponentInformation,
                      [&](TacticalOpponentInformation &opponent) { opponent.Process(archive); });
    }
};
struct TeamState {
    bool hasPossession;
    int32_t timeNeededToGetToBallMilliseconds;
    int32_t designatedTeamPossessionPlayer;
    float teamPossessionAmount, fadingTeamPossessionAmount;
    TeamAIState teamController;
    FixedVector<HumanGamerState, MAXIMUM_TEAM_PLAYERS> humanGamers;
    FixedVector<int32_t, MAXIMUM_TEAM_PLAYERS> switchPriority;
    int32_t lastTouchPlayer, mainSelectedPlayer;
    void Process(Archive &archive) {
        PROCESS_FIELD(hasPossession, TYPE_BOOLEAN);
        PROCESS_FIELD(timeNeededToGetToBallMilliseconds, TYPE_INTEGER);
        PROCESS_POINTER_FIELD(designatedTeamPossessionPlayer, PLAYER);
        PROCESS_FIELD(teamPossessionAmount, TYPE_FLOAT);
        PROCESS_FIELD(fadingTeamPossessionAmount, TYPE_FLOAT);
        PROCESS_STRUCT(teamController);
        archive.field = "humanGamers";
        ProcessVector(archive, humanGamers,
                      [&](HumanGamerState &gamer) { gamer.Process(archive); });
        archive.field = "switchPriority";
        ProcessVector(archive, switchPriority,
                      [&](int32_t &player) { archive.Process(player, TYPE_INTEGER); });
        PROCESS_POINTER_FIELD(lastTouchPlayer, PLAYER);
        PROCESS_POINTER_FIELD(mainSelectedPlayer, PLAYER);
    }
};

struct MentalImageState {
    uint32_t timeStampMilliseconds;
    float maximumDistanceDeviation, maximumMovementDeviation;
    FixedVector<PlayerImage, 2 * MAXIMUM_TEAM_PLAYERS> players;
    FixedVector<Vector3, 320> ballPredictions;
    // The state omits whether the predictions are mirrored, which the simulation applies only
    // when it reads them. The codec never mirrors.
    bool lazyFlip = false;
    void Process(Archive &archive) {
        PROCESS_FIELD(timeStampMilliseconds, TYPE_UNSIGNED_INTEGER);
        PROCESS_FIELD(maximumDistanceDeviation, TYPE_FLOAT);
        PROCESS_FIELD(maximumMovementDeviation, TYPE_FLOAT);
        archive.field = "players";
        ProcessVector(archive, players, [&](PlayerImage &image) { image.Process(archive); });
        archive.field = "ballPredictions";
        ProcessVector(archive, ballPredictions,
                      [&](Vector3 &prediction) { archive.Process(prediction, TYPE_VECTOR3); });
    }
};

struct BallSerialized {
    Vector3 momentum;
    Quaternion rotationPerMillisecond;
    Vector3 predictions[401];
    int32_t validPredictions;
    Quaternion orientationPrediction;
    FixedVector<Vector3, 402> ballPositionHistory;
    Vector3 positionBuffer;
    Quaternion orientationBuffer;
    bool ballTouchesNet;
    // The state omits whether `predictions` and `ballPositionHistory` are mirrored, as above.
    bool lazyFlip = false;
    void Process(Archive &archive) {
        PROCESS_FIELD(momentum, TYPE_VECTOR3);
        PROCESS_FIELD(rotationPerMillisecond, TYPE_QUATERNION);
        for (int tick = 0; tick < 401; tick++) {
            ArchiveScope scope(archive, "predictions", tick);
            archive.field = archive.layout ? "" : "predictions";
            archive.Process(predictions[tick], TYPE_VECTOR3);
        }
        PROCESS_FIELD(validPredictions, TYPE_INTEGER);
        PROCESS_FIELD(orientationPrediction, TYPE_QUATERNION);
        archive.field = "ballPositionHistory";
        ProcessVector(archive, ballPositionHistory,
                      [&](Vector3 &position) { archive.Process(position, TYPE_VECTOR3); });
        PROCESS_FIELD(positionBuffer, TYPE_VECTOR3);
        PROCESS_FIELD(orientationBuffer, TYPE_QUATERNION);
        PROCESS_FIELD(ballTouchesNet, TYPE_BOOLEAN);
    }
};

// `teamID` is a GRF team ID and `setPieceTeam` a team index. The caller of the codec mirrors
// `teamID` and `restartPosition` when the teams swap sides.
struct RefereeBufferState {
    bool active;
    int32_t desiredSetPiece;
    int32_t teamID;
    int32_t setPieceTeam;
    uint64_t stopTime, prepareTime, startTime;
    Vector3 restartPosition;
    int32_t taker;
    bool endPhase;
    void Process(Archive &archive) {
        PROCESS_FIELD(active, TYPE_BOOLEAN);
        PROCESS_FIELD(desiredSetPiece, TYPE_GAME_MODE);
        PROCESS_FIELD(teamID, TYPE_INTEGER);
        PROCESS_POINTER_FIELD(setPieceTeam, TEAM);
        PROCESS_FIELD(stopTime, TYPE_UNSIGNED_LONG);
        PROCESS_FIELD(prepareTime, TYPE_UNSIGNED_LONG);
        PROCESS_FIELD(startTime, TYPE_UNSIGNED_LONG);
        PROCESS_FIELD(restartPosition, TYPE_VECTOR3);
        PROCESS_POINTER_FIELD(taker, PLAYER);
        PROCESS_FIELD(endPhase, TYPE_BOOLEAN);
    }
};
struct FoulState {
    int32_t foulPlayer, foulVictim;
    int32_t foulType;
    bool advantage;
    uint64_t foulTime;
    Vector3 foulPosition;
    bool hasBeenProcessed;
    void Process(Archive &archive) {
        PROCESS_POINTER_FIELD(foulPlayer, PLAYER);
        PROCESS_POINTER_FIELD(foulVictim, PLAYER);
        PROCESS_FIELD(foulType, TYPE_INTEGER);
        PROCESS_FIELD(advantage, TYPE_BOOLEAN);
        PROCESS_FIELD(foulTime, TYPE_UNSIGNED_LONG);
        PROCESS_FIELD(foulPosition, TYPE_VECTOR3);
        PROCESS_FIELD(hasBeenProcessed, TYPE_BOOLEAN);
    }
};
struct RefereeState {
    RefereeBufferState buffer;
    int32_t afterSetPieceRelaxTimeMilliseconds;
    FixedVector<int32_t, MAXIMUM_TEAM_PLAYERS> offsidePlayers;
    FoulState foul;
    void Process(Archive &archive) {
        PROCESS_STRUCT(buffer);
        PROCESS_FIELD(afterSetPieceRelaxTimeMilliseconds, TYPE_INTEGER);
        archive.field = "offsidePlayers";
        ProcessVector(archive, offsidePlayers,
                      [&](int32_t &player) { archive.Process(player, TYPE_INTEGER); });
        PROCESS_STRUCT(foul);
    }
};

// The members up to `controlAllPlayers` stay constant over an episode, and the rest change.
struct ScenarioState {
    Vector3 ballPosition;
    int32_t leftTeamSize, rightTeamSize, leftAgents, rightAgents;
    bool useMagnet, offsides;
    float leftTeamDifficulty, rightTeamDifficulty;
    bool deterministic, endEpisodeOnScore, endEpisodeOnPossessionChange, endEpisodeOnOutOfPlay;
    // In steps.
    int32_t gameDuration, secondHalf;
    bool controlAllPlayers;
    bool realTime;
    uint32_t gameEngineRandomSeed;
    // The teams swap sides, so the second team plays from the left.
    bool reverseTeamProcessing;
    FormationEntry leftTeam[MAXIMUM_TEAM_PLAYERS], rightTeam[MAXIMUM_TEAM_PLAYERS];
    void Process(Archive &archive) {
        PROCESS_FIELD(ballPosition, TYPE_VECTOR3);
        PROCESS_FIELD(leftTeamSize, TYPE_INTEGER);
        PROCESS_FIELD(rightTeamSize, TYPE_INTEGER);
        if (leftTeamSize < 0 || rightTeamSize < 0 || leftTeamSize > MAXIMUM_TEAM_PLAYERS ||
            rightTeamSize > MAXIMUM_TEAM_PLAYERS)
            throw std::runtime_error("the state holds a team size out of range");
        PROCESS_FIELD(leftAgents, TYPE_INTEGER);
        PROCESS_FIELD(rightAgents, TYPE_INTEGER);
        PROCESS_FIELD(useMagnet, TYPE_BOOLEAN);
        PROCESS_FIELD(offsides, TYPE_BOOLEAN);
        PROCESS_FIELD(leftTeamDifficulty, TYPE_FLOAT);
        PROCESS_FIELD(rightTeamDifficulty, TYPE_FLOAT);
        PROCESS_FIELD(deterministic, TYPE_BOOLEAN);
        PROCESS_FIELD(endEpisodeOnScore, TYPE_BOOLEAN);
        PROCESS_FIELD(endEpisodeOnPossessionChange, TYPE_BOOLEAN);
        PROCESS_FIELD(endEpisodeOnOutOfPlay, TYPE_BOOLEAN);
        PROCESS_FIELD(gameDuration, TYPE_INTEGER);
        PROCESS_FIELD(secondHalf, TYPE_INTEGER);
        PROCESS_FIELD(controlAllPlayers, TYPE_BOOLEAN);
        PROCESS_FIELD(realTime, TYPE_BOOLEAN);
        PROCESS_FIELD(gameEngineRandomSeed, TYPE_UNSIGNED_INTEGER);
        PROCESS_FIELD(reverseTeamProcessing, TYPE_BOOLEAN);
        for (int player = 0; player < archive.Count(leftTeamSize, MAXIMUM_TEAM_PLAYERS); player++) {
            ArchiveScope scope(archive, "leftTeam", player);
            leftTeam[player].Process(archive);
        }
        for (int player = 0; player < archive.Count(rightTeamSize, MAXIMUM_TEAM_PLAYERS);
             player++) {
            ArchiveScope scope(archive, "rightTeam", player);
            rightTeam[player].Process(archive);
        }
    }
};

constexpr int MAXIMUM_PLAYERS = 2 * MAXIMUM_TEAM_PLAYERS;
constexpr int MAXIMUM_CONTROLLERS = 32;
// GRF: `run_game` creates a keyboard for every player of two teams of 11.
constexpr int MATCH_KEYBOARDS = 22;

struct MatchState {
    // Team indices of the teams stored first and second, which swap when the teams swap sides.
    int32_t firstTeam, secondTeam;
    FixedVector<MentalImageState, 4> mentalImages;
    TeamState teams[2];
    // Not in the state.
    int playerCount = MAXIMUM_PLAYERS;
    PlayerState players[MAXIMUM_PLAYERS];
    int32_t goalCount[2];
    float possessionLastMinute;
    // The referee, then the two linesmen.
    PlayerBaseState<NoController> officials[3];
    // GRF: a second copy of each human gamer's keyboard, first team first, after the officials.
    KeyboardState secondKeyboardCopies[MAXIMUM_CONTROLLERS];
    // Not in the state.
    int freeControllerCount = 0;
    KeyboardState freeControllers[MAXIMUM_CONTROLLERS];
    BallSerialized ball;
    uint64_t matchTimeMilliseconds, actualTimeMilliseconds, goalScoredTimer;
    int32_t matchPhase;
    bool inPlay, inSetPiece, goalScored, ballIsInGoal;
    int32_t lastGoalTeam;
    int32_t lastGoalScorer;
    // GRF team IDs, one per `e_TouchType`.
    int32_t lastTouchTeamIDs[4];
    int32_t lastTouchTeamID;
    int32_t bestPossessionTeam;
    int32_t designatedPossessionPlayer;
    int32_t ballRetainer;
    uint32_t possessionSideHistoryMaximumTimeMilliseconds;
    // One value per tick over the last 6000 ms, so at most 600.
    FixedVector<float, 600> possessionSideHistory;
    bool autoUpdateInGameCamera;
    Quaternion cameraOrientation, cameraNodeOrientation;
    Vector3 cameraNodePosition;
    float cameraFieldOfView, cameraNearCap, cameraFarCap;
    FixedVector<Vector3, 160> cameraPosition;
    uint32_t lastBodyBallCollisionTimeMilliseconds;
    RefereeState referee;
};

struct EnvironmentStateFull {
    int32_t gameState;
    int32_t waitingForGameCount;
    uint8_t rng[2512];
    ScenarioState scenario;
    int32_t step;
    MatchState match;
};

// The state stores neither `freeControllerCount` nor the player count, which follows from the
// team sizes.
void ProcessEnvironmentState(Archive &archive, EnvironmentStateFull &environment,
                             int freeControllerCount);
// The start of a state, up to and including its scenario.
void ProcessEnvironmentScenario(Archive &archive, EnvironmentStateFull &environment);

}  // namespace state
}  // namespace cuju

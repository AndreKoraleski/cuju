// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The referee, for half time, balls out of play, fouls, offside and the start of each set piece.
#pragma once

#include "../profiling.h"

#include "collisions.h"
#include "reset.h"
#include "team_ai.h"

namespace cuju {
namespace simulation {

// `reset.h` defines these two, and through `team_ai.h` it includes this file before it does.
CUJU_FUNCTION inline bool LeftTeamOwnsBall(const World &world);
CUJU_FUNCTION inline bool RefereePrepareSetPiece(World &world, const SimulationContext &context,
                                                 int setPiece);

enum e_MatchPhase { e_MatchPhase_PreMatch, e_MatchPhase_FirstHalf, e_MatchPhase_SecondHalf };
// GRF: without referee animations, the clock skips the wait of a stoppage at once.
constexpr bool REFEREE_ANIMATIONS = false;

// `time` is in milliseconds.
CUJU_FUNCTION_NO_INLINE inline void BumpActualTime(World &world, const SimulationContext &context,
                                                   uint64_t time) {
    state::MatchState &match = world.serialized.match;
    if (match.inPlay)
        match.matchTimeMilliseconds =
            (uint64_t)((float)match.matchTimeMilliseconds +
                       time * (1.0f / context.staticData->matchDurationFactor));
    match.actualTimeMilliseconds += time;
    if (match.goalScored)
        match.goalScoredTimer += time;
    else
        match.goalScoredTimer = 0;
    if (match.inPlay && !match.inSetPiece) {
        // The state stores the possession relative to the team processed first.
        const float sign = world.FirstTeamID() == 1 ? -1.0f : 1.0f;
        float possession = match.possessionLastMinute * sign;
        int teamID = world.playerTeamID[match.designatedPossessionPlayer] == 0 ? 0 : 1;
        float delta = 0.001f * time;
        // Exact: the operand order of `std::max(moved, limit)` and `std::min(moved, limit)`, which
        // return `moved` on a tie.
        if (teamID == 0) {
            float moved = possession - delta, limit = -60.0f;
            possession = (moved < limit) ? limit : moved;
        } else {
            float moved = possession + delta, limit = 60.0f;
            possession = (limit < moved) ? limit : moved;
        }
        match.possessionLastMinute = possession * sign;
    }
}

CUJU_FUNCTION inline void MatchSetMatchPhase(World &world, int phase) {
    world.serialized.match.matchPhase = phase;
    for (int player = 0; player < world.serialized.match.playerCount; player++) {
        state::PlayerBaseState<state::AIControllerState> &playerBase =
            world.serialized.match.players[player].base;
        if (!playerBase.isActive) continue;
        playerBase.fatigueFactorInverse += 1.0f;
        playerBase.fatigueFactorInverse = clamp(playerBase.fatigueFactorInverse, 0.01f, 1.0f);
    }
}

// GRF: before any touch, the team processed first counts as the last to touch.
CUJU_FUNCTION inline int MatchLastTouchTeam(const World &world) {
    return world.serialized.match.lastTouchTeamID != -1 ? world.serialized.match.lastTouchTeamID
                                                        : world.FirstTeamID();
}
CUJU_FUNCTION inline int StaticSide(int teamID) { return teamID == 0 ? -1 : 1; }

CUJU_FUNCTION inline void ResetFoul(state::FoulState &foul) {
    foul.foulPlayer = -1;
    foul.foulType = 0;
    foul.advantage = false;
    foul.foulTime = 0;
    foul.hasBeenProcessed = true;
}

// `setPiece` is an `e_GameMode`. The buffer times are in `actualTimeMilliseconds`. The caller sets
// the restart position, the team and the active flag of the buffer.
CUJU_FUNCTION_NO_INLINE inline void RefereeStop(World &world, const SimulationContext &context,
                                                int setPiece,
                                                uint64_t extraPrepareMilliseconds = 0) {
    state::MatchState &match = world.serialized.match;
    state::RefereeBufferState &buffer = match.referee.buffer;
    match.inPlay = false;
    buffer.desiredSetPiece = setPiece;
    buffer.stopTime = match.actualTimeMilliseconds;
    buffer.prepareTime = match.actualTimeMilliseconds + 2000;
    if (!REFEREE_ANIMATIONS) BumpActualTime(world, context, 1900);
    if (extraPrepareMilliseconds) {
        buffer.prepareTime += extraPrepareMilliseconds;
        if (!REFEREE_ANIMATIONS) BumpActualTime(world, context, extraPrepareMilliseconds);
    }
    buffer.startTime = buffer.prepareTime + 2000;
}

// Returns `true` when the foul stops play.
CUJU_FUNCTION_NO_INLINE inline bool RefereeCheckFoul(World &world,
                                                     const SimulationContext &context) {
    state::MatchState &match = world.serialized.match;
    state::RefereeState &referee = match.referee;
    state::FoulState &foul = referee.foul;
    bool penalty = false;
    if (foul.foulType != 0) {
        // Exact: the penalty-area bounds, 16.5 m deep and 20.15 m to each side, compare in double.
        if ((double)fabsf(foul.foulPosition.coordinates[1]) < 20.15 - (double)LINE_HALF_WIDTH &&
            (double)(foul.foulPosition.coordinates[0] *
                     -StaticSide(world.playerTeamID[foul.foulVictim])) >
                (double)PITCH_HALF_LENGTH - 16.5 + (double)LINE_HALF_WIDTH)
            penalty = true;
    }
    if (foul.advantage) {
        if (penalty) {
            foul.advantage = false;
        } else if (match.actualTimeMilliseconds - 600 > foul.foulTime) {
            if (match.actualTimeMilliseconds - 3000 > foul.foulTime) {
                foul.foulPlayer = -1;
                foul.foulType = 0;
            } else if (TeamByID(world, world.playerTeamID[foul.foulVictim])
                           .fadingTeamPossessionAmount < 1.0) {
                foul.advantage = false;
            }
        }
    }
    if (foul.foulType != 0 && foul.advantage == false && !foul.hasBeenProcessed) {
        state::RefereeBufferState &buffer = referee.buffer;
        const uint64_t extraTime = foul.foulType >= 2 ? 10000 : 0;
        if (!penalty) {
            RefereeStop(world, context, e_GameMode_FreeKick, extraTime);
            buffer.restartPosition = foul.foulPosition;
        } else {
            RefereeStop(world, context, e_GameMode_Penalty, extraTime);
            buffer.restartPosition =
                Vector3((float)(((double)PITCH_HALF_LENGTH - 11.0) *
                                StaticSide(world.playerTeamID[foul.foulPlayer])),
                        0, 0);
        }
        buffer.teamID = world.playerTeamID[foul.foulVictim];
        buffer.active = true;
        state::PlayerState &offender = match.players[foul.foulPlayer];
        // A foul type of 2 is a yellow card and 3 a red card.
        if (foul.foulType == 2) {
            offender.cards++;
            offender.cardEffectiveTimeMilliseconds = match.actualTimeMilliseconds + 6000;
        }
        if (foul.foulType == 3) {
            offender.cards += 3;
            offender.cardEffectiveTimeMilliseconds = match.actualTimeMilliseconds + 6000;
        }
        foul.hasBeenProcessed = true;
        return true;
    }
    return false;
}

CUJU_FUNCTION inline void Randomize(World &world, uint32_t seed) {
    RNGView{world.serialized.rng}.Seed(seed);
}

CUJU_FUNCTION_NO_INLINE inline bool RefereeProcess(World &world, const SimulationContext &context) {
    CUJU_PROFILE_SCOPE(REFEREE_PROCESS);
    state::MatchState &match = world.serialized.match;
    state::RefereeState &referee = match.referee;
    state::RefereeBufferState &buffer = referee.buffer;
    state::FoulState &foul = referee.foul;
    const state::ScenarioState &scenario = world.serialized.scenario;
    if (match.inPlay && !match.inSetPiece) {
        Vector3 ballPosition = BallPredict(match.ball, 0);
        // GRF: the half time is an `int` product that wraps. The comparison with the unsigned match
        // time converts it, so after a wrap to a negative product half time never comes.
        const int64_t halfTime = (int32_t)((uint32_t)1800 * (uint32_t)scenario.secondHalf);
        if (match.matchTimeMilliseconds >= (uint64_t)halfTime &&
            match.matchPhase == e_MatchPhase_FirstHalf) {
            match.inPlay = false;
            buffer.desiredSetPiece = e_GameMode_KickOff;
            buffer.stopTime = match.actualTimeMilliseconds;
            buffer.prepareTime = buffer.stopTime + 100;
            buffer.startTime = buffer.prepareTime + 200;
            buffer.restartPosition = scenario.ballPosition;
            buffer.active = true;
            buffer.endPhase = true;
            buffer.teamID = LeftTeamOwnsBall(world) ? 1 : 0;
            buffer.setPieceTeam = TeamSlot(world, buffer.teamID);
            buffer.taker = -1;
            ResetFoul(foul);
            MatchSetMatchPhase(world, e_MatchPhase_SecondHalf);
        }
        if (scenario.reverseTeamProcessing) ballPosition.Mirror();

        if ((double)fabsf(ballPosition.coordinates[0]) >
                (double)(PITCH_HALF_LENGTH + LINE_HALF_WIDTH) + 0.11 ||
            match.goalScored) {
            foul.advantage = false;
            bool isFoul = false;
            if (!match.goalScored)
                isFoul = RefereeCheckFoul(world, context);
            else
                foul.foulType = 0;
            if (!isFoul) {
                match.inPlay = false;
                const int lastTouchTeam = MatchLastTouchTeam(world);
                const int lastSide = StaticSide(lastTouchTeam);
                if (match.goalScored) {
                    buffer.desiredSetPiece = e_GameMode_KickOff;
                    buffer.stopTime = match.actualTimeMilliseconds;
                    buffer.prepareTime = match.actualTimeMilliseconds + 500;
                    if (!REFEREE_ANIMATIONS) BumpActualTime(world, context, 400);
                    buffer.startTime = buffer.prepareTime + 500;
                    buffer.restartPosition = Vector3(0, 0, 0);
                    buffer.teamID = world.FirstTeamID();
                    // `lastGoalTeam` is a slot, and the other slot kicks off.
                    buffer.setPieceTeam = 1 - match.lastGoalTeam;
                } else if ((ballPosition.coordinates[0] > 0 && lastSide > 0) ||
                           (ballPosition.coordinates[0] < 0 && lastSide < 0)) {
                    RefereeStop(world, context, e_GameMode_Corner);
                    float y = ballPosition.coordinates[1];
                    if (y > 0)
                        y = PITCH_HALF_WIDTH;
                    else
                        y = -PITCH_HALF_WIDTH;
                    buffer.restartPosition = Vector3(PITCH_HALF_LENGTH * lastSide, y, 0);
                    buffer.teamID = 1 - lastTouchTeam;
                } else {
                    RefereeStop(world, context, e_GameMode_GoalKick);
                    buffer.restartPosition =
                        Vector3((float)((double)PITCH_HALF_LENGTH * 0.92 * -lastSide), 0, 0);
                    buffer.teamID = 1 - lastTouchTeam;
                }
                buffer.active = true;
            }
        }

        if (referee.afterSetPieceRelaxTimeMilliseconds == 0) {
            if ((double)fabsf(ballPosition.coordinates[1]) >
                (double)(PITCH_HALF_WIDTH + LINE_HALF_WIDTH) + 0.11) {
                foul.advantage = false;
                if (!RefereeCheckFoul(world, context)) {
                    match.inPlay = false;
                    const int lastTouchTeam = MatchLastTouchTeam(world);
                    buffer.teamID = 1 - lastTouchTeam;
                    RefereeStop(world, context, e_GameMode_ThrowIn);
                    buffer.restartPosition.coordinates[0] =
                        clamp(ballPosition.coordinates[0], -PITCH_HALF_LENGTH + 0.6f,
                              PITCH_HALF_LENGTH - 0.6f);
                    if (ballPosition.coordinates[1] > 0)
                        buffer.restartPosition.coordinates[1] = PITCH_HALF_WIDTH;
                    if (ballPosition.coordinates[1] <= 0)
                        buffer.restartPosition.coordinates[1] = -PITCH_HALF_WIDTH;
                    buffer.restartPosition.coordinates[2] = 0;
                    buffer.active = true;
                }
            }
        }
        RefereeCheckFoul(world, context);
    } else if (!match.inPlay && !match.inSetPiece && buffer.active == true) {
        if (buffer.prepareTime == match.actualTimeMilliseconds) {
            if (buffer.endPhase == true) {
                if (match.matchPhase == e_MatchPhase_PreMatch)
                    MatchSetMatchPhase(world, e_MatchPhase_FirstHalf);
                buffer.endPhase = false;
            }
            Randomize(world, scenario.gameEngineRandomSeed);
            if (!RefereePrepareSetPiece(world, context, buffer.desiredSetPiece)) return false;
        }
        if (buffer.startTime == match.actualTimeMilliseconds) {
            match.inPlay = true;
            match.inSetPiece = true;
        }
    }

    if (match.inSetPiece) {
        bool taken = buffer.desiredSetPiece == e_GameMode_KickOff;
        if (!taken) {
            if (buffer.taker < 0) return Unsupported("the set piece has no taker");
            const state::AnimationState &takerAnimation =
                Humanoid(world, buffer.taker).currentAnimation;
            taken = takerAnimation.touchFrame != -1 &&
                    !(takerAnimation.frameNumber < takerAnimation.touchFrame);
        }
        if (taken) {
            buffer.active = false;
            match.inSetPiece = false;
            TeamByID(world, 0).teamController.setPieceType = e_GameMode_Normal;
            TeamByID(world, 1).teamController.setPieceType = e_GameMode_Normal;
            referee.afterSetPieceRelaxTimeMilliseconds = 400;
            foul.foulPlayer = -1;
            foul.foulType = 0;
            if (match.matchPhase == e_MatchPhase_PreMatch)
                MatchSetMatchPhase(world, e_MatchPhase_FirstHalf);
        }
    }
    if (referee.afterSetPieceRelaxTimeMilliseconds > 0)
        referee.afterSetPieceRelaxTimeMilliseconds -= 10;
    return true;
}

CUJU_FUNCTION_NO_INLINE inline bool RefereeBallTouched(World &world,
                                                       const SimulationContext &context) {
    state::MatchState &match = world.serialized.match;
    state::RefereeState &referee = match.referee;
    state::RefereeBufferState &buffer = referee.buffer;
    if (!world.serialized.scenario.offsides) return true;
    const int lastTouchTeamID = match.lastTouchTeamID;
    if (lastTouchTeamID == -1) return true;
    if (match.inPlay && !match.inSetPiece && buffer.active == false &&
        TeamActivePlayers(world, 1 - lastTouchTeamID).size() > 1) {
        const int ballOwner = MatchLastTouchPlayer(world);
        for (int offsideIndex = 0; offsideIndex < referee.offsidePlayers.count; offsideIndex++) {
            if (referee.offsidePlayers[offsideIndex] != ballOwner) continue;
            referee.foul.advantage = false;
            if (!RefereeCheckFoul(world, context)) {
                RefereeStop(world, context, e_GameMode_FreeKick);
                buffer.restartPosition = PitchPosition(world, ballOwner);
                buffer.teamID = 1 - lastTouchTeamID;
                buffer.active = true;
            }
        }
    }
    referee.offsidePlayers.count = 0;
    if (match.inPlay && (buffer.active == false ||
                         (buffer.active == true && buffer.desiredSetPiece != e_GameMode_ThrowIn &&
                          buffer.desiredSetPiece != e_GameMode_Corner))) {
        // GRF: the offside line comes from a fresh mental image, so the defenders' position and
        // movement and the ball enter as they are this tick.
        ImageList images;
        for (int player : TeamActivePlayers(world, 1 - lastTouchTeamID)) {
            const state::SpatialState &spatialState = Humanoid(world, player).spatialState;
            images.push_back(PlayerImagePosition{spatialState.position, spatialState.movement,
                                                 match.players[player].dynamicFormationEntry.role});
        }
        const int side = TeamDynamicSide(world, lastTouchTeamID);
        float offsideLine = OffsideLineFromImages(images, BallPredict(match.ball, 0),
                                                  TeamDynamicSide(world, 1 - lastTouchTeamID), 0);
        const int teamLastTouchPlayer = TeamByID(world, lastTouchTeamID).lastTouchPlayer;
        for (int player : TeamActivePlayers(world, lastTouchTeamID)) {
            if (player == teamLastTouchPlayer) continue;
            if (Humanoid(world, player).spatialState.position.coordinates[0] * side <
                offsideLine * side) {
                referee.offsidePlayers.push_back(player);
            }
        }
    }
    return true;
}

}  // namespace simulation
}  // namespace cuju

// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The world of one environment and the mirroring between its coordinate frames.
// Exact: a mirror turns the pitch by negating x and y only, so a second mirror restores the bits.
#pragma once

#include "../warp.h"

#include "../profiling.h"

#include "../state/environment_state.h"

namespace cuju {
namespace simulation {

using state::EnvironmentStateFull;
using state::HumanoidState;
using state::MatchState;
using state::MentalImageState;
using state::PlayerImage;
using state::Radian;

// In the rest frame between ticks, each team is in its own frame, side -1, no mirror, the ball in
// team 0's. `MirrorMatch` moves to the call frame of the team being processed, and a state stores
// the canonical frame of `CanonicalToRest`. These flags are not in the state.
struct FrameFlags {
    bool teamMirrored[2] = {false, false};
    int teamSide[2] = {-1, -1};
    bool ballMirrored = false;
    bool humanoidMirrored[state::MAXIMUM_PLAYERS] = {};
    bool mentalBallMirrored[4] = {};
};

// A state together with the data that a state lacks.
struct World {
    EnvironmentStateFull serialized;
    FrameFlags flags;
    int playerTeamID[state::MAXIMUM_PLAYERS];
    // 0 selects the player's own formation entry, any other value 1 plus the player whose entry it
    // uses. GRF: a goalkeeper sent off hands its entry to the first active player of its team.
    int8_t formationOverride[state::MAXIMUM_PLAYERS];
    CUJU_FUNCTION int FirstTeamID() const {
        return serialized.match.firstTeam ^ (serialized.scenario.reverseTeamProcessing ? 1 : 0);
    }
};

CUJU_FUNCTION inline void MirrorVector(Vector3 &vector) { vector.Mirror(); }
CUJU_FUNCTION inline void MirrorRadian(Radian &radian) { radian.rotated = !radian.rotated; }

CUJU_FUNCTION_NO_INLINE inline void MirrorHumanoid(HumanoidState &humanoid, bool &mirrored) {
    CUJU_PROFILE_SCOPE(MIRROR_HUMANOID);
    state::NodeState &root = humanoid.nodes[0];
    root.position = root.position * Vector3(-1, -1, 1);
    Quaternion rotation = root.rotation;
    if (!mirrored) {
        root.rotation = Quaternion(rotation.elements[1], rotation.elements[0], rotation.elements[3],
                                   -rotation.elements[2]);
    } else {
        root.rotation = Quaternion(rotation.elements[1], rotation.elements[0],
                                   -rotation.elements[3], rotation.elements[2]);
    }
    mirrored = !mirrored;
    MirrorVector(humanoid.startPosition);
    MirrorRadian(humanoid.startAngle);
    MirrorVector(humanoid.nextStartPosition);
    MirrorRadian(humanoid.nextStartAngle);
    // A spatial state mirrors its positions and movements and keeps its angles.
    state::SpatialState &spatialState = humanoid.spatialState;
    MirrorVector(spatialState.position);
    MirrorVector(spatialState.actualMovement);
    MirrorVector(spatialState.physicsMovement);
    MirrorVector(spatialState.animationMovement);
    MirrorVector(spatialState.movement);
    MirrorVector(spatialState.actionSmuggleMovement);
    MirrorVector(spatialState.movementSmuggleMovement);
    MirrorVector(spatialState.positionOffsetMovement);
    MirrorVector(humanoid.previousPosition2D);
    MirrorVector(humanoid.tripDirection);
    MirrorVector(humanoid.decayingPositionOffset);
    // The scratch values of the animation selection are not state and stay as they are.
}

CUJU_FUNCTION inline void MirrorTeam(World &world, int teamID) {
    world.flags.teamSide[teamID] *= -1;
    world.flags.teamMirrored[teamID] = !world.flags.teamMirrored[teamID];
    WarpFor(world.serialized.match.playerCount, [&](int player) {
        if (world.playerTeamID[player] == teamID)
            MirrorHumanoid(world.serialized.match.players[player].base.humanoid,
                           world.flags.humanoidMirrored[player]);
    });
}

// The prediction and history arrays mirror lazily. `lazyFlip` says they hold negated x and y, a
// reader applies it per value, and a user of the whole arrays calls `BallMaterialize` first.
CUJU_FUNCTION inline void MirrorBall(state::BallSerialized &ball) {
    MirrorVector(ball.momentum);
    ball.lazyFlip = !ball.lazyFlip;
    MirrorVector(ball.positionBuffer);
}
CUJU_FUNCTION inline void BallMaterialize(state::BallSerialized &ball) {
    if (!ball.lazyFlip) return;
    constexpr int predictionCount = sizeof(ball.predictions) / sizeof(ball.predictions[0]);
    WarpFor(predictionCount + ball.ballPositionHistory.count, [&](int index) {
        MirrorVector(index < predictionCount ? ball.predictions[index]
                                             : ball.ballPositionHistory[index - predictionCount]);
    });
    ball.lazyFlip = false;
}
CUJU_FUNCTION inline void MentalMaterialize(MentalImageState &image) {
    if (!image.lazyFlip) return;
    WarpFor(image.ballPredictions.count,
            [&](int index) { MirrorVector(image.ballPredictions[index]); });
    image.lazyFlip = false;
}

CUJU_FUNCTION inline void MirrorPlayerImage(PlayerImage &image) {
    MirrorVector(image.position);
    MirrorVector(image.directionVector);
    MirrorVector(image.movement);
}

CUJU_FUNCTION inline void MirrorMentalImage(World &world, int imageIndex, bool team0, bool team1,
                                            bool ball) {
    MentalImageState &image = world.serialized.match.mentalImages[imageIndex];
    WarpFor(image.players.count, [&](int index) {
        int teamID = world.playerTeamID[image.players[index].player];
        if (teamID == 0 ? team0 : team1) MirrorPlayerImage(image.players[index]);
    });
    if (ball) {
        image.lazyFlip = !image.lazyFlip;
        world.flags.mentalBallMirrored[imageIndex] = !world.flags.mentalBallMirrored[imageIndex];
    }
}

CUJU_FUNCTION inline void MirrorMatch(World &world, bool team0, bool team1, bool ball) {
    CUJU_PROFILE_SCOPE(MIRROR_MATCH);
    if (team0) MirrorTeam(world, 0);
    if (team1) MirrorTeam(world, 1);
    if (ball) {
        world.flags.ballMirrored = !world.flags.ballMirrored;
        MirrorBall(world.serialized.match.ball);
    }
    for (int image = 0; image < world.serialized.match.mentalImages.count; image++)
        MirrorMentalImage(world, image, team0, team1, ball);
}

// A state stores the ball, the last-touch team IDs and the referee's team and positions relative
// to processing order, and the goal counts and possession times in state order. The flags stay.
// The conversion is an involution, so `RestToCanonical` is the same function.
CUJU_FUNCTION_NO_INLINE inline void CanonicalToRest(World &world) {
    MatchState &match = world.serialized.match;
    if (world.FirstTeamID() == 1) {
        for (int32_t &teamID : match.lastTouchTeamIDs)
            if (teamID != -1) teamID = 1 - teamID;
        if (match.lastTouchTeamID != -1) match.lastTouchTeamID = 1 - match.lastTouchTeamID;
    }
    if (!world.serialized.scenario.reverseTeamProcessing) return;
    MirrorBall(match.ball);
    for (int image = 0; image < match.mentalImages.count; image++)
        match.mentalImages[image].lazyFlip = !match.mentalImages[image].lazyFlip;
    match.referee.buffer.teamID = 1 - match.referee.buffer.teamID;
    MirrorVector(match.referee.buffer.restartPosition);
    MirrorVector(match.referee.foul.foulPosition);
}
CUJU_FUNCTION inline void RestToCanonical(World &world) { CanonicalToRest(world); }

}  // namespace simulation
}  // namespace cuju

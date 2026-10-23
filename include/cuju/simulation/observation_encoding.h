// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC
// SPDX-License-Identifier: Apache-2.0

// The view of each agent on an observation, its representations and its checkpoint reward.
// Exact: as GRF's Python, this computes in double on float values, with `pow` from `libm`.
#pragma once

#include "../libm/pow.h"
#include "environment.h"

namespace cuju {
namespace simulation {

// The agent's own team is `team[0]`. `active` and `designated` index it, `active` -1 for none.
struct AgentView {
    Observation observation;
    int active;
    int designated;
    int stickyActionCount;
    uint8_t stickyActions[MAXIMUM_STICKY_ACTIONS];
};

CUJU_FUNCTION inline void StickyActions(World &world, int actionSet, bool leftTeam, int agent,
                                        uint8_t *out) {
    const int count = StickyActionCount(actionSet);
    for (int index = 0; index < count; index++)
        out[index] = StickyActionState(world, StickyAction(index), leftTeam, agent);
}

// `source` from the side of the right team. Positions stay, and `NegateGeometry` turns them.
CUJU_FUNCTION inline void SwapTeams(const Observation &source, Observation &swapped) {
    swapped = source;
    swapped.ballOwnedTeam = source.ballOwnedTeam > -1 ? 1 - source.ballOwnedTeam : -1;
    swapped.score[0] = source.score[1];
    swapped.score[1] = source.score[0];
    for (int team = 0; team < 2; team++) {
        swapped.team[team] = source.team[1 - team];
        for (int controller = 0; controller < MAXIMUM_TEAM_CONTROLLERS; controller++)
            swapped.controlledPlayer[team][controller] =
                source.controlledPlayer[1 - team][controller];
    }
}

// `negate` combines `AxisNegation` bits. Both together turn the pitch by 180 degrees.
CUJU_FUNCTION inline void NegateGeometry(Observation &observation, int negate) {
    for (int coordinate = 0; coordinate < 2; coordinate++) {
        if (!(negate & (1 << coordinate))) continue;
        observation.ball[coordinate] = -observation.ball[coordinate];
        observation.ballDirection[coordinate] = -observation.ballDirection[coordinate];
    }
    for (int team = 0; team < 2; team++)
        for (int player = 0; player < observation.team[team].count; player++)
            for (int coordinate = 0; coordinate < 2; coordinate++) {
                if (!(negate & (1 << coordinate))) continue;
                observation.team[team].players[player].position[coordinate] =
                    -observation.team[team].players[player].position[coordinate];
                observation.team[team].players[player].direction[coordinate] =
                    -observation.team[team].players[player].direction[coordinate];
            }
}

// The agent sees the axes of `negate` with their signs changed, and its sticky actions, given
// unturned, reorder to match. GRF negates both axes for the right team and none for the left.
CUJU_FUNCTION inline void MakeAgentView(const Observation &raw, bool right, int negate, int agent,
                                        int actionSet, const uint8_t *stickyActions,
                                        AgentView &view) {
    if (right)
        SwapTeams(raw, view.observation);
    else
        view.observation = raw;
    if (negate) NegateGeometry(view.observation, negate);
    view.designated = view.observation.team[0].designatedPlayer;
    view.active = view.observation.controlledPlayer[0][agent];
    view.stickyActionCount = StickyActionCount(actionSet);
    // Entry `index` of the view takes the state of the sticky action that points its way.
    for (int index = 0; index < view.stickyActionCount; index++) {
        int source = index;
        if (negate) {
            const int flippedAction = NegateActionAxes(
                StickyAction(index), (negate & NEGATE_X) != 0, (negate & NEGATE_Y) != 0);
            for (int other = 0; other < view.stickyActionCount; other++)
                if (StickyAction(other) == flippedAction) source = other;
        }
        view.stickyActions[index] = stickyActions[source];
    }
}

// 115 values, the positions and directions of the left team, then of the right team, 22 values
// each with `fixedPositions` as in `simple115v2`, then the ball, its direction, and one-hots of
// the owning team in 3, the active player in 11 and the game mode in 7. Missing values are -1.
CUJU_FUNCTION inline void EncodeSimple115(const AgentView &view, bool fixedPositions, float *out) {
    int count = 0;
    auto teamValues = [&](int team, bool directions, float *destination) {
        int written = 0;
        for (int player = 0; player < view.observation.team[team].count; player++)
            for (int coordinate = 0; coordinate < 2; coordinate++)
                destination[written++] =
                    directions ? view.observation.team[team].players[player].direction[coordinate]
                               : view.observation.team[team].players[player].position[coordinate];
        return written;
    };
    for (int part = 0; part < 4; part++) {  // left, left directions, right, right directions
        count += teamValues(part / 2, part % 2 == 1, out + count);
        if (fixedPositions)
            while (count < (part + 1) * 22) out[count++] = -1;
    }
    while (count < 88) out[count++] = -1;
    for (int index = 0; index < 3; index++) out[count++] = view.observation.ball[index];
    for (int index = 0; index < 3; index++) out[count++] = view.observation.ballDirection[index];
    for (int index = 0; index < 3; index++)
        out[count++] = (view.observation.ballOwnedTeam + 1 == index) ? 1.0f : 0.0f;
    for (int index = 0; index < 11; index++)
        out[count++] = (view.active != -1 && view.active == index) ? 1.0f : 0.0f;
    for (int index = 0; index < 7; index++)
        out[count++] = (view.observation.gameMode == index) ? 1.0f : 0.0f;
}

// GRF's default SMM `channel_dimensions`.
constexpr int SMM_WIDTH = 96, SMM_HEIGHT = 72, SMM_LAYERS = 4;
// One mark per player, plus the ball and the active player.
constexpr int SMM_MARKS = 2 * state::MAXIMUM_TEAM_PLAYERS + 2;
// Layer 0 is the left team, 1 the right team, 2 the ball and 3 the active player.
// GRF: the cell index truncates toward zero, then clamps into the minimap.
CUJU_FUNCTION inline int SmmIndex(int layer, double positionX, double positionY, int width,
                                  int height) {
    const double minimumX = -1.0, maximumX = 1.0, minimumY = -1.0 / 2.25, maximumY = 1.0 / 2.25;
    int column = (int)((positionX - minimumX) / (maximumX - minimumX) * width);
    int row = (int)((positionY - minimumY) / (maximumY - minimumY) * height);
    column = column < 0 ? 0 : (column > width - 1 ? width - 1 : column);
    row = row < 0 ? 0 : (row > height - 1 ? height - 1 : row);
    return (row * width + column) * SMM_LAYERS + layer;
}
// The SMM holds 255 at the marked bytes and 0 everywhere else.
CUJU_FUNCTION inline int SmmMarks(const AgentView &view, int *marks, int width, int height) {
    int count = 0;
    for (int team = 0; team < 2; team++)
        for (int player = 0; player < view.observation.team[team].count; player++)
            marks[count++] =
                SmmIndex(team, view.observation.team[team].players[player].position[0],
                         view.observation.team[team].players[player].position[1], width, height);
    marks[count++] = SmmIndex(2, view.observation.ball[0], view.observation.ball[1], width, height);
    if (view.active != -1)
        marks[count++] =
            SmmIndex(3, view.observation.team[0].players[view.active].position[0],
                     view.observation.team[0].players[view.active].position[1], width, height);
    return count;
}

// Exact: numpy's type promotion. Before numpy 2, as `numpyLegacy`, the sum runs in double and
// rounds to float. From numpy 2 on, `increment` rounds to float and the sum runs in float.
CUJU_FUNCTION inline float AddFloat(float reward, double increment, bool numpyLegacy) {
    return numpyLegacy ? (float)((double)reward + increment) : reward + (float)increment;
}
// GRF's `checkpoints` reward on top of the scoring `reward`. `collected` counts the checkpoints
// of this episode, and a goal pays the rest.
CUJU_FUNCTION inline float CheckpointReward(const AgentView &view, float reward, int &collected,
                                            bool numpyLegacy) {
    const int checkpointCount = 10;
    const double checkpointReward = 0.1;
    if (reward == 1) {
        reward = AddFloat(reward, checkpointReward * (checkpointCount - collected), numpyLegacy);
        collected = checkpointCount;
        return reward;
    }
    if (view.observation.ballOwnedTeam != 0 || view.observation.ballOwnedPlayer != view.active)
        return reward;
    const float ballX = view.observation.ball[0];
    const float ballY = view.observation.ball[1];
    const double distance =
        libm::pow(libm::pow((double)ballX - 1, 2.0) + libm::pow((double)ballY, 2.0), 0.5);
    while (collected < checkpointCount) {
        const double threshold = 0.99 - 0.8 / (checkpointCount - 1) * collected;
        if (distance > threshold) break;
        reward = AddFloat(reward, checkpointReward, numpyLegacy);
        collected++;
    }
    return reward;
}

}  // namespace simulation
}  // namespace cuju

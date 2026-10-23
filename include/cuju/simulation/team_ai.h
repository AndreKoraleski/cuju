// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// Team AI for tactics, offside trap, danger ranking, attacking runs, dynamic roles and man marking.
#pragma once

#include "../profiling.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../real_to_string.h"
#include "ai_controller.h"

namespace cuju {
namespace simulation {

// GRF: a property passes through a string with six decimals, so a read returns the round trip.
CUJU_FUNCTION inline void PropertiesSet(state::PropertiesState &properties, int id, float value) {
    properties.present |= 1u << id;
    properties.value[id] = RoundTripSixDecimals(value);
}

// The state holds the goals and the possession of the last 60 s in processing order. These
// return them by team ID.
CUJU_FUNCTION inline int MatchGoalCount(const World &world, int teamID) {
    return world.serialized.match.goalCount[teamID == world.FirstTeamID() ? 0 : 1];
}
CUJU_FUNCTION inline float MatchPossessionLastMinute(const World &world) {
    return world.FirstTeamID() == 1 ? -world.serialized.match.possessionLastMinute
                                    : world.serialized.match.possessionLastMinute;
}

// GRF: every team uses the same user tactics, kept in the alphabetical order of their names.
struct UserTactic {
    // The base tactic, or -1 for a dribble tactic, which the update skips.
    int id;
    float value;
    // 0 when the tactic has no modifier.
    float modifierMultiplier;
};
constexpr int USER_TACTICS = 12;
CUJU_FUNCTION_NO_INLINE inline const UserTactic &UserTacticAt(int index) {
    using namespace state;
    static const UserTactic tactics[USER_TACTICS] = {
        {-1, 0.72f, 0.0f},  // dribble_centermagnet
        {-1, 0.5f, 0.0f},   // dribble_offensiveness
        {DEFENSE_DEPTH, 0.3f, 0.1f},
        {DEFENSE_MICRO_FOCUS_STRENGTH, 0.96f, 0.15f},
        {DEFENSE_MIDFIELD_FOCUS, 0.96f, 0.3f},
        {DEFENSE_SIDE_FOCUS_STRENGTH, 0.16f, 0.1f},
        {DEFENSE_WIDTH, 0.7f, 0.1f},
        {OFFENSE_DEPTH, 0.34f, 0.1f},
        {OFFENSE_MICRO_FOCUS_STRENGTH, 0.92f, 0.15f},
        {OFFENSE_MIDFIELD_FOCUS, 0.88f, 0.3f},
        {OFFENSE_SIDE_FOCUS_STRENGTH, 0.88f, 0.1f},
        {OFFENSE_WIDTH, 0.74f, 0.1f},
    };
    return tactics[index];
}

CUJU_FUNCTION_NO_INLINE inline bool TeamUpdateTactics(World &world, int teamID) {
    state::TeamAIState &teamAI = TeamByID(world, teamID).teamController;
    const state::MatchState &match = world.serialized.match;
    int goals = MatchGoalCount(world, teamID);
    int opponentGoals = MatchGoalCount(world, 1 - teamID);
    teamAI.liveTeamTactics = teamAI.baseTeamTactics;
    float goalFactor = clamp((float)(0.5 + (double)((opponentGoals - goals) * 0.25f)), 0.0f, 1.0f);
    float timeFactor =
        0.5f + 0.5f * clamp((float)match.matchTimeMilliseconds / 6300000.0f, 0.0f, 1.0f);
    float offenseBias = clamp(0.5f + (goalFactor - 0.5f) * (timeFactor * 1.0f), 0.0f, 1.0f);
    float possessionFactor = MatchPossessionLastMinute(world) / 60.0f;
    float recentPossessionBias;
    if (teamID == 0)
        recentPossessionBias = (float)(0.5 - (double)(possessionFactor * 0.5f));
    else
        recentPossessionBias = (float)(0.5 + (double)(possessionFactor * 0.5f));
    teamAI.offensivenessBias = offenseBias * 0.5f + recentPossessionBias * 0.5f;
    for (int index = 0; index < USER_TACTICS; index++) {
        const UserTactic &tactic = UserTacticAt(index);
        // GRF: a stored tactic passes through a string with six decimals.
        float multiplier = tactic.modifierMultiplier == 0.0f
                               ? 0.0f
                               : RoundTripSixDecimals(tactic.modifierMultiplier);
        float userOffset = RoundTripSixDecimals(tactic.value);
        float offset = userOffset;
        offset = (offset - 0.5f) * 2.0f * multiplier;
        float baseValue =
            tactic.id < 0 ? -1.0f : PropertiesGetReal(teamAI.baseTeamTactics, tactic.id, -1.0f);
        if (baseValue >= 0.0f)
            PropertiesSet(teamAI.liveTeamTactics, tactic.id, clamp(baseValue + offset, 0.0f, 1.0f));
    }
    return true;
}

// Only players without a gamer qualify. Returns -1 when none does.
CUJU_FUNCTION inline int SelectAttackingRunPlayer(const World &world,
                                                  const SimulationContext &context, int teamID) {
    int possessionPlayer = TeamByID(world, teamID).designatedTeamPossessionPlayer;
    Vector3 offenseFocusPosition = Humanoid(world, possessionPlayer).spatialState.position +
                                   Vector3(-TeamDynamicSide(world, teamID) * 26.0f, 0, 0);
    return GetClosestPlayer(world, context, teamID, offenseFocusPosition, true, possessionPlayer);
}

CUJU_FUNCTION inline void TeamApplyAttackingRun(World &world, const SimulationContext &context,
                                                int teamID, int manualPlayer) {
    state::TeamAIState &teamAI = TeamByID(world, teamID).teamController;
    teamAI.endApplyAttackingRunMilliseconds = world.serialized.match.actualTimeMilliseconds + 4000;
    teamAI.attackingRunPlayer =
        manualPlayer >= 0 ? manualPlayer : SelectAttackingRunPlayer(world, context, teamID);
}

CUJU_FUNCTION_NO_INLINE inline bool TeamAIProcess(World &world, const SimulationContext &context,
                                                  int teamID) {
    CUJU_PROFILE_SCOPE(TEAM_AI_PROCESS);
    state::MatchState &match = world.serialized.match;
    state::TeamState &team = TeamByID(world, teamID);
    const state::TeamState &opponentTeam = TeamByID(world, 1 - teamID);
    state::TeamAIState &teamAI = team.teamController;
    const int side = TeamDynamicSide(world, teamID);
    auto positionOf = [&](int player) { return Humanoid(world, player).spatialState.position; };
    auto movementOf = [&](int player) { return Humanoid(world, player).spatialState.movement; };

    if (match.actualTimeMilliseconds % 1000 == 0 && !TeamUpdateTactics(world, teamID)) return false;

    teamAI.teamHasPossession = team.hasPossession;
    teamAI.teamHasUniquePossession = TeamHasUniquePossession(world, teamID);
    teamAI.opponentTeamHasPossession = opponentTeam.hasPossession;
    teamAI.opponentTeamHasUniquePossession = TeamHasUniquePossession(world, 1 - teamID);
    teamAI.teamHasBestPossession = match.bestPossessionTeam == TeamSlot(world, teamID);
    teamAI.teamPossessionAmount = team.teamPossessionAmount;
    teamAI.fadingTeamPossessionAmount = team.fadingTeamPossessionAmount;
    teamAI.timeNeededToGetToBall = team.timeNeededToGetToBallMilliseconds;
    teamAI.opponentTimeNeededToGetToBall = opponentTeam.timeNeededToGetToBallMilliseconds;

    const float offensivenessBias = teamAI.offensivenessBias;
    float startDistance = 30.0f + 20.0f * offensivenessBias;
    float forceDistance = 6.0f;
    float deepestDanger = (PITCH_HALF_LENGTH - startDistance) * side;
    float adaptedBallX = BallPredict(match.ball, 0).coordinates[0];
    adaptedBallX *= side;
    float offsetX = 20.0f + 10.0f * (1.0f - offensivenessBias);
    float startToForcedBias =
        NormalizedClamp(adaptedBallX, PITCH_HALF_LENGTH - startDistance - offsetX,
                        PITCH_HALF_LENGTH - forceDistance);
    adaptedBallX += offsetX * (1.0f - startToForcedBias);
    adaptedBallX *= side;
    if (adaptedBallX * side > deepestDanger * side) deepestDanger = adaptedBallX;
    if (BallPredict(match.ball, 700).coordinates[0] * side > deepestDanger * side)
        deepestDanger = BallPredict(match.ball, 700).coordinates[0];
    const int opponentPossessionPlayer = opponentTeam.designatedTeamPossessionPlayer;
    if (opponentPossessionPlayer < 0)
        return Unsupported("the team AI finds no opponent possession player");
    float cautionDistance = 4.0f * side;
    // GRF: the comparison predicts 0.15 s ahead, the stored danger 0.1 s.
    if ((positionOf(opponentPossessionPlayer).coordinates[0] +
         movementOf(opponentPossessionPlayer).coordinates[0] * 0.15f + cautionDistance) *
            side >
        deepestDanger * side)
        deepestDanger = positionOf(opponentPossessionPlayer).coordinates[0] +
                        movementOf(opponentPossessionPlayer).coordinates[0] * 0.1f +
                        cautionDistance;
    float lineX = GetOffsideLine(world, MentalImageIndex(world, 0), teamID);
    float allowSlackDistance = 4.0f;
    if (lineX * side - allowSlackDistance > deepestDanger * side)
        deepestDanger = lineX - allowSlackDistance * side;
    teamAI.offsideTrapX = deepestDanger;

    PlayerList players = TeamActivePlayers(world, 1 - teamID);
    Vector3 mostDangerousPosition =
        Vector3((float)(((double)PITCH_HALF_LENGTH - 2.0) * side), 0, 0);
    mostDangerousPosition =
        mostDangerousPosition * 0.8f + BallPredict(match.ball, 100).Get2D() * 0.2f;
    InlineVector<state::TacticalOpponentInformation, state::MAXIMUM_TEAM_PLAYERS> information;
    for (int player : players) {
        state::TacticalOpponentInformation entry;
        entry.player = player;
        entry.dangerFactor =
            (1.0f - NormalizedClamp((positionOf(player) - mostDangerousPosition).GetLength(), 0,
                                    PITCH_HALF_LENGTH * 2));
        entry.dangerFactor *= 0.95f;
        if (player == opponentTeam.designatedTeamPossessionPlayer) entry.dangerFactor += 0.05f;
        information.push_back(entry);
    }
    // GRF: `std::sort` is an insertion sort up to 16 elements, so equal ratings keep their order.
    // This insertion sort matches it.
    if (information.size() > 16) return Unsupported("the team AI ranks more than 16 opponents");
    for (size_t next = 1; next < information.size(); next++) {
        state::TacticalOpponentInformation moving = information[next];
        size_t slot = next;
        while (slot > 0 && moving.dangerFactor > information[slot - 1].dangerFactor) {
            information[slot] = information[slot - 1];
            slot--;
        }
        information[slot] = moving;
    }
    teamAI.tacticalOpponentInformation.count = 0;
    for (const auto &entry : information) teamAI.tacticalOpponentInformation.push_back(entry);

    if (match.actualTimeMilliseconds % 500 == 0 &&
        teamAI.endApplyAttackingRunMilliseconds <= match.actualTimeMilliseconds) {
        if (HumanGamerCount(team) < 2) {
            if (match.bestPossessionTeam == TeamSlot(world, teamID)) {
                float neededRating = 0.5f;
                int runner = SelectAttackingRunPlayer(world, context, teamID);
                if (runner >= 0) {
                    float distance =
                        (positionOf(runner) - positionOf(team.designatedTeamPossessionPlayer))
                            .GetLength();
                    float distanceRating =
                        libm::powf(1.0f - NormalizedClamp(distance, 0, 40), 0.5f);
                    Vector3 spot = positionOf(runner) * Vector3(1.0f, 0.8f, 0.0f) +
                                   Vector3(side * 10.0f, 0, 0);
                    PlayerList opponents =
                        GetClosestPlayers(world, context, 1 - teamID, spot, false, 4);
                    float opponentDensityRating = 1.0f;
                    for (int opponent : opponents) {
                        float opponentDistance = (positionOf(opponent) - spot).GetLength();
                        float opponentDistanceRatingInverse = libm::powf(
                            Curve(1.0f - NormalizedClamp(opponentDistance, 0, 15), 1.0f), 0.5f);
                        opponentDensityRating -= opponentDistanceRatingInverse * 0.3f;
                    }
                    float runConditionsRating = distanceRating * opponentDensityRating;
                    if (runConditionsRating >= neededRating)
                        TeamApplyAttackingRun(world, context, teamID);
                }
            }
        }
    }

    if (match.actualTimeMilliseconds % 1500 == 0) {
        const int designated = team.designatedTeamPossessionPlayer;
        teamAI.forwardSupportPlayer = GetClosestPlayer(
            world, context, teamID,
            positionOf(designated) * Vector3(1.0f, 1.0f, 0.0f) + Vector3(-side * 1.5f, 0, 0), false,
            designated);
    }
    return true;
}

using HungarianRow = InlineVector<int, state::MAXIMUM_TEAM_PLAYERS>;
using HungarianMatrix =
    InlineVector<int, state::MAXIMUM_TEAM_PLAYERS * state::MAXIMUM_TEAM_PLAYERS>;
// GRF: libhungarian, step for step. Reduces `cost` in place and returns the total cost, or -1 when
// the reduced costs fail to prove the result optimal. Writes `assignedColumns` only on success.
CUJU_FUNCTION_NO_INLINE inline int HungarianSolve(HungarianMatrix &cost, int size,
                                                  HungarianRow &assignedColumns) {
    const int INT_INFINITY = 0x7FFFFFFF;
    auto costAt = [&](int rowIndex, int columnIndex) -> int & {
        return cost[rowIndex * size + columnIndex];
    };
    int index, otherColumn, row = 0, column = 0, reduction, unchosenCount, queuePosition, unmatched,
                            total = 0;
    const int rowCount = size;
    HungarianRow columnOfRow, unchosenRows, rowDecrease, slackRow, rowOfColumn, parentRow,
        columnIncrease, slack;
    for (HungarianRow *buffer : {&columnOfRow, &unchosenRows, &rowDecrease, &slackRow, &rowOfColumn,
                                 &parentRow, &columnIncrease, &slack}) {
        buffer->resize(size);
        for (int entry = 0; entry < size; entry++) (*buffer)[entry] = 0;
    }
    for (column = 0; column < size; column++) {
        reduction = costAt(0, column);
        for (row = 1; row < rowCount; row++)
            if (costAt(row, column) < reduction) reduction = costAt(row, column);
        total += reduction;
        if (reduction != 0)
            for (row = 0; row < rowCount; row++) costAt(row, column) -= reduction;
    }
    unchosenCount = 0;
    for (column = 0; column < size; column++) {
        rowOfColumn[column] = -1;
        parentRow[column] = -1;
        columnIncrease[column] = 0;
        slack[column] = INT_INFINITY;
    }
    for (row = 0; row < rowCount; row++) {
        reduction = costAt(row, 0);
        for (column = 1; column < size; column++)
            if (costAt(row, column) < reduction) reduction = costAt(row, column);
        rowDecrease[row] = reduction;
        bool matched = false;
        for (column = 0; column < size; column++)
            if (reduction == costAt(row, column) && rowOfColumn[column] < 0) {
                columnOfRow[row] = column;
                rowOfColumn[column] = row;
                matched = true;
                break;
            }
        if (!matched) {
            columnOfRow[row] = -1;
            unchosenRows[unchosenCount++] = row;
        }
    }
    if (unchosenCount == 0) goto done;
    unmatched = unchosenCount;
    while (1) {
        queuePosition = 0;
        while (1) {
            while (queuePosition < unchosenCount) {
                row = unchosenRows[queuePosition];
                reduction = rowDecrease[row];
                for (column = 0; column < size; column++)
                    if (slack[column]) {
                        int delta = costAt(row, column) - reduction + columnIncrease[column];
                        if (delta < slack[column]) {
                            if (delta == 0) {
                                if (rowOfColumn[column] < 0) goto breakthrough;
                                slack[column] = 0;
                                parentRow[column] = row;
                                unchosenRows[unchosenCount++] = rowOfColumn[column];
                            } else {
                                slack[column] = delta;
                                slackRow[column] = row;
                            }
                        }
                    }
                queuePosition++;
            }
            reduction = INT_INFINITY;
            for (column = 0; column < size; column++)
                if (slack[column] && slack[column] < reduction) reduction = slack[column];
            for (queuePosition = 0; queuePosition < unchosenCount; queuePosition++)
                rowDecrease[unchosenRows[queuePosition]] += reduction;
            for (column = 0; column < size; column++)
                if (slack[column]) {
                    slack[column] -= reduction;
                    if (slack[column] == 0) {
                        row = slackRow[column];
                        if (rowOfColumn[column] < 0) {
                            for (otherColumn = column + 1; otherColumn < size; otherColumn++)
                                if (slack[otherColumn] == 0)
                                    columnIncrease[otherColumn] += reduction;
                            goto breakthrough;
                        } else {
                            parentRow[column] = row;
                            unchosenRows[unchosenCount++] = rowOfColumn[column];
                        }
                    }
                } else {
                    columnIncrease[column] += reduction;
                }
        }
    breakthrough:
        while (1) {
            otherColumn = columnOfRow[row];
            columnOfRow[row] = column;
            rowOfColumn[column] = row;
            if (otherColumn < 0) break;
            row = parentRow[otherColumn];
            column = otherColumn;
        }
        if (--unmatched == 0) goto done;
        unchosenCount = 0;
        for (column = 0; column < size; column++) {
            parentRow[column] = -1;
            slack[column] = INT_INFINITY;
        }
        for (row = 0; row < rowCount; row++)
            if (columnOfRow[row] < 0) unchosenRows[unchosenCount++] = row;
    }
done:
    for (row = 0; row < rowCount; row++)
        for (column = 0; column < size; column++)
            if (costAt(row, column) < rowDecrease[row] - columnIncrease[column]) return -1;
    for (row = 0; row < rowCount; row++) {
        column = columnOfRow[row];
        if (column < 0 || costAt(row, column) != rowDecrease[row] - columnIncrease[column])
            return -1;
    }
    row = 0;
    for (column = 0; column < size; column++)
        if (columnIncrease[column]) row++;
    if (row > rowCount) return -1;
    assignedColumns = columnOfRow;
    for (index = 0; index < rowCount; index++) total += rowDecrease[index];
    for (index = 0; index < size; index++) total -= columnIncrease[index];
    return total;
}

CUJU_FUNCTION_NO_INLINE inline bool TeamCalculateDynamicRoles(World &world,
                                                              const SimulationContext &context,
                                                              int teamID) {
    PlayerList players = TeamActivePlayers(world, teamID);
    if (!players.empty()) {
        // GRF: removes the first goalkeeper. Only one that comes first among the active players is
        // supported, and any other order has no defined result.
        if (Formation(world, context, players[0]).role != e_PlayerRole_Goalkeeper)
            return Unsupported("the dynamic roles need the goalkeeper as the first active player");
        players.erase(players.begin());
    }
    const unsigned int playerCount = (unsigned int)players.size();
    auto positionOf = [&](int player) { return Humanoid(world, player).spatialState.position; };
    auto movementOf = [&](int player) { return Humanoid(world, player).spatialState.movement; };
    InlineVector<Vector3, state::MAXIMUM_TEAM_PLAYERS> adaptedFormationPositions;
    for (unsigned int slotIndex = 0; slotIndex < playerCount; slotIndex++)
        adaptedFormationPositions.push_back(
            TeamAdaptedFormationPosition(world, context, players[slotIndex], false));
    auto integerCost = [&](unsigned int playerIndex, unsigned int slotIndex) {
        Vector3 playerPosition =
            positionOf(players[playerIndex]) + movementOf(players[playerIndex]) * 0.5f;
        float distance = (playerPosition - adaptedFormationPositions[slotIndex]).GetLength();
        return FloatToInt32(roundf(distance * 10));
    };
    HungarianMatrix distances;
    for (unsigned int playerIndex = 0; playerIndex < playerCount; playerIndex++)
        for (unsigned int slotIndex = 0; slotIndex < playerCount; slotIndex++)
            distances.push_back(integerCost(playerIndex, slotIndex));
    // Exact: every sort of integers gives the same order as GRF's `std::sort`.
    for (int unsorted = 1; unsorted < distances.count; unsorted++)
        for (int slot = unsorted; slot > 0 && distances[slot] < distances[slot - 1]; slot--) {
            int swapped = distances[slot];
            distances[slot] = distances[slot - 1];
            distances[slot - 1] = swapped;
        }

    for (unsigned int cutoff = playerCount; cutoff < distances.size(); cutoff += 5) {
        // A row is a formation slot and a column a player.
        HungarianMatrix cost;
        cost.resize(playerCount * playerCount);
        for (unsigned int playerIndex = 0; playerIndex < playerCount; playerIndex++)
            for (unsigned int slotIndex = 0; slotIndex < playerCount; slotIndex++) {
                int entryCost = integerCost(playerIndex, slotIndex);
                if (entryCost >= distances[cutoff]) entryCost = 50000;
                cost[playerIndex + slotIndex * playerCount] = entryCost;
            }
        HungarianRow columnOfRow;
        int totalCost = HungarianSolve(cost, (int)playerCount, columnOfRow);
        if (totalCost != -1 && (totalCost < 50000 || cutoff >= distances.size() - 5)) {
            for (unsigned int playerIndex = 0; playerIndex < playerCount; playerIndex++)
                for (unsigned int slotIndex = 0; slotIndex < playerCount; slotIndex++)
                    if (columnOfRow[slotIndex] == (int)playerIndex) {
                        const StaticFormation &formation =
                            Formation(world, context, players[slotIndex]);
                        state::FormationEntry &entry =
                            world.serialized.match.players[players[playerIndex]]
                                .dynamicFormationEntry;
                        entry.role = formation.role;
                        entry.position = formation.position;
                        entry.startPosition = formation.startPosition;
                        entry.lazy = formation.lazy;
                        entry.controllable = formation.controllable;
                    }
            break;
        }
    }
    return true;
}

CUJU_FUNCTION_NO_INLINE inline float TeamCalculateMarkingQuality(const World &world, int teamID,
                                                                 int player, int opponent) {
    const int side = TeamDynamicSide(world, teamID);
    const state::SpatialState &playerState = Humanoid(world, player).spatialState,
                              &opponentState = Humanoid(world, opponent).spatialState;
    Vector3 opponentPosition = opponentState.position + opponentState.movement * 0.1f;
    Vector3 playerPosition = playerState.position + playerState.movement * 0.1f;
    Vector3 goalPosition = Vector3(PITCH_HALF_LENGTH * side, 0, 0);
    Vector3 toGoal = goalPosition - playerPosition;
    float lineLength = clamp((toGoal).GetLength(), 4.0f, 14.0f);
    Vector3 toGoalDirection = toGoal.GetNormalized(Vector3(side, 0, 0));
    Vector3 safetyVector = -toGoalDirection * 0.5f;
    Line line(playerPosition + safetyVector +
                  toGoalDirection.GetRotated2D((float)(-0.5 * (double)PI)) * lineLength,
              playerPosition + safetyVector +
                  toGoalDirection.GetRotated2D((float)(0.5 * (double)PI)) * lineLength);
    bool opponentIsOnRightSideOfLine =
        ((line.points[1].coordinates[0] - line.points[0].coordinates[0]) *
             (opponentPosition.coordinates[1] - line.points[0].coordinates[1]) -
         (line.points[1].coordinates[1] - line.points[0].coordinates[1]) *
             (opponentPosition.coordinates[0] - line.points[0].coordinates[0])) > 0;
    float lineParameter = 0.0f;
    float opponentFromLineDistance = line.GetDistanceToPoint(opponentPosition, lineParameter);
    float adaptedOpponentFromLineDistance = opponentFromLineDistance;
    if (opponentIsOnRightSideOfLine)
        adaptedOpponentFromLineDistance = fabsf(opponentFromLineDistance - 2.0f);
    float opponentFromLineDistanceFactor =
        libm::powf(NormalizedClamp(adaptedOpponentFromLineDistance, 0.0f, 60.0f), 0.5f);
    float opponentOnLineDistanceFactor =
        libm::powf(clamp(fabsf(lineParameter * 2.0f - 1.0f), 0.0f, 1.0f), 0.5f);
    float result = 1.0f;
    result -= opponentFromLineDistanceFactor * 0.5f;
    result -= opponentOnLineDistanceFactor * 0.5f;
    result = clamp(result, 0.0f, 1.0f);
    if (!opponentIsOnRightSideOfLine) result *= 0.6f;
    float opponentDistance =
        (1.0f - NormalizedClamp((playerPosition - opponentPosition).GetLength(), 0.0f,
                                PITCH_HALF_LENGTH * 2.0f));
    result = result * 0.8f + opponentDistance * 0.2f;
    return result;
}

CUJU_FUNCTION_NO_INLINE inline bool TeamCalculateManMarking(World &world,
                                                            const SimulationContext &context,
                                                            int teamID) {
    const int markedOpponentCount = 3;
    const state::TeamAIState &teamAI = TeamByID(world, teamID).teamController;
    PlayerList players = TeamActivePlayers(world, teamID);
    for (int player : players) world.serialized.match.players[player].manMarking = -1;
    const int markedCount =
        std::min((int)teamAI.tacticalOpponentInformation.count, markedOpponentCount);
    for (int rank = 0; rank < markedCount; rank++) {
        int closestPlayer = -1;
        float bestMarkingQuality = -1.0f;
        size_t closestIndex = 0;
        const int opponentPlayer = teamAI.tacticalOpponentInformation[rank].player;
        for (size_t index = 0; index < players.size(); index++) {
            if (Formation(world, context, players[index]).role == e_PlayerRole_Goalkeeper) continue;
            float markingQuality =
                TeamCalculateMarkingQuality(world, teamID, players[index], opponentPlayer);
            if (markingQuality > bestMarkingQuality) {
                closestPlayer = players[index];
                bestMarkingQuality = markingQuality;
                closestIndex = index;
            }
        }
        if (closestPlayer >= 0) {
            world.serialized.match.players[closestPlayer].manMarking = opponentPlayer;
            players.erase(players.begin() + closestIndex);
        }
        if (players.empty()) break;
    }
    return true;
}

}  // namespace simulation
}  // namespace cuju

// The referee needs the team AI above.
#include "referee.h"

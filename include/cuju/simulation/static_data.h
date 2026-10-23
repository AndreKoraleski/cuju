// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The static data of a match, which a state does not hold, and its parser.
#pragma once

#include <cstring>
#include <stdexcept>
#include <string>

#include "../state/environment_state.h"
#include "../vector3.h"

namespace cuju {
namespace simulation {

constexpr int PLAYER_STATISTICS = 22;
enum PlayerStatistic {
    e_Statistic_PhysicalBalance,
    e_Statistic_PhysicalReaction,
    e_Statistic_PhysicalAcceleration,
    e_Statistic_PhysicalVelocity,
    e_Statistic_PhysicalStamina,
    e_Statistic_PhysicalAgility,
    e_Statistic_PhysicalShotPower,
    e_Statistic_TechnicalStandingTackle,
    e_Statistic_TechnicalSlidingTackle,
    e_Statistic_TechnicalBallControl,
    e_Statistic_TechnicalDribble,
    e_Statistic_TechnicalShortPass,
    e_Statistic_TechnicalHighPass,
    e_Statistic_TechnicalHeader,
    e_Statistic_TechnicalShot,
    e_Statistic_TechnicalVolley,
    e_Statistic_MentalCalmness,
    e_Statistic_MentalWorkRate,
    e_Statistic_MentalResilience,
    e_Statistic_MentalDefensivePositioning,
    e_Statistic_MentalOffensivePositioning,
    e_Statistic_MentalVision
};

struct PlayerData {
    float statistics[PLAYER_STATISTICS];
    float velocityStatistic;  // The statistic that the maximum velocity of a player reads.
    float height;
    int32_t skinColor;
};

struct StaticFormation {
    int32_t role;  // An `e_PlayerRole`.
    Vector3 position, startPosition;
    bool lazy, controllable;
};

constexpr int MAXIMUM_STATIC_ENTRIES = 2 * state::MAXIMUM_TEAM_PLAYERS + 3;

// The entries are the players in state order, then the referee and the two linesmen.
// `formation` holds the players only.
struct StaticData {
    int count = 0;
    PlayerData data[MAXIMUM_STATIC_ENTRIES];
    StaticFormation formation[MAXIMUM_STATIC_ENTRIES];
    // GRF: `match_duration * 0.2f + 0.05f` in float, with the `match_duration` of 0.027 it sets.
    float matchDurationFactor = 0x1.c5d638p-5f;
    int32_t stableID[MAXIMUM_STATIC_ENTRIES] = {};
};

// The bytes are an `int32_t` count, the `PlayerData` of each entry, then the section `FORM` with
// each player's formation, `MDUR` with `matchDurationFactor` and `SIDS` with `stableID`. Throws
// `std::runtime_error` on bytes of any other shape.
inline StaticData ParseStatic(const std::string &bytes) {
    StaticData staticData;
    if (bytes.size() < 4) throw std::runtime_error("static data: too short");
    memcpy(&staticData.count, bytes.data(), 4);
    size_t offset = 4 + (size_t)staticData.count * sizeof(PlayerData);
    if (staticData.count < 3 || staticData.count > MAXIMUM_STATIC_ENTRIES || bytes.size() < offset)
        throw std::runtime_error("static data: bad size");
    memcpy(staticData.data, bytes.data() + 4, staticData.count * sizeof(PlayerData));
    const int playerCount = staticData.count - 3;
    // A role, a position, a start position and the two flags.
    const size_t entryBytes = 4 + 12 + 12 + 1 + 1;
    if (bytes.compare(offset, 4, "FORM") != 0 ||
        bytes.size() < offset + 4 + playerCount * entryBytes)
        throw std::runtime_error("static data: missing or bad formation section");
    offset += 4;
    for (int player = 0; player < playerCount; player++) {
        StaticFormation &formation = staticData.formation[player];
        memcpy(&formation.role, &bytes[offset], 4);
        memcpy(&formation.position, &bytes[offset + 4], 12);
        memcpy(&formation.startPosition, &bytes[offset + 16], 12);
        formation.lazy = bytes[offset + 28] != 0;
        formation.controllable = bytes[offset + 29] != 0;
        offset += entryBytes;
    }
    if (bytes.compare(offset, 4, "MDUR") != 0 || bytes.size() < offset + 8)
        throw std::runtime_error("static data: missing or bad match duration section");
    memcpy(&staticData.matchDurationFactor, &bytes[offset + 4], 4);
    offset += 8;
    if (bytes.compare(offset, 4, "SIDS") != 0 ||
        bytes.size() != offset + 4 + (size_t)staticData.count * 4)
        throw std::runtime_error("static data: missing or bad stable ID section");
    memcpy(staticData.stableID, &bytes[offset + 4], (size_t)staticData.count * 4);
    return staticData;
}

}  // namespace simulation
}  // namespace cuju

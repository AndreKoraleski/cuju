// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// Builds the agentless bundle of a GRF scenario file without GRF. Every constant table is GRF's.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../animation/animation_derived.h"
#include "../animation/keyframes.h"
#include "../animation/skeleton.h"
#include "../rng.h"
#include "../simulation/environment.h"
#include "../simulation/state_io.h"
#include "../simulation/static_data.h"
#include "../simulation/team_select.h"

namespace cuju {
namespace assets {

// GRF: `TeamData` has 11 formation slots and 11 persons, and a larger team crashes GRF.
constexpr int MAXIMUM_SCENARIO_TEAM_PLAYERS = 11;
constexpr int PLAYER_ROLES = 10;

// One `AddPlayer` call, in environment coordinates. `role` is an `e_PlayerRole`.
struct ScenarioPlayer {
    // x in [-1, 1], y in [-0.42, 0.42]. GRF rounds both to float, so double keeps Python's value.
    double x = 0, y = 0;
    int32_t role = 0;
    bool lazy = false, controllable = true;
};

// What a scenario file builds for one parity of the episode number. Python has already added the
// lazy goalkeeper of `Scenario._FakePlayersForEmptyTeam` to an empty team.
// GRF: the defaults are those of `ScenarioConfig` in `main.hpp`.
struct ScenarioDescription {
    // As `SetBallPosition` takes it.
    double ballX = 0, ballY = 0;
    // In `AddPlayer` order.
    std::vector<ScenarioPlayer> leftTeam, rightTeam;
    bool useMagnet = true, offsides = true, realTime = false;
    double leftTeamDifficulty = 1.0, rightTeamDifficulty = 0.6;
    bool deterministic = false, endEpisodeOnScore = false, endEpisodeOnPossessionChange = false,
         endEpisodeOnOutOfPlay = false;
    int32_t gameDuration = 3000, secondHalf = 999999999;
    bool controlAllPlayers = false;
    // GRF: only a deterministic scenario keeps this seed, and GRF replaces it otherwise.
    uint32_t gameEngineRandomSeed = 42;
};

// GRF: `TeamData::TeamData` gives player i of every team the person `PlayerData::PlayerData(id)`
// for the IDs 398, 11, 254, 320, 103, 188, 74, 332, 290, 391 and 264, with the 22 statistics
// that `CalculateStat` gives in `PlayerStatistic` order, the velocity statistic and the height.
constexpr float PLAYER_TABLE[MAXIMUM_SCENARIO_TEAM_PLAYERS][simulation::PLAYER_STATISTICS + 2] = {
    {0x1.e1e3eap-1f, 0x1.000000p+0f, 0x1.97c0d2p-1f, 0x1.4d9ddcp-1f, 0x1.97c0d2p-1f,
     0x1.e1e3eap-1f, 0x1.e1e3eap-1f, 0x1.72af58p-2f, 0x1.72af58p-2f, 0x1.037ac4p-1f,
     0x1.72af58p-2f, 0x1.e1e3eap-1f, 0x1.000000p+0f, 0x1.72af58p-2f, 0x1.037ac4p-1f,
     0x1.72af58p-2f, 0x1.97c0d2p-1f, 0x1.97c0d2p-1f, 0x1.97c0d2p-1f, 0x1.000000p+0f,
     0x1.bcd24ep-3f, 0x1.97c0d2p-1f, 0x1.4d9ddcp-1f, 0x1.deb852p+0f},
    {0x1.ab1b9cp-1f, 0x1.921e1ep-1f, 0x1.921e1ep-1f, 0x1.921e1ep-1f, 0x1.602300p-1f,
     0x1.921e1ep-1f, 0x1.ab1b9cp-1f, 0x1.921e1ep-1f, 0x1.921e1ep-1f, 0x1.79207ep-1f,
     0x1.602300p-1f, 0x1.ab1b9cp-1f, 0x1.921e1ep-1f, 0x1.602300p-1f, 0x1.472580p-1f,
     0x1.c65e8ap-2f, 0x1.602300p-1f, 0x1.602300p-1f, 0x1.602300p-1f, 0x1.921e1ep-1f,
     0x1.472580p-1f, 0x1.79207ep-1f, 0x1.921e1ep-1f, 0x1.b33334p+0f},
    {0x1.000000p+0f, 0x1.d4dd94p-1f, 0x1.6240fap-1f, 0x1.6240fap-1f, 0x1.6240fap-1f,
     0x1.6240fap-1f, 0x1.aea960p-1f, 0x1.000000p+0f, 0x1.fb11c6p-1f, 0x1.6240fap-1f,
     0x1.df48c2p-2f, 0x1.aea960p-1f, 0x1.aea960p-1f, 0x1.88752ep-1f, 0x1.df48c2p-2f,
     0x1.4677f6p-2f, 0x1.6240fap-1f, 0x1.6240fap-1f, 0x1.6240fap-1f, 0x1.fb11c6p-1f,
     0x1.df48c2p-2f, 0x1.aea960p-1f, 0x1.6240fap-1f, 0x1.bd70a4p+0f},
    {0x1.000000p+0f, 0x1.000000p+0f, 0x1.75c39cp-1f, 0x1.75c39cp-1f, 0x1.75c39cp-1f,
     0x1.4f570cp-1f, 0x1.e9098ep-1f, 0x1.000000p+0f, 0x1.000000p+0f, 0x1.28ea5cp-1f,
     0x1.6b4916p-2f, 0x1.4f570cp-1f, 0x1.4f570cp-1f, 0x1.c29cdcp-1f, 0x1.6b4916p-2f,
     0x1.1e6fb4p-2f, 0x1.75c39cp-1f, 0x1.75c39cp-1f, 0x1.75c39cp-1f, 0x1.000000p+0f,
     0x1.6b4916p-2f, 0x1.75c39cp-1f, 0x1.75c39cp-1f, 0x1.ee147ap+0f},
    {0x1.8bd2b6p-1f, 0x1.7459dap-1f, 0x1.7459dap-1f, 0x1.7459dap-1f, 0x1.456840p-1f,
     0x1.5ce11ep-1f, 0x1.7459dap-1f, 0x1.a34b94p-1f, 0x1.8bd2b6p-1f, 0x1.7459dap-1f,
     0x1.2def62p-1f, 0x1.bac472p-1f, 0x1.a34b94p-1f, 0x1.456840p-1f, 0x1.fdfb50p-2f,
     0x1.712664p-2f, 0x1.456840p-1f, 0x1.456840p-1f, 0x1.456840p-1f, 0x1.a34b94p-1f,
     0x1.fdfb50p-2f, 0x1.a34b94p-1f, 0x1.7459dap-1f, 0x1.b851ecp+0f},
    {0x1.39d0a6p-1f, 0x1.8b09a6p-1f, 0x1.dc42a6p-1f, 0x1.dc42a6p-1f, 0x1.8b09a6p-1f,
     0x1.000000p+0f, 0x1.000000p+0f, 0x1.2ebd4cp-2f, 0x1.2ebd4cp-2f, 0x1.dc42a6p-1f,
     0x1.000000p+0f, 0x1.dc42a6p-1f, 0x1.8b09a6p-1f, 0x1.39d0a6p-1f, 0x1.000000p+0f,
     0x1.8b09a6p-1f, 0x1.8b09a6p-1f, 0x1.8b09a6p-1f, 0x1.8b09a6p-1f, 0x1.d12f4cp-2f,
     0x1.000000p+0f, 0x1.39d0a6p-1f, 0x1.dc42a6p-1f, 0x1.b5c290p+0f},
    {0x1.000000p+0f, 0x1.bd7214p-1f, 0x1.6e6a34p-1f, 0x1.6e6a34p-1f, 0x1.6e6a34p-1f,
     0x1.6e6a34p-1f, 0x1.bd7214p-1f, 0x1.000000p+0f, 0x1.bd7214p-1f, 0x1.6e6a34p-1f,
     0x1.1f6252p-1f, 0x1.000000p+0f, 0x1.000000p+0f, 0x1.6e6a34p-1f, 0x1.1f6252p-1f,
     0x1.a0b4e2p-2f, 0x1.6e6a34p-1f, 0x1.6e6a34p-1f, 0x1.6e6a34p-1f, 0x1.bd7214p-1f,
     0x1.1f6252p-1f, 0x1.bd7214p-1f, 0x1.6e6a34p-1f, 0x1.e3d70ap+0f},
    {0x1.985b5cp-1f, 0x1.985b5cp-1f, 0x1.985b5cp-1f, 0x1.985b5cp-1f, 0x1.717742p-1f,
     0x1.bf3f96p-1f, 0x1.e623b0p-1f, 0x1.4a932ap-1f, 0x1.23af10p-1f, 0x1.985b5cp-1f,
     0x1.985b5cp-1f, 0x1.e623b0p-1f, 0x1.bf3f96p-1f, 0x1.4a932ap-1f, 0x1.985b5cp-1f,
     0x1.23af10p-1f, 0x1.717742p-1f, 0x1.717742p-1f, 0x1.717742p-1f, 0x1.4a932ap-1f,
     0x1.985b5cp-1f, 0x1.717742p-1f, 0x1.985b5cp-1f, 0x1.d70a3ep+0f},
    {0x1.328fa0p-1f, 0x1.82d9d0p-1f, 0x1.fb4916p-1f, 0x1.fb4916p-1f, 0x1.82d9d0p-1f,
     0x1.000000p+0f, 0x1.000000p+0f, 0x1.23f680p-2f, 0x1.a7589ep-3f, 0x1.aafee6p-1f,
     0x1.000000p+0f, 0x1.d323fep-1f, 0x1.aafee6p-1f, 0x1.328fa0p-1f, 0x1.000000p+0f,
     0x1.5ab4b8p-1f, 0x1.82d9d0p-1f, 0x1.82d9d0p-1f, 0x1.82d9d0p-1f, 0x1.9c65c8p-2f,
     0x1.000000p+0f, 0x1.5ab4b8p-1f, 0x1.fb4916p-1f, 0x1.bd70a4p+0f},
    {0x1.452fa0p-1f, 0x1.9a5a8ap-1f, 0x1.000000p+0f, 0x1.000000p+0f, 0x1.9a5a8ap-1f,
     0x1.000000p+0f, 0x1.000000p+0f, 0x1.35b35cp-2f, 0x1.c110a2p-3f, 0x1.c4efeep-1f,
     0x1.000000p+0f, 0x1.ef8572p-1f, 0x1.c4efeep-1f, 0x1.452fa0p-1f, 0x1.000000p+0f,
     0x1.6fc504p-1f, 0x1.9a5a8ap-1f, 0x1.9a5a8ap-1f, 0x1.9a5a8ap-1f, 0x1.b573a8p-2f,
     0x1.000000p+0f, 0x1.6fc504p-1f, 0x1.000000p+0f, 0x1.cf5c28p+0f},
    {0x1.51083ep-1f, 0x1.a94d72p-1f, 0x1.000000p+0f, 0x1.000000p+0f, 0x1.a94d72p-1f,
     0x1.000000p+0f, 0x1.000000p+0f, 0x1.40fba8p-2f, 0x1.d16ca4p-3f, 0x1.d56ffcp-1f,
     0x1.000000p+0f, 0x1.000000p+0f, 0x1.d56ffcp-1f, 0x1.51083ep-1f, 0x1.000000p+0f,
     0x1.7d2ac8p-1f, 0x1.a94d72p-1f, 0x1.a94d72p-1f, 0x1.a94d72p-1f, 0x1.c56366p-2f,
     0x1.000000p+0f, 0x1.7d2ac8p-1f, 0x1.000000p+0f, 0x1.b0a3d8p+0f}};
// The skin colour of each of those persons, which the simulation never reads.
constexpr int32_t PLAYER_SKIN[MAXIMUM_SCENARIO_TEAM_PLAYERS] = {1, 2, 4, 1, 1, 4, 3, 1, 1, 2, 3};
// GRF: the officials get `PlayerData::PlayerData()`. Their skin colour is a draw of the generator.
constexpr float OFFICIAL_STATISTIC = (float)0.6, OFFICIAL_HEIGHT = 1.8f;
// GRF: `Match::matchDurationFactor`, `match_duration * 0.2f + 0.05f` with the 0.027 that
// `GameEnv::start_game` sets.
constexpr float MATCH_DURATION_FACTOR = 0x1.c5d638p-5f;

// GRF: a node of `player.object` and its collision geometry, if any.
struct SkeletonNodeConstant {
    const char *name;
    int32_t parent, localMode;
    const char *geometry;
    int32_t geometryLocalMode;
    // Position, rotation, scale, then the two corners of the mesh box.
    float geometryValues[16];
};
constexpr int32_t SKELETON_NODE_MAP[state::BODY_PARTS] = {2,  3, 8, 11, 9, 12, 10,
                                                          13, 4, 6, 5,  7, 1,  0};
// GRF: `0x1.47aep-133f`, `numeric_limits<float>::min() / 100`, is an empty mesh box's maximum.
constexpr SkeletonNodeConstant SKELETON_NODES[state::HUMANOID_NODES] = {
    {"player", -1, 1, nullptr, 0, {}},
    {"body",
     0,
     0,
     "pelvis",
     0,
     {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -0x1.90ff98p-3f, -0x1.cf41f2p-4f, -0x1.fd8adap-5f,
      0x1.913404p-3f, 0x1.044d02p-3f, 0x1.dab9f6p-3f}},
    {"middle",
     1,
     0,
     "trunk",
     0,
     {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -0x1.d4fdf4p-3f, -0x1.0a3d70p-3f, -0x1.4fdf3cp-4f,
      0x1.d4fdf4p-3f, 0x1.df3b64p-4f, 0x1.0cccccp-1f}},
    {"neck",
     2,
     0,
     "head",
     0,
     {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -0x1.68db8cp-4f, -0x1.0aa64cp-3f, -0x1.6f0068p-5f,
      0x1.6b50b0p-4f, 0x1.b22d0ep-4f, 0x1.2617c2p-2f}},
    {"left_shoulder",
     2,
     0,
     "left_upperarm",
     0,
     {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -0x1.e353f8p-5f, -0x1.1eb852p-4f, -0x1.5c28f6p-2f,
      0x1.e353f8p-5f, 0x1.22d0e6p-4f, 0x1.47aep-133f}},
    {"left_elbow",
     4,
     0,
     "left_lowerarm",
     0,
     {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -0x1.41f212p-5f, -0x1.a92a30p-5f, -0x1.5d97f6p-2f,
      0x1.3d07c8p-5f, 0x1.4af4f0p-5f, 0x1.2617c2p-5f}},
    {"right_shoulder",
     2,
     0,
     "right_upperarm",
     0,
     {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -0x1.e353f8p-5f, -0x1.1eb852p-4f, -0x1.5c28f6p-2f,
      0x1.e353f8p-5f, 0x1.22d0e6p-4f, 0x1.47aep-133f}},
    {"right_elbow",
     6,
     0,
     "right_lowerarm",
     0,
     {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -0x1.41f212p-5f, -0x1.a92a30p-5f, -0x1.5d97f6p-2f,
      0x1.3d07c8p-5f, 0x1.4af4f0p-5f, 0x1.2617c2p-5f}},
    {"left_thigh",
     1,
     0,
     "left_upperleg",
     0,
     {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -0x1.9eecc0p-4f, -0x1.6e2eb2p-4f, -0x1.d13404p-2f,
      0x1.9eecc0p-4f, 0x1.ae7d56p-4f, 0x1.41205cp-4f}},
    {"left_knee",
     8,
     0,
     "left_lowerleg",
     0,
     {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -0x1.e353f8p-5f, -0x1.94af50p-5f, -0x1.fd2200p-2f,
      0x1.eecbfcp-5f, 0x1.5a8588p-4f, 0x1.6f0068p-5f}},
    {"left_ankle",
     9,
     0,
     "left_foot",
     0,
     {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -0x1.5182aap-5f, -0x1.491d14p-3f, -0x1.d288cep-4f,
      0x1.573eacp-5f, 0x1.89a028p-4f, 0x1.47aep-133f}},
    {"right_thigh",
     1,
     0,
     "right_upperleg",
     0,
     {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -0x1.9eecc0p-4f, -0x1.6e2eb2p-4f, -0x1.d13404p-2f,
      0x1.9eecc0p-4f, 0x1.ae7d56p-4f, 0x1.41205cp-4f}},
    {"right_knee",
     11,
     0,
     "right_lowerleg",
     0,
     {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -0x1.e353f8p-5f, -0x1.94af50p-5f, -0x1.fd2200p-2f,
      0x1.eecbfcp-5f, 0x1.5a8588p-4f, 0x1.6f0068p-5f}},
    {"right_ankle",
     12,
     0,
     "right_foot",
     0,
     {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -0x1.5182aap-5f, -0x1.491d14p-3f, -0x1.d288cep-4f,
      0x1.573eacp-5f, 0x1.89a028p-4f, 0x1.47aep-133f}}};
// GRF: the bone offsets of `player.object`, as `HumanoidBase::ResetPosition` leaves them. The root
// position and identity rotations it sets already hold. The root rotation keeps the file's -0.
struct TemplateNode {
    float position[3], rotation[4];
    int32_t localMode;
};
constexpr TemplateNode TEMPLATE_NODES[state::HUMANOID_NODES] = {
    {{0, 0, 0}, {0, 0, -0.0f, 1}, 1},
    {{0, 0, 0x1.eb851ep-1f}, {0, 0, 0, 1}, 0},
    {{0, 0, 0x1.333334p-3f}, {0, 0, 0, 1}, 0},
    {{0, -0x1.eb851ep-6f, 0x1.000000p-1f}, {0, 0, 0, 1}, 0},
    {{0x1.47ae14p-3f, -0x1.47ae14p-7f, 0x1.eb851ep-2f}, {0, 0, 0, 1}, 0},
    {{-0x1.47ae14p-7f, 0, -0x1.51eb86p-2f}, {0, 0, 0, 1}, 0},
    {{-0x1.47ae14p-3f, -0x1.47ae14p-7f, 0x1.eb851ep-2f}, {0, 0, 0, 1}, 0},
    {{0x1.47ae14p-7f, 0, -0x1.51eb86p-2f}, {0, 0, 0, 1}, 0},
    {{0x1.645a1cp-4f, 0, -0x1.47ae14p-7f}, {0, 0, 0, 1}, 0},
    {{0, 0, -0x1.ae147ap-2f}, {0, 0, 0, 1}, 0},
    {{0, -0x1.47ae14p-5f, -0x1.c28f5cp-2f}, {0, 0, 0, 1}, 0},
    {{-0x1.645a1cp-4f, 0, -0x1.47ae14p-7f}, {0, 0, 0, 1}, 0},
    {{0, 0, -0x1.ae147ap-2f}, {0, 0, 0, 1}, 0},
    {{0, -0x1.47ae14p-5f, -0x1.c28f5cp-2f}, {0, 0, 0, 1}, 0}};

// GRF: the tactics of the `TeamAIController` constructor in `state::TacticID` order, the live ones
// mixed with those of `TeamData`. GRF keeps `%f` text, so each is its `atof`, rounded to float.
constexpr double BASE_TACTICS[state::TACTIC_KEYS] = {0.75, 0.8, 0.5, 0.35, 0.54, 0.4, 0.8,
                                                     0.9,  0.7, 0.6, 0.35, 0.52, 0.1, 0.9};
constexpr double LIVE_TACTICS[state::TACTIC_KEYS] = {0.71,  0.938, 0.776, 0.35, 0.54, 0.332, 0.84,
                                                     0.868, 0.826, 0.828, 0.35, 0.52, 0.176, 0.948};

// GRF: the 11 formation slots of `TeamData`, as `atof` reads them.
constexpr double FORMATION_SLOTS[MAXIMUM_SCENARIO_TEAM_PLAYERS][2] = {
    {-1.0, 0.0}, {-0.7, 0.75}, {-1.0, 0.25}, {-1.0, -0.25}, {-0.7, -0.75}, {0.0, 0.5},
    {-0.2, 0.0}, {0.0, -0.5},  {0.6, 0.75},  {1.0, 0.0},    {0.6, -0.75}};
// GK, LB, CB, CB, RB, CM, CM, CM, LM, CF and RM.
constexpr int32_t FORMATION_SLOT_ROLES[MAXIMUM_SCENARIO_TEAM_PLAYERS] = {0, 2, 1, 1, 3, 5,
                                                                         5, 5, 6, 9, 7};

// GRF: `GetDefaultRolePosition`.
inline Vector3 DefaultRolePosition(int32_t role) {
    switch (role) {
        case 0:
        case 1:
            return Vector3((float)-1.0, (float)0.0, 0);
        case 2:
            return Vector3((float)-0.8, (float)0.8, 0);
        case 3:
            return Vector3((float)-0.8, (float)-0.8, 0);
        case 4:
            return Vector3((float)-0.5, (float)0.0, 0);
        case 5:
            return Vector3((float)0.0, (float)0.0, 0);
        case 6:
            return Vector3((float)0.0, (float)1.0, 0);
        case 7:
            return Vector3((float)0.0, (float)-1.0, 0);
        case 8:
            return Vector3((float)0.5, (float)0.0, 0);
        case 9:
            return Vector3((float)1.0, (float)0.0, 0);
        default:
            return Vector3((float)0.0, (float)0.0, 0);
    }
}

// GRF: `clamp`, which returns an in-range value itself and so keeps the sign of a zero.
inline float EngineClamp(float value, float minimum, float maximum) {
    if (minimum > value) return minimum;
    if (maximum < value) return maximum;
    return value;
}

// GRF: the constant that turns the y of a scenario into the y of a formation.
constexpr float FORMATION_Y_SCALE = -2.36f;

// GRF: as `FormationEntry(x, y, ...)` builds it, the position equal to the start position.
inline state::FormationEntry ScenarioEntry(const ScenarioPlayer &player) {
    state::FormationEntry entry{};
    entry.role = player.role;
    entry.position = Vector3((float)player.x, (float)player.y * FORMATION_Y_SCALE, 0);
    entry.startPosition = entry.position;
    entry.lazy = player.lazy;
    entry.controllable = player.controllable;
    return entry;
}

// GRF: `TeamData::TeamData` builds the spots from the fixed slots. The role, start position and
// flags come from the scenario afterwards, so a player's spot follows its index, not its role.
inline std::vector<state::FormationEntry> TeamFormation(const std::vector<ScenarioPlayer> &team) {
    const int count = (int)team.size();
    std::vector<state::FormationEntry> formation(count);
    for (int slot = 0; slot < count; slot++) {
        const Vector3 shape((float)FORMATION_SLOTS[slot][0], (float)FORMATION_SLOTS[slot][1], 0);
        formation[slot].role = FORMATION_SLOT_ROLES[slot];
        formation[slot].position = shape * 0.6f + DefaultRolePosition(formation[slot].role) * 0.4f;
    }
    const float minimumDistance = 0.5f;
    bool changed = true;
    for (int iteration = 0; changed && iteration < 10; iteration++) {
        std::vector<Vector3> offset(count);
        changed = false;
        for (int first = 0; first < count - 1; first++) {
            if (formation[first].role == simulation::e_PlayerRole_Goalkeeper) continue;
            for (int second = first + 1; second < count; second++) {
                if (formation[second].role == simulation::e_PlayerRole_Goalkeeper) continue;
                const Vector3 difference = formation[first].position - formation[second].position;
                if (difference.GetLength() < minimumDistance) {
                    changed = true;
                    const float distanceFactor = 1.0f - (difference.GetLength() / minimumDistance);
                    offset[first] += difference.GetNormalized(Vector3(0, 1, 0)) * minimumDistance *
                                     distanceFactor * 0.5f;
                    offset[second] -= difference.GetNormalized(Vector3(0, 1, 0)) * minimumDistance *
                                      distanceFactor * 0.5f;
                }
            }
        }
        if (changed) {
            for (int slot = 0; slot < count; slot++) {
                formation[slot].position += offset[slot];
                formation[slot].position.coordinates[0] =
                    EngineClamp(formation[slot].position.coordinates[0], -1, 1);
                formation[slot].position.coordinates[1] =
                    EngineClamp(formation[slot].position.coordinates[1], -1, 1);
            }
        }
    }
    for (int slot = 0; slot < count; slot++) {
        const state::FormationEntry scenario = ScenarioEntry(team[slot]);
        formation[slot].startPosition = scenario.startPosition;
        formation[slot].lazy = scenario.lazy;
        formation[slot].role = scenario.role;
        formation[slot].controllable = scenario.controllable;
    }
    return formation;
}

inline void CheckDescription(const ScenarioDescription &description) {
    for (const std::vector<ScenarioPlayer> *team :
         {&description.leftTeam, &description.rightTeam}) {
        const char *side = team == &description.leftTeam ? "left" : "right";
        if (team->empty())
            throw std::invalid_argument(std::string("the ") + side +
                                        " team is empty; the scenario module adds its goalkeeper");
        if ((int)team->size() > MAXIMUM_SCENARIO_TEAM_PLAYERS)
            throw std::invalid_argument(std::string("the ") + side + " team has " +
                                        std::to_string(team->size()) +
                                        " players, more than the 11 that GRF holds");
        if ((int)team->size() > state::MAXIMUM_TEAM_PLAYERS)
            throw std::invalid_argument(std::string("the ") + side + " team has " +
                                        std::to_string(team->size()) + " players, more than the " +
                                        std::to_string(state::MAXIMUM_TEAM_PLAYERS) +
                                        " of this build (CUJU_TEAM_PLAYERS)");
        for (const ScenarioPlayer &player : *team)
            if (player.role < 0 || player.role >= PLAYER_ROLES)
                throw std::invalid_argument("expected a player role from 0 to 9, found " +
                                            std::to_string(player.role));
        // GRF: `TeamAIController::CalculateDynamicRoles` seeks the goalkeeper without advancing its
        // iterator, so it hangs in open play within 400 ms unless each first player is a
        // goalkeeper.
        if (team->front().role != simulation::e_PlayerRole_Goalkeeper)
            throw std::invalid_argument(std::string("the first player of the ") + side +
                                        " team is not a goalkeeper, which hangs GRF "
                                        "once the match is in open play");
    }
}

struct BuiltVariant {
    std::string staticData, state, skeletons;
};

template <class T>
inline void AppendRaw(std::string &out, const T &value) {
    out.append(reinterpret_cast<const char *>(&value), sizeof(T));
}
inline void AppendString(std::string &out, const std::string &text) {
    AppendRaw(out, (int32_t)text.size());
    out += text;
}

inline std::string StaticBytes(const std::vector<state::FormationEntry> &formation,
                               const std::vector<int> &teamIndex, int32_t officialSkin) {
    std::string out;
    const int players = (int)formation.size();
    AppendRaw(out, (int32_t)(players + 3));
    for (int player = 0; player < players + 3; player++) {
        simulation::PlayerData data{};
        if (player < players) {
            const float *row = PLAYER_TABLE[teamIndex[player]];
            for (int statistic = 0; statistic < simulation::PLAYER_STATISTICS; statistic++)
                data.statistics[statistic] = row[statistic];
            data.velocityStatistic = row[simulation::PLAYER_STATISTICS];
            data.height = row[simulation::PLAYER_STATISTICS + 1];
            data.skinColor = PLAYER_SKIN[teamIndex[player]];
        } else {
            for (float &statistic : data.statistics) statistic = OFFICIAL_STATISTIC;
            data.velocityStatistic = OFFICIAL_STATISTIC;
            data.height = OFFICIAL_HEIGHT;
            data.skinColor = officialSkin;
        }
        AppendRaw(out, data);
    }
    out += "FORM";
    for (const state::FormationEntry &entry : formation) {
        AppendRaw(out, entry.role);
        AppendRaw(out, entry.position);
        AppendRaw(out, entry.startPosition);
        AppendRaw(out, (uint8_t)entry.lazy);
        AppendRaw(out, (uint8_t)entry.controllable);
    }
    out += "MDUR";
    AppendRaw(out, MATCH_DURATION_FACTOR);
    out += "SIDS";
    // GRF: stable IDs follow creation order, which is state order, the officials last.
    for (int32_t entry = 0; entry < players + 3; entry++) AppendRaw(out, entry);
    return out;
}

// One skeleton for all, since GRF scales only the rendered mesh by the height, never the skeleton.
inline std::vector<animation::Skeleton> ConstantSkeletons(int humanoids) {
    animation::Skeleton skeleton{};
    for (int bodyPart = 0; bodyPart < state::BODY_PARTS; bodyPart++)
        skeleton.nodeMap[bodyPart] = SKELETON_NODE_MAP[bodyPart];
    skeleton.nodes.resize(state::HUMANOID_NODES);
    for (int node = 0; node < state::HUMANOID_NODES; node++) {
        const SkeletonNodeConstant &constant = SKELETON_NODES[node];
        animation::SkeletonNode &target = skeleton.nodes[node];
        target.parent = constant.parent;
        target.localMode = constant.localMode;
        target.position =
            Vector3(TEMPLATE_NODES[node].position[0], TEMPLATE_NODES[node].position[1],
                    TEMPLATE_NODES[node].position[2]);
        target.rotation =
            Quaternion(TEMPLATE_NODES[node].rotation[0], TEMPLATE_NODES[node].rotation[1],
                       TEMPLATE_NODES[node].rotation[2], TEMPLATE_NODES[node].rotation[3]);
        target.scale = Vector3(1);
        target.geometries.clear();
        if (constant.geometry) {
            animation::SkeletonGeometry geometry{};
            const std::string name = constant.geometry;
            geometry.footOrLowerLeg = name == "left_foot" || name == "right_foot" ||
                                      name == "left_lowerleg" || name == "right_lowerleg";
            geometry.localMode = constant.geometryLocalMode;
            const float *values = constant.geometryValues;
            geometry.position = Vector3(values[0], values[1], values[2]);
            geometry.rotation = Quaternion(values[3], values[4], values[5], values[6]);
            geometry.scale = Vector3(values[7], values[8], values[9]);
            geometry.boundingBoxMinimum = Vector3(values[10], values[11], values[12]);
            geometry.boundingBoxMaximum = Vector3(values[13], values[14], values[15]);
            target.geometries.push_back(geometry);
        }
    }
    return std::vector<animation::Skeleton>(humanoids, skeleton);
}

inline std::string SkeletonBytes(const std::vector<const state::NodeState *> &nodes) {
    std::string out = "CUJHUM1";
    AppendRaw(out, (int32_t)nodes.size());
    for (const state::NodeState *humanoid : nodes) {
        for (int bodyPart = 0; bodyPart < state::BODY_PARTS; bodyPart++)
            AppendRaw(out, SKELETON_NODE_MAP[bodyPart]);
        AppendRaw(out, (int32_t)state::HUMANOID_NODES);
        for (int node = 0; node < state::HUMANOID_NODES; node++) {
            const SkeletonNodeConstant &constant = SKELETON_NODES[node];
            AppendString(out, constant.name);
            AppendRaw(out, constant.parent);
            AppendRaw(out, humanoid[node].localMode);
            AppendRaw(out, humanoid[node].position);
            AppendRaw(out, humanoid[node].rotation);
            AppendRaw(out, humanoid[node].scale);
            AppendRaw(out, (int32_t)(constant.geometry ? 1 : 0));
            if (!constant.geometry) continue;
            AppendString(out, constant.geometry);
            AppendRaw(out, constant.geometryLocalMode);
            for (float value : constant.geometryValues) AppendRaw(out, value);
        }
    }
    return out;
}

// GRF: a humanoid as the `HumanoidBase` constructor leaves it, before `ResetPosition`.
inline state::HumanoidState NewHumanoid() {
    state::HumanoidState humanoid{};
    for (int node = 0; node < state::HUMANOID_NODES; node++) {
        humanoid.nodes[node].position =
            Vector3(TEMPLATE_NODES[node].position[0], TEMPLATE_NODES[node].position[1],
                    TEMPLATE_NODES[node].position[2]);
        humanoid.nodes[node].rotation =
            Quaternion(TEMPLATE_NODES[node].rotation[0], TEMPLATE_NODES[node].rotation[1],
                       TEMPLATE_NODES[node].rotation[2], TEMPLATE_NODES[node].rotation[3]);
        humanoid.nodes[node].scale = Vector3(1);
        humanoid.nodes[node].localMode = TEMPLATE_NODES[node].localMode;
    }
    // GRF: a default `PlayerCommand` selected the idle animation.
    state::PlayerCommand &command = humanoid.currentAnimation.originatingCommand;
    command.desiredFunctionType = simulation::e_FunctionType_Movement;
    command.strictMovement = simulation::e_StrictMovement_Dynamic;
    command.touchInformation.targetPlayer = -1;
    command.touchInformation.forcedTargetPlayer = -1;
    command.tripType = 1;
    return humanoid;
}

// GRF: the shared part of a player controller after `PlayerController::Reset` and `SetPlayer`.
inline state::PlayerControllerState NewPlayerController(int player, int teamSlot) {
    state::PlayerControllerState controller{};
    controller.player = player;
    controller.inputDirection = Vector3(0, -1, 0);
    controller.team = teamSlot;
    controller.opponentTeam = 1 - teamSlot;
    controller.opponentPlayer = -1;
    // GRF: `lastSwitchTime_ms = -10000` in an unsigned 64-bit field.
    controller.lastSwitchTimeMilliseconds = (uint64_t)(int64_t)-10000;
    controller.possessionAmount = 0.9f;
    controller.teamPossessionAmount = 1.0f;
    controller.fadingTeamPossessionAmount = 1.0f;
    controller.opponentTimeNeededToGetToBall = 100;
    return controller;
}

inline state::PropertiesState Tactics(const double (&values)[state::TACTIC_KEYS]) {
    state::PropertiesState tactics{};
    tactics.present = (1u << state::TACTIC_KEYS) - 1;
    for (int tactic = 0; tactic < state::TACTIC_KEYS; tactic++)
        tactics.value[tactic] = (float)values[tactic];
    return tactics;
}

struct BuilderTables {
    std::unique_ptr<animation::AnimationStore> store;
    std::vector<animation::AnimationDerived> derived;
};

inline BuilderTables LoadBuilderTables(const std::string &animationBytes) {
    BuilderTables tables;
    tables.store = animation::ParseAnimationTables(animationBytes);
    tables.derived = animation::DeriveAll(*tables.store);
    return tables;
}

// GRF: the template is the state that `GameEnv::reset` leaves before the ticks that bring the match
// into play. Each field is set as the GRF constructors named below set it.
inline BuiltVariant BuildVariant(const ScenarioDescription &description, bool reverse,
                                 const BuilderTables &tables) {
    CheckDescription(description);
    using simulation::World;
    auto world = std::make_unique<World>();
    state::EnvironmentStateFull &environment = world->serialized;
    state::ScenarioState &scenario = environment.scenario;
    state::MatchState &match = environment.match;

    // 1000 + reverse is the seed of the shipped bundles. Every episode reseeds anyway.
    const uint32_t seed =
        description.deterministic ? description.gameEngineRandomSeed : 1000u + (reverse ? 1u : 0u);

    // GRF: `GameEnv::reset` runs in the state `game_running` and resets the step to -1.
    environment.gameState = simulation::e_GameState_Running;
    environment.waitingForGameCount = 0;
    environment.step = -1;
    // GRF: the generator draws from `uniform_real<float>(0, 1)`, and the seed sets the rest.
    const float distribution[2] = {0.0f, 1.0f};
    memcpy(environment.rng + 2504, distribution, sizeof(distribution));

    // GRF: `GameEnv::setConfig` scales the ball position in double.
    scenario.ballPosition = Vector3((float)((double)(float)description.ballX * 54.4),
                                    (float)((double)(float)description.ballY * -83.6), 0);
    scenario.leftTeamSize = (int32_t)description.leftTeam.size();
    scenario.rightTeamSize = (int32_t)description.rightTeam.size();
    scenario.leftAgents = scenario.rightAgents = 0;
    scenario.useMagnet = description.useMagnet;
    scenario.offsides = description.offsides;
    scenario.leftTeamDifficulty = (float)description.leftTeamDifficulty;
    scenario.rightTeamDifficulty = (float)description.rightTeamDifficulty;
    scenario.deterministic = description.deterministic;
    scenario.endEpisodeOnScore = description.endEpisodeOnScore;
    scenario.endEpisodeOnPossessionChange = description.endEpisodeOnPossessionChange;
    scenario.endEpisodeOnOutOfPlay = description.endEpisodeOnOutOfPlay;
    scenario.gameDuration = description.gameDuration;
    scenario.secondHalf = description.secondHalf;
    scenario.controlAllPlayers = description.controlAllPlayers;
    scenario.realTime = description.realTime;
    scenario.gameEngineRandomSeed = seed;
    scenario.reverseTeamProcessing = reverse;
    for (int player = 0; player < scenario.leftTeamSize; player++)
        scenario.leftTeam[player] = ScenarioEntry(description.leftTeam[player]);
    for (int player = 0; player < scenario.rightTeamSize; player++)
        scenario.rightTeam[player] = ScenarioEntry(description.rightTeam[player]);

    // Slot 0 holds the team processed first, the right team when reversed.
    match.firstTeam = 0;
    match.secondTeam = 1;
    match.playerCount = scenario.leftTeamSize + scenario.rightTeamSize;
    const int firstTeamID = reverse ? 1 : 0;
    const std::vector<ScenarioPlayer> *teams[2] = {&description.leftTeam, &description.rightTeam};
    std::vector<state::FormationEntry> formation;
    std::vector<int> teamIndex;
    for (int slot = 0; slot < 2; slot++) {
        const std::vector<state::FormationEntry> entries =
            TeamFormation(*teams[slot == 0 ? firstTeamID : 1 - firstTeamID]);
        for (int index = 0; index < (int)entries.size(); index++) {
            formation.push_back(entries[index]);
            teamIndex.push_back(index);
        }
    }

    // GRF: as `Player::Player` and `Player::Activate` leave them, first team first.
    const int firstTeamSize = (int)teams[firstTeamID]->size();
    for (int player = 0; player < match.playerCount; player++) {
        const int slot = player < firstTeamSize ? 0 : 1;
        state::PlayerState &entry = match.players[player];
        entry.base.isActive = true;
        entry.base.humanoid = NewHumanoid();
        entry.base.controller.base = NewPlayerController(player, slot);
        entry.base.externalController = -1;
        entry.base.lastTouchType = simulation::e_TouchType_None;
        entry.base.fatigueFactorInverse = 1.0f;
        entry.manMarking = -1;
        entry.dynamicFormationEntry = formation[player];
        // GRF: `Player::ResetSituation`, through `Activate`.
        entry.timeNeededToGetToBallMilliseconds = 1000;
        entry.timeNeededToGetToBallOptimisticMilliseconds = 1000;
        entry.timeNeededToGetToBallPreviousMilliseconds = 1000;
    }
    for (state::PlayerBaseState<state::NoController> &official : match.officials) {
        official.isActive = true;
        official.humanoid = NewHumanoid();
        official.externalController = -1;
        official.lastTouchType = simulation::e_TouchType_None;
        official.fatigueFactorInverse = 1.0f;
    }
    // GRF: after the `Team` and `TeamAIController` constructors.
    for (state::TeamState &team : match.teams) {
        team.timeNeededToGetToBallMilliseconds = 100;
        team.teamPossessionAmount = 1.0f;
        team.fadingTeamPossessionAmount = 1.0f;
        state::TeamAIState &ai = team.teamController;
        ai.taker = -1;
        ai.setPieceType = simulation::e_GameMode_Normal;
        ai.baseTeamTactics = Tactics(BASE_TACTICS);
        ai.liveTeamTactics = Tactics(LIVE_TACTICS);
        ai.offensivenessBias = 0.5f;
        ai.depth = 0.45f;
        ai.width = 0.95f;
        ai.attackingRunPlayer = ai.teamPressurePlayer = ai.forwardSupportPlayer = -1;
        team.lastTouchPlayer = -1;
        team.mainSelectedPlayer = -1;
    }
    // The possession time is stored negated when the right team comes first, so it starts at -0.
    match.goalCount[0] = match.goalCount[1] = 0;
    match.possessionLastMinute = reverse ? -0.0f : 0.0f;
    // GRF: the 22 keyboards, disabled and unmirrored, since no human gamer owns one.
    match.freeControllerCount = 2 * simulation::MAXIMUM_TEAM_CONTROLLERS;
    for (int keyboard = 0; keyboard < match.freeControllerCount; keyboard++) {
        state::KeyboardState &device = match.freeControllers[keyboard];
        device.mirror = 1.0f;
        device.direction = Vector3(0);
        device.disabled = true;
        for (bool &button : device.buttonsPressed) button = false;
    }
    // GRF: `Ball::Ball` predicts the flight of a ball at rest at the origin.
    BallCalculatePrediction(match.ball, false);
    match.lastGoalTeam = match.lastGoalScorer = -1;
    for (int32_t &teamID : match.lastTouchTeamIDs) teamID = -1;
    match.lastTouchTeamID = -1;
    match.bestPossessionTeam = -1;
    match.designatedPossessionPlayer = 0;
    match.ballRetainer = -1;
    match.possessionSideHistoryMaximumTimeMilliseconds = 6000;
    match.autoUpdateInGameCamera = true;
    // The referee prepares the kickoff of the first team, in the frame of the rest.
    state::RefereeState &referee = match.referee;
    referee.buffer.active = true;
    referee.buffer.desiredSetPiece = simulation::e_GameMode_KickOff;
    referee.buffer.teamID = firstTeamID;
    referee.buffer.setPieceTeam = 0;
    referee.buffer.startTime = 2000;
    referee.buffer.restartPosition = scenario.ballPosition;
    referee.buffer.taker = -1;
    referee.buffer.endPhase = true;
    referee.foul.foulPlayer = referee.foul.foulVictim = -1;
    referee.foul.hasBeenProcessed = true;

    const int humanoids = match.playerCount + 3;
    uint8_t skinGenerator[RNG_BYTES];
    memcpy(skinGenerator, environment.rng, RNG_BYTES);
    RNGView{skinGenerator}.Seed(seed);
    // GRF: each player resets its position twice before the officials draw their skin.
    for (int draw = 0; draw < 2 * match.playerCount; draw++) RandomInRange(skinGenerator, 0, 1);
    const int32_t officialSkin = (int32_t)std::round(RandomInRange(skinGenerator, 1, 4));
    BuiltVariant variant;
    variant.staticData = StaticBytes(formation, teamIndex, officialSkin);
    const simulation::StaticData staticData = simulation::ParseStatic(variant.staticData);
    const std::vector<animation::Skeleton> skeletons = ConstantSkeletons(humanoids);
    simulation::SimulationContext context{&staticData, &tables.store->view, tables.derived.data(),
                                          skeletons.data(), nullptr};
    context.ticksPerStep = 10;

    // GRF: the creation draws the idle frames, the skin of the officials and the lighting.
    simulation::InitializePlayerTeams(*world);
    if (!simulation::EnvironmentResetFromTemplate(*world, context, seed))
        throw std::runtime_error("the simulation does not support the creation of this scenario");
    // GRF: `Team::AddHumanGamers` picks the player of each team closest to the ball at the origin.
    for (int teamID = 0; teamID < 2; teamID++) {
        const Vector3 ball = simulation::BallPredict(match.ball, 0).Get2D();
        simulation::TeamByID(*world, teamID).designatedTeamPossessionPlayer =
            simulation::GetClosestPlayer(*world, context, teamID, ball, false);
    }

    simulation::StateCodec codec;
    codec.freeControllerCount = match.freeControllerCount;
    variant.state = codec.SaveFrame(*world, 0);

    // The skeletons hold the pose after the ticks into play. The simulation reads the state
    // instead.
    auto played = std::make_unique<World>(*world);
    simulation::EnvironmentCore core;
    if (!simulation::EnvironmentCoreAfterReset(*played, context, core))
        throw std::runtime_error(
            "the simulation does not support the ticks that bring this scenario into play");
    std::vector<const state::NodeState *> nodes;
    for (int player = 0; player < match.playerCount; player++)
        nodes.push_back(played->serialized.match.players[player].base.humanoid.nodes);
    for (int official = 0; official < 3; official++)
        nodes.push_back(played->serialized.match.officials[official].humanoid.nodes);
    variant.skeletons = SkeletonBytes(nodes);
    return variant;
}

// The four variants of an agentless `l0_r0` bundle, as `LoadAssets` reads it. The loader adds the
// agents. GRF: the first episode, `parity[0]`, is even, as GRF's second scenario build is.
inline std::string BuildBundle(const std::string &name, const ScenarioDescription (&parity)[2],
                               const BuilderTables &tables) {
    if (parity[0].leftTeam.size() != parity[1].leftTeam.size() ||
        parity[0].rightTeam.size() != parity[1].rightTeam.size())
        throw std::invalid_argument(
            "the team sizes depend on the parity of the episode, which a bundle does not support");
    std::string out = "CUJASSET1";
    AppendString(out, name);
    AppendRaw(out, (int32_t)0);
    AppendRaw(out, (int32_t)0);
    AppendRaw(out, (int32_t)4);
    for (int reverse = 0; reverse < 2; reverse++)
        for (int episodeParity = 0; episodeParity < 2; episodeParity++) {
            const BuiltVariant variant = BuildVariant(parity[episodeParity], reverse != 0, tables);
            AppendRaw(out, (int32_t)reverse);
            AppendRaw(out, (int32_t)episodeParity);
            for (const std::string *part :
                 {&variant.staticData, &variant.state, &variant.skeletons}) {
                AppendRaw(out, (uint32_t)part->size());
                out += *part;
            }
        }
    return out;
}

}  // namespace assets
}  // namespace cuju
